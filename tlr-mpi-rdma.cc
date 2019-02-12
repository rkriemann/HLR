//
// Project     : HLib
// File        : tlr-mpi.cc
// Description : TLR arithmetic with MPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <fstream>

using namespace std;

#include "mpi.hh"

#include "cmdline.inc"
#include "problem.inc"
#include "tlr.hh"
#include "tlr.inc"
#include "tlr-mpi.inc"
#include "tensor.hh"
#include "RedirectOutput.hh"

#include "parallel/TDistrBC.hh"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace MPI
{

template < typename value_t >
mpi::request
rget ( mpi::window &                    win,
       HLIB::BLAS::Matrix< value_t > &  M,
       const int                        root )
{
    mpi::request  req;
    const size_t  count = M.nrows() * M.ncols() * sizeof(value_t);
    
    MPI_CHECK_RESULT( MPI_Rget,
                      ( M.data(), count, MPI_BYTE, // local destination buffer
                        root,                      // from which node
                        0, count, MPI_BYTE,        // offset in root buffer
                        MPI_Win( win ), & req.mpi_request ) );

    return req;
}

template < typename value_t >
std::vector< mpi::window >
setup_rdma ( mpi::communicator &  comm,
             TMatrix *            A )
{
    std::vector< mpi::window >  wins;
    
    if ( is_dense( A ) )
    {
        wins.reserve( 1 );
        wins[0] = mpi::window( comm, blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ).data(), A->nrows() * A->ncols() );
        wins[0].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = ptrcast( A, TRkMatrix );
        
        R->set_rank( RANK );
        
        wins.reserve( 2 );
        wins[0] = mpi::window( comm, blas_mat_A< value_t >( R ).data(), A->nrows() * RANK );
        wins[0].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
        wins[1] = mpi::window( comm, blas_mat_B< value_t >( R ).data(), A->ncols() * RANK );
        wins[1].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
    }// if
    else
        HERROR( ERR_MAT_TYPE, "setup_rdma", "unsupported matrix type" );

    return wins;
}

template < typename value_t >
std::vector< mpi::request >
request_rdma ( std::vector< mpi::window > &  wins,
               TMatrix *                     A,
               const int                     root )
{
    std::vector< mpi::request >  reqs;

    reqs.reserve( wins.size() );

    if ( is_dense( A ) )
    {
        reqs.push_back( rget( wins[0], blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ), root ) );
    }// if
    else if ( is_lowrank( A ) )
    {
        reqs.push_back( rget( wins[0], blas_mat_A< value_t >( ptrcast( A, TRkMatrix ) ), root ) );
        reqs.push_back( rget( wins[1], blas_mat_B< value_t >( ptrcast( A, TRkMatrix ) ), root ) );
    }// if
    else
        HERROR( ERR_MAT_TYPE, "request_rdma", "unsupported matrix type" );

    return reqs;
}

void
wait_rdma ( std::vector< mpi::request > &  reqs )
{
    std::vector< MPI_Request >  requests;
    
    requests.reserve( reqs.size() );
    
    for ( auto  req : reqs )
        requests.push_back( MPI_Request(req) );

    MPI_CHECK_RESULT( MPI_Waitall,
                      ( reqs.size(), & requests[0], MPI_STATUSES_IGNORE ) );
}

void
finish_rdma( std::vector< mpi::window > &  wins )
{
    for ( auto &  win : wins )
        win.fence( MPI_MODE_NOSUCCEED );
}

size_t  max_add_mem = 0;

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    assert( RANK != 0 );
    
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

    auto  nbr = A->nblock_rows();
    auto  nbc = A->nblock_cols();

    // setup global types (IMPORTANT!!!)
    type_dense = RTTI::type_to_id( "TDenseMatrix" );
    type_lr    = RTTI::type_to_id( "TRkMatrix" );
    type_ghost = RTTI::type_to_id( "TGhostMatrix" );
    
    //
    // set up global map of matrix types in A
    //

    const auto  mat_types = build_type_matrix( A );

    //
    // set up communicators for rows/cols
    //

    vector< mpi::communicator >          row_comms( nbr ), col_comms( nbc );  // communicators for rows/columns
    vector< list< int > >                row_procs( nbr ), col_procs( nbc );  // set of processors for rows/columns
    vector< unordered_map< int, int > >  row_maps( nbr ),  col_maps( nbc );   // mapping of global ranks to row/column ranks

    build_row_comms( A, row_comms, row_procs, row_maps );
    build_col_comms( A, col_comms, col_procs, col_maps );

    //
    // LU factorization
    //

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        // DBG::print(  "────────────────────────────────────────────────" );
        // DBG::printf( "step %d", i );
        
        auto  A_ii = ptrcast( A->block( i, i ), TDenseMatrix );
        auto  p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            // DBG::printf( "invert( %d )", A_ii->id() );
            B::invert( blas_mat< value_t >( A_ii ) );
        }// if

        if ( contains( col_procs[i], pid ) );
        {
            //
            // set up RDMA for diagonal block
            //

            unique_ptr< TMatrix >   T_ii;        // temporary storage with auto-delete
            TMatrix *               H_ii = A_ii; // handle for A_ii/T_ii
            vector< mpi::window >   diag_wins;
            vector< mpi::request >  diag_reqs;
            
            if ( pid != p_ii )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if

            // set up windows/attached memory 
            diag_wins = setup_rdma< value_t >( col_comms[i], H_ii );

            // and start get requests
            if ( pid != p_ii )
                diag_reqs = request_rdma< value_t >( diag_wins, H_ii, col_maps[i][p_ii] );
            
            //
            // off-diagonal solve
            //

            bool  have_diag = ( pid == p_ii );
            
            for ( uint  j = i+1; j < nbr; ++j )
            {
                // L is unit diagonal !!! Only solve with U
                auto        A_ji = A->block( j, i );
                const auto  p_ji = A_ji->procs().master();
                
                if ( pid == p_ji )
                {
                    if ( ! have_diag )
                    {
                        wait_rdma( diag_reqs );
                        have_diag = true;

                        if ( pid != p_ii )
                            add_mem += H_ii->byte_size();
                    }// if
                    
                    trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A->block( j, i ) );
                }// if
            }// for

            finish_rdma( diag_wins );
        }

        //
        // set up MPI RDMA for row/column matrices
        //

        vector< unique_ptr< TMatrix > >   row_i_mat( nbr );        // for autodeletion of temp. matrices
        vector< unique_ptr< TMatrix > >   col_i_mat( nbc );
        vector< TMatrix * >               row_i( nbr, nullptr );   // actual matrix handles
        vector< TMatrix * >               col_i( nbc, nullptr );
        vector< vector< mpi::window > >   row_wins( nbr );         // holds MPI windows for matrices
        vector< vector< mpi::window > >   col_wins( nbc );
        vector< vector< mpi::request > >  row_reqs( nbr );         // holds MPI RDMA requests for matrices
        vector< vector< mpi::request > >  col_reqs( nbc );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();

            if ( contains( row_procs[j], pid ) )
            {
                row_i[j] = A_ji;
            
                if ( pid != p_ji )
                {
                    row_i_mat[j] = create_matrix( A_ji, mat_types(j,i), pid );
                    row_i[j]     = row_i_mat[j].get();
                }// if
                
                // set up windows/attached memory 
                row_wins[j] = setup_rdma< value_t >( row_comms[j], row_i[j] );

                // and start get requests
                if ( pid != p_ji )
                    row_reqs[j] = request_rdma< value_t >( row_wins[j], row_i[j], row_maps[j][p_ji] );
            }// if
        }// for
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = A->block( i, l );
            const auto  p_il = A_il->procs().master();
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                col_i[l] = A_il;
                
                if ( pid != p_il )
                {
                    col_i_mat[l] = create_matrix( A_il, mat_types(i,l), pid );
                    col_i[l]     = col_i_mat[l].get();
                }// if
                
                // set up windows/attached memory 
                col_wins[l] = setup_rdma< value_t >( col_comms[l], col_i[l] );

                // and start get requests
                if ( pid != p_il )
                    col_reqs[l] = request_rdma< value_t >( col_wins[l], col_i[l], col_maps[l][p_il] );
            }// if
        }// for

        //
        // update of trailing sub-matrix
        //
        
        vector< bool >  row_done( nbr, false );  // signals received matrix
        vector< bool >  col_done( nbc, false );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  p_ji = A->block( j, i )->procs().master();
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const auto  p_il = A->block( i, l )->procs().master();
                
                //
                // update local matrix block
                //
                
                auto        A_jl = A->block( j, l );
                const auto  p_jl = A_jl->procs().master();
                
                if ( pid == p_jl )
                {
                    //
                    // ensure broadcasts fir A_ji and A_il have finished
                    //

                    if (( p_ji != pid ) && ! row_done[j] )
                    {
                        // DBG::printf( "waiting for %d", A->block(j,i)->id() );
                        wait_rdma( row_reqs[j] );
                        row_done[j] = true;
                        add_mem += row_i[j]->byte_size();
                    }// if
                    else

                    if (( p_il != pid ) && ! col_done[l] )
                    {
                        // DBG::printf( "waiting for %d", A->block(i,l)->id() );
                        wait_rdma( col_reqs[l] );
                        col_done[l] = true;
                        add_mem += col_i[l]->byte_size();
                    }// if

                    //
                    // finally compute update
                    //
                    
                    update< value_t >( row_i[j], col_i[l], A_jl, acc );
                }// if
            }// for
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // finish epoch (started in "setup_rdma")
        //

        for ( uint  j = i+1; j < nbr; ++j )
            finish_rdma( row_wins[j] );

        for ( uint  l = i+1; l < nbc; ++l )
            finish_rdma( col_wins[l] );
    }// for

    std::cout << "  time in MPI : " << format( "%.2fs" ) % time_mpi << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( ! is_blocked( A ) )
        HERROR( ERR_ARG, "", "" );

    lu< value_t >( ptrcast( A, TBlockMatrix ), acc );
}

}// namespace MPI

}// namespace TLR

//
// main function
//
void
mymain ( int argc, char ** argv )
{
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();

    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TLR::cluster( coord.get(), ntile );
    
    {
        TBlockCyclicDistrBC  distr;
        
        distr.distribute( nprocs, bct->root(), nullptr );
    }
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "tlrmpi_bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "tlrmpi_bct_distr" );
    }// if
    
    auto  A = problem->build_matrix( bct.get(), fixed_rank( k ) );
    
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "tlrmpi_A_%03d", pid ) );
    }// if
    
    TLR::MPI::RANK = k;
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::MPI::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        std::cout << "    done in " << toc << std::endl;

        // compare with otherwise computed result
        auto  D  = read_matrix( "LU.hm" );
        auto  BC = ptrcast( C.get(), TBlockMatrix );
        auto  BD = ptrcast( D.get(), TBlockMatrix );
        bool  correct = true;
        
        D->set_procs( ps_single( pid ), recursive );
        
        for ( uint i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( ! is_ghost( BC->block( i, j ) ) )
                {
                    const auto  f = diff_norm_F( BD->block( i, j ), BC->block( i, j ) );

                    if ( f > 1e-10 )
                    {
                        DBG::printf( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BC->block( i, j ) ) );
                        correct = false;
                    }// if
                }// if
            }// for
        }// for

        if ( correct )
            std::cout << "    no error" << std::endl;
        
        // TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        // std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }
}

int
main ( int argc, char ** argv )
{
    // init MPI before anything else
    mpi::environment   env{ argc, argv };
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();

    // redirect output for all except proc 0
    unique_ptr< RedirectOutput >  redir_out = ( pid != 0
                                                ? make_unique< RedirectOutput >( to_string( "tlrmpi_%03d.out", pid ) )
                                                : nullptr );

    parse_cmdline( argc, argv );
    
    // {
    //     double  f[4] = { 1.0*(pid+1), 2.0*(pid+1), 3.0*(pid+1), 4.0*(pid+1) };
    //     double  g[4] = { 5.0*(pid+1), 6.0*(pid+1), 7.0*(pid+1), 8.0*(pid+1) };
        
    //     mpi::window  win_f( world, f, 4 );
        
    //     win_f.fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
        
    //     if ( pid == 1 )
    //         win_f.get( g, 4,   // destination buffer
    //                    0,      // from process
    //                    0, 4 ); // position in from buffer
    //     else
    //         win_f.get( g, 4,   // destination buffer
    //                    1,      // from process
    //                    0, 4 ); // position in from buffer
        
    //     win_f.fence( MPI_MODE_NOSUCCEED );
        
    //     std::cout << g[0] << g[1] << g[2] << g[3] << std::endl;
    // }
    
    // {
    //     double  f[4] = { 1.0*(pid+1), 2.0*(pid+1), 3.0*(pid+1), 4.0*(pid+1) };
    //     double  g[4] = { 5.0*(pid+1), 6.0*(pid+1), 7.0*(pid+1), 8.0*(pid+1) };

    //     std::cout << f[0] << f[1] << f[2] << f[3] << std::endl;
    //     // std::cout << g[0] << g[1] << g[2] << g[3] << std::endl;
    //     std::cout << std::endl;
        
    //     MPI_Win  win_f;

    //     MPI_Win_create( f, 4*sizeof(double), 1, MPI_INFO_NULL, MPI_Comm(world), & win_f );

    //     MPI_Win_fence( ( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE ), win_f ); 

    //     if ( pid == 1 )
    //         MPI_Get( g, 4*sizeof(double), MPI_BYTE,          // local destination
    //                  0, // from
    //                  0, 4*sizeof(double), MPI_BYTE, win_f ); // from
    //     else
    //         MPI_Get( g, 4*sizeof(double), MPI_BYTE,          // local destination
    //                  1, // from
    //                  0, 4*sizeof(double), MPI_BYTE, win_f ); // from
                 
    //     MPI_Win_fence( MPI_MODE_NOSUCCEED, win_f );
        
    //     // std::cout << f[0] << f[1] << f[2] << f[3] << std::endl;
    //     std::cout << g[0] << g[1] << g[2] << g[3] << std::endl;
    // }
    
    try
    {
        INIT();

        // adjust HLIB network data
        NET::set_nprocs( nprocs );
        NET::set_pid( pid );
    
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        mymain( argc, argv );

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }

    return 0;
}
