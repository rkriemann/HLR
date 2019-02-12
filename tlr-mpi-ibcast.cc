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

// #include <boost/mpi.hpp>

// namespace mpi = boost::mpi;

#include "mpi.hh"

#include "cmdline.inc"
#include "problem.inc"
#include "tlr.hh"
#include "tlr.inc"
#include "tlr-mpi.inc"
#include "tensor.hh"
#include "RedirectOutput.hh"

#include "parallel/TDistrBC.hh"

namespace TLR
{

namespace MPI
{

template < typename value_t >
std::vector< mpi::request >
ibroadcast ( mpi::communicator &  comm,
             TMatrix *            A,
             const int            root_proc )
{
    // DBG::printf( "broadcast( %d ) from %d", A->id(), root_proc );

    std::vector< mpi::request >  reqs;
    
    auto  tic = Time::Wall::now();
    
    if ( is_dense( A ) )
    {
        auto  D = ptrcast( A, TDenseMatrix );
        
        reqs = { comm.ibroadcast( blas_mat< value_t >( D ).data(), D->nrows() * D->ncols(), root_proc ) };
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = ptrcast( A, TRkMatrix );
        auto  k = R->rank();

        R->set_rank( RANK );
        
        reqs.reserve( 2 );
        
        // reqs.push_back( comm.ibroadcast( k, root_proc ) );
        // std::cout << "ibcast : " << reqs[0] << std::endl;

        reqs.push_back( comm.ibroadcast( blas_mat_A< value_t >( R ).data(), R->nrows() * RANK, root_proc ) );
        reqs.push_back( comm.ibroadcast( blas_mat_B< value_t >( R ).data(), R->ncols() * RANK, root_proc ) );
    }// if
    else
        HERROR( ERR_MAT_TYPE, "ibroadcast", "" );

    return reqs;
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

    // std::cout << "exchanging matrix types" << std::endl;
    
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

        if ( contains( col_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            unique_ptr< TMatrix >  T_ii;        // temporary storage with auto-delete
            TMatrix *              H_ii = A_ii; // handle for A_ii/T_ii
            
            if ( pid != p_ii )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if
            
            // DBG::printf( "broadcast( %d ) from %d", A_ii->id(), p_ii );
            auto  diag_reqs = ibroadcast< value_t >( col_comms[i], H_ii, col_maps[i][p_ii] );

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
                        wait_all( diag_reqs );
                        have_diag = true;

                        if ( pid != p_ii )
                            add_mem += H_ii->byte_size();
                    }// if
                    
                    // DBG::printf( "solve_U( %d, %d )", H_ii->id(), A->block( j, i )->id() );
                    trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A->block( j, i ) );
                }// if
            }// for
        }
        
        //
        // broadcast blocks in row/column for update phase
        //
        
        vector< unique_ptr< TMatrix > >   row_i_mat( nbr );        // for autodeletion of temp. matrices
        vector< unique_ptr< TMatrix > >   col_i_mat( nbc );
        vector< TMatrix * >               row_i( nbr, nullptr );   // actual matrix handles
        vector< TMatrix * >               col_i( nbc, nullptr );
        vector< vector< mpi::request > >  row_reqs( nbr );         // holds MPI requests for matrices
        vector< vector< mpi::request > >  col_reqs( nbc );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                row_i[j] = A_ji;
                
                if ( pid != p_ji )
                {
                    row_i_mat[j] = create_matrix( A_ji, mat_types(j,i), pid );
                    row_i[j]     = row_i_mat[j].get();
                }// if
                
                // DBG::printf( "broadcasting %d from %d", A_ji->id(), p_ji );
                row_reqs[j] = ibroadcast< value_t >( row_comms[j], row_i[j], row_maps[j][p_ji] );
            }// if
        }
        
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
                
                // DBG::printf( "broadcasting %d from %d", A_il->id(), p_il );
                col_reqs[l] = ibroadcast< value_t >( col_comms[l], col_i[l], col_maps[l][p_il] );
            }// if
        }// for
        
        // for ( uint  j = i+1; j < nbr; ++j )
        // {
        //     DBG::printf( "in row %d waiting for %d requests", j, row_reqs[j].size() );
        //     for ( auto  req : row_reqs[j] )
        //         mpi::wait( req );
        // }// for
        
        // for ( uint  l = i+1; l < nbc; ++l )
        // {
        //     DBG::printf( "in column %d waiting for %d requests", l, col_reqs[l].size() );
        //     for ( auto  req : col_reqs[l] )
        //         mpi::wait( req );
        // }// for
            
        // continue;
        
        //
        // update of trailing sub-matrix
        //
        
        vector< bool >  row_done( nbr, false );  // signals finished broadcast
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
                        mpi::wait_all( row_reqs[j] );
                        row_done[j] = true;
                        add_mem    += row_i[j]->byte_size();
                    }// if
                    else

                    if (( p_il != pid ) && ! col_done[l] )
                    {
                        // DBG::printf( "waiting for %d", A->block(i,l)->id() );
                        mpi::wait_all( col_reqs[l] );
                        col_done[l] = true;
                        add_mem    += col_i[l]->byte_size();
                    }// if

                    //
                    // finally compute update
                    //
                    
                    update< value_t >( row_i[j], col_i[l], A_jl, acc );
                }// if
            }// for
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );
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

std::vector< mpi::request >
test_bcast ( mpi::communicator &  comm,
             double *  data1,
             double *  data2,
             int       size,
             int       root )
{
    auto  req1 = comm.ibroadcast( data1, size, root );
    auto  req2 = comm.ibroadcast( data2, size, root );

    return { req1, req2 };
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

    // for ( int  i = 0; i < 10; ++i )
    // {
    //     std::cout << i << std::endl;
        
    //     double  f[4] = { 1.0*(pid+1), 2.0*(pid+1), 3.0*(pid+1), 4.0*(pid+1) };
    //     double  g[4] = { 5.0*(pid+1), 6.0*(pid+1), 7.0*(pid+1), 8.0*(pid+1) };

    //     std::cout << f[0] << f[1] << f[2] << f[3] << std::endl;
    //     std::cout << g[0] << g[1] << g[2] << g[3] << std::endl;

    //     // auto  f_req = test_bcast( world, f, 4, 0 );
    //     // auto  g_req = test_bcast( world, g, 4, 1 );

    //     std::vector< mpi::request >  reqs = test_bcast( world, f, g, 4, 0 );
        
    //     mpi::wait_all( reqs );

    //     std::cout << f[0] << f[1] << f[2] << f[3] << std::endl;
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
