//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : TLR arithmetic with TBB+MPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <fstream>

using namespace std;

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

// #include <boost/mpi.hpp>

// namespace mpi = boost::mpi;

#include "mpi.hh"

#include "cmdline.inc"
#include "problem.inc"
#include "tlr.hh"
#include "tlr.inc"
#include "tensor.hh"
#include "RedirectOutput.hh"

#include "parallel/TDistrBC.hh"

namespace
{

//
// simplifies test if <val> is in <cont>
//
template < typename T_container >
bool
contains ( T_container const &                    cont,
           typename T_container::const_reference  val )
{
    for ( const auto &  c : cont )
    {
        if ( c == val )
            return true;
    }// for

    return false;
    
    return std::find( cont.begin(), cont.end(), val ) != cont.end();
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace MPI
{

typeid_t  type_dense, type_lr, type_ghost;

std::unique_ptr< TMatrix >
create_matrix ( const TMatrix *  A,
                const typeid_t   type,
                const int        proc )
{
    std::unique_ptr< TMatrix >  T;
    
    if ( type == type_dense )
        T = make_unique< TDenseMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    else if ( type == type_lr )
        T = make_unique< TRkMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    else
        HERROR( ERR_MAT_TYPE, "create_matrix", "" );

    T->set_id( A->id() );
    T->set_procs( ps_single( proc ) );
    
    return T;
}

double  time_mpi = 0;

template < typename value_t >
void
broadcast ( mpi::communicator &  comm,
            TMatrix *            A,
            const int            root_proc )
{
    // DBG::printf( "broadcast( %d ) from %d", A->id(), root_proc );

    auto  tic = Time::Wall::now();
    
    if ( is_dense( A ) )
    {
        auto  D = ptrcast( A, TDenseMatrix );
        
        mpi::broadcast( comm, blas_mat< value_t >( D ).data(), D->nrows() * D->ncols(), root_proc );
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = ptrcast( A, TRkMatrix );
        auto  k = R->rank();
        
        mpi::broadcast( comm, k, root_proc );

        R->set_rank( k );
        
        mpi::broadcast( comm, blas_mat_A< value_t >( R ).data(), R->nrows() * k, root_proc );
        mpi::broadcast( comm, blas_mat_B< value_t >( R ).data(), R->ncols() * k, root_proc );
    }// if
    else
        HERROR( ERR_MAT_TYPE, "broadcast", "" );

    auto  toc = Time::Wall::since( tic );

    time_mpi += toc.seconds();
}

tensor2< typeid_t >
build_type_matrix ( const TBlockMatrix *  A )
{
    const auto           nbr = A->nblock_rows();
    const auto           nbc = A->nblock_cols();
    tensor2< typeid_t >  mat_types( nbr, nbc );
    mpi::communicator    world;
    const auto           nprocs = world.size();

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            mat_types(i,j) = A->block( i, j )->type();
    
    for ( uint  p = 0; p < nprocs; ++p )
    {
        tensor2< typeid_t >  rem_types( mat_types );

        mpi::broadcast( world, & rem_types(0,0), nbr * nbc, p );
        
        for ( uint  i = 0; i < nbr; ++i )
            for ( uint  j = 0; j < nbc; ++j )
                if ( rem_types(i,j) != type_ghost )
                    mat_types(i,j) = rem_types(i,j);
    }// for

    // for ( uint  i = 0; i < nbr; ++i )
    // {
    //     for ( uint  j = 0; j < nbc; ++j )
    //         std::cout << RTTI::id_to_type( mat_types(i,j) ) << "  ";
    //     std::cout << std::endl;
    // }// for

    return mat_types;
}

void
build_row_comms ( const TBlockMatrix *                   A,
                  vector< mpi::communicator > &          row_comms,
                  vector< list< int > > &                row_procs,
                  vector< unordered_map< int, int > > &  row_maps )
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        list< int >  procs;
        
        for ( uint  j = 0; j < nbc; ++j )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbr;
            
        for ( uint  l = 0; l < i; ++l )
        {
            if ( procs == row_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        // std::cout << i << " : ";
        // for ( auto  p : procs ) std::cout << p << ", ";
        // std::cout << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( pos < nbr )
        {
            row_comms[i] = row_comms[pos];
            row_maps[i]  = row_maps[pos];
        }// if
        else
        {
            row_comms[i] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                row_maps[i][p] = comm_rank++;
        }// else
            
        row_procs[i] = std::move( procs );
    }// for
}

void
build_col_comms ( const TBlockMatrix *                   A,
                  vector< mpi::communicator > &          col_comms,
                  vector< list< int > > &                col_procs,
                  vector< unordered_map< int, int > > &  col_maps )
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();

    for ( uint  j = 0; j < nbc; ++j )
    {
        list< int >  procs;
            
        for ( uint  i = 0; i < nbr; ++i )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbc;
            
        for ( uint  l = 0; l < j; ++l )
        {
            if ( procs == col_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for
            
        // std::cout << j << " : ";
        // for ( auto  p : procs ) std::cout << p << ", ";
        // std::cout << " (" << ( pos == nbc ? j : pos ) << ")" << std::endl;

        // use previously created communicator or create new if none found
        if ( pos < nbc )
        {
            col_comms[j] = col_comms[pos];
            col_maps[j]  = col_maps[pos];
        }// if
        else
        {
            col_comms[j] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                col_maps[j][p] = comm_rank++;
        }// else
            
        col_procs[j] = std::move( procs );
    }// for
}

size_t  max_add_mem = 0;

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
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
        // DBG::print(  "────────────────────────────────────────────────" );
        // DBG::printf( "step %d", i );
        
        auto  A_ii = ptrcast( A->block( i, i ), TDenseMatrix );
        auto  p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            // DBG::printf( "invert( %d )", A_ii->id() );
            B::invert( blas_mat< value_t >( A_ii ) );
        }// if

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
            
            if ( contains( col_procs[i], pid ) )
            {
                // DBG::printf( "broadcast( %d ) from %d", A_ii->id(), p_ii );
                broadcast< value_t >( col_comms[i], H_ii, col_maps[i][p_ii] );
            }// if
            
            //
            // off-diagonal solve
            //
            
            tbb::parallel_for( i+1, nbr,
                               [A,H_ii,i,pid] ( uint  j )
                               // for ( uint  j = i+1; j < nbr; ++j )
                               {
                                   // L is unit diagonal !!! Only solve with U
                                   auto        A_ji = A->block( j, i );
                                   const auto  p_ji = A_ji->procs().master();
                                   
                                   if ( pid == p_ji )
                                   {
                                       // DBG::printf( "solve_U( %d, %d )", H_ii->id(), A->block( j, i )->id() );
                                       trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A->block( j, i ) );
                                   }// if
                               } );
        }
        
        #if 1
        //
        // broadcast blocks in row/column for update phase
        //

        vector< unique_ptr< TMatrix > >  row_i_mat( nbr ); // for autodeletion
        vector< unique_ptr< TMatrix > >  col_i_mat( nbc );
        vector< TMatrix * >              row_i( nbr, nullptr ); // matrix handles
        vector< TMatrix * >              col_i( nbc, nullptr );
        size_t                           add_mem = 0;
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                if ( pid != p_ji )
                {
                    row_i_mat[j] = create_matrix( A_ji, mat_types(j,i), pid );
                    row_i[j]     = row_i_mat[j].get();
                }// if
                else
                    row_i[j]     = A_ji;
                
                broadcast< value_t >( row_comms[j], row_i[j], row_maps[j][p_ji] );

                if ( pid != p_ji )
                    add_mem += row_i[j]->byte_size();
            }// if
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = A->block( i, l );
            const auto  p_il = A_il->procs().master();
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                if ( pid != p_il )
                {
                    col_i_mat[l] = create_matrix( A_il, mat_types(i,l), pid );
                    col_i[l]     = col_i_mat[l].get();
                }// if
                else
                    col_i[l]     = A_il;
                
                broadcast< value_t >( col_comms[l], col_i[l], col_maps[l][p_il] );

                if ( pid != p_il )
                    add_mem += col_i[l]->byte_size();
            }// if
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // update of trailing sub-matrix
        //
        
        tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
                                                         i+1, nbc ),
                           [A,i,pid,&row_i,&col_i,&acc] ( const tbb::blocked_range2d< uint > & r )
                           {
                               for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                               {
                                   const auto  A_ji = row_i[j];
                                   
                                   for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                   {
                                       const auto  A_il = col_i[l];
                                       
                                       //
                                       // update local matrix block
                                       //
                
                                       auto        A_jl = A->block( j, l );
                                       const auto  p_jl = A_jl->procs().master();
                
                                       if ( pid == p_jl )
                                           update< value_t >( A_ji, A_il, A_jl, acc );
                                   }// for
                               }// for
                           } );
        
        #else
        
        //
        // update of trailing sub-matrix
        //
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto             A_ji = A->block( j, i );
            unique_ptr< TMatrix >  T_ji;
            TMatrix *              H_ji = A_ji;
            const auto             p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                if ( pid != p_ji )
                {
                    T_ji = create_matrix( A_ji, mat_types(j,i), pid );
                    H_ji = T_ji.get();
                }// if
                
                broadcast< value_t >( row_comms[j], H_ji, row_maps[j][p_ji] );
            }// if
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const auto             A_il = A->block( i, l );
                unique_ptr< TMatrix >  T_il;
                TMatrix *              H_il = A_il;
                const auto             p_il = A_il->procs().master();
                
                // broadcast A_il to all processors in column l
                if ( contains( col_procs[l], pid ) )
                {
                    if ( pid != p_il )
                    {
                        T_il = create_matrix( A_il, mat_types(i,l), pid );
                        H_il = T_il.get();
                    }// if
                
                    broadcast< value_t >( col_comms[l], H_il, col_maps[l][p_il] );
                }// if

                //
                // update local matrix block
                //
                
                auto        A_jl = A->block( j, l );
                const auto  p_jl = A_jl->procs().master();
                
                if ( pid == p_jl )
                    update< value_t >( H_ji, H_il, A_jl, acc );
            }// for
        }// for
        #endif
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
    
    parse_cmdline( argc, argv );
    
    // redirect output for all except proc 0
    unique_ptr< RedirectOutput >  redir_out = ( pid != 0
                                                ? make_unique< RedirectOutput >( to_string( "tlrmpi_%03d.out", pid ) )
                                                : nullptr );
    
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
