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

#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

#include "common.inc"
#include "tlr.hh"
#include "tlr.inc"

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
        DBG::printf( "%d", c );
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

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    mpi::communicator  world;
    const auto         my_proc = world.rank();
    
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

    auto  nbr = A->nblock_rows();
    auto  nbc = A->nblock_cols();

    //
    // set up communicators for rows/cols
    //

    vector< mpi::communicator >          row_comms( nbr ), col_comms( nbc );  // communicators for rows/columns
    vector< list< int > >                row_procs( nbr ), col_procs( nbc );  // set of processors for rows/columns
    vector< unordered_map< int, int > >  row_maps( nbr ),  col_maps( nbc );   // mapping of global ranks to row/column ranks

    std::cout << "rows" << std::endl;
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

        std::cout << i << " : ";
        for ( auto  p : procs ) std::cout << p << ", ";
        std::cout << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( pos < nbr )
        {
            row_comms[i] = row_comms[pos];
            row_maps[i]  = row_maps[pos];
        }// if
        else
        {
            row_comms[i] = world.split( contains( procs, my_proc ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                row_maps[i][p] = comm_rank++;
        }// else
            
        row_procs[i] = std::move( procs );
    }// for

    std::cout << "columns" << std::endl;
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
            
        std::cout << j << " : ";
        for ( auto  p : procs ) std::cout << p << ", ";
        std::cout << " (" << ( pos == nbc ? j : pos ) << ")" << std::endl;

        // use previously created communicator or create new if none found
        if ( pos < nbc )
        {
            col_comms[j] = col_comms[pos];
            col_maps[j]  = col_maps[pos];
        }// if
        else
        {
            col_comms[j] = world.split( contains( procs, my_proc ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                col_maps[j][p] = comm_rank++;
        }// else
            
        col_procs[j] = std::move( procs );
    }// for




    std::cout << "LU" << std::endl;
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        DBG::print(  "────────────────────────────────────────────────" );
        DBG::printf( "step %d", i );
        
        auto  A_ii = ptrcast( A->block( i, i ), TDenseMatrix );
        auto  p_ii = A_ii->procs().master();

        if ( my_proc == p_ii )
        {
            DBG::printf( "invert( %d )", A_ii->id() );
            DBG::write( blas_mat< value_t >( A_ii ), to_string( "A_%d%d.mat", i, i ), "A" );
            B::invert( blas_mat< value_t >( A_ii ) );
        }// if

        //
        // broadcast diagonal block
        //

        unique_ptr< TDenseMatrix >  T_ii;           // temporary storage with auto-delete
        TDenseMatrix *              D_ii = nullptr; // dense handle for A_ii/T_ii

        if ( my_proc != p_ii )
        {
            T_ii = make_unique< TDenseMatrix >( A_ii->row_is(), A_ii->col_is(), A_ii->is_complex() );
            D_ii = T_ii.get();
            D_ii->set_id( A_ii->id() );
            D_ii->set_procs( ps_single( my_proc ) );
        }// if
        else
            D_ii = ptrcast( A_ii, TDenseMatrix );
                
        if ( contains( col_procs[i], my_proc ) )
        {
            DBG::printf( "broadcast( %d ) from %d", A_ii->id(), p_ii );
            mpi::broadcast( col_comms[i], blas_mat< value_t >( D_ii ).data(), D_ii->nrows() * D_ii->ncols(), col_maps[i][p_ii] );
        }// if

        //
        // off-diagonal solve
        //
            
        // tbb::parallel_for( i+1, nbr,
        //                    [D_ii,A,i,my_proc] ( uint  j )
        for ( uint  j = i+1; j < nbr; ++j )
        {
            // L is unit diagonal !!! Only solve with U
            auto        A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();
            
            if ( my_proc == p_ji )
            {
                DBG::printf( "solve_U( %d, %d )", D_ii->id(), A->block( j, i )->id() );
                trsmuh< value_t >( D_ii, A->block( j, i ) );
            }// if
        }
        // );

        //
        // update of trailing sub-matrix
        //
            
        // tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
        //                                                  i+1, nbc ),
        //                    [A,i,&acc] ( const tbb::blocked_range2d< uint > & r )
                           // {
                           //     for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                           //     {
                           //         for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                           //         {
                           //             update< value_t >( A->block( j, i ), A->block( i, l ), A->block( j, l ), acc );
                           //         }// for
                           //     }// for
                           // } );
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();
            
            if ( my_proc == p_ji )
            {
                // broadcast A_ji to all processors in row j
            }// if
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const auto  A_il = A->block( i, l );
                const auto  p_il = A_il->procs().master();
                
                if ( my_proc == p_il )
                {
                    // broadcast A_il to all processors in column l
                }// if

                auto        A_jl = A->block( j, l );
                const auto  p_jl = A_jl->procs().master();
                
                if ( my_proc == p_jl )
                {
                    update< value_t >( A->block( j, i ), A->block( i, l ), A->block( j, l ), acc );
                }// if
            }// for
        }// for
    }// for
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
    // init MPI before anything else
    boost::mpi::environment   env{ argc, argv };
    boost::mpi::communicator  world;
    const auto                my_proc = world.rank();
    const auto                nprocs  = world.size();

    //
    // adjust HLIB network data
    //
    
    NET::set_nprocs( nprocs );
    NET::set_pid( my_proc );
    
    //
    // redirect output
    //

    string                  output( to_string( "tlrmpi_%03d.out", my_proc ) );
    unique_ptr< ofstream >  fout;
    streambuf *             orig_cout = cout.rdbuf();
    streambuf *             orig_cerr = cerr.rdbuf();

    if ( my_proc != 0 )
    {
        fout = make_unique< ofstream >( output.c_str() );
        cout.rdbuf( fout->rdbuf() );
        cerr.rdbuf( fout->rdbuf() );
    }// if

    std::cout << "━━ " << Mach::hostname() << std::endl
              << "    CPU cores : " << Mach::cpuset()
              << std::endl;
    
    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TLR::cluster( coord.get(), ntile );
    
    {
        TBlockCyclicDistrBC  distr;
        
        distr.distribute( nprocs, bct->root(), nullptr );
    }
    
    if (( my_proc == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "bct_distr" );
    }// if
    
    auto  A = problem->build_matrix( bct.get(), fixed_rank( k ) );
    
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "hlrtest_A_%03d", my_proc ) );
    }// if
    
    {
        std::cout << "━━ LU facorisation ( TLR MPI )" << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::MPI::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        // std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

    // reset cout buffers
    if ( cout.rdbuf() != orig_cout ) cout.rdbuf( orig_cout );
    if ( cerr.rdbuf() != orig_cerr ) cerr.rdbuf( orig_cerr );
}
