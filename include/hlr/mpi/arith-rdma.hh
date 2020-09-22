#ifndef __HLR_MPI_ARITH_RDMA_HH
#define __HLR_MPI_ARITH_RDMA_HH
//
// Project     : HLib
// File        : arith-rdma.hh
// Description : arithmetic functions based on MPI blocking broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <vector>
#include <list>
#include <mutex>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hpro/matrix/structure.hh>
#include <hpro/matrix/TBSHMBuilder.hh>
#include <hpro/algebra/mat_fac.hh>
#include <hpro/algebra/solve_tri.hh>
#include <hpro/algebra/mat_mul.hh>

#include "hlr/utils/tools.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/mpi/arith.hh"

namespace hlr
{

using namespace HLIB;

namespace mpi
{

namespace rdma
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

template < typename value_t >
mpi::request
rget ( mpi::window &                    win,
       HLIB::BLAS::Matrix< value_t > &  M,
       const int                        root )
{
    hlr::log( 5, HLIB::to_string( "rget: %d × %d from %d", M.nrows(), M.ncols(), root ) );
    
    mpi::request  req;
    const size_t  count = M.nrows() * M.ncols() * sizeof(value_t);
    
    MPI_CHECK_RESULT( MPI_Rget,
                      ( M.data(), count, MPI_BYTE, // local destination buffer
                        root,                      // from which node
                        0, count, MPI_BYTE,        // offset in root buffer
                        MPI_Win( win ), & req.mpi_request ) );

    return std::move( req );
}

template < typename value_t >
std::vector< mpi::window >
setup_rdma ( mpi::communicator &  comm,
             TMatrix *            A,
             const uint           rank )
{
    std::vector< mpi::window >  wins;
    
    if ( is_dense( A ) )
    {
        hlr::log( 5, HLIB::to_string( "setup_rdma: dense, %d × %d", A->nrows(), A->ncols() ) );
                  
        wins.reserve( 1 );
        wins[0] = mpi::window( comm, blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ).data(), A->nrows() * A->ncols() );
        wins[0].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
    }// if
    else if ( is_lowrank( A ) )
    {
        hlr::log( 5, HLIB::to_string( "setup_rdma: lowrank, %d × %d, %d", A->nrows(), A->ncols(), rank ) );
        
        auto  R = ptrcast( A, TRkMatrix );
        
        R->set_rank( rank );
        
        wins.reserve( 2 );
        wins[0] = mpi::window( comm, blas_mat_A< value_t >( R ).data(), R->nrows() * rank );
        wins[0].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
        wins[1] = mpi::window( comm, blas_mat_B< value_t >( R ).data(), R->ncols() * rank );
        wins[1].fence( MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE );
    }// if
    else
        hlr::error( "(setup_rdma) unsupported matrix type : " + A->typestr() ) ;

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
        hlr::log( 5, HLIB::to_string( "request_rdma: dense, %d × %d", A->nrows(), A->ncols() ) );
        
        reqs.push_back( rget( wins[0], blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ), root ) );
    }// if
    else if ( is_lowrank( A ) )
    {
        hlr::log( 5, HLIB::to_string( "request_rdma: lowrank, %d × %d, %d", A->nrows(), A->ncols(), ptrcast( A, TRkMatrix )->rank() ) );
        
        reqs.push_back( rget( wins[0], blas_mat_A< value_t >( ptrcast( A, TRkMatrix ) ), root ) );
        reqs.push_back( rget( wins[1], blas_mat_B< value_t >( ptrcast( A, TRkMatrix ) ), root ) );
    }// if
    else
        hlr::error( "(request_rdma) unsupported matrix type : " + A->typestr() ) ;

    return std::move( reqs );
}

void
wait_rdma ( std::vector< mpi::request > &  reqs )
{
    wait_all( reqs );
}

void
finish_rdma( std::vector< mpi::window > &  wins )
{
    for ( auto &  win : wins )
    {
        win.fence( MPI_MODE_NOSUCCEED );
        win.free();
    }// for
}

template < typename value_t,
           typename approx_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc,
     const approx_t &   approx )
{
    // assert( RANK != 0 );
    assert( is_blocked( A ) );
    
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    // setup global types (IMPORTANT!!!)
    // type_dense = RTTI::type_to_id( "TDenseMatrix" );
    // type_lr    = RTTI::type_to_id( "TRkMatrix" );
    // type_ghost = RTTI::type_to_id( "TGhostMatrix" );
    
    //
    // set up global map of matrix types in A
    //

    const auto  mat_types = build_type_matrix( BA );

    //
    // set up communicators for rows/cols
    //

    std::vector< mpi::communicator >               row_comms( nbr ), col_comms( nbc );  // communicators for rows/columns
    std::vector< std::list< int > >                row_procs( nbr ), col_procs( nbc );  // set of processors for rows/columns
    std::vector< std::unordered_map< int, int > >  row_maps( nbr ),  col_maps( nbc );   // mapping of global ranks to row/column ranks

    mpi::matrix::build_row_comms( BA, row_comms, row_procs, row_maps );
    mpi::matrix::build_col_comms( BA, col_comms, col_procs, col_maps );

    //
    // LU factorization
    //

    const int  rank        = acc.rank();
    size_t     max_add_mem = 0;

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        hlr::log( 4, HLIB::to_string( "──────────────── step %d ────────────────", i ) );
        
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
        int   p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            hlr::log( 4, HLIB::to_string( "  invert( %d )", A_ii->id() ) );
            BLAS::invert( blas_mat< value_t >( A_ii ) );
        }// if

        if ( contains( col_procs[i], pid ) )
        {
            //
            // set up RDMA for diagonal block
            //

            std::unique_ptr< TMatrix >   T_ii;        // temporary storage with auto-delete
            TMatrix *                    H_ii = A_ii; // handle for A_ii/T_ii
            std::vector< mpi::request >  diag_reqs;
            
            if ( pid != p_ii )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if

            // set up windows/attached memory 
            auto  diag_wins = setup_rdma< value_t >( col_comms[i], H_ii, rank );

            // and start get requests
            if ( pid != p_ii )
                diag_reqs = request_rdma< value_t >( diag_wins, H_ii, col_maps[i][p_ii] );
            
            //
            // off-diagonal solve
            //

            bool        have_diag = ( pid == p_ii );
            std::mutex  req_mtx;
            
            // tbb::parallel_for(
            //     i+1, nbr,
            //     [&,BA,H_ii,i,pid,p_ii] ( uint  j )
            for ( uint  j = i+1; j < nbr; ++j )
                {
                    // L is unit diagonal !!! Only solve with U
                    auto       A_ji = BA->block( j, i );
                    const int  p_ji = A_ji->procs().master();

                    std::cout << "solve : " << A_ji->id() << ", " << pid << ", " << p_ji << std::endl;
                    
                    if ( pid == p_ji )
                    {
                        {
                            std::lock_guard< std::mutex >  lock( req_mtx );
                        
                            if ( ! have_diag )
                            {
                                // DBG::printf( "waiting for %d", H_ii->id() );
                                wait_rdma( diag_reqs );
                                have_diag = true;
                                
                                if ( pid != p_ii )
                                    add_mem += H_ii->byte_size();
                            }// if
                        }

                        // DBG::printf( "solving %d", A_ji->id() );
                        trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A_ji );
                    }// if
                }// );
            
            finish_rdma( diag_wins );
        }

        //
        // set up MPI RDMA for row/column matrices
        //

        std::vector< std::unique_ptr< TMatrix > >   row_i_mat( nbr );        // for autodeletion of temp. matrices
        std::vector< std::unique_ptr< TMatrix > >   col_i_mat( nbc );
        std::vector< TMatrix * >                    row_i( nbr, nullptr );   // actual matrix handles
        std::vector< TMatrix * >                    col_i( nbc, nullptr );
        std::vector< std::vector< mpi::window > >   row_wins( nbr );         // holds MPI windows for matrices
        std::vector< std::vector< mpi::window > >   col_wins( nbc );
        std::vector< std::vector< mpi::request > >  row_reqs( nbr );         // holds MPI RDMA requests for matrices
        std::vector< std::vector< mpi::request > >  col_reqs( nbc );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
            const int   p_ji = A_ji->procs().master();

            if ( contains( row_procs[j], pid ) )
            {
                row_i[j] = A_ji;
            
                if ( pid != p_ji )
                {
                    row_i_mat[j] = create_matrix( A_ji, mat_types(j,i), pid );
                    row_i[j]     = row_i_mat[j].get();
                }// if
                
                // set up windows/attached memory 
                row_wins[j] = setup_rdma< value_t >( row_comms[j], row_i[j], rank );

                // and start get requests
                if ( pid != p_ji )
                    row_reqs[j] = request_rdma< value_t >( row_wins[j], row_i[j], row_maps[j][p_ji] );
            }// if
        }// for
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = BA->block( i, l );
            const int   p_il = A_il->procs().master();
            
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
                col_wins[l] = setup_rdma< value_t >( col_comms[l], col_i[l], rank );

                // and start get requests
                if ( pid != p_il )
                    col_reqs[l] = request_rdma< value_t >( col_wins[l], col_i[l], col_maps[l][p_il] );
            }// if
        }// for

        //
        // update of trailing sub-matrix
        //
        
        std::vector< bool >        row_done( nbr, false );  // signals received matrix
        std::vector< bool >        col_done( nbc, false );
        std::vector< std::mutex >  row_mtx( nbr );         // mutices for access to requests
        std::vector< std::mutex >  col_mtx( nbc );
        
        // tbb::parallel_for(
        //     tbb::blocked_range2d< uint >( i+1, nbr,
        //                                   i+1, nbc ),
        //     [&,BA,i,pid] ( const tbb::blocked_range2d< uint > & r )
        //     {
        //         for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
        //         {
        //             const int  p_ji = BA->block( j, i )->procs().master();
                    
        //             for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
        //             {
        //                 const int  p_il = BA->block( i, l )->procs().master();
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    const int  p_ji = BA->block( j, i )->procs().master();
                    
                    for ( uint  l = i+1; l < nbc; ++l )
                    {
                        const int  p_il = BA->block( i, l )->procs().master();
                        
                        //
                        // update local matrix block
                        //
                
                        auto       A_jl = BA->block( j, l );
                        const int  p_jl = A_jl->procs().master();
                
                        if ( pid == p_jl )
                        {
                            //
                            // ensure broadcasts fir A_ji and A_il have finished
                            //

                            {
                                std::lock_guard< std::mutex >  lock( row_mtx[j] );
                        
                                if (( p_ji != pid ) && ! row_done[j] )
                                {
                                    // DBG::printf( "waiting for %d", BA->block(j,i)->id() );
                                    wait_rdma( row_reqs[j] );
                                    row_done[j] = true;
                                    add_mem += row_i[j]->byte_size();
                                }// if
                            }
                            
                            {
                                std::lock_guard< std::mutex >  lock( col_mtx[l] );
                        
                                if (( p_il != pid ) && ! col_done[l] )
                                {
                                    // DBG::printf( "waiting for %d", BA->block(i,l)->id() );
                                    wait_rdma( col_reqs[l] );
                                    col_done[l] = true;
                                    add_mem += col_i[l]->byte_size();
                                }// if
                            }

                            //
                            // finally compute update
                            //
                    
                            // DBG::printf( "updating %d with %d × %d", A_jl->id(), row_i[j]->id(), col_i[l]->id() );
                            hlr::multiply< value_t >( value_t(-1),
                                                      apply_normal, *row_i[j],
                                                      apply_normal, *col_i[l],
                                                      *A_jl, acc, approx );
                        }// if
                    }// for
                }// for
                // } );

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // finish epoch (started in "setup_rdma")
        //

        for ( uint  j = i+1; j < nbr; ++j )
            finish_rdma( row_wins[j] );

        for ( uint  l = i+1; l < nbc; ++l )
            finish_rdma( col_wins[l] );
    }// for

    // std::cout << "  time in MPI : " << to_string( "%.2fs", time_mpi ) << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

}// namespace tlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace hodlr
{

}// namespace hodlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace tileh
{

}// namespace tileh

}// namespace rdma

}// namespace mpi

}// namespace hlr

#endif // __HLR_MPI_ARITH_RDMA_HH
