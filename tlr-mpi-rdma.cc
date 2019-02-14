//
// Project     : HLib
// File        : tlr-mpi.cc
// Description : TLR arithmetic with MPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <fstream>
#include <mutex>

using namespace std;

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "mpi.hh"

#include "cmdline.inc"
#include "problem.inc"
#include "tlr.hh"
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

            bool   have_diag = ( pid == p_ii );
            mutex  req_mtx;
            
            // tbb::parallel_for(
            //     i+1, nbr,
            //     [&,A,H_ii,i,pid,p_ii] ( uint  j )
            for ( uint  j = i+1; j < nbr; ++j )
                {
                    // L is unit diagonal !!! Only solve with U
                    auto        A_ji = A->block( j, i );
                    const auto  p_ji = A_ji->procs().master();
                    
                    if ( pid == p_ji )
                    {
                        {
                            lock_guard< mutex >  lock( req_mtx );
                        
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
        
        vector< bool >   row_done( nbr, false );  // signals received matrix
        vector< bool >   col_done( nbc, false );
        vector< mutex >  row_mtx( nbr );         // mutices for access to requests
        vector< mutex >  col_mtx( nbc );
        
        // tbb::parallel_for(
        //     tbb::blocked_range2d< uint >( i+1, nbr,
        //                                   i+1, nbc ),
        //     [&,A,i,pid] ( const tbb::blocked_range2d< uint > & r )
        //     {
        //         for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
        //         {
        //             const auto  p_ji = A->block( j, i )->procs().master();
                    
        //             for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
        //             {
        //                 const auto  p_il = A->block( i, l )->procs().master();
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

                            {
                                lock_guard< mutex >  lock( row_mtx[j] );
                        
                                if (( p_ji != pid ) && ! row_done[j] )
                                {
                                    // DBG::printf( "waiting for %d", A->block(j,i)->id() );
                                    wait_rdma( row_reqs[j] );
                                    row_done[j] = true;
                                    add_mem += row_i[j]->byte_size();
                                }// if
                            }
                            
                            {
                                lock_guard< mutex >  lock( col_mtx[l] );
                        
                                if (( p_il != pid ) && ! col_done[l] )
                                {
                                    // DBG::printf( "waiting for %d", A->block(i,l)->id() );
                                    wait_rdma( col_reqs[l] );
                                    col_done[l] = true;
                                    add_mem += col_i[l]->byte_size();
                                }// if
                            }

                            //
                            // finally compute update
                            //
                    
                            // DBG::printf( "updating %d with %d × %d", A_jl->id(), row_i[j]->id(), col_i[l]->id() );
                            update< value_t >( row_i[j], col_i[l], A_jl, acc );
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

#include "tlr-mpi-main.inc"
