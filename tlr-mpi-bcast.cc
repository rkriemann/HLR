//
// Project     : HLib
// File        : tlr-mpi.cc
// Description : TLR arithmetic with MPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <fstream>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

using namespace std;

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
        
        comm.broadcast( blas_mat< value_t >( D ).data(), D->nrows() * D->ncols(), root_proc );
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = ptrcast( A, TRkMatrix );
        auto  k = R->rank();
        
        comm.broadcast( k, root_proc );

        R->set_rank( k );
        
        comm.broadcast( blas_mat_A< value_t >( R ).data(), R->nrows() * k, root_proc );
        comm.broadcast( blas_mat_B< value_t >( R ).data(), R->ncols() * k, root_proc );
    }// if
    else
        HERROR( ERR_MAT_TYPE, "broadcast", "" );

    auto  toc = Time::Wall::since( tic );

    time_mpi += toc.seconds();
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
            broadcast< value_t >( col_comms[i], H_ii, col_maps[i][p_ii] );
            
            if ( pid != p_ii )
                add_mem += H_ii->byte_size();
            
            //
            // off-diagonal solve
            //
            
            tbb::parallel_for(
                i+1, nbr,
                [A,H_ii,i,pid] ( uint  j )
                // for ( uint  j = i+1; j < nbr; ++j )
                {
                    // L is unit diagonal !!! Only solve with U
                    auto        A_ji = A->block( j, i );
                    const auto  p_ji = A_ji->procs().master();
                    
                    if ( pid == p_ji )
                    {
                        // DBG::printf( "solve_U( %d, %d )", H_ii->id(), A->block( j, i )->id() );
                        trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A_ji );
                    }// if
                } );
        }
        
        //
        // broadcast blocks in row/column for update phase
        //

        vector< unique_ptr< TMatrix > >  row_i_mat( nbr );      // for autodeletion
        vector< unique_ptr< TMatrix > >  col_i_mat( nbc );
        vector< TMatrix * >              row_i( nbr, nullptr ); // matrix handles
        vector< TMatrix * >              col_i( nbc, nullptr );
        
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
        
        tbb::parallel_for(
            tbb::blocked_range2d< uint >( i+1, nbr,
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
