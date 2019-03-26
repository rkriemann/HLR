#ifndef __HLR_GASPI_ARITH_HH
#define __HLR_GASPI_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : arithmetic functions based on GASPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <vector>
#include <list>
#include <mutex>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <matrix/structure.hh>
#include <matrix/TBSHMBuilder.hh>
#include <algebra/mat_fac.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "utils/tools.hh"
#include "utils/tensor.hh"
#include "common/multiply.hh"
#include "common/solve.hh"

#include "gaspi/gaspi.hh"

namespace HLR
{

using namespace HLIB;

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace GASPI
{

namespace TLR
{

const typeid_t  TYPE_DENSE = RTTI::type_to_id( "TDenseMatrix" );
const typeid_t  TYPE_LR    = RTTI::type_to_id( "TRkMatrix" );
const typeid_t  TYPE_GHOST = RTTI::type_to_id( "TGhostMatrix" );

constexpr segment_id_t  ID_DENSE = 1;
constexpr segment_id_t  ID_LR_A  = 2;
constexpr segment_id_t  ID_LR_B  = 3;

//
// create n×m block matrix type info for full block matrix
//
tensor2< typeid_t >
build_type_matrix ( const TBlockMatrix *  A )
{
    const auto           nbr = A->nblock_rows();
    const auto           nbc = A->nblock_cols();
    GASPI::process  proc;
    const auto           pid    = proc.rank();
    const auto           nprocs = proc.size();
    tensor2< typeid_t >  mat_types( nbr, nbc );

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            mat_types(i,j) = A->block( i, j )->type();

    tensor2< typeid_t >  rem_types( nbr, nbc );
    GASPI::group         world;
    GASPI::segment       loc_seg( 0, & mat_types(0,0), nbr*nbc, world );
    GASPI::segment       rem_seg( 1, & rem_types(0,0), nbr*nbc, world );
    GASPI::queue         queue;
    
    for ( uint  p = 0; p < nprocs; ++p )
    {
        if ( p != pid )
        {
            queue.read( rem_seg, p, 0 );
            queue.wait();
            // GASPI::notify_wait( rem_seg, 1 );
                               
            for ( uint  i = 0; i < nbr; ++i )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( rem_types(i,j) != TYPE_GHOST )
                        mat_types(i,j) = rem_types(i,j);
        }// if

        world.barrier();
    }// for

    if ( verbose( 5 ) )
    {
        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
                std::cout << RTTI::id_to_type( mat_types(i,j) ) << "  ";
            std::cout << std::endl;
        }// for
    }// if

    loc_seg.release();
    rem_seg.release();
    
    return mat_types;
}

//
// create matrix defined by <type>
//
std::unique_ptr< TMatrix >
create_matrix ( const TMatrix *  A,
                const typeid_t   type,
                const int        proc )
{
    assert(( type == TYPE_DENSE ) || ( type == TYPE_LR ));
    
    std::unique_ptr< TMatrix >  T;

    if ( type == TYPE_DENSE )
    {
        HLR::log( 4, HLIB::to_string( "create_matrix( %d ) : dense", A->id() ) );
        T = std::make_unique< TDenseMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    }// if
    else if ( type == TYPE_LR )
    {
        HLR::log( 4, HLIB::to_string( "create_matrix( %d ) : lowrank", A->id() ) );
        T = std::make_unique< TRkMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    }// if

    T->set_id( A->id() );
    T->set_procs( ps_single( proc ) );
    
    return T;
}

//
// define GASPI segments for matrix data
//
template < typename value_t >
std::vector< GASPI::segment >
create_segments ( const segment_id_t  sid,
                  TMatrix *           A,
                  const uint          rank,
                  GASPI::group &      group )
{
    std::vector< GASPI::segment >  segs;

    // we need space for low-rank blocks
    assert( sid % 2 == 0 );
    
    if ( is_dense( A ) )
    {
        HLR::log( 5, HLIB::to_string( "create_segments: dense, %d, %d × %d", A->id(), A->nrows(), A->ncols() ) );

        segs.reserve( 1 );
        segs.push_back( GASPI::segment( sid, blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ).data(), A->nrows() * A->ncols(), group ) );
    }// if
    else if ( is_lowrank( A ) )
    {
        HLR::log( 5, HLIB::to_string( "create_segments: lowrank, %d, %d × %d, %d", A->id(), A->nrows(), A->ncols(), rank ) );
        
        auto  R = ptrcast( A, TRkMatrix );
        
        R->set_rank( rank );
        
        segs.reserve( 2 );
        segs[0] = GASPI::segment( sid,   blas_mat_A< value_t >( R ).data(), R->nrows() * rank, group );
        segs[1] = GASPI::segment( sid+1, blas_mat_B< value_t >( R ).data(), R->ncols() * rank, group );
    }// if
    else
        HLR::error( "(create_segments) unsupported matrix type : " + A->typestr() ) ;

    return segs;
}

//
// enqueue read requests for matrix data
//
template < typename value_t >
void
read_matrix ( std::vector< GASPI::segment > &  segs,
              TMatrix *                        A,
              const int                        source,
              GASPI::queue &                   queue )
{
    if ( is_dense( A ) )
    {
        HLR::log( 5, HLIB::to_string( "read_matrix: dense, %d × %d", A->nrows(), A->ncols() ) );

        queue.read( segs[0], source, segs[0].id() );
    }// if
    else if ( is_lowrank( A ) )
    {
        HLR::log( 5, HLIB::to_string( "read_matrix: lowrank, %d × %d, %d", A->nrows(), A->ncols(), ptrcast( A, TRkMatrix )->rank() ) );
        
        queue.read( segs[0], source, segs[0].id() );
        queue.read( segs[1], source, segs[1].id() );
    }// if
    else
        HLR::error( "(request_rdma) unsupported matrix type : " + A->typestr() ) ;
}

//
// return list of processors in row <row> starting after diagonal block
//
std::list< GASPI::rank_t >
get_row_procs ( const TBlockMatrix *  A,
                const uint            row )
{
    const auto                  nbr = A->nblock_rows();
    const auto                  nbc = A->nblock_cols();
    std::list< GASPI::rank_t >  procs;
    
    for ( uint  j = 0; j < nbc; ++j )
        procs.push_back( A->block( row, j )->procs().master() );
    
    procs.sort();
    procs.unique();

    return procs;
}

//
// return list of processors in column <col> starting after diagonal block
//
std::list< GASPI::rank_t >
get_col_procs ( const TBlockMatrix *  A,
                const uint            col )
{
    const auto                  nbr = A->nblock_rows();
    const auto                  nbc = A->nblock_cols();
    std::list< GASPI::rank_t >  procs;
    
    for ( uint  i = 0; i < nbr; ++i )
        procs.push_back( A->block( i, col )->procs().master() );
    
    procs.sort();
    procs.unique();

    return procs;
}

//
// create groups for each block row of the matrix
// - if processor sets of different rows are identical, so are the groups
//
void
build_row_groups ( const TBlockMatrix *                         A,
                   std::vector< GASPI::group > &                groups,  // communicator per row
                   std::vector< std::list< GASPI::rank_t > > &  procs )  // set of processors per row
{
    const auto      nbr = A->nblock_rows();
    const auto      nbc = A->nblock_cols();
    GASPI::process  process;
    const auto      pid = process.rank();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        const auto  procs_i = get_row_procs( A, i );
        uint        pos     = nbr;
            
        for ( uint  l = 0; l < i; ++l )
        {
            if ( procs_i == procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        if ( HLIB::verbose( 4 ) )
            std::cout << i << " : " << to_string( procs_i ) << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( contains( procs_i, pid ) )
        {
            if ( pos < nbr ) groups[i] = groups[pos];
            else             groups[i] = GASPI::group( procs_i );
        }// if
            
        procs[i] = std::move( procs_i );

        process.barrier();
    }// for
}

//
// same as above but for block columns
//
void
build_col_groups ( const TBlockMatrix *                         A,
                   std::vector< GASPI::group > &                groups,  // group per column
                   std::vector< std::list< GASPI::rank_t > > &  procs )  // set of processors per column
{
    const auto      nbr = A->nblock_rows();
    const auto      nbc = A->nblock_cols();
    GASPI::process  process;
    const auto      pid = process.rank();
    
    for ( uint  i = 0; i < nbc; ++i )
    {
        const auto  procs_i = get_col_procs( A, i );
        uint        pos     = nbc;
            
        for ( uint  l = 0; l < i; ++l )
        {
            if ( procs_i == procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        if ( HLIB::verbose( 4 ) )
            std::cout << i << " : " << to_string( procs_i ) << " (" << ( pos == nbc ? i : pos ) << ")" << std::endl;
            
        // use previously created group or create new if none found
        if ( contains( procs_i, pid ) )
        {
            if ( pos < nbc ) groups[i] = groups[pos];
            else             groups[i] = GASPI::group( procs_i );
        }// if
            
        procs[i] = std::move( procs_i );

        process.barrier();
    }// for
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    // assert( RANK != 0 );
    assert( is_blocked( A ) );
    
    GASPI::process  process;
    const auto      pid    = process.rank();
    const auto      nprocs = process.size();
    
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
    // queues for communication with all nodes
    //

    std::vector< GASPI::queue >  queues( nprocs );  // one queue to each remote processor
    GASPI::group                 world;

    //
    // set up groups for all rows/columns
    //

    std::vector< GASPI::group >                row_groups( nbr ), col_groups( nbc );
    std::vector< std::list< GASPI::rank_t > >  row_procs( nbr ), col_procs( nbc );

    build_row_groups( BA, row_groups, row_procs );
    build_col_groups( BA, col_groups, col_procs );

    //
    // set up IDs for blocks in A
    //

    tensor2< segment_id_t >  id( nbr, nbc );
    segment_id_t             nid = 0;

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            id(i,j) = ( nid += 2 );

    HLR::log( 4, HLIB::to_string( "maximal ID: %d", nid ) );
    
    //
    // LU factorization
    // - GASPI segment Ids are given for blocks (i,j) as i-j for blocks in current column
    //   and j-i+(n-i) for blocks in current row
    //

    const int  rank        = acc.rank();
    size_t     max_add_mem = 0;
    // auto       id          =
    //     [nbr] ( const uint  i, const uint  j )
    //     {
    //         if ( j <= i ) return i-j;
    //         else          return j-i+(nbr-i);
    //     };

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        HLR::log( 4, HLIB::to_string( "──────────────── step %d ────────────────", i ) );

        HLR::log( 4, HLIB::to_string( "#segments: %d/%d", process.nalloc_segments(), process.nmax_segments() ) );

        //
        // factorization of diagonal block
        //
        
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
        auto  p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            HLR::log( 4, HLIB::to_string( "  invert( %d )", A_ii->id() ) );
            BLAS::invert( blas_mat< value_t >( A_ii ) );
        }// if

        //
        // solve off-diagonal blocks in current block column
        // (since L is identity, no solving in block row)
        //

        if ( contains( col_procs[i], pid ) )
        {
            //
            // set up destination matrices
            //

            std::unique_ptr< TMatrix >   T_ii;        // temporary storage with auto-delete
            TMatrix *                    H_ii = A_ii; // handle for A_ii/T_ii
            
            if ( pid != p_ii )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if

            // set up windows/attached memory 
            auto  diag_seg = create_segments< value_t >( id(i,i), H_ii, rank, col_groups[i] );

            // and initiate reading remote memory
            if ( pid != p_ii )
                read_matrix< value_t >( diag_seg, H_ii, p_ii, queues[p_ii] );
            
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
                    auto        A_ji = BA->block( j, i );
                    const auto  p_ji = A_ji->procs().master();

                    HLR::log( 4, HLIB::to_string( "  solve( %d )", A_ji->id() ) );
                    
                    if ( pid == p_ji )
                    {
                        {
                            std::lock_guard< std::mutex >  lock( req_mtx );
                        
                            if ( ! have_diag )
                            {
                                DBG::printf( "waiting for %d", H_ii->id() );
                                queues[p_ii].wait();
                                have_diag = true;
                                
                                if ( pid != p_ii )
                                    add_mem += H_ii->byte_size();
                            }// if
                        }

                        // DBG::printf( "solving %d", A_ji->id() );
                        trsmuh< value_t >( ptrcast( H_ii, TDenseMatrix ), A_ji );
                    }// if
                }// );

            // finish segments
            for ( auto &  seg : diag_seg )
                seg.release();
        }

        world.barrier();
        
        continue;

        //
        // set up MPI RDMA for row/column matrices
        //

        std::vector< std::unique_ptr< TMatrix > >     row_i_mat( nbr );        // for autodeletion of temp. matrices
        std::vector< std::unique_ptr< TMatrix > >     col_i_mat( nbc );
        std::vector< TMatrix * >                      row_i( nbr, nullptr );   // actual matrix handles
        std::vector< TMatrix * >                      col_i( nbc, nullptr );
        std::vector< std::vector< GASPI::segment > >  row_segs( nbr );         // holds MPI windows for matrices
        std::vector< std::vector< GASPI::segment > >  col_segs( nbc );

        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
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
                row_segs[j] = create_segments< value_t >( id(j,i), row_i[j], rank, row_groups[j] );

                // and start get requests
                if ( pid != p_ji )
                    read_matrix< value_t >( row_segs[j], row_i[j], p_ji, queues[ p_ji ] );
            }// if
        }// for
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = BA->block( i, l );
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
                col_segs[l] = create_segments< value_t >( id(i,l), col_i[l], rank, col_groups[l] );

                // and start get requests
                if ( pid != p_il )
                    read_matrix< value_t >( col_segs[l], col_i[l], p_il, queues[ p_il ] );
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
        //             const auto  p_ji = BA->block( j, i )->procs().master();
                    
        //             for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
        //             {
        //                 const auto  p_il = BA->block( i, l )->procs().master();
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    const auto  p_ji = BA->block( j, i )->procs().master();
                    
                    for ( uint  l = i+1; l < nbc; ++l )
                    {
                        const auto  p_il = BA->block( i, l )->procs().master();
                        
                        //
                        // update local matrix block
                        //
                
                        auto        A_jl = BA->block( j, l );
                        const auto  p_jl = A_jl->procs().master();
                
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
                                    queues[p_ji].wait();
                                    row_done[j] = true;
                                    add_mem += row_i[j]->byte_size();
                                }// if
                            }
                            
                            {
                                std::lock_guard< std::mutex >  lock( col_mtx[l] );
                        
                                if (( p_il != pid ) && ! col_done[l] )
                                {
                                    // DBG::printf( "waiting for %d", BA->block(i,l)->id() );
                                    queues[p_il].wait();
                                    col_done[l] = true;
                                    add_mem += col_i[l]->byte_size();
                                }// if
                            }

                            //
                            // finally compute update
                            //
                    
                            // DBG::printf( "updating %d with %d × %d", A_jl->id(), row_i[j]->id(), col_i[l]->id() );
                            multiply< value_t >( value_t(-1), row_i[j], col_i[l], A_jl, acc );
                        }// if
                    }// for
                }// for
                // } );

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // finish epoch (started in "setup_rdma")
        //

        for ( uint  j = i+1; j < nbr; ++j )
            for ( auto & seg : row_segs[j] )
                seg.release();

        for ( uint  l = i+1; l < nbc; ++l )
            for ( auto & seg : col_segs[l] )
                seg.release();
    }// for

    // std::cout << "  time in MPI : " << to_string( "%.2fs", time_mpi ) << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

}// namespace TLR

}// namespace GASPI

}// namespace HLR

#endif // __HLR_GASPI_ARITH_HH