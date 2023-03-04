#ifndef __HLR_GASPI_ARITH_HH
#define __HLR_GASPI_ARITH_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : arithmetic functions based on GASPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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
#include "hlr/utils/tensor.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"

#include "hlr/gaspi/gaspi.hh"

namespace hlr { namespace gaspi { namespace tlr {

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace hpro = HLIB;

const hpro::typeid_t  TYPE_DENSE = hpro::RTTI::type_to_id( "TDenseMatrix" );
const hpro::typeid_t  TYPE_LR    = hpro::RTTI::type_to_id( "TRkMatrix" );
const hpro::typeid_t  TYPE_GHOST = hpro::RTTI::type_to_id( "TGhostMatrix" );

constexpr segment_id_t  ID_DENSE = 1;
constexpr segment_id_t  ID_LR_A  = 2;
constexpr segment_id_t  ID_LR_B  = 3;

//
// create n×m block matrix type info for full block matrix
//
tensor2< hpro::typeid_t >
build_type_matrix ( const hpro::TBlockMatrix *  A )
{
    const auto                 nbr = A->nblock_rows();
    const auto                 nbc = A->nblock_cols();
    gaspi::process             proc;
    const auto                 pid    = proc.rank();
    const auto                 nprocs = proc.size();
    tensor2< hpro::typeid_t >  mat_types( nbr, nbc );

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            mat_types(i,j) = A->block( i, j )->type();

    tensor2< hpro::typeid_t >  rem_types( nbr, nbc );
    gaspi::group               world;
    gaspi::segment             loc_seg( 0, & mat_types(0,0), nbr*nbc, world );
    gaspi::segment             rem_seg( 1, & rem_types(0,0), nbr*nbc, world );
    gaspi::queue               queue;
    
    for ( uint  p = 0; p < nprocs; ++p )
    {
        if ( p != pid )
        {
            queue.read( rem_seg, p, 0 );
            queue.wait();
            // gaspi::notify_wait( rem_seg, 1 );
                               
            for ( uint  i = 0; i < nbr; ++i )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( rem_types(i,j) != TYPE_GHOST )
                        mat_types(i,j) = rem_types(i,j);
        }// if

        world.barrier();
    }// for

    if ( hpro::verbose( 5 ) )
    {
        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
                std::cout << hpro::RTTI::id_to_type( mat_types(i,j) ) << "  ";
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
std::unique_ptr< hpro::TMatrix >
create_matrix ( const hpro::TMatrix *  A,
                const hpro::typeid_t   type,
                const int              proc )
{
    assert(( type == TYPE_DENSE ) || ( type == TYPE_LR ));
    
    std::unique_ptr< hpro::TMatrix >  T;

    if ( type == TYPE_DENSE )
    {
        hlr::log( 4, hpro::to_string( "create_matrix( %d ) : dense", A->id() ) );
        T = std::make_unique< hpro::TDenseMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    }// if
    else if ( type == TYPE_LR )
    {
        hlr::log( 4, hpro::to_string( "create_matrix( %d ) : lowrank", A->id() ) );
        T = std::make_unique< hpro::TRkMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    }// if

    T->set_id( A->id() );
    T->set_procs( hpro::ps_single( proc ) );
    
    return T;
}

//
// define GASPI segments for matrix data
//
template < typename value_t >
std::vector< gaspi::segment >
create_segments ( const segment_id_t  sid,
                  hpro::TMatrix *     A,
                  const uint          rank,
                  gaspi::group &      group )
{
    std::vector< gaspi::segment >  segs;

    // we need space for low-rank blocks
    assert( sid % 2 == 0 );
    
    if ( is_dense( A ) )
    {
        hlr::log( 4, hpro::to_string( "create_segments: dense, %d, %d × %d on ", A->id(), A->nrows(), A->ncols() ) + group.to_string() );

        segs.reserve( 1 );
        segs.push_back( gaspi::segment( sid, hpro::blas_mat< value_t >( ptrcast( A, hpro::TDenseMatrix ) ).data(), A->nrows() * A->ncols(), group ) );
    }// if
    else if ( is_lowrank( A ) )
    {
        hlr::log( 4, hpro::to_string( "create_segments: lowrank, %d, %d × %d, %d on ", A->id(), A->nrows(), A->ncols(), rank ) + group.to_string() );
        
        auto  R = ptrcast( A, hpro::TRkMatrix );
        
        R->set_rank( rank );
        
        segs.reserve( 2 );
        segs.push_back( gaspi::segment( sid,   hpro::blas_mat_A< value_t >( R ).data(), R->nrows() * rank, group ) );
        segs.push_back( gaspi::segment( sid+1, hpro::blas_mat_B< value_t >( R ).data(), R->ncols() * rank, group ) );
    }// if
    else
        hlr::error( "(create_segments) unsupported matrix type : " + A->typestr() ) ;

    return segs;
}

//
// enqueue read requests for matrix data
//
template < typename value_t >
void
read_matrix ( std::vector< gaspi::segment > &  segs,
              hpro::TMatrix *                  A,
              const int                        source,
              gaspi::queue &                   queue )
{
    if ( is_dense( A ) )
    {
        hlr::log( 4, hpro::to_string( "read_matrix: dense, %d, %d × %d from %d", A->id(), A->nrows(), A->ncols(), source ) );

        queue.read( segs[0], source, segs[0].id() );
    }// if
    else if ( is_lowrank( A ) )
    {
        hlr::log( 4, hpro::to_string( "read_matrix: lowrank, %d, %d × %d, %d from %d", A->id(), A->nrows(), A->ncols(),
                                      ptrcast( A, hpro::TRkMatrix )->rank(), source ) );
        
        queue.read( segs[0], source, segs[0].id() );
        queue.read( segs[1], source, segs[1].id() );
    }// if
    else
        hlr::error( "(read_matrix) unsupported matrix type : " + A->typestr() ) ;
}

//
// enqueue send requests for matrix data
//
template < typename value_t >
void
send_matrix ( std::vector< gaspi::segment > &  segs,
              hpro::TMatrix *                  A,
              const int                        dest,
              gaspi::queue &                   queue )
{
    if ( is_dense( A ) )
    {
        hlr::log( 4, hpro::to_string( "send_matrix: dense, %d, %d × %d to %d", A->id(), A->nrows(), A->ncols(), dest ) );

        queue.write_notify( segs[0], dest, segs[0].id(), segs[0].id() );
    }// if
    else if ( is_lowrank( A ) )
    {
        hlr::log( 4, hpro::to_string( "send_matrix: lowrank, %d, %d × %d, %d to %d", A->id(), A->nrows(), A->ncols(),
                                      ptrcast( A, hpro::TRkMatrix )->rank(), dest ) );
        
        queue.write_notify( segs[0], dest, segs[0].id(), segs[0].id() );
        queue.write_notify( segs[1], dest, segs[1].id(), segs[1].id() );
    }// if
    else
        hlr::error( "(send_matrix) unsupported matrix type : " + A->typestr() ) ;
}

//
// wait for matrix communication to finish 
//
void
wait_matrix ( std::vector< gaspi::segment > &  segs )
{
    if ( segs.size() == 1 )
    {
        gaspi::notify_wait( segs[0], segs[0].id() );
    }// if
    else if ( segs.size() == 2 )
    {
        gaspi::notify_wait( segs[0], segs[0].id() );
        gaspi::notify_wait( segs[1], segs[1].id() );
    }// if
    else
        hlr::error( hpro::to_string( "(wait_matrix) unsupported number of segments: %d", segs.size() ) );
}

//
// return list of processors in row <row> starting after diagonal block
//
std::list< gaspi::rank_t >
get_row_procs ( const hpro::TBlockMatrix *  A,
                const uint                  row )
{
    const auto                  nbr = A->nblock_rows();
    const auto                  nbc = A->nblock_cols();
    std::list< gaspi::rank_t >  procs;
    
    for ( uint  j = 0; j < nbc; ++j )
        procs.push_back( A->block( row, j )->procs().master() );
    
    procs.sort();
    procs.unique();

    return procs;
}

//
// return list of processors in column <col> starting after diagonal block
//
std::list< gaspi::rank_t >
get_col_procs ( const hpro::TBlockMatrix *  A,
                const uint                  col )
{
    const auto                  nbr = A->nblock_rows();
    const auto                  nbc = A->nblock_cols();
    std::list< gaspi::rank_t >  procs;
    
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
build_row_groups ( const hpro::TBlockMatrix *                   A,
                   std::vector< gaspi::group > &                groups,  // communicator per row
                   std::vector< std::list< gaspi::rank_t > > &  procs )  // set of processors per row
{
    const auto      nbr = A->nblock_rows();
    const auto      nbc = A->nblock_cols();
    gaspi::process  process;
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

        if ( hpro::verbose( 4 ) )
            std::cout << i << " : " << to_string( procs_i ) << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( contains( procs_i, pid ) )
        {
            if ( pos < nbr ) groups[i] = groups[pos];
            else             groups[i] = gaspi::group( procs_i );
        }// if
            
        procs[i] = std::move( procs_i );

        process.barrier();
    }// for
}

//
// same as above but for block columns
//
void
build_col_groups ( const hpro::TBlockMatrix *                   A,
                   std::vector< gaspi::group > &                groups,  // group per column
                   std::vector< std::list< gaspi::rank_t > > &  procs )  // set of processors per column
{
    const auto      nbr = A->nblock_rows();
    const auto      nbc = A->nblock_cols();
    gaspi::process  process;
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

        if ( hpro::verbose( 4 ) )
            std::cout << i << " : " << to_string( procs_i ) << " (" << ( pos == nbc ? i : pos ) << ")" << std::endl;
            
        // use previously created group or create new if none found
        if ( contains( procs_i, pid ) )
        {
            if ( pos < nbc ) groups[i] = groups[pos];
            else             groups[i] = gaspi::group( procs_i );
        }// if
            
        procs[i] = std::move( procs_i );

        process.barrier();
    }// for
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    // assert( RANK != 0 );
    assert( is_blocked( A ) );
    
    gaspi::process  process;
    const auto      pid    = process.rank();
    const auto      nprocs = process.size();
    
    hlr::log( 4, hpro::to_string( "lu( %d )", A->id() ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
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

    std::vector< gaspi::queue >  queues( nprocs );  // one queue to each remote processor
    gaspi::group                 world;

    //
    // set up groups for all rows/columns
    //

    std::vector< gaspi::group >                row_groups( nbr ), col_groups( nbc );
    std::vector< std::list< gaspi::rank_t > >  row_procs( nbr ), col_procs( nbc );

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

    hlr::log( 4, hpro::to_string( "maximal ID: %d", nid ) );
    
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
        
        hlr::log( 4, hpro::to_string( "──────────────── step %d ────────────────", i ) );

        hlr::log( 5, hpro::to_string( "#alloc. segments: %d of %d", process.nalloc_segments(), process.nmax_segments() ) );

        //
        // factorization of diagonal block
        //
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            hlr::log( 4, hpro::to_string( "  invert( %d )", A_ii->id() ) );
            blas::invert( hpro::blas_mat< value_t >( A_ii ) );
        }// if

        //
        // solve off-diagonal blocks in current block column
        // (since L is identity, no solving in block row)
        //

        if ( contains( col_procs[i], pid ) )
        {
            // only communicate if more than one processor involved
            const bool  with_comm = ( col_procs[i].size() > 1 );
            const bool  is_local  = ( pid == p_ii );
            
            //
            // set up destination matrices and GASPI segments
            //

            std::unique_ptr< hpro::TMatrix >   T_ii;        // temporary storage with auto-delete
            hpro::TMatrix *                    H_ii = A_ii; // handle for A_ii/T_ii
            
            if ( ! is_local )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if

            // set up windows/attached memory 
            std::vector< gaspi::segment >  diag_seg;

            if ( with_comm )
                diag_seg = create_segments< value_t >( id(i,i), H_ii, rank, col_groups[i] );
            
            //
            // start sending matrices
            //
            
            if ( with_comm && is_local )
            {
                for ( auto  dest : col_procs[i] )
                    send_matrix< value_t >( diag_seg, H_ii, dest, queues[dest] );
            }// if
            
            //
            // off-diagonal solve
            //

            bool  have_diag = ( pid == p_ii );
            
            for ( uint  j = i+1; j < nbr; ++j )
            {
                // L is unit diagonal !!! Only solve with U
                auto        A_ji = BA->block( j, i );
                const auto  p_ji = A_ji->procs().master();
                
                hlr::log( 4, hpro::to_string( "  solve( %d )", A_ji->id() ) );
                
                if ( pid == p_ji )
                {
                    if ( ! have_diag )
                    {
                        hlr::log( 4, hpro::to_string( "waiting for %d", H_ii->id() ) );
                        wait_matrix( diag_seg );
                        have_diag = true;
                        
                        if ( pid != p_ii )
                            add_mem += H_ii->byte_size();
                    }// if
                    
                    // DBG::printf( "solving %d", A_ji->id() );
                    trsmuh< value_t >( *ptrcast( H_ii, hpro::TDenseMatrix ), *A_ji );
                }// if
            }// for
            
            // finish segments
            for ( auto &  seg : diag_seg )
                seg.release();
        }

        world.barrier();
        
        //
        // start sending matrices to remote processors
        //

        std::vector< std::unique_ptr< hpro::TMatrix > >  row_i_mat( nbr );        // for autodeletion of temp. matrices
        std::vector< std::unique_ptr< hpro::TMatrix > >  col_i_mat( nbc );
        std::vector< hpro::TMatrix * >                   row_i( nbr, nullptr );   // actual matrix handles
        std::vector< hpro::TMatrix * >                   col_i( nbc, nullptr );
        std::vector< std::vector< gaspi::segment > >     row_segs( nbr );         // holds GASPI segments for matrices
        std::vector< std::vector< gaspi::segment > >     col_segs( nbc );

        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji      = BA->block( j, i );
            const auto  p_ji      = A_ji->procs().master();
            const bool  with_comm = ( row_procs[j].size() > 1 );
            const bool  is_local  = ( pid == p_ji );

            if ( contains( row_procs[j], pid ) )
            {
                row_i[j] = A_ji;
            
                if ( ! is_local )
                {
                    row_i_mat[j] = create_matrix( A_ji, mat_types(j,i), pid );
                    row_i[j]     = row_i_mat[j].get();
                }// if
                
                // set up windows/attached memory
                if ( with_comm )
                    row_segs[j] = create_segments< value_t >( id(j,i), row_i[j], rank, row_groups[j] );

                // and start get requests
                if ( with_comm && is_local )
                {
                    for ( auto  dest : row_procs[j] )
                        send_matrix< value_t >( row_segs[j], A_ji, dest, queues[ dest ] );
                }// if
            }// if
        }// for
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il      = BA->block( i, l );
            const auto  p_il      = A_il->procs().master();
            const bool  with_comm = ( col_procs[l].size() > 1 );
            const bool  is_local  = ( pid == p_il );
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                col_i[l] = A_il;
                
                if ( ! is_local )
                {
                    col_i_mat[l] = create_matrix( A_il, mat_types(i,l), pid );
                    col_i[l]     = col_i_mat[l].get();
                }// if
                
                // set up windows/attached memory
                if ( with_comm )
                    col_segs[l] = create_segments< value_t >( id(i,l), col_i[l], rank, col_groups[l] );

                // and start get requests
                if ( with_comm && is_local )
                {
                    for ( auto  dest : col_procs[l] )
                        send_matrix< value_t >( col_segs[l], A_il, dest, queues[ dest ] );
                }// if
            }// if
        }// for

        //
        // update of trailing sub-matrix
        //
        
        std::vector< bool >  row_done( nbr, false );  // signals received matrix
        std::vector< bool >  col_done( nbc, false );
        
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

                    if (( p_ji != pid ) && ! row_done[j] )
                    {
                        hlr::log( 5, hpro::to_string( "waiting for row matrix %d", BA->block(j,i)->id() ) );
                        wait_matrix( row_segs[j] );
                        row_done[j] = true;
                        add_mem += row_i[j]->byte_size();
                    }// if
                            
                    if (( p_il != pid ) && ! col_done[l] )
                    {
                        hlr::log( 5, hpro::to_string( "waiting for col matrix %d", BA->block(i,l)->id() ) );
                        wait_matrix( col_segs[l] );
                        col_done[l] = true;
                        add_mem += col_i[l]->byte_size();
                    }// if

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

        world.barrier();
    }// for

    std::cout << "  add memory  : " << hpro::Mem::to_string( max_add_mem ) << std::endl;
}

}}}// namespace hlr::gaspi::tlr

#endif // __HLR_GASPI_ARITH_HH
