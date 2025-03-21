#ifndef __HLR_MPI_ARITH_IBCAST_HH
#define __HLR_MPI_ARITH_IBCAST_HH
//
// Project     : HLR
// Module      : arith-bcast.hh
// Description : arithmetic functions based on MPI blocking broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
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
#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/mpi/arith.hh"
#include "hlr/dag/lu.hh"
#include "hlr/tbb/dag.hh"
#include "hlr/tbb/arith.hh"

namespace hlr
{

namespace hpro = HLIB;

namespace mpi
{

namespace ibcast
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// initiate broadcast of <A> from <root_proc> to all processors in <comm>
//
template < typename value_t >
std::vector< mpi::request >
ibroadcast ( mpi::communicator &  comm,
             hpro::TMatrix *      A,
             const int            root_proc,
             const int            rank )
{
    log( 4, hpro::to_string( "broadcast( %d ) from %d", A->id(), root_proc ) );

    std::vector< mpi::request >  reqs;
    
    auto  tic = Time::Wall::now();
    
    if ( is_dense( A ) )
    {
        auto  D = ptrcast( A, hpro::TDenseMatrix );
        
        reqs.reserve( 1 );
        reqs.push_back( comm.ibroadcast( hpro::blas_mat< value_t >( D ).data(), D->nrows() * D->ncols(), root_proc ) );
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = ptrcast( A, hpro::TRkMatrix );
        auto  k = R->rank();

        R->set_rank( rank );
        
        reqs.reserve( 2 );
        
        // reqs.push_back( comm.ibroadcast( k, root_proc ) );
        // std::cout << "ibcast : " << reqs[0] << std::endl;

        reqs.push_back( comm.ibroadcast( hpro::blas_mat_A< value_t >( R ).data(), R->nrows() * rank, root_proc ) );
        reqs.push_back( comm.ibroadcast( hpro::blas_mat_B< value_t >( R ).data(), R->ncols() * rank, root_proc ) );
    }// if
    else
        assert( false );
    
    return std::move( reqs );
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    assert( is_blocked( A ) );
    // assert( RANK != 0 );
    
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    if ( hpro::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

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

    // std::cout << "exchanging matrix types" << std::endl;
    
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
        
        hlr::log( 4, hpro::to_string( "──────────────── step %d ────────────────", i ) );
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        int   p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            // DBG::printf( "invert( %d )", A_ii->id() );
            blas::invert( blas_mat< value_t >( A_ii ) );
        }// if

        // nothing to solve/update at last step
        if ( i == nbr-1 )
            break;
        
        if ( contains( col_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            std::unique_ptr< hpro::TMatrix >  T_ii;        // temporary storage with auto-delete
            TMatrix *                         H_ii = A_ii; // handle for A_ii/T_ii
            
            if ( pid != p_ii )
            {
                T_ii = create_matrix( A_ii, mat_types(i,i), pid );
                H_ii = T_ii.get();
            }// if
            
            auto  diag_reqs{ ibroadcast< value_t >( col_comms[i], H_ii, col_maps[i][p_ii], rank ) };

            //
            // off-diagonal solve
            //

            bool        have_diag = ( pid == p_ii );
            std::mutex  req_mtx;
            
            // tbb::parallel_for(
            //     i+1, nbr,
            //     [&,A,H_ii,i,pid,p_ii] ( uint  j )
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    // L is unit diagonal !!! Only solve with U
                    auto       A_ji = BA->block( j, i );
                    const int  p_ji = A_ji->procs().master();
                
                    if ( pid == p_ji )
                    {
                        {
                            std::lock_guard< std::mutex >  lock( req_mtx );
                        
                            if ( ! have_diag )
                            {
                                wait_all( diag_reqs );
                                have_diag = true;
                            
                                if ( pid != p_ii )
                                    add_mem += H_ii->byte_size();
                            }// if
                        }
                    
                        trsmuh< value_t >( *ptrcast( H_ii, hpro::TDenseMatrix ), *A_ji );
                    }// if
                } // );

            // wait also on sending processor
            if ( pid == p_ii )
                wait_all( diag_reqs );

            for ( auto & req : diag_reqs )
            {
                if ( req.mpi_request != MPI_REQUEST_NULL )
                    hlr::log( 0, hpro::to_string( "open request at %d", __LINE__ ) );
            }// for
        }
        
        //
        // broadcast blocks in row/column for update phase
        //
        
        std::vector< std::unique_ptr< hpro::TMatrix > >   row_i_mat( nbr );        // for autodeletion of temp. matrices
        std::vector< std::unique_ptr< hpro::TMatrix > >   col_i_mat( nbc );
        std::vector< hpro::TMatrix * >                    row_i( nbr, nullptr );   // actual matrix handles
        std::vector< hpro::TMatrix * >                    col_i( nbc, nullptr );
        std::vector< std::vector< mpi::request > >        row_reqs( nbr );         // holds MPI requests for matrices
        std::vector< std::vector< mpi::request > >        col_reqs( nbc );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
            const int   p_ji = A_ji->procs().master();
            
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
                row_reqs[j] = ibroadcast< value_t >( row_comms[j], row_i[j], row_maps[j][p_ji], rank );
            }// if
        }
        
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
                
                // DBG::printf( "broadcasting %d from %d", A_il->id(), p_il );
                col_reqs[l] = ibroadcast< value_t >( col_comms[l], col_i[l], col_maps[l][p_il], rank );
            }// if
        }// for
        
        //
        // update of trailing sub-matrix
        //
        
        std::vector< bool >             row_done( nbr, false );  // signals finished broadcast
        std::vector< bool >             col_done( nbc, false );
        std::vector< std::mutex >       row_mtx( nbr );          // mutices for access to requests
        std::vector< std::mutex >       col_mtx( nbc );
        ::tbb::blocked_range2d< uint >  range( i+1, nbr,
                                               i+1, nbc );
        
        // ::tbb::parallel_for( blocks,
        //       [&,BA,i,pid] ( const ::tbb::blocked_range2d< uint > & range )
            {
                for ( auto  j = range.rows().begin(); j != range.rows().end(); ++j )
                {
                    const int  p_ji = BA->block( j, i )->procs().master();
                    
                    for ( uint  l = range.cols().begin(); l != range.cols().end(); ++l )
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
                                    // DBG::printf( "waiting for %d", A->block(j,i)->id() );
                                    mpi::wait_all( row_reqs[j] );
                                    row_done[j] = true;
                                    add_mem    += row_i[j]->byte_size();
                                }// if
                            }

                            {
                                std::lock_guard< std::mutex >  lock( col_mtx[l] );
                        
                                if (( p_il != pid ) && ! col_done[l] )
                                {
                                    // DBG::printf( "waiting for %d", A->block(i,l)->id() );
                                    mpi::wait_all( col_reqs[l] );
                                    col_done[l] = true;
                                    add_mem    += col_i[l]->byte_size();
                                }// if
                            }

                            //
                            // finally compute update
                            //
                    
                            hlr::multiply< value_t >( value_t(-1),
                                                      apply_normal, *row_i[j],
                                                      apply_normal, *col_i[l],
                                                      *A_jl, acc, approx );
                        }// if
                    }// for
                }// for
            } // );

        //
        // wait also on sending processor
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            if ( contains( row_procs[j], pid ) && ( pid == BA->block( j, i )->procs().master() ))
                wait_all( row_reqs[j] );
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            if ( contains( col_procs[l], pid ) && ( pid == BA->block( i, l )->procs().master() ))
                wait_all( col_reqs[l] );
        }// for
        
        max_add_mem = std::max( max_add_mem, add_mem );

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( auto & req : row_reqs[j] )
            {
                if ( req.mpi_request != MPI_REQUEST_NULL )
                    hlr::log( 0, hpro::to_string( "open request at %d", __LINE__ ) );
            }// for
        }// for

        for ( uint  l = i+1; l < nbc; ++l )
        {
            for ( auto & req : col_reqs[l] )
            {
                if ( req.mpi_request != MPI_REQUEST_NULL )
                    hlr::log( 0, hpro::to_string( "open request at %d", __LINE__ ) );
            }// for
        }// for
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

mpi::request
ibroadcast ( mpi::communicator &  comm,
             hpro::TByteStream &  bs,
             const int            root_proc )
{
    // first exchange size information
    size_t  size = bs.size();

    comm.broadcast( size, root_proc );

    log( 5, hpro::to_string( "bs size = %d", size ) );
    
    if ( bs.size() != size )
        bs.set_size( size );

    // then the actual data
    return comm.ibroadcast( bs.data(), size, root_proc );
}

size_t  max_add_mem = 0;

template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    assert( is_blocked( A ) );

    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    log( 4, hpro::to_string( "lu( %d )", A->id() ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

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

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        hlr::log( 4, hpro::to_string( "──────────────── step %d ────────────────", i ) );
        
        auto  A_ii = BA->block( i, i );
        int   p_ii = A_ii->procs().master();

        assert( A_ii->procs().size() == 1 );
        
        if ( pid == p_ii )
        {
            log( 4, hpro::to_string( "lu( %d )", A_ii->id() ) );

            // hpro::LU::factorise_rec( A_ii, acc );
            auto  dag = std::move( dag::gen_dag_lu_oop_auto( *A_ii, 128, tbb::dag::refine ) );

            tbb::dag::run( dag, acc );
        }// if

        // nothing to solve/update at last step
        if ( i == nbr-1 )
            break;
        
        if ( contains( col_procs[i], pid ) || contains( row_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            std::unique_ptr< hpro::TMatrix >  T_ii;        // temporary storage with auto-delete
            hpro::TMatrix *                   H_ii = A_ii; // handle for A_ii/T_ii
            hpro::TByteStream                 bs;
            mpi::request                      col_req_ii, row_req_ii;

            if ( pid == p_ii )
            {
                log( 4, hpro::to_string( "serialization of %d ", A_ii->id() ) );
                
                bs.set_size( A_ii->bs_size() );
                A_ii->write( bs );
            }// if

            // broadcast serialized data
            if ( contains( col_procs[i], pid ) )
            {
                log( 4, hpro::to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + hlr::to_string( col_procs[i] ) );
                col_req_ii = ibroadcast( col_comms[i], bs, col_maps[i][p_ii] );
            }// if

            if (( col_procs[i] != row_procs[i] ) && contains( row_procs[i], pid ))
            {
                log( 4, hpro::to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + hlr::to_string( row_procs[i] ) );
                row_req_ii = ibroadcast( row_comms[i], bs, row_maps[i][p_ii] );
            }// if
            
            //
            // off-diagonal solves
            //

            bool  recv_ii = false;

            auto  wait_ii =
                [&,A_ii,i,pid] ()
                {
                    if ( ! recv_ii )
                    {
                        log( 4, hpro::to_string( "construction of %d ", A_ii->id() ) );

                        if ( contains( col_procs[i], pid ) ) col_req_ii.wait();
                        else                                 row_req_ii.wait();
                        
                        hpro::TBSHMBuilder  bs_hbuild;

                        T_ii    = bs_hbuild.build( bs );
                        T_ii->set_procs( ps_single( pid ) );
                        H_ii    = T_ii.get();
                        recv_ii = true;
                        
                        add_mem += H_ii->bs_size();
                    }// if
                };
            
            for ( uint  j = i+1; j < nbr; ++j )
            {
                // L is unit diagonal !!! Only solve with U
                auto       A_ji = BA->block( j, i );
                const int  p_ji = A_ji->procs().master();

                assert( A_ji->procs().size() == 1 );
                
                if ( pid == p_ji )
                {
                    if ( pid != p_ii )
                        wait_ii();
            
                    log( 4, hpro::to_string( "solve_U( %d, %d )", H_ii->id(), A_ji->id() ) );

                    // solve_upper_right( A_ji, H_ii, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
                    auto  dag = std::move( gen_dag_solve_upper( *H_ii, *A_ji, 128, tbb::dag::refine ) );

                    tbb::dag::run( dag, acc );
                }// if
            }// for

            for ( uint  l = i+1; l < nbc; ++l )
            {
                // L is unit diagonal !!! Only solve with U
                auto       A_il = BA->block( i, l );
                const int  p_il = A_il->procs().master();
                
                assert( A_il->procs().size() == 1 );
                
                if ( pid == p_il )
                {
                    if ( pid != p_ii )
                        wait_ii();
                    
                    log( 4, hpro::to_string( "solve_L( %d, %d )", H_ii->id(), A_il->id() ) );

                    //solve_lower_left( apply_normal, H_ii, nullptr, A_il, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
                    auto  dag = std::move( gen_dag_solve_lower( *H_ii, *A_il, 128, tbb::dag::refine ) );

                    tbb::dag::run( dag, acc );
                }// if
            }// for

            if ( pid == p_ii )
            {
                col_req_ii.wait();
                if ( col_procs[i] != row_procs[i] )
                    row_req_ii.wait();
            }// if
        }
        
        //
        // broadcast blocks in row/column for update phase
        //

        std::vector< hpro::TByteStream >                 row_i_bs( nbr );       // bytestreams for communication
        std::vector< hpro::TByteStream >                 col_i_bs( nbc );       // 
        std::vector< std::unique_ptr< hpro::TMatrix > >  row_i_mat( nbr );      // for autodeletion
        std::vector< std::unique_ptr< hpro::TMatrix > >  col_i_mat( nbc );
        std::vector< hpro::TMatrix * >                   row_i( nbr, nullptr ); // matrix handles
        std::vector< hpro::TMatrix * >                   col_i( nbc, nullptr );
        std::vector< mpi::request >                      row_reqs( nbr );       // holds MPI requests for matrices
        std::vector< mpi::request >                      col_reqs( nbc );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
            const int   p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                if ( pid == p_ji )
                {
                    log( 4, hpro::to_string( "serialisation of %d ", A_ji->id() ) );
                    
                    row_i_bs[j].set_size( A_ji->bs_size() );
                    A_ji->write( row_i_bs[j] );
                    row_i[j] = A_ji;
                }// if
                
                log( 4, hpro::to_string( "broadcast %d from %d to ", A_ji->id(), p_ji ) + hlr::to_string( row_procs[j] ) );

                row_reqs[j] = ibroadcast( row_comms[j], row_i_bs[j], row_maps[j][p_ji] );
                add_mem    += row_i_bs[j].size();
            }// if
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = BA->block( i, l );
            const int   p_il = A_il->procs().master();
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                if ( pid == p_il )
                {
                    log( 4, hpro::to_string( "serialisation of %d ", A_il->id() ) );
                    
                    col_i_bs[l].set_size( A_il->bs_size() );
                    A_il->write( col_i_bs[l] );
                    col_i[l] = A_il;
                }// if
                
                log( 4, hpro::to_string( "broadcast %d from %d to ", A_il->id(), p_il ) + hlr::to_string( col_procs[l] ) );
                
                col_reqs[l] = ibroadcast( col_comms[l], col_i_bs[l], col_maps[l][p_il] );
                add_mem    += col_i_bs[l].size();
            }// if
        }// for
        
        //
        // update of trailing sub-matrix
        //
        
        std::vector< bool >  row_done( nbr, false );  // signals finished broadcast
        std::vector< bool >  col_done( nbc, false );

        for ( uint  j = i+1; j < nbr; ++j )
        {
            const int  p_ji = BA->block( j, i )->procs().master();
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const int  p_il = BA->block( i, l )->procs().master();
                
                auto       A_jl = BA->block( j, l );
                const int  p_jl = A_jl->procs().master();
                
                if ( pid == p_jl )
                {
                    //
                    // ensure broadcasts fir A_ji and A_il have finished
                    //
                    
                    if (( p_ji != pid ) && ! row_done[j] )
                    {
                        // DBG::printf( "waiting for %d", A->block(j,i)->id() );
                        row_reqs[j].wait();
                        
                        log( 4, hpro::to_string( "construction of %d ", BA->block( j, i )->id() ) );
                        
                        hpro::TBSHMBuilder  bs_hbuild;
                        
                        row_i_mat[j] = bs_hbuild.build( row_i_bs[j] );
                        row_i_mat[j]->set_procs( ps_single( pid ) );
                        row_i[j]     = row_i_mat[j].get();
                        row_done[j]  = true;
                        add_mem     += row_i[j]->byte_size();
                    }// if

                    if (( p_il != pid ) && ! col_done[l] )
                    {
                        // DBG::printf( "waiting for %d", A->block(i,l)->id() );
                        col_reqs[l].wait();

                        log( 4, hpro::to_string( "construction of %d ", BA->block( i, l )->id() ) );
                    
                        hpro::TBSHMBuilder  bs_hbuild;

                        col_i_mat[l] = bs_hbuild.build( col_i_bs[l] );
                        col_i_mat[l]->set_procs( ps_single( pid ) );
                        col_i[l]     = col_i_mat[l].get();
                        col_done[l]  = true;
                        add_mem     += col_i[l]->byte_size();
                    }// if

                    //
                    // update local matrix block
                    //
                
                    log( 4, hpro::to_string( "update of %d with %d × %d", A_jl->id(), row_i[j]->id(), col_i[l]->id() ) );
                    
                    // recursive method has same degree of parallelism as DAG method
                    hlr::tbb::multiply( -1.0, apply_normal, *row_i[j], apply_normal, *col_i[l], *A_jl, acc );
                }// if
            }// for
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );

        //
        // wait also on sending processor
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            if ( pid == BA->block( j, i )->procs().master() )
                row_reqs[j].wait();
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            if ( pid == BA->block( i, l )->procs().master() )
                col_reqs[l].wait();
        }// for
        
        max_add_mem = std::max( max_add_mem, add_mem );

        for ( uint  j = i+1; j < nbr; ++j )
        {
            if ( row_reqs[j].mpi_request != MPI_REQUEST_NULL )
                hlr::log( 0, hpro::to_string( "open request at %d", __LINE__ ) );
        }// for

        for ( uint  l = i+1; l < nbc; ++l )
        {
            if ( col_reqs[l].mpi_request != MPI_REQUEST_NULL )
                hlr::log( 0, hpro::to_string( "open request at %d", __LINE__ ) );
        }// for
    }// for

    // std::cout << "  time in MPI : " << hpro::to_string( "%.2fs", time_mpi ) << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

}// namespace tileh

}// namespace ibcast

}// namespace mpi

}// namespace hlr

#endif // __HLR_MPI_ARITH_IBCAST_HH
