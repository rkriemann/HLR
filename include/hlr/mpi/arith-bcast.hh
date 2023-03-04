#ifndef __HLR_MPI_ARITH_BCAST_HH
#define __HLR_MPI_ARITH_BCAST_HH
//
// Project     : HLR
// Module      : arith-bcast.hh
// Description : arithmetic functions based on MPI blocking broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <list>

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
#include "hlr/dag/lu.hh"
#include "hlr/tbb/dag.hh"
#include "hlr/tbb/arith.hh"

namespace hlr
{

namespace mpi
{

namespace bcast
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// broadcast matrix <A> from <root_proc> to all processors in <comm>
//
template < typename value_t >
void
broadcast ( mpi::communicator &  comm,
            TMatrix *            A,
            const int            root_proc )
{
    // DBG::printf( "broadcast( %d ) from %d", A->id(), root_proc );

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
        assert( false );
}

//
// compute LU factorization of <A>
//
template < typename value_t,
           typename approx_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc,
     const approx_t &   approx )
{
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

    size_t  max_add_mem = 0;

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        // DBG::print(  "────────────────────────────────────────────────" );
        // DBG::printf( "step %d", i );
        
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
        int   p_ii = A_ii->procs().master();

        if ( pid == p_ii )
        {
            // DBG::printf( "invert( %d )", A_ii->id() );
            BLAS::invert( blas_mat< value_t >( A_ii ) );
        }// if

        if ( contains( col_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            std::unique_ptr< TMatrix >  T_ii;        // temporary storage with auto-delete
            TMatrix *                   H_ii = A_ii; // handle for A_ii/T_ii
            
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
            
            ::tbb::parallel_for(
                i+1, nbr,
                [BA,H_ii,i,pid] ( uint  j )
                // for ( uint  j = i+1; j < nbr; ++j )
                {
                    // L is unit diagonal !!! Only solve with U
                    auto       A_ji = BA->block( j, i );
                    const int  p_ji = A_ji->procs().master();
                    
                    if ( pid == p_ji )
                    {
                        // DBG::printf( "solve_U( %d, %d )", H_ii->id(), BA->block( j, i )->id() );
                        trsmuh< value_t >( *ptrcast( H_ii, TDenseMatrix ), *A_ji );
                    }// if
                } );
        }
        
        //
        // broadcast blocks in row/column for update phase
        //

        std::vector< std::unique_ptr< TMatrix > >  row_i_mat( nbr );      // for autodeletion
        std::vector< std::unique_ptr< TMatrix > >  col_i_mat( nbc );
        std::vector< TMatrix * >                   row_i( nbr, nullptr ); // matrix handles
        std::vector< TMatrix * >                   col_i( nbc, nullptr );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
            const int   p_ji = A_ji->procs().master();
            
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
            const auto  A_il = BA->block( i, l );
            const int   p_il = A_il->procs().master();
            
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
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( i+1, nbr,
                                            i+1, nbc ),
            [&,BA,i,pid] ( const ::tbb::blocked_range2d< uint > & r )
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
                        
                        auto       A_jl = BA->block( j, l );
                        const int  p_jl = A_jl->procs().master();
                        
                        if ( pid == p_jl )
                            hlr::multiply< value_t >( value_t(-1),
                                                      apply_normal, *A_ji,
                                                      apply_normal, *A_il,
                                                      *A_jl, acc, approx );
                    }// for
                }// for
            } );
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

//
// broadcast bytestream <bs> from <root_proc> to all processors in <comm>
// 
void
broadcast ( mpi::communicator &  comm,
            TByteStream &        bs,
            const int            root_proc )
{
    // first exchange size information
    size_t  size = bs.size();

    comm.broadcast( size, root_proc );

    log( 5, HLIB::to_string( "bs size = %d", size ) );
    
    if ( bs.size() != size )
        bs.set_size( size );

    // then the actual data
    comm.broadcast( bs.data(), size, root_proc );
}

//
// compute LU factorization of <A>
//
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    assert( is_blocked( A ) );

    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    log( 4, HLIB::to_string( "lu( %d )", A->id() ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
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

    size_t  max_add_mem = 0;

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        log( 4, "────────────────────────────────────────────────" );
        log( 4, HLIB::to_string( "step %d", i ) );
        
        auto  A_ii = BA->block( i, i );
        int   p_ii = A_ii->procs().master();

        assert( A_ii->procs().size() == 1 );
        
        if ( pid == p_ii )
        {
            log( 4, HLIB::to_string( "lu( %d )", A_ii->id() ) );

            // HLIB::LU::factorise_rec( A_ii, acc );
            auto  dag = std::move( dag::gen_dag_lu_oop_auto( *A_ii, 128, tbb::dag::refine ) );

            tbb::dag::run( dag, acc );
        }// if

        if ( contains( col_procs[i], pid ) || contains( row_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            auto  tic = Time::Wall::now();
            
            std::unique_ptr< TMatrix >  T_ii;        // temporary storage with auto-delete
            TMatrix *                   H_ii = A_ii; // handle for A_ii/T_ii
            TByteStream                 bs;

            if ( pid == p_ii )
            {
                log( 4, HLIB::to_string( "serialization of %d ", A_ii->id() ) );
                
                bs.set_size( A_ii->bs_size() );
                A_ii->write( bs );
            }// if

            // broadcast serialized data
            if ( contains( col_procs[i], pid ) )
            {
                log( 4, HLIB::to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + hlr::to_string( col_procs[i] ) );
                broadcast( col_comms[i], bs, col_maps[i][p_ii] );
            }// if

            if (( col_procs[i] != row_procs[i] ) && contains( row_procs[i], pid ))
            {
                log( 4, HLIB::to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + hlr::to_string( row_procs[i] ) );
                broadcast( row_comms[i], bs, row_maps[i][p_ii] );
            }// if
            
            // and reconstruct matrix
            if ( pid != p_ii )
            {
                log( 4, HLIB::to_string( "construction of %d ", A_ii->id() ) );
                
                TBSHMBuilder  bs_hbuild;

                T_ii = bs_hbuild.build( bs );
                T_ii->set_procs( ps_single( pid ) );
                H_ii = T_ii.get();

                add_mem += H_ii->bs_size();
            }// if
            
            auto  toc = Time::Wall::since( tic );

            // time_mpi += toc.seconds();
            
            //
            // off-diagonal solves
            //
            
            for ( uint  j = i+1; j < nbr; ++j )
            {
                // L is unit diagonal !!! Only solve with U
                auto       A_ji = BA->block( j, i );
                const int  p_ji = A_ji->procs().master();

                assert( A_ji->procs().size() == 1 );
                
                if ( pid == p_ji )
                {
                    log( 4, HLIB::to_string( "solve_U( %d, %d )", H_ii->id(), A_ji->id() ) );

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
                    log( 4, HLIB::to_string( "solve_L( %d, %d )", H_ii->id(), A_il->id() ) );

                    // solve_lower_left( apply_normal, H_ii, nullptr, A_il, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
                    auto  dag = std::move( gen_dag_solve_lower( *H_ii, *A_il, 128, tbb::dag::refine ) );

                    tbb::dag::run( dag, acc );
                }// if
            }// for
        }
        
        //
        // broadcast blocks in row/column for update phase
        //

        std::vector< std::unique_ptr< TMatrix > >  row_i_mat( nbr );      // for autodeletion
        std::vector< std::unique_ptr< TMatrix > >  col_i_mat( nbc );
        std::vector< TMatrix * >                   row_i( nbr, nullptr ); // matrix handles
        std::vector< TMatrix * >                   col_i( nbc, nullptr );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = BA->block( j, i );
            const int   p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                TByteStream  bs;
                
                if ( pid == p_ji )
                {
                    log( 4, HLIB::to_string( "serialisation of %d ", A_ji->id() ) );
                    
                    bs.set_size( A_ji->bs_size() );
                    A_ji->write( bs );
                    row_i[j] = A_ji;
                }// if
                
                log( 4, HLIB::to_string( "broadcast %d from %d to ", A_ji->id(), p_ji ) + hlr::to_string( row_procs[j] ) );
                
                broadcast( row_comms[j], bs, row_maps[j][p_ji] );

                if ( pid != p_ji )
                {
                    log( 4, HLIB::to_string( "construction of %d ", A_ji->id() ) );
                    
                    TBSHMBuilder  bs_hbuild;

                    row_i_mat[j] = bs_hbuild.build( bs );
                    row_i_mat[j]->set_procs( ps_single( pid ) );
                    row_i[j]     = row_i_mat[j].get();
                    add_mem     += row_i[j]->byte_size();
                }// if
            }// if
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = BA->block( i, l );
            const int   p_il = A_il->procs().master();
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                TByteStream  bs;
                
                if ( pid == p_il )
                {
                    log( 4, HLIB::to_string( "serialisation of %d ", A_il->id() ) );
                    
                    bs.set_size( A_il->bs_size() );
                    A_il->write( bs );
                    col_i[l] = A_il;
                }// if
                
                log( 4, HLIB::to_string( "broadcast %d from %d to ", A_il->id(), p_il ) + hlr::to_string( col_procs[l] ) );
                
                broadcast( col_comms[l], bs, col_maps[l][p_il] );

                if ( pid != p_il )
                {
                    log( 4, HLIB::to_string( "construction of %d ", A_il->id() ) );
                    
                    TBSHMBuilder  bs_hbuild;

                    col_i_mat[l] = bs_hbuild.build( bs );
                    col_i_mat[l]->set_procs( ps_single( pid ) );
                    col_i[l]     = col_i_mat[l].get();
                    add_mem     += col_i[l]->byte_size();
                }// if
            }// if
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // update of trailing sub-matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = row_i[j];
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const auto  A_il = col_i[l];
                
                //
                // update local matrix block
                //
                
                auto       A_jl = BA->block( j, l );
                const int  p_jl = A_jl->procs().master();
                
                if ( pid == p_jl )
                {
                    log( 4, HLIB::to_string( "update of %d with %d × %d", A_jl->id(), A_ji->id(), A_il->id() ) );

                    // recursive method shows best parallel performance
                    hlr::tbb::multiply( -1.0, apply_normal, *A_ji, apply_normal, *A_il, *A_jl, acc );
                }// if
            }// for
        }// for
    }// for

    // std::cout << "  time in MPI : " << HLIB::to_string( "%.2fs", time_mpi ) << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

}// namespace tileh

}// namespace bcast

}// namespace mpi

}// namespace hlr

#endif // __HLR_MPI_ARITH_BCAST_HH
