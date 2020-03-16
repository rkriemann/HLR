#ifndef __HLR_TBB_ARITH_ACCU_HH
#define __HLR_TBB_ARITH_ACCU_HH
//
// Project     : HLib
// File        : arith.hh
// Description : arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/algebra/mat_add.hh>
#include <hpro/algebra/mat_conv.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/matrix/restrict.hh"

namespace hlr { namespace tbb { namespace accu {

namespace hpro = HLIB;

using namespace hpro;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

namespace detail
{

using  upd_list_t = std::list< std::pair< const TMatrix *, const TMatrix * > >;

//
// forward decl.
//
template < typename value_t >
void
multiply ( const value_t                 alpha,
           const hpro::matop_t           op_A,
           const hpro::matop_t           op_B,
           hpro::TMatrix &               C,
           upd_list_t &                  upd_C,
           std::unique_ptr< TMatrix > &  upd_accu,
           const hpro::TTruncAcc &       acc );

//
// special case : C is low-rank matrix
// - construct sub-blocks of C for all blocked updates with corresponding accumulators
//
template < typename value_t >
void
multiply ( const value_t                 alpha,
           const hpro::matop_t           op_A,
           const hpro::matop_t           op_B,
           hpro::TRkMatrix &             C,
           upd_list_t &                  upd_C,
           std::unique_ptr< TMatrix > &  upd_accu,
           const hpro::TTruncAcc &       acc )
{
    //
    // first handle updates to C with at least one leaf block
    //

    std::unique_ptr< TMatrix >       U( std::move( upd_accu ) );
    std::unique_ptr< TBlockMatrix >  BC;
    ::tbb::affinity_partitioner      ap; // to ensure (???) allocated data is used in equal threads
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( is_blocked_all( *A, *B ) )
        {
            //
            // set up block matrix with low-rank sub-blocks
            //

            if ( is_null( BC ) )
            {
                auto  BA = cptrcast( A, TBlockMatrix );
                auto  BB = cptrcast( B, TBlockMatrix );
                
                BC = std::make_unique< TBlockMatrix >( C.row_is(), C.col_is() );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                ::tbb::parallel_for(
                    ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                    0, BC->nblock_cols() ),
                    [&,alpha,op_A,op_B,BA,BB] ( const auto & r )
                    {
                        for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                            for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                            {
                                HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                                
                                BC->set_block( i, j, new TRkMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                    BB->block( 0, j, op_B )->col_is( op_B ),
                                                                    C.value_type() ) );
                            }// for
                    },
                    ap );
            }// if
        }// if
        else
        {
            auto  T = multiply< value_t >( alpha, op_A, A, op_B, B );
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else
            {
                if ( is_dense( T.get() ) )
                {
                    hpro::add( value_t(1), U.get(), value_t(1), T.get(), acc );
                    U = std::move( T );
                }// if
                else
                    hpro::add( value_t(1), T.get(), value_t(1), U.get(), acc );
            }// else
        }// else
    }// for

    //
    // now handle recursive updates
    //
    
    if ( ! is_null( BC ) )
    {
        //
        // first, split update U into subblock updates
        // (to release U before recursion)
        //

        tensor2< std::unique_ptr< TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        
        if ( ! is_null( U ) )
        {
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha,op_A,op_B] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                            sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                },
                ap );
            
            U.reset( nullptr );
        }// if

        //
        // apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,alpha,op_A,op_B] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        upd_list_t  upd_ij;
                
                        for ( auto  [ A, B ] : upd_C )
                        {
                            if ( is_blocked_all( A, B ) )
                            {
                                auto  BA = cptrcast( A, TBlockMatrix );
                                auto  BB = cptrcast( B, TBlockMatrix );
                        
                                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                                {
                                    auto  A_il = BA->block( i, l, op_A );
                                    auto  B_lj = BB->block( l, j, op_B );
                            
                                    if ( is_null_any( A_il, B_lj ) )
                                        continue;
                            
                                    upd_ij.push_back( { A_il, B_lj } );
                                }// for
                            }// if
                        }// for

                        HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                        multiply< value_t >( alpha, op_A, op_B, *BC->block( i, j ), upd_ij, sub_U( i, j ), acc );
                    }// for
                }// for
            },
            ap );

        //
        // finally convert subblocks to single low-rank matrix for new accumulated updates
        //

        U = to_rank( BC.get(), acc );
    }// if

    //
    // apply all accumulated updates
    //

    if ( ! is_null( U ) )
        hpro::add( alpha, U.get(), value_t(1), &C, acc );
}

//
// general version to compute C = C + α op( A ) op( B )
//
template < typename value_t >
void
multiply ( const value_t                 alpha,
           const hpro::matop_t           op_A,
           const hpro::matop_t           op_B,
           hpro::TMatrix &               C,
           upd_list_t &                  upd_C,
           std::unique_ptr< TMatrix > &  upd_accu,
           const hpro::TTruncAcc &       acc )
{
    if ( is_lowrank( C ) )
    {
        multiply< value_t >( alpha, op_A, op_B, *ptrcast( &C, TRkMatrix ), upd_C, upd_accu, acc );
        return;
    }// if
    
    //
    // first handle computable updates to C, including non-blocked C
    //

    std::unique_ptr< TMatrix >  U( std::move( upd_accu ) );
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( ! is_blocked_all( *A, *B, C ) )
        {
            auto  T = std::unique_ptr< TMatrix >();
            
            if ( is_blocked_all( *A, *B ) )
            {
                T = std::make_unique< TRkMatrix >( C.row_is(), C.col_is(), hpro::value_type< value_t >::value );

                hpro::multiply( value_t(1), op_A, A, op_B, B, value_t(0), T.get(), acc );
            }// if
            else
            {
                // either A or B is low-rank or dense
                T = multiply< value_t >( alpha, op_A, A, op_B, B );
            }// else
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else
            {
                if ( is_dense( T.get() ) )
                {
                    hpro::add( value_t(1), U.get(), value_t(1), T.get(), acc );
                    U = std::move( T );
                }// if
                else
                    hpro::add( value_t(1), T.get(), value_t(1), U.get(), acc );
            }// else
        }// if
    }// for

    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, TBlockMatrix );

        //
        // first, split update U into subblock updates
        // (to release U before recursion)
        //

        tensor2< std::unique_ptr< TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        ::tbb::affinity_partitioner            ap;
        
        if ( ! is_null( U ) )
        {
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,BC,alpha,op_A,op_B] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            if ( ! is_null( BC->block( i, j ) ) )
                                sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                        }// for
                    }// for
                },
                ap );
            
            U.reset( nullptr );
        }// if

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,BC,alpha,op_A,op_B] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  C_ij = BC->block(i,j);

                        HLR_ASSERT( ! is_null( C_ij ) );

                        upd_list_t  upd_ij;
                
                        for ( auto  [ A, B ] : upd_C )
                        {
                            if ( is_blocked_all( A, B ) )
                            {
                                auto  BA = cptrcast( A, TBlockMatrix );
                                auto  BB = cptrcast( B, TBlockMatrix );
                                
                                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                                {
                                    auto  A_il = BA->block( i, l, op_A );
                                    auto  B_lj = BB->block( l, j, op_B );
                                    
                                    if ( is_null_any( A_il, B_lj ) )
                                        continue;
                                    
                                    upd_ij.push_back( { A_il, B_lj } );
                                }// for
                            }// if
                        }// for
                        
                        multiply< value_t >( alpha, op_A, op_B, *C_ij, upd_ij, sub_U( i, j ), acc );
                    }// for
                }// for
            },
            ap );
    }// if
    else
    {
        // apply accumulated updates
        hpro::add( alpha, U.get(), value_t(1), &C, acc );
    }// else
}

}// namespace detail

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc )
{
    std::unique_ptr< TMatrix >  U;
    detail::upd_list_t          upd{ { &A, &B } };
    
    detail::multiply< value_t >( alpha, op_A, op_B, C, upd, U, acc );
}

}}}// namespace hlr::tbb::accu

#endif // __HLR_TBB_ARITH_ACCU_HH
