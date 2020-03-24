#ifndef __HLR_SEQ_ARITH_ACCU_HH
#define __HLR_SEQ_ARITH_ACCU_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_add.hh>
#include <hpro/algebra/mat_conv.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/tensor.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/add.hh"
#include "hlr/matrix/restrict.hh"

namespace hlr { namespace seq { namespace accu {

namespace hpro = HLIB;

namespace detail
{

using  upd_list_t = std::list< std::pair< const hpro::TMatrix *, const hpro::TMatrix * > >;

//
// forward decl.
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TMatrix &                     C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx );

//
// special case : C is low-rank matrix
// - construct sub-blocks of C for all blocked updates with corresponding accumulators
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TRkMatrix &                   C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx )
{
    //
    // first handle all updates to C which have at least one leaf block factor
    //

    std::unique_ptr< hpro::TMatrix >       U( std::move( upd_accu ) );
    std::unique_ptr< hpro::TBlockMatrix >  BC;
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( is_blocked_all( *A, *B ) )
        {
            //
            // for recursive part below, we set up a temporary block matrix
            // with low-rank sub-blocks
            //

            if ( is_null( BC ) )
            {
                auto  BA = cptrcast( A, hpro::TBlockMatrix );
                auto  BB = cptrcast( B, hpro::TBlockMatrix );
                
                BC = std::make_unique< hpro::TBlockMatrix >( C.row_is(), C.col_is() );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                        
                        BC->set_block( i, j, new hpro::TRkMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                  BB->block( 0, j, op_B )->col_is( op_B ),
                                                                  C.value_type() ) );
                    }// for
            }// if
        }// if
        else
        {
            auto  T = multiply< value_t >( alpha, op_A, A, op_B, B );
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else if ( is_dense( T.get() ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *U, *T, acc, approx );
                U = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *U, acc, approx );
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

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        
        if ( ! is_null( U ) )
        {
            for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );

            U.reset( nullptr );
        }// if

        //
        // apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                upd_list_t  upd_ij;
                
                for ( auto  [ A, B ] : upd_C )
                {
                    // filter out above handled updates
                    if ( ! is_blocked_all( A, B ) )
                        continue;
                    
                    auto  BA = cptrcast( A, hpro::TBlockMatrix );
                    auto  BB = cptrcast( B, hpro::TBlockMatrix );
                        
                    for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                    {
                        auto  A_il = BA->block( i, l, op_A );
                        auto  B_lj = BB->block( l, j, op_B );
                        
                        if ( is_null_any( A_il, B_lj ) )
                            continue;
                        
                        upd_ij.push_back( { A_il, B_lj } );
                    }// for
                }// for

                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                multiply< value_t >( alpha, op_A, op_B, *BC->block( i, j ), upd_ij, sub_U( i, j ), acc, approx );
            }// for
        }// for

        //
        // finally convert subblocks to single low-rank matrix for new accumulated updates
        //

        if ( ! is_null( U ) )
            HLR_ERROR( "accumulator non-null" );
        
        U = to_rank( BC.get(), acc );
    }// if

    //
    // apply all accumulated updates
    //

    if ( ! is_null( U ) )
        hlr::add( alpha, *U, C, acc, approx );
}

//
// general version to compute C = C + α op( A ) op( B )
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TMatrix &                     C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx )
{
    if ( is_lowrank( C ) )
    {
        multiply< value_t >( alpha, op_A, op_B, *ptrcast( &C, hpro::TRkMatrix ), upd_C, upd_accu, acc, approx );
        return;
    }// if
    
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    std::unique_ptr< hpro::TMatrix >  U( std::move( upd_accu ) );
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( ! is_blocked_all( *A, *B, C ) )
        {
            auto  T = std::unique_ptr< hpro::TMatrix >();

            //
            // compute update
            //
            
            if ( is_blocked_all( *A, *B ) )
            {
                //
                // assumption: dense matrices only on lowest level in matrix,
                //             therefore C has to be of low-rank format
                //

                HLR_ERROR( C.typestr() + " += blocked × blocked" );
                
                T = std::make_unique< hpro::TRkMatrix >( C.row_is(), C.col_is(), hpro::value_type< value_t >::value );

                hpro::multiply( value_t(1), op_A, A, op_B, B, value_t(0), T.get(), acc );
            }// if
            else
            {
                // either A or B is low-rank or dense
                T = multiply< value_t >( alpha, op_A, A, op_B, B );
            }// else

            //
            // apply update to accumulator
            //
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else if ( is_dense( T.get() ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *U, *T, acc, approx );
                U = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *U, acc, approx );
            }// else
        }// if
    }// for

    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // first, split update U into subblock updates
        // (to release U before recursion and by that avoid
        //  memory consumption dependent on hierarchy depth)
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        
        if ( ! is_null( U ) )
        {
            for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    if ( ! is_null( BC->block( i, j ) ) )
                        sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );

            U.reset( nullptr );
        }// if

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);

                HLR_ASSERT( ! is_null( C_ij ) );

                upd_list_t  upd_ij;
                
                for ( auto  [ A, B ] : upd_C )
                {
                    // filter out above handled updates
                    if ( ! is_blocked_all( A, B ) )
                        continue;
                    
                    auto  BA = cptrcast( A, hpro::TBlockMatrix );
                    auto  BB = cptrcast( B, hpro::TBlockMatrix );
                        
                    for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                    {
                        auto  A_il = BA->block( i, l, op_A );
                        auto  B_lj = BB->block( l, j, op_B );
                            
                        if ( is_null_any( A_il, B_lj ) )
                            continue;
                        
                        upd_ij.push_back( { A_il, B_lj } );
                    }// for
                }// for

                multiply< value_t >( alpha, op_A, op_B, *C_ij, upd_ij, sub_U( i, j ), acc, approx );
            }// for
        }// for
    }// if
    else if ( ! is_null( U ) )
    {
        // apply accumulated updates
        hlr::add( alpha, *U, C, acc, approx );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    std::unique_ptr< hpro::TMatrix >  U;
    detail::upd_list_t               upd{ { &A, &B } };
    
    detail::multiply< value_t >( alpha, op_A, op_B, C, upd, U, acc, approx );
}

}}}// namespace hlr::seq::accu

#endif // __HLR_SEQ_ARITH_ACCU_HH
