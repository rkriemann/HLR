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
#include "hlr/arith/solve.hh"
#include "hlr/matrix/restrict.hh"
#include "hlr/seq/norm.hh"

namespace hlr { namespace seq { namespace accu {

namespace hpro = HLIB;

namespace detail
{

////////////////////////////////////////////////////////////////////////////////
//
// general functions for handling accumulated updates
//
////////////////////////////////////////////////////////////////////////////////

using  upd_list_t = std::list< std::pair< const hpro::TMatrix *, const hpro::TMatrix * > >;

//
// restrict given recursive updates to sub block (i,j)
// (of corresponding block matrix updates belong to)
//
upd_list_t
restrict_rec ( const matop_t     op_A,
               const matop_t     op_B,
               const upd_list_t  upd_rec,
               const uint        i,
               const uint        j )
{
    auto  upd = upd_list_t();
                                        
    for ( auto  [ A, B ] : upd_rec )
    {
        // filter out non-recursive updates
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
                                                
            upd.push_back( { A_il, B_lj } );
        }// for
    }// for
                                        
    return upd;
}

//
// evaluate all computable updates and apply to accumulator matrix
// - given accumulator matrix may be replaced and newly created is returned
// - may proceed recursively if M is a leaf, e.g., low-rank block
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< hpro::TMatrix >
eval_nonrec ( const value_t                       alpha,
              const hpro::matop_t                 op_A,
              const hpro::matop_t                 op_B,
              hpro::TMatrix &                     M,
              upd_list_t &                        upd_rec,
              std::unique_ptr< hpro::TMatrix > &  upd_mat,
              const hpro::TTruncAcc &             acc,
              const approx_t &                    approx )
{
    std::unique_ptr< hpro::TMatrix >       U( std::move( upd_mat ) );
    std::unique_ptr< hpro::TBlockMatrix >  BC; // for recursive handling

    //
    // handle all, actually computable updates, i.e., one factor is a leaf block
    //
    
    for ( auto  [ A, B ] : upd_rec )
    {
        if ( is_blocked_all( *A, *B, M ) )
            continue;
        
        if ( is_blocked_all( A, B ) )
        {
            //
            // if M is a leaf and A _and_ B are blocked, a temporary matrix
            // is created for further recursive update handling
            //

            if ( ! is_null( BC ) )
                continue;
                
            // TODO: non low-rank M
            HLR_ASSERT( is_lowrank( M ) );
                
            auto  BA = cptrcast( A, hpro::TBlockMatrix );
            auto  BB = cptrcast( B, hpro::TBlockMatrix );
                
            BC = std::make_unique< hpro::TBlockMatrix >( A->row_is( op_A ), B->col_is( op_B ) );

            BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

            for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                {
                    HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                        
                    BC->set_block( i, j, new hpro::TRkMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                              BB->block( 0, j, op_B )->col_is( op_B ),
                                                              M.value_type() ) );
                }// for
            }// for
        }// if
        else
        {
            //
            // compute update (either A or B is a leaf)
            //

            auto  T = multiply< value_t >( alpha, op_A, A, op_B, B );

            //
            // apply update to accumulator
            //
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else if ( ! is_dense( *U ) && is_dense( *T ) )
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
    // now handle recursive updates if M is a leaf block
    //
    
    if ( ! is_null( BC ) )
    {
        //
        // TODO: try with empty sub_mat, don't release U and add sub results later
        //
        
        //
        // first, split update U into subblock updates
        // (to release U before recursion)
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_mat( BC->nblock_rows(), BC->nblock_cols() );
        tensor2< upd_list_t >                        sub_rec( BC->nblock_rows(), BC->nblock_cols() );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                if ( ! is_null( U ) )
                    sub_mat(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                sub_rec(i,j) = detail::restrict_rec( op_A, op_B, upd_rec, i, j );
            }// for
        }// for

        U.reset( nullptr );
        
        //
        // apply recursive updates
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  U_ij = eval_nonrec< value_t >( alpha, op_A, op_B, *BC->block(i,j), sub_rec(i,j), sub_mat(i,j), acc, approx );

                // replace block in BC by accumulator matrix for agglomeration below
                BC->delete_block( i, j );
                BC->set_block( i, j, U_ij.release() );
            }// for
        }// for

        //
        // finally convert subblocks to single low-rank matrix for new accumulated updates
        //

        U = matrix::convert_to_lowrank( *BC, acc, approx );
    }// if

    //
    // finished: all updates are applied now
    //
    
    return U;
}

}// namespace detail

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

//
// compute C = C + α op( A ) op( B ) where A and B are provides as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TMatrix &                     C,
           upd_list_t &                        upd_rec,
           std::unique_ptr< hpro::TMatrix > &  upd_mat,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx )
{
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    auto  U = eval_nonrec( alpha, op_A, op_B, C, upd_rec, upd_mat, acc, approx );
    
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

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_mat( BC->nblock_rows(), BC->nblock_cols() );
        tensor2< upd_list_t >                        sub_rec( BC->nblock_rows(), BC->nblock_cols() );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                if ( ! is_null( U ) )
                    sub_mat(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                sub_rec(i,j) = detail::restrict_rec( op_A, op_B, upd_rec, i, j );
            }// for
        }// for

        U.reset( nullptr );

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                multiply< value_t >( alpha, op_A, op_B, *BC->block(i,j), sub_rec(i,j), sub_mat(i,j), acc, approx );
    }// if
    else 
    {
        HLR_ASSERT( upd_rec.size() == 0 );

        // apply accumulated updates
        if ( ! is_null( U ) )
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
    detail::upd_list_t                upd{ { &A, &B } };
    
    detail::multiply< value_t >( alpha, op_A, op_B, C, upd, U, acc, approx );
}

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based LU factorization
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const hpro::TMatrix &               L,
                  hpro::TMatrix &                     M,
                  upd_list_t &                        upd_rec,
                  std::unique_ptr< hpro::TMatrix > &  upd_mat,
                  const hpro::TTruncAcc &             acc,
                  const approx_t &                    approx )
{
    // apply computable updates
    auto  U_accu = eval_nonrec( value_t(1), apply_normal, apply_normal, M, upd_rec, upd_mat, acc, approx );
    
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_mat( BM->nblock_rows(), BM->nblock_cols() );
        tensor2< upd_list_t >                        sub_rec( BM->nblock_rows(), BM->nblock_cols() );
        
        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BM->block( i, j ) ) );
                
                if ( ! is_null( U_accu ) )
                    sub_mat(i,j) = hlr::matrix::restrict( *U_accu, BM->block( i, j )->block_is() );
                sub_rec(i,j) = restrict_rec( apply_normal, apply_normal, upd_rec, i, j );
            }// for
        }// for

        U_accu.reset( nullptr );

        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j), sub_rec(i,j), sub_mat(i,j), acc, approx );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        sub_rec(k,j).push_back( { BL->block(k,i), BM->block(i,j) } ); // -1
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        if ( ! is_null( U_accu ) )
            hlr::add( value_t(-1), *U_accu, M, acc, approx );

        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const hpro::TMatrix &               U,
                  hpro::TMatrix &                     M,
                  upd_list_t &                        upd_rec,
                  std::unique_ptr< hpro::TMatrix > &  upd_mat,
                  const hpro::TTruncAcc &             acc,
                  const approx_t &                    approx )
{
    // apply computable updates
    auto  U_accu = eval_nonrec( value_t(1), apply_normal, apply_normal, M, upd_rec, upd_mat, acc, approx );
    
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_mat( BM->nblock_rows(), BM->nblock_cols() );
        tensor2< upd_list_t >                        sub_rec( BM->nblock_rows(), BM->nblock_cols() );
        
        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BM->block( i, j ) ) );
                
                if ( ! is_null( U_accu ) )
                    sub_mat(i,j) = hlr::matrix::restrict( *U_accu, BM->block( i, j )->block_is() );
                sub_rec(i,j) = restrict_rec( apply_normal, apply_normal, upd_rec, i, j );
            }// for
        }// for

        U_accu.reset( nullptr );

        if ( side == from_left )
        {
            HLR_ASSERT( false );
        }// if
        else
        {
            for ( uint j = 0; j < BM->nblock_cols(); ++j )
            {
                const auto  U_jj = BU->block( j, j );
            
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_rec(i,j), sub_mat(i,j), acc, approx );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_rec(i,k).push_back( {BM->block(i,j), BU->block(j,k) } ); // -1
            }// for
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        if ( ! is_null( U_accu ) )
            hlr::add( value_t(-1), *U_accu, M, acc, approx );
        
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &                     M,
     upd_list_t &                        upd_rec,
     std::unique_ptr< hpro::TMatrix > &  upd_mat,
     const hpro::TTruncAcc &             acc,
     const approx_t &                    approx )
{
    //
    // evaluate all computable updates to M
    //

    auto  U_accu = eval_nonrec( value_t(1), apply_normal, apply_normal, M, upd_rec, upd_mat, acc, approx );
    
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, hpro::TBlockMatrix );

        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_mat( BM->nblock_rows(), BM->nblock_cols() );
        tensor2< upd_list_t >                        sub_rec( BM->nblock_rows(), BM->nblock_cols() );
        
        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BM->block( i, j ) ) );
                
                if ( ! is_null( U_accu ) )
                    sub_mat(i,j) = hlr::matrix::restrict( *U_accu, BM->block( i, j )->block_is() );
                sub_rec(i,j) = restrict_rec( apply_normal, apply_normal, upd_rec, i, j );
            }// for
        }// for

        U_accu.reset( nullptr );

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_rec(i,i), sub_mat(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_rec(j,i), sub_mat(j,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
                solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_rec(i,j), sub_mat(i,j), acc, approx );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_rec(j,l).push_back( { BM->block( j, i ), BM->block( i, l ) } ); // -1
        }// for
    }// if
    else
    {
        //
        // no recursive updates left, apply accumulated updates
        // and factorize
        //

        if ( ! is_null( U_accu ) )
            hlr::add( value_t(-1), *U_accu, M, acc, approx );
        
        if ( is_dense( M ) )
        {
            auto  D = ptrcast( &M, hpro::TDenseMatrix );

            invert< value_t >( *D );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          M,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    std::unique_ptr< hpro::TMatrix >  U;
    detail::upd_list_t                upd{};
    
    detail::lu< value_t >( M, upd, U, acc, approx );
}

}}}// namespace hlr::seq::accu

#endif // __HLR_SEQ_ARITH_ACCU_HH
