#ifndef __HLR_SEQ_ARITH_ACCU_HH
#define __HLR_SEQ_ARITH_ACCU_HH
//
// Project     : HLib
// Module      : seq/arith_accu.hh
// Description : sequential arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/add.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/accumulator.hh"

namespace hlr { namespace seq { namespace accu {

namespace hpro = HLIB;

using hlr::seq::matrix::accumulator;

namespace timer = HLIB::Time::Wall;

extern double  t_apply;
extern double  t_eval;

// #define ACCU_TIC       auto  __tic = timer::now()
// #define ACCU_TOC( t )  { auto  __toc = timer::since( __tic ); t += __toc.seconds(); }
#define ACCU_TIC       {}
#define ACCU_TOC( t )  {}

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

//
// compute C = C + α op( A ) op( B ) where A and B are provided as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           hpro::TMatrix &          C,
           accumulator &            accu,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    accu.eval( alpha, C, acc, approx );
    
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

        auto  sub_accu = accu.restrict< value_t >( *BC );

        accu.clear_matrix();

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                multiply< value_t >( alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
    }// if
    else 
    {
        // apply accumulated updates
        accu.apply( alpha, C, acc, approx );
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
    accumulator::update_list          upd{ { op_A, &A, op_B, &B } };
    accumulator                       accu{ std::move( U ), std::move( upd ) };
    
    detail::multiply< value_t >( alpha, C, accu, acc, approx );
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
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  accumulator &            accu,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    // apply computable updates
    {
    ACCU_TIC;

    accu.eval( value_t(1), M, acc, approx );

    ACCU_TOC( t_eval );
    }
    
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

        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j), sub_accu(i,j), acc, approx );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        sub_accu(k,j).pending.push_back( { apply_normal, BL->block(k,i),
                                                           apply_normal, BM->block(i,j) } );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else if ( is_lowrank( M ) )
    {
        // no recursive updates left, apply accumulated updates and solve
        {
        ACCU_TIC;
        
        accu.apply( value_t(-1), M, acc, approx );

        ACCU_TOC( t_apply );
        }

        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), M, acc, approx );

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
                  accumulator &                       accu,
                  const hpro::TTruncAcc &             acc,
                  const approx_t &                    approx )
{
    // apply computable updates
    {
    ACCU_TIC;

    accu.eval( value_t(1), M, acc, approx );

    ACCU_TOC( t_eval );
    }
    
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

        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

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
                    solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_accu(i,j), acc, approx );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_accu(i,k).pending.push_back( { apply_normal, BM->block(i,j),
                                                           apply_normal, BU->block(j,k) } );
            }// for
        }// else
    }// if
    else if ( is_lowrank( M ) )
    {
        // no recursive updates left, apply accumulated updates and solve
        {
        ACCU_TIC;
        
        accu.apply( value_t(-1), M, acc, approx );

        ACCU_TOC( t_apply );
        }
        
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), M, acc, approx );
        
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          M,
     accumulator &            accu,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    //
    // evaluate all computable updates to M
    //

    {
    ACCU_TIC;

    accu.eval( value_t(1), M, acc, approx );

    ACCU_TOC( t_eval );
    }
    
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

        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_accu(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
                solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_accu(j,l).pending.push_back( { apply_normal, BM->block( j, i ),
                                                       apply_normal, BM->block( i, l ) } );
        }// for
    }// if
    else
    {
        //
        // no recursive updates left, apply accumulated updates
        // and factorize
        //

        accu.apply( value_t(-1), M, acc, approx );
        
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
    accumulator  accu;
    
    detail::lu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::seq::accu

#endif // __HLR_SEQ_ARITH_ACCU_HH
