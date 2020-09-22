#ifndef __HLR_ARITH_MULVEC_HH
#define __HLR_ARITH_MULVEC_HH
//
// Project     : HLib
// Module      : mul_vec
// Description : matrix-vector multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/vector/scalar_vector.hh"

namespace hlr
{

//
// compute y = y + α op( M ) x for blas::vector
//

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TMatrix &            M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y );

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TBlockMatrix &       M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    if ( alpha == value_t(0) )
        return;

    const auto  row_ofs = M.row_is( op_M ).first();
    const auto  col_ofs = M.col_is( op_M ).first();
    
    for ( uint  i = 0; i < M.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < M.nblock_cols(); ++j )
        {
            auto  B_ij = M.block( i, j );
            
            if ( ! is_null( B_ij ) )
            {
                auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );
                
                mul_vec( alpha, op_M, *B_ij, x_j, y_i );
            }// if
        }// for
    }// for
}

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TDenseMatrix &       M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    blas::mulvec( alpha, blas::mat_view( op_M, hpro::blas_mat< value_t >( M ) ), x, value_t(1), y );
}

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TRkMatrix &          M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    switch ( op_M )
    {
        case apply_normal :
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( M ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_A< value_t >( M ), t, value_t(1), y );
        }
        break;
        
        case apply_conjugate :
        {
            assert( ! hpro::is_complex_type< value_t >::value );

            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( M ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_A< value_t >( M ), t, value_t(1), y );
        }
        break;
        
        case apply_transposed :
        {
            assert( ! hpro::is_complex_type< value_t >::value );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( hpro::blas_mat_A< value_t >( M ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( M ), t, value_t(1), y );
        }
        break;
        
        case apply_adjoint :
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( M ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( M ), t, value_t(1), y );
        }
        break;
    }// switch
}

template < typename value_t >
void
mul_vec ( const value_t                                alpha,
          const hpro::matop_t                          op_M,
          const matrix::uniform_lrmatrix< value_t > &  M,
          const blas::vector< value_t > &              x,
          blas::vector< value_t > &                    y )
{
    switch ( op_M )
    {
        case apply_normal :
        {
            //
            // y = y + U·S·V^H x
            //
        
            auto  t = M.col_cb().transform_forward( x );
            auto  s = blas::mulvec( value_t(1), M.coeff(), t );
            auto  r = M.row_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }
        break;
        
        case apply_conjugate :
        {
            assert( ! hpro::is_complex_type< value_t >::value );

            auto  t = M.col_cb().transform_forward( x );
            auto  s = blas::mulvec( value_t(1), M.coeff(), t );
            auto  r = M.row_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }
        break;
        
        case apply_transposed :
        {
            //
            // y = y + (U·S·V^H)^T x
            //   = y + conj(V)·S^T·U^T x
            //
        
            auto  cx = blas::copy( x );

            blas::conj( cx );
        
            auto  t  = M.row_cb().transform_forward( cx );

            blas::conj( t );
        
            auto  s = blas::mulvec( value_t(1), blas::transposed(M.coeff()), t );
            auto  r = M.col_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }
        break;

        case apply_adjoint :
        {
            //
            // y = y + (U·S·V^H)^H x
            //   = y + V·S^H·U^H x
            //
        
            auto  t = M.row_cb().transform_forward( x );
            auto  s = blas::mulvec( value_t(1), blas::adjoint(M.coeff()), t );
            auto  r = M.col_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }
        break;
    }// switch
}

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TMatrix &            M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    // assert( M.ncols( op_M ) == x.length() );
    // assert( M.nrows( op_M ) == y.length() );

    if ( is_blocked( M ) )
        mul_vec( alpha, op_M, * cptrcast( &M, hpro::TBlockMatrix ), x, y );
    else if ( is_lowrank( M ) )
        mul_vec( alpha, op_M, * cptrcast( &M, hpro::TRkMatrix ), x, y );
    else if ( matrix::is_uniform_lowrank( M ) )
        mul_vec( alpha, op_M, * cptrcast( &M, matrix::uniform_lrmatrix< value_t > ), x, y );
    else if ( is_dense( M ) )
        mul_vec( alpha, op_M, * cptrcast( &M, hpro::TDenseMatrix ), x, y );
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x for scalar vectors
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const hpro::matop_t                       op_M,
          const hpro::TMatrix &                     M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );

    mul_vec( alpha, op_M, M, hpro::blas_vec< value_t >( x ), hpro::blas_vec< value_t >( y ) );
}

template < typename value_t >
void
mul_vec ( const value_t                alpha,
          const hpro::matop_t          op_M,
          const hpro::TMatrix &        M,
          const hpro::TScalarVector &  x,
          hpro::TScalarVector &        y )
{
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );

    mul_vec( alpha, op_M, M, hpro::blas_vec< value_t >( x ), hpro::blas_vec< value_t >( y ) );
}

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH
