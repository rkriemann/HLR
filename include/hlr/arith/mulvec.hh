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
#include "hlr/arith/h2.hh"
#include "hlr/matrix/dense_matrix.hh"
#include "hlr/matrix/lrmatrix.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/matrix/mplrmatrix.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/vector/scalar_vector.hh"

namespace hlr
{

//
// compute y = y + α op( M ) x for blas::vector
//

template < typename value_t >
void
mul_vec ( const value_t                     alpha,
          const Hpro::matop_t               op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y );

template < typename value_t >
void
mul_vec ( const value_t                          alpha,
          const Hpro::matop_t                    op_M,
          const Hpro::TBlockMatrix< value_t > &  M,
          const blas::vector< value_t > &        x,
          blas::vector< value_t > &              y )
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
mul_vec ( const value_t                           alpha,
          const Hpro::matop_t                     op_M,
          const matrix::dense_matrix< value_t > & M,
          const blas::vector< value_t > &         x,
          blas::vector< value_t > &               y )
{
    M.apply_add( alpha, x, y, op_M );
}

template < typename value_t >
void
mul_vec ( const value_t                        alpha,
          const Hpro::matop_t                  op_M,
          const matrix::lrmatrix< value_t > &  M,
          const blas::vector< value_t > &      x,
          blas::vector< value_t > &            y )
{
    M.apply_add( alpha, x, y, op_M );
}

template < typename value_t >
void
mul_vec ( const value_t                         alpha,
          const Hpro::matop_t                   op_M,
          const matrix::lrsmatrix< value_t > &  M,
          const blas::vector< value_t > &       x,
          blas::vector< value_t > &             y )
{
    M.apply_add( alpha, x, y, op_M );
}

template < typename value_t >
void
mul_vec ( const value_t                         alpha,
          const Hpro::matop_t                   op_M,
          const matrix::mplrmatrix< value_t > & M,
          const blas::vector< value_t > &       x,
          blas::vector< value_t > &             y )
{
    M.apply_add( alpha, x, y, op_M );
}

template < typename value_t >
void
mul_vec ( const value_t                          alpha,
          const Hpro::matop_t                    op_M,
          const Hpro::TDenseMatrix< value_t > &  M,
          const blas::vector< value_t > &        x,
          blas::vector< value_t > &              y )
{
    M.apply_add( alpha, x, y, op_M );
    // blas::mulvec( alpha, blas::mat_view( op_M, Hpro::blas_mat< value_t >( M ) ), x, value_t(1), y );
}

template < typename value_t >
void
mul_vec ( const value_t                       alpha,
          const Hpro::matop_t                 op_M,
          const Hpro::TRkMatrix< value_t > &  M,
          const blas::vector< value_t > &     x,
          blas::vector< value_t > &           y )
{
    M.apply_add( alpha, x, y, op_M );

    // switch ( op_M )
    // {
    //     case apply_normal :
    //     {
    //         auto  t = blas::mulvec( value_t(1), blas::adjoint( Hpro::blas_mat_B< value_t >( M ) ), x );

    //         blas::mulvec( alpha, Hpro::blas_mat_A< value_t >( M ), t, value_t(1), y );
    //     }
    //     break;
        
    //     case apply_conjugate :
    //     {
    //         assert( ! Hpro::is_complex_type_v< value_t > );

    //         auto  t = blas::mulvec( value_t(1), blas::adjoint( Hpro::blas_mat_B< value_t >( M ) ), x );

    //         blas::mulvec( alpha, Hpro::blas_mat_A< value_t >( M ), t, value_t(1), y );
    //     }
    //     break;
        
    //     case apply_transposed :
    //     {
    //         assert( ! Hpro::is_complex_type_v< value_t > );
            
    //         auto  t = blas::mulvec( value_t(1), blas::transposed( Hpro::blas_mat_A< value_t >( M ) ), x );

    //         blas::mulvec( alpha, Hpro::blas_mat_B< value_t >( M ), t, value_t(1), y );
    //     }
    //     break;
        
    //     case apply_adjoint :
    //     {
    //         auto  t = blas::mulvec( value_t(1), blas::adjoint( Hpro::blas_mat_A< value_t >( M ) ), x );

    //         blas::mulvec( alpha, Hpro::blas_mat_B< value_t >( M ), t, value_t(1), y );
    //     }
    //     break;
    // }// switch
}

template < typename value_t >
void
mul_vec ( const value_t                                alpha,
          const Hpro::matop_t                          op_M,
          const matrix::uniform_lrmatrix< value_t > &  M,
          const blas::vector< value_t > &              x,
          blas::vector< value_t > &                    y )
{
    M.apply_add( alpha, x, y, op_M );

    // switch ( op_M )
    // {
    //     case apply_normal :
    //     {
    //         //
    //         // y = y + U·S·V^H x
    //         //
        
    //         auto  t = M.col_cb().transform_forward( x );
    //         auto  s = blas::mulvec( value_t(1), M.coeff(), t );
    //         auto  r = M.row_cb().transform_backward( s );

    //         blas::add( alpha, r, y );
    //     }
    //     break;
        
    //     case apply_conjugate :
    //     {
    //         assert( ! Hpro::is_complex_type_v< value_t > );

    //         auto  t = M.col_cb().transform_forward( x );
    //         auto  s = blas::mulvec( value_t(1), M.coeff(), t );
    //         auto  r = M.row_cb().transform_backward( s );

    //         blas::add( alpha, r, y );
    //     }
    //     break;
        
    //     case apply_transposed :
    //     {
    //         //
    //         // y = y + (U·S·V^H)^T x
    //         //   = y + conj(V)·S^T·U^T x
    //         //
        
    //         auto  cx = blas::copy( x );

    //         blas::conj( cx );
        
    //         auto  t  = M.row_cb().transform_forward( cx );

    //         blas::conj( t );
        
    //         auto  s = blas::mulvec( value_t(1), blas::transposed(M.coeff()), t );
    //         auto  r = M.col_cb().transform_backward( s );

    //         blas::add( alpha, r, y );
    //     }
    //     break;

    //     case apply_adjoint :
    //     {
    //         //
    //         // y = y + (U·S·V^H)^H x
    //         //   = y + V·S^H·U^H x
    //         //
        
    //         auto  t = M.row_cb().transform_forward( x );
    //         auto  s = blas::mulvec( value_t(1), blas::adjoint(M.coeff()), t );
    //         auto  r = M.col_cb().transform_backward( s );

    //         blas::add( alpha, r, y );
    //     }
    //     break;
    // }// switch
}

#if defined(HAS_H2)

template < typename value_t >
void
mul_vec ( const value_t                           alpha,
          const Hpro::matop_t                     op_M,
          const Hpro::TUniformMatrix< value_t > & M,
          const blas::vector< value_t > &         x,
          blas::vector< value_t > &               y )
{
    M.apply_add( alpha, x, y, op_M );

    // switch ( op_M )
    // {
    //     case apply_normal :
    //     {
    //         //
    //         // y = y + U·S·V^H x
    //         //
        
    //         // U·S·V' x + y
    //         auto  tx = blas::vector< value_t >( Hpro::col_basis( &M )->rank() );
    //         auto  ty = blas::vector< value_t >( y.length() );
                                                 
    //         Hpro::col_basis( &M )->transform_forward( x, tx );
    //         auto  tt = blas::mulvec( value_t(1), Hpro::coeff( &M ), tx );
    //         Hpro::row_basis( &M )->transform_backward( tt, ty );

    //         blas::add( alpha, ty, y );
    //     }
    //     break;
        
    //     case apply_conjugate :
    //     {
    //         assert( ! Hpro::is_complex_type_v< value_t > );

    //         HLR_ERROR( "todo" );
    //     }
    //     break;
        
    //     case apply_transposed :
    //     {
    //         //
    //         // y = y + (U·S·V^H)^T x
    //         //   = y + conj(V)·S^T·U^T x
    //         //
        
    //         HLR_ERROR( "todo" );
    //     }
    //     break;

    //     case apply_adjoint :
    //     {
    //         // (U·S·V')' x + y = V·S'·U' x + y
    //         auto  tx = blas::vector< value_t >( Hpro::row_basis( &M )->rank() );
    //         auto  ty = blas::vector< value_t >( y.length() );
                                                 
    //         Hpro::row_basis( &M )->transform_forward( x, tx );
    //         auto  tt = blas::mulvec( value_t(1), blas::adjoint( Hpro::coeff( &M ) ), tx );
    //         Hpro::col_basis( &M )->transform_backward( tt, ty );

    //         blas::add( alpha, ty, y );
    //     }
    //     break;
    // }// switch
}

#endif

template < typename value_t >
void
mul_vec ( const value_t                     alpha,
          const Hpro::matop_t               op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y )
{
    using namespace hlr::matrix;
    
    // assert( M.ncols( op_M ) == x.length() );
    // assert( M.nrows( op_M ) == y.length() );

    if      ( is_blocked( M ) )
        mul_vec( alpha, op_M, * cptrcast( &M, Hpro::TBlockMatrix< value_t > ), x, y );
    else
        M.apply_add( alpha, x, y, op_M );
        
    // else if ( is_compressible_lowrank( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, matrix::lrmatrix< value_t > ), x, y );
    // else if ( is_compressible_lowrankS( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, matrix::lrsmatrix< value_t > ), x, y );
    // else if ( is_mixedprec_lowrank( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, matrix::mplrmatrix< value_t > ), x, y );
    // else if ( is_lowrank( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, Hpro::TRkMatrix< value_t > ), x, y );
    // else if ( is_uniform_lowrank( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, matrix::uniform_lrmatrix< value_t > ), x, y );
    // #if defined(HAS_H2) 
    // else if ( is_uniform( &M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, Hpro::TUniformMatrix< value_t > ), x, y );
    // #endif
    // else if ( is_compressible_dense( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, matrix::dense_matrix< value_t > ), x, y );
    // else if ( is_dense( M ) )
    //     mul_vec( alpha, op_M, * cptrcast( &M, Hpro::TDenseMatrix< value_t > ), x, y );
    // else
    //     HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x for scalar vectors
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    mul_vec( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

// template < typename value_t >
// void
// mul_vec ( const value_t                           alpha,
//           const Hpro::matop_t                     op_M,
//           const Hpro::TMatrix< value_t > &        M,
//           const Hpro::TScalarVector< value_t > &  x,
//           Hpro::TScalarVector< value_t > &        y )
// {
//     mul_vec( alpha, op_M, M, Hpro::blas_vec< value_t >( x ), Hpro::blas_vec< value_t >( y ) );
// }

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH
