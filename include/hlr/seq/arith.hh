#ifndef __HLR_SEQ_ARITH_HH
#define __HLR_SEQ_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_fac.hh>
#include <hpro/algebra/solve_tri.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/approx/svd.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/vector/scalar_vector.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/seq/norm.hh"

namespace hlr { namespace seq {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TMatrix &            M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    // assert( ! is_null( M ) );
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, hpro::TBlockMatrix );
        const auto  row_ofs = B->row_is( op_M ).first();
        const auto  col_ofs = B->col_is( op_M ).first();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                    auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );

                    mul_vec( alpha, op_M, *B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = cptrcast( &M, hpro::TDenseMatrix );
        
        blas::mulvec( alpha, blas::mat_view( op_M, hpro::blas_mat< value_t >( D ) ), x, value_t(1), y );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, hpro::TRkMatrix );

        if ( op_M == hpro::apply_normal )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_A< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            assert( is_complex_type< value_t >::value == false );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( hpro::blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
        
        if ( op_M == hpro::apply_normal )
        {
            //
            // y = y + U·S·V^H x
            //
        
            auto  t = R->col_cb().transform_forward( x );
            auto  s = blas::mulvec( value_t(1), R->coeff(), t );
            auto  r = R->row_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            //
            // y = y + (U·S·V^H)^T x
            //   = y + conj(V)·S^T·U^T x
            //
        
            auto  cx = blas::copy( x );

            blas::conj( cx );
        
            auto  t  = R->row_cb().transform_forward( cx );

            blas::conj( t );
        
            auto  s = blas::mulvec( value_t(1), blas::transposed(R->coeff()), t );
            auto  r = R->col_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            //
            // y = y + (U·S·V^H)^H x
            //   = y + V·S^H·U^H x
            //
        
            auto  t = R->row_cb().transform_forward( x );
            auto  s = blas::mulvec( value_t(1), blas::adjoint(R->coeff()), t );
            auto  r = R->col_cb().transform_backward( s );

            blas::add( alpha, r, y );
        }// if
    }// if
    else
        assert( false );
}

//
// compute y = y + α op( M ) x
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

//
// compute C = C + α op( A ) op( B )
//
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
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc );
                }// for
            }// for
        }// for
    }// if
    else
        hpro::multiply< value_t >( alpha, op_A, &A, op_B, &B, value_t(1), &C, acc );
}

//
// compute C = C + α op( A ) op( B )
//
template < typename value_t >
void
multiply_apx ( const value_t            alpha,
               const hpro::matop_t      op_A,
               const hpro::TMatrix &    A,
               const hpro::matop_t      op_B,
               const hpro::TMatrix &    B,
               hpro::TMatrix &          C,
               const hpro::TTruncAcc &  acc,
               typename hpro::real_type< value_t >::type_t  tol )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_rows( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_apx< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, tol );
                }// for
            }// for
        }// for
    }// if
    else
    {
        if ( is_lowrank( C ) )
        {
            //
            // look for Frobenius norm of factors and return if too small
            //

            const auto  norm_A = norm::norm_F( A );
            const auto  norm_B = norm::norm_F( B );

            if ( norm_A * norm_B < tol )
                return;
        }// if
        
        hpro::multiply< value_t >( alpha, op_A, &A, op_B, &B, value_t(1), &C, acc );
    }// else
}

//
// solve op(L) x = y with lower triangular L
//
void
trsvl ( const hpro::matop_t      op_L,
        const hpro::TMatrix &    L,
        hpro::TScalarVector &    x,
        const hpro::diag_type_t  diag_mode );

//
// solve op(U) x = y with upper triangular U
//
void
trsvu ( const hpro::matop_t      op_U,
        const hpro::TMatrix &    U,
        hpro::TScalarVector &    x,
        const hpro::diag_type_t  diag_mode );

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
inline void
gauss_elim ( hpro::TMatrix *          A,
             hpro::TMatrix *          T,
             const hpro::TTruncAcc &  acc )
{
    assert( ! is_null_any( A, T ) );
    assert( A->type() == T->type() );

    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( A, hpro::TBlockMatrix );
        auto  BT = ptrcast( T, hpro::TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        hlr::seq::gauss_elim( MA(0,0), MT(0,0), acc );
        // hlr::log( 0, hpro::to_string( "                               %d = %.8e", MA(0,0)->id(), norm_F( MA(0,0) ) ) );

        // T_01 = A_00⁻¹ · A_01
        hpro::multiply( 1.0, hpro::apply_normal, MA(0,0), hpro::apply_normal, MA(0,1), 0.0, MT(0,1), acc );
        // seq::matrix::clear( *MT(0,1) );
        // multiply( 1.0, MA(0,0), MA(0,1), MT(0,1), acc );
        
        // T_10 = A_10 · A_00⁻¹
        hpro::multiply( 1.0, hpro::apply_normal, MA(1,0), hpro::apply_normal, MA(0,0), 0.0, MT(1,0), acc );
        // seq::matrix::clear( *MT(1,0) );
        // multiply( 1.0, MA(1,0), MA(0,0), MT(1,0), acc );

        // A_11 = A_11 - T_10 · A_01
        hpro::multiply( -1.0, hpro::apply_normal, MT(1,0), hpro::apply_normal, MA(0,1), 1.0, MA(1,1), acc );
        // multiply( -1.0, MT(1,0), MA(0,1), MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        hlr::seq::gauss_elim( MA(1,1), MT(1,1), acc );
        // hlr::log( 0, hpro::to_string( "                               %d = %.8e", MA(1,1)->id(), norm_F( MA(1,1) ) ) );

        // A_01 = - T_01 · A_11
        hpro::multiply( -1.0, hpro::apply_normal, MT(0,1), hpro::apply_normal, MA(1,1), 0.0, MA(0,1), acc );
        // seq::matrix::clear( *MA(0,1) );
        // multiply( -1.0, MT(0,1), MA(1,1), MA(0,1), acc );
            
        // A_10 = - A_11 · T_10
        hpro::multiply( -1.0, hpro::apply_normal, MA(1,1), hpro::apply_normal, MT(1,0), 0.0, MA(1,0), acc );
        // seq::matrix::clear( *MA(1,0) );
        // multiply( -1.0, MA(1,1), MT(1,0), MA(1,0), acc );

        // A_00 = T_00 - A_01 · T_10
        hpro::multiply( -1.0, hpro::apply_normal, MA(0,1), hpro::apply_normal, MT(1,0), 1.0, MA(0,0), acc );
        // multiply( -1.0, MA(0,1), MT(1,0), MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        if ( A->is_complex() ) blas::invert( DA->blas_cmat() );
        else                   blas::invert( DA->blas_rmat() );
    }// if
    else
        HLR_ASSERT( false );

    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d )", A->id() ) );
}

namespace tlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

//
// LU factorization for TLR block format
// 
template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
            
        blas::invert( hpro::blas_mat< value_t >( A_ii ) );

        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal !!!
            // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
            trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( j, i ),
                                               hpro::apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc );
            }// for
        }// for
    }// for
}

}// namespace tlr

namespace hodlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const hpro::TMatrix *      L,
        blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, hpro::TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), hpro::TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        blas::matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), blas::range::all );
            
        trsml( L00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( L10 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, TDenseMatrix );
        
        // blas::matrix< value_t >  Y( X, copy_value );

        // blas::prod( value_t(1), blas_mat< value_t >( DL ), Y, value_t(0), X );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const hpro::TMatrix *      U,
         blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, hpro::TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), hpro::TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        blas::matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), blas::range::all );
            
        trsmuh( U00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( U01 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( U, hpro::TDenseMatrix );
        
        blas::matrix< value_t >  Y( X, hpro::copy_value );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( blas::matrix< value_t > &  U,
        blas::matrix< value_t > &  V,
        hpro::TMatrix *            A,
        const hpro::TTruncAcc &    acc )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), blas::range::all );

        addlr( U0, V0, A00, acc );
        addlr( U1, V1, A11, acc );

        {
            auto [ U01, V01 ] = hlr::approx::svd< value_t >( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                                             { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                                             acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = hlr::approx::svd< value_t >( { hpro::blas_mat_A< value_t >( A10 ), U1 },
                                                             { hpro::blas_mat_B< value_t >( A10 ), V0 },
                                                             acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), hpro::blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( A00, acc );
        
        trsml(  A00, hpro::blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, hpro::blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A10 ) ), hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( A10 ), T );

        seq::hodlr::addlr< value_t >( UT, hpro::blas_mat_B< value_t >( A01 ), A11, acc );
        
        seq::hodlr::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        blas::invert( hpro::blas_mat< value_t >( DA ) );
    }// else
}

}// namespace hodlr

namespace tileh
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        hpro::LU::factorise_rec( BA->block( i, i ), acc );

        for ( uint j = i+1; j < nbr; ++j )
        {
            solve_upper_right( BA->block( j, i ),
                               BA->block( i, i ), nullptr, acc,
                               hpro::solve_option_t( hpro::block_wise, hpro::general_diag, hpro::store_inverse ) );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            solve_lower_left( hpro::apply_normal, BA->block( i, i ), nullptr,
                              BA->block( i, l ), acc,
                              hpro::solve_option_t( hpro::block_wise, hpro::unit_diag, hpro::store_inverse ) );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply( -1.0,
                                    hpro::apply_normal, * BA->block( j, i ),
                                    hpro::apply_normal, * BA->block( i, l ),
                                    * BA->block( j, l ), acc );
            }// for
        }// for
    }// for
}

}// namespace tileh

}}// namespace hlr::seq

#endif // __HLR_SEQ_ARITH_HH
