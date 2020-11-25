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
#include "hlr/arith/mulvec.hh"
#include "hlr/arith/lu.hh"
#include "hlr/arith/solve.hh"
#include "hlr/arith/invert.hh"
#include "hlr/approx/svd.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/vector/scalar_vector.hh"
#include "hlr/seq/matrix.hh"

namespace hlr { namespace seq {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α · op( M ) · x
//
using hlr::mul_vec;

//
// compute C = C + α · op( A ) · op( B ) 
// and     C = C + α · op( A ) · op( D ) · op( B )
//
using hlr::multiply;

//
// compute C = C + α · op( A ) · op( B ) with additional approximation
// by omitting sub products based on Frobenius norm of factors
//
using hlr::multiply_apx;

//
// compute Hadamard product A = α A*B 
//
using hlr::multiply_hadamard;

//
// LU factorization
//
using hlr::lu;     

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

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
// 
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
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
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
lu_lazy ( hpro::TMatrix &          A,
          const hpro::TTruncAcc &  acc,
          const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( & A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        for ( int  k = 0; k < int(i); k++ )
            hlr::seq::multiply< value_t >( value_t(-1),
                                           hpro::apply_normal, *BA->block( i, k ),
                                           hpro::apply_normal, *BA->block( k, i ),
                                           *BA->block( i, i ), acc, approx );
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  D_ii = blas::mat< value_t >( A_ii );
            
        blas::invert( D_ii );

        //
        // solve with L, e.g. L_ii X_ij = M_ij
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  A_ij = BA->block( i, j );

            // only update block as L = I
            for ( int  k = 0; k < int(i); k++ )
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( i, k ),
                                               hpro::apply_normal, *BA->block( k, j ),
                                               *A_ij, acc, approx );
        }// for
        
        //
        // solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            for ( int  k = 0; k < int(i); k++ )
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( j, k ),
                                               hpro::apply_normal, *BA->block( k, i ),
                                               *A_ji, acc, approx );

            if ( is_lowrank( A_ji ) )
            {
                // A_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                // ⟶ W = U, X = D_ii^-T·V
                auto  R_ji = ptrcast( A_ji, hpro::TRkMatrix );
                auto  V    = blas::copy( blas::mat_V< value_t >( R_ji ) );

                blas::prod( value_t(1), blas::adjoint( D_ii ), V, value_t(0), blas::mat_V< value_t >( R_ji ) );
            }// if
            else if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + A_ji->typestr() );
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
// 
template < typename value_t,
           typename approx_t >
void
ldu ( hpro::TMatrix &          A,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );

        HLR_ASSERT( is_dense( A_ii ) );
        
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat< value_t >( ptrcast( A_ii, hpro::TDenseMatrix ) );
            
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //

        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  L_ji = BA->block( j, i );

            if ( is_lowrank( L_ji ) )
            {
                // L_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                // ⟶ W = U, X = D_ii^-T·V
                auto  R_ji = ptrcast( L_ji, hpro::TRkMatrix );
                auto  V    = blas::copy( blas::mat_V< value_t >( R_ji ) );

                blas::prod( value_t(1), blas::adjoint( D_ii ), V, value_t(0), blas::mat_V< value_t >( R_ji ) );
            }// if
            else if ( is_dense( L_ji ) )
            {
                auto  D_ji = ptrcast( L_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// else
        }// for

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  U_ij = BA->block( i, j );

            if ( is_lowrank( U_ij ) )
            {
                // U_ij = W·X' = D_ii^-1·U·V' = D_ii^-1·A_ij
                // ⟶ W = D_ii^-1·U, X = V
                auto  R_ij = ptrcast( U_ij, hpro::TRkMatrix );
                auto  U    = blas::copy( blas::mat_U< value_t >( R_ij ) );

                blas::prod( value_t(1), D_ii, U, value_t(0), blas::mat_U< value_t >( R_ij ) );
            }// if
            else if ( is_dense( U_ij ) )
            {
                auto  D_ij = ptrcast( U_ij, hpro::TDenseMatrix );
                auto  T_ij = blas::copy( blas::mat< value_t >( D_ij ) );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), blas::mat< value_t >( D_ij ) );
            }// else
        }// for

        //
        // update trailing sub matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( j, i ),
                                               hpro::apply_normal, *T_ii,
                                               hpro::apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace hodlr
{

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const hpro::TMatrix &      L,
        blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L.id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( &L, hpro::TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), hpro::TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        blas::matrix< value_t >  X0( X, L00->row_is() - L.row_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, L11->row_is() - L.row_ofs(), blas::range::all );
            
        trsml( *L00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( L10 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( *L11, X1 );
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
trsmuh ( const hpro::TMatrix &      U,
         blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U.id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( &U, hpro::TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), hpro::TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        blas::matrix< value_t >  X0( X, U00->col_is() - U.col_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, U11->col_is() - U.col_ofs(), blas::range::all );
            
        trsmuh( *U00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( U01 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( *U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( &U, hpro::TDenseMatrix );
        
        blas::matrix< value_t >  Y( X, hpro::copy_value );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t,
           typename approx_t >
void
addlr ( blas::matrix< value_t > &  U,
        blas::matrix< value_t > &  V,
        hpro::TMatrix &            A,
        const hpro::TTruncAcc &    acc,
        const approx_t &           approx )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A.col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A.col_ofs(), blas::range::all );

        addlr( U0, V0, *A00, acc, approx );
        addlr( U1, V1, *A11, acc, approx );

        {
            auto [ U01, V01 ] = approx( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                        { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                        acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = approx( { hpro::blas_mat_A< value_t >( A10 ), U1 },
                                        { hpro::blas_mat_B< value_t >( A10 ), V0 },
                                        acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( &A, hpro::TDenseMatrix );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), hpro::blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( &A00, acc, approx );
        
        trsml(  A00, hpro::blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, hpro::blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A10 ) ), hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( A10 ), T );

        seq::hodlr::addlr< value_t >( UT, hpro::blas_mat_B< value_t >( A01 ), A11, acc, approx );
        
        seq::hodlr::lu< value_t >( &A11, acc, approx );
    }// if
    else
    {
        auto  DA = ptrcast( &A, hpro::TDenseMatrix );
        
        blas::invert( hpro::blas_mat< value_t >( DA ) );
    }// else
}

}// namespace hodlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace tileh
{

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
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        hlr::seq::lu( * BA->block( i, i ), acc, approx );

        for ( uint j = i+1; j < nbr; ++j )
        {
            hlr::solve_upper_tri( from_right, general_diag, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            hlr::solve_lower_tri( from_left, unit_diag, *BA->block( i, i ), *BA->block( i, l ), acc, approx );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::multiply( value_t(-1),
                               hpro::apply_normal, *BA->block( j, i ),
                               hpro::apply_normal, *BA->block( i, l ),
                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tileh

}}// namespace hlr::seq

#endif // __HLR_SEQ_ARITH_HH
