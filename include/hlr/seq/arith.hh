#ifndef __HLR_SEQ_ARITH_HH
#define __HLR_SEQ_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/matrix.hh"
// #include "hlr/seq/mat_mul.hh"

#include "hlr/seq/arith_tile.hh"
#include "hlr/seq/arith_tiled_v2.hh"

namespace hlr { namespace seq {

using namespace HLIB;

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// LU factorization for TLR block format
// 
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    HLR_LOG( 4, HLIB::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
        BLAS::invert( blas_mat< value_t >( A_ii ) );

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
                multiply< value_t >( value_t(-1), BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
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
trsml ( const TMatrix *            L,
        BLAS::Matrix< value_t > &  X )
{
    HLR_LOG( 4, HLIB::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), BLAS::Range::all );
            
        trsml( L00, X0 );

        auto  T = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( L10 ) ), X0 );
        
        BLAS::prod( value_t(-1), blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, TDenseMatrix );
        
        // BLAS::Matrix< value_t >  Y( X, copy_value );

        // BLAS::prod( value_t(1), blas_mat< value_t >( DL ), Y, value_t(0), X );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const TMatrix *            U,
         BLAS::Matrix< value_t > &  X )
{
    HLR_LOG( 4, HLIB::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), BLAS::Range::all );
            
        trsmuh( U00, X0 );

        auto  T = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_A< value_t >( U01 ) ), X0 );
        
        BLAS::prod( value_t(-1), blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( U, TDenseMatrix );
        
        BLAS::Matrix< value_t >  Y( X, copy_value );

        BLAS::prod( value_t(1), BLAS::adjoint( blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( BLAS::Matrix< value_t > &  U,
        BLAS::Matrix< value_t > &  V,
        TMatrix *                  A,
        const TTruncAcc &          acc )
{
    HLR_LOG( 4, HLIB::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        BLAS::Matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), BLAS::Range::all );

        addlr( U0, V0, A00, acc );
        addlr( U1, V1, A11, acc );

        {
            auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                { blas_mat_B< value_t >( A01 ), V1 },
                                                                acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                { blas_mat_B< value_t >( A10 ), V0 },
                                                                acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );

        BLAS::prod( value_t(1), U, BLAS::adjoint( V ), value_t(1), blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    HLR_LOG( 4, HLIB::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( A00, acc );
        
        trsml(  A00, blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = BLAS::prod(  value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = BLAS::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        seq::hodlr::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        
        seq::hodlr::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        BLAS::invert( blas_mat< value_t >( DA ) );
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
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    HLR_LOG( 4, HLIB::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        LU::factorise_rec( BA->block( i, i ), acc );

        for ( uint j = i+1; j < nbr; ++j )
        {
            solve_upper_right( BA->block( j, i ),
                               BA->block( i, i ), nullptr, acc,
                               solve_option_t( block_wise, general_diag, store_inverse ) );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            solve_lower_left( apply_normal, BA->block( i, i ), nullptr,
                              BA->block( i, l ), acc,
                              solve_option_t( block_wise, unit_diag, store_inverse ) );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                multiply( -1.0, BA->block( j, i ), BA->block( i, l ), 1.0, BA->block( j, l ), acc );
            }// for
        }// for
    }// for
}

}// namespace tileh

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
          const matop_t                    op_M,
          const TMatrix *                  M,
          const BLAS::Vector< value_t > &  x,
          BLAS::Vector< value_t > &        y )
{
    assert( ! is_null( M ) );
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( M, TBlockMatrix );
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
                    auto  y_i = x( B_ij->row_is( op_M ) - row_ofs );

                    mul_vec( alpha, op_M, B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = cptrcast( M, TDenseMatrix );
        
        BLAS::mulvec( alpha, BLAS::mat_view( op_M, blas_mat< value_t >( D ) ), x, value_t(1), y );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( M, TRkMatrix );

        if ( op_M == apply_normal )
        {
            auto  t = BLAS::mulvec( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( R ) ), x );

            BLAS::mulvec( alpha, blas_mat_A< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_transposed )
        {
            assert( is_complex_type< value_t >::value == false );
            
            auto  t = BLAS::mulvec( value_t(1), BLAS::transposed( blas_mat_A< value_t >( R ) ), x );

            BLAS::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_adjoint )
        {
            auto  t = BLAS::mulvec( value_t(1), BLAS::adjoint( blas_mat_A< value_t >( R ) ), x );

            BLAS::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
    }// if
    else
        assert( false );
}

//
// solve op(L) x = y with lower triangular L
//
void
trsvl ( const HLIB::matop_t      op_L,
        const HLIB::TMatrix &    L,
        HLIB::TScalarVector &    x,
        const HLIB::diag_type_t  diag_mode );

//
// solve op(U) x = y with upper triangular U
//
void
trsvu ( const HLIB::matop_t      op_U,
        const HLIB::TMatrix &    U,
        HLIB::TScalarVector &    x,
        const HLIB::diag_type_t  diag_mode );

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
inline void
gauss_elim ( HLIB::TMatrix *    A,
             HLIB::TMatrix *    T,
             const TTruncAcc &  acc )
{
    assert( ! is_null_any( A, T ) );
    assert( A->type() == T->type() );

    HLR_LOG( 4, HLIB::to_string( "gauss_elim( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );
        auto  BT = ptrcast( T, TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        hlr::seq::gauss_elim( MA(0,0), MT(0,0), acc );
        // hlr::log( 0, HLIB::to_string( "                               %d = %.8e", MA(0,0)->id(), HLIB::norm_F( MA(0,0) ) ) );

        // T_01 = A_00⁻¹ · A_01
        multiply( 1.0, apply_normal, MA(0,0), apply_normal, MA(0,1), 0.0, MT(0,1), acc );
        // seq::matrix::clear( *MT(0,1) );
        // multiply( 1.0, MA(0,0), MA(0,1), MT(0,1), acc );
        
        // T_10 = A_10 · A_00⁻¹
        multiply( 1.0, apply_normal, MA(1,0), apply_normal, MA(0,0), 0.0, MT(1,0), acc );
        // seq::matrix::clear( *MT(1,0) );
        // multiply( 1.0, MA(1,0), MA(0,0), MT(1,0), acc );

        // A_11 = A_11 - T_10 · A_01
        multiply( -1.0, apply_normal, MT(1,0), apply_normal, MA(0,1), 1.0, MA(1,1), acc );
        // multiply( -1.0, MT(1,0), MA(0,1), MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        hlr::seq::gauss_elim( MA(1,1), MT(1,1), acc );
        // hlr::log( 0, HLIB::to_string( "                               %d = %.8e", MA(1,1)->id(), HLIB::norm_F( MA(1,1) ) ) );

        // A_01 = - T_01 · A_11
        multiply( -1.0, apply_normal, MT(0,1), apply_normal, MA(1,1), 0.0, MA(0,1), acc );
        // seq::matrix::clear( *MA(0,1) );
        // multiply( -1.0, MT(0,1), MA(1,1), MA(0,1), acc );
            
        // A_10 = - A_11 · T_10
        multiply( -1.0, apply_normal, MA(1,1), apply_normal, MT(1,0), 0.0, MA(1,0), acc );
        // seq::matrix::clear( *MA(1,0) );
        // multiply( -1.0, MA(1,1), MT(1,0), MA(1,0), acc );

        // A_00 = T_00 - A_01 · T_10
        multiply( -1.0, apply_normal, MA(0,1), apply_normal, MT(1,0), 1.0, MA(0,0), acc );
        // multiply( -1.0, MA(0,1), MT(1,0), MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        if ( A->is_complex() ) HLIB::BLAS::invert( DA->blas_cmat() );
        else                   HLIB::BLAS::invert( DA->blas_rmat() );
    }// if
    else
        assert( false );

    HLR_LOG( 4, HLIB::to_string( "gauss_elim( %d )", A->id() ) );
}

}}// namespace hlr::seq

#endif // __HLR_SEQ_ARITH_HH
