#ifndef __HLR_MULTIPLY_HH
#define __HLR_MULTIPLY_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common/approx.hh"
#include "utils/log.hh"

namespace HLR
{

using namespace HLIB;

//
// compute C := C + α A·B
//
template < typename value_t >
void
multiply ( const value_t      alpha,
           const TRkMatrix *  A,
           const TRkMatrix *  B,
           TRkMatrix *        C,
           const TTruncAcc &  acc )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A ) ), blas_mat_A< value_t >( B ) );
    auto  UT = BLAS::prod(      alpha, blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), UT },
                                                    { blas_mat_B< value_t >( C ), blas_mat_B< value_t >( B ) },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TRkMatrix *     A,
           const TDenseMatrix *  B,
           TRkMatrix *           C,
           const TTruncAcc &     acc )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = BLAS::prod( alpha, BLAS::adjoint( blas_mat< value_t >( B ) ), blas_mat_B< value_t >( A ) );

    auto [ U, V ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), blas_mat_A< value_t >( A ) },
                                                    { blas_mat_B< value_t >( C ), VB },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TDenseMatrix *  A,
           const TRkMatrix *     B,
           TRkMatrix *           C,
           const TTruncAcc &     acc )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = BLAS::prod( alpha, blas_mat< value_t >( A ), blas_mat_A< value_t >( B ) );

    auto [ U, V ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), AU },
                                                    { blas_mat_B< value_t >( C ), blas_mat_B< value_t >( B ) },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TDenseMatrix *  A,
           const TDenseMatrix *  B,
           TRkMatrix *           C,
           const TTruncAcc &     acc )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = BLAS::prod( alpha, blas_mat< value_t >( A ), blas_mat< value_t >( B ) );

    BLAS::prod( value_t(1), blas_mat_A< value_t >( C ), BLAS::adjoint( blas_mat_B< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = HLR::approx_svd< value_t >( AB, acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t      alpha,
           const TRkMatrix *  A,
           const TRkMatrix *  B,
           TDenseMatrix *     C,
           const TTruncAcc & )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A ) ), blas_mat_A< value_t >( B ) );
    auto  UT = BLAS::prod( value_t(1), blas_mat_A< value_t >( A ), T );

    BLAS::prod( alpha, UT, BLAS::adjoint( blas_mat_B< value_t >( B ) ), value_t(1), blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TDenseMatrix *  A,
           const TRkMatrix *     B,
           TDenseMatrix *        C,
           const TTruncAcc & )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = BLAS::prod( value_t(1), blas_mat< value_t >( A ), blas_mat_A< value_t >( B ) );

    BLAS::prod( alpha, AU, BLAS::adjoint( blas_mat_B< value_t >( B ) ), value_t(1), blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TRkMatrix *     A,
           const TDenseMatrix *  B,
           TDenseMatrix *        C,
           const TTruncAcc & )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A ) ), blas_mat< value_t >( B ) );

    BLAS::prod( alpha, blas_mat_A< value_t >( A ), VB, value_t(1), blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t         alpha,
           const TDenseMatrix *  A,
           const TDenseMatrix *  B,
           TDenseMatrix *        C,
           const TTruncAcc & )
{
    if ( verbose( 4 ) )
        DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + A B
    BLAS::prod( alpha, blas_mat< value_t >( A ), blas_mat< value_t >( B ), value_t(1), blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "multiply" function
//

template < typename value_t,
           typename matrix1_t,
           typename matrix2_t >
void
multiply ( const value_t      alpha,
           const matrix1_t *  A,
           const matrix2_t *  B,
           TMatrix *          C,
           const TTruncAcc &  acc )
{
    if      ( is_dense(   C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, TDenseMatrix ), acc );
    else if ( is_lowrank( C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, TRkMatrix ),    acc );
    else
        assert( false );
}

template < typename value_t,
           typename matrix1_t >
void
multiply ( const value_t      alpha,
           const matrix1_t *  A,
           const TMatrix *    B,
           TMatrix *          C,
           const TTruncAcc &  acc )
{
    if      ( is_dense(   B ) ) multiply< value_t, matrix1_t, TDenseMatrix >( alpha, A, cptrcast( B, TDenseMatrix ), C, acc );
    else if ( is_lowrank( B ) ) multiply< value_t, matrix1_t, TRkMatrix >(    alpha, A, cptrcast( B, TRkMatrix ),    C, acc );
    else
        HLR::error( "unsupported matrix type : " + B->typestr() );
}

template < typename value_t >
void
multiply ( const value_t      alpha,
           const TMatrix *    A,
           const TMatrix *    B,
           TMatrix *          C,
           const TTruncAcc &  acc )
{
    if      ( is_dense(   A ) ) multiply< value_t, TDenseMatrix >( alpha, cptrcast( A, TDenseMatrix ), B, C, acc );
    else if ( is_lowrank( A ) ) multiply< value_t, TRkMatrix >(    alpha, cptrcast( A, TRkMatrix ),    B, C, acc );
    else
        HLR::error( "unsupported matrix type : " + B->typestr() );
}

}// namespace HLR

#endif // __HLR_MULTIPLY_HH
