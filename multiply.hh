#ifndef __HLR_MULTIPLY_HH
#define __HLR_MULTIPLY_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "approx.hh"

//
// compute C := C + α A·B
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const HLIB::TRkMatrix *  A,
           const HLIB::TRkMatrix *  B,
           HLIB::TRkMatrix *        C,
           const HLIB::TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat_A< value_t >( B ) );
    auto  UT = HLIB::BLAS::prod(      alpha, HLIB::blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), UT },
                                                   { HLIB::blas_mat_B< value_t >( C ), HLIB::blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TRkMatrix *     A,
           const HLIB::TDenseMatrix *  B,
           HLIB::TRkMatrix *           C,
           const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = HLIB::BLAS::prod( alpha, HLIB::BLAS::adjoint( HLIB::blas_mat< value_t >( B ) ), HLIB::blas_mat_B< value_t >( A ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), HLIB::blas_mat_A< value_t >( A ) },
                                                   { HLIB::blas_mat_B< value_t >( C ), VB },
                                                   acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TDenseMatrix *  A,
           const HLIB::TRkMatrix *     B,
           HLIB::TRkMatrix *           C,
           const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = HLIB::BLAS::prod( alpha, HLIB::blas_mat< value_t >( A ), HLIB::blas_mat_A< value_t >( B ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), AU },
                                                   { HLIB::blas_mat_B< value_t >( C ), HLIB::blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TDenseMatrix *  A,
           const HLIB::TDenseMatrix *  B,
           HLIB::TRkMatrix *           C,
           const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = HLIB::BLAS::prod( alpha, HLIB::blas_mat< value_t >( A ), HLIB::blas_mat< value_t >( B ) );

    HLIB::BLAS::prod( value_t(1), HLIB::blas_mat_A< value_t >( C ), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = LR::approx_svd< value_t >( AB, acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const HLIB::TRkMatrix *  A,
           const HLIB::TRkMatrix *  B,
           HLIB::TDenseMatrix *     C,
           const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiplyd( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat_A< value_t >( B ) );
    auto  UT = HLIB::BLAS::prod( value_t(1), HLIB::blas_mat_A< value_t >( A ), T );

    HLIB::BLAS::prod( alpha, UT, HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( B ) ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TDenseMatrix *  A,
           const HLIB::TRkMatrix *     B,
           HLIB::TDenseMatrix *        C,
           const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiplyd( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = HLIB::BLAS::prod( value_t(1), HLIB::blas_mat< value_t >( A ), HLIB::blas_mat_A< value_t >( B ) );

    HLIB::BLAS::prod( alpha, AU, HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( B ) ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TRkMatrix *     A,
           const HLIB::TDenseMatrix *  B,
           HLIB::TDenseMatrix *        C,
           const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiplyd( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat< value_t >( B ) );

    HLIB::BLAS::prod( alpha, HLIB::blas_mat_A< value_t >( A ), VB, value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const HLIB::TDenseMatrix *  A,
           const HLIB::TDenseMatrix *  B,
           HLIB::TDenseMatrix *        C,
           const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "multiplyd( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + A B
    HLIB::BLAS::prod( alpha, HLIB::blas_mat< value_t >( A ), HLIB::blas_mat< value_t >( B ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "multiply" function
//

template < typename value_t,
           typename matrix1_t,
           typename matrix2_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const matrix2_t *        B,
           HLIB::TMatrix *          C,
           const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, HLIB::TDenseMatrix ), acc );
    else if ( HLIB::is_lowrank( C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, HLIB::TRkMatrix ),    acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

template < typename value_t,
           typename matrix1_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const HLIB::TMatrix *    B,
           HLIB::TMatrix *          C,
           const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   B ) ) multiply< value_t, matrix1_t, HLIB::TDenseMatrix >( alpha, A, cptrcast( B, HLIB::TDenseMatrix ), C, acc );
    else if ( HLIB::is_lowrank( B ) ) multiply< value_t, matrix1_t, HLIB::TRkMatrix >(    alpha, A, cptrcast( B, HLIB::TRkMatrix ),    C, acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const HLIB::TMatrix *    A,
           const HLIB::TMatrix *    B,
           HLIB::TMatrix *          C,
           const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   A ) ) multiply< value_t, HLIB::TDenseMatrix >( alpha, cptrcast( A, HLIB::TDenseMatrix ), B, C, acc );
    else if ( HLIB::is_lowrank( A ) ) multiply< value_t, HLIB::TRkMatrix >(    alpha, cptrcast( A, HLIB::TRkMatrix ),    B, C, acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

#endif // __HLR_MULTIPLY_HH
