#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/arith/approx.hh"
#include "hlr/utils/log.hh"

namespace hlr
{

namespace hpro = HLIB;

//
// compute C := C + α A·B
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::TRkMatrix *  A,
           const hpro::TRkMatrix *  B,
           hpro::TRkMatrix *        C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat_A< value_t >( B ) );
    auto  UT = blas::prod(      alpha, hpro::blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = approx( { hpro::blas_mat_A< value_t >( C ), UT },
                            { hpro::blas_mat_B< value_t >( C ), hpro::blas_mat_B< value_t >( B ) },
                            acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TRkMatrix *     A,
           const hpro::TDenseMatrix *  B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( alpha, blas::adjoint( hpro::blas_mat< value_t >( B ) ), hpro::blas_mat_B< value_t >( A ) );

    auto [ U, V ] = approx( { hpro::blas_mat_A< value_t >( C ), hpro::blas_mat_A< value_t >( A ) },
                            { hpro::blas_mat_B< value_t >( C ), VB },
                            acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TRkMatrix *     B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat_A< value_t >( B ) );

    auto [ U, V ] = approx( { hpro::blas_mat_A< value_t >( C ), AU },
                            { hpro::blas_mat_B< value_t >( C ), hpro::blas_mat_B< value_t >( B ) },
                            acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TDenseMatrix *  B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( B ) );

    blas::prod( value_t(1), hpro::blas_mat_A< value_t >( C ), blas::adjoint( hpro::blas_mat_B< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = approx( AB, acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::TRkMatrix *  A,
           const hpro::TRkMatrix *  B,
           hpro::TDenseMatrix *     C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat_A< value_t >( B ) );
    auto  UT = blas::prod( value_t(1), hpro::blas_mat_A< value_t >( A ), T );

    blas::prod( alpha, UT, blas::adjoint( hpro::blas_mat_B< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TRkMatrix *     B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), hpro::blas_mat< value_t >( A ), hpro::blas_mat_A< value_t >( B ) );

    blas::prod( alpha, AU, blas::adjoint( hpro::blas_mat_B< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TRkMatrix *     A,
           const hpro::TDenseMatrix *  B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat< value_t >( B ) );

    blas::prod( alpha, hpro::blas_mat_A< value_t >( A ), VB, value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TDenseMatrix *  B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + A B
    blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( B ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "multiply" function
//

template < typename value_t,
           typename approx_t,
           typename matrix1_t,
           typename matrix2_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const matrix2_t *        B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if      ( is_dense(   C ) ) multiply< value_t, approx_t >( alpha, A, B, ptrcast( C, hpro::TDenseMatrix ), acc, approx );
    else if ( is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, A, B, ptrcast( C, hpro::TRkMatrix ),    acc, approx );
    else
        assert( false );
}

template < typename value_t,
           typename approx_t,
           typename matrix1_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const hpro::TMatrix *    B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if      ( is_dense(   B ) ) multiply< value_t, approx_t, matrix1_t, hpro::TDenseMatrix >( alpha, A, cptrcast( B, hpro::TDenseMatrix ), C, acc, approx );
    else if ( is_lowrank( B ) ) multiply< value_t, approx_t, matrix1_t, hpro::TRkMatrix >(    alpha, A, cptrcast( B, hpro::TRkMatrix ),    C, acc, approx );
    else
        hlr::error( "unsupported matrix type : " + B->typestr() );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::TMatrix *    A,
           const hpro::TMatrix *    B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if      ( is_dense(   A ) ) multiply< value_t, approx_t, hpro::TDenseMatrix >( alpha, cptrcast( A, hpro::TDenseMatrix ), B, C, acc, approx );
    else if ( is_lowrank( A ) ) multiply< value_t, approx_t, hpro::TRkMatrix >(    alpha, cptrcast( A, hpro::TRkMatrix ),    B, C, acc, approx );
    else
        hlr::error( "unsupported matrix type : " + B->typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_MULTIPLY_HH
