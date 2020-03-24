#ifndef __HLR_ARITH_ADD_HH
#define __HLR_ARITH_ADD_HH
//
// Project     : HLib
// File        : add.hh
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/arith/approx.hh"
#include "hlr/utils/log.hh"

namespace hlr
{

namespace hpro = HLIB;

//
// compute C := C + α A
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TRkMatrix &  A,
      hpro::TRkMatrix &        C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  UA = blas::copy( hpro::blas_mat_A< value_t >( A ) );

        blas::scale( alpha, UA );

        auto [ U, V ] = approx( {                               UA, hpro::blas_mat_A< value_t >( C ) },
                                { hpro::blas_mat_B< value_t >( A ), hpro::blas_mat_B< value_t >( C ) },
                                acc );
        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        auto [ U, V ] = approx( { hpro::blas_mat_A< value_t >( A ), hpro::blas_mat_A< value_t >( C ) },
                                { hpro::blas_mat_B< value_t >( A ), hpro::blas_mat_B< value_t >( C ) },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ) );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TDenseMatrix &  A,
      hpro::TRkMatrix &           C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto  TA = blas::copy( hpro::blas_mat< value_t >( A ) );

    blas::prod( alpha,
                hpro::blas_mat_A< value_t >( C ),
                blas::adjoint( hpro::blas_mat_B< value_t >( C ) ),
                value_t( 1 ),
                TA );

    auto [ U, V ] = approx( TA, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TRkMatrix &  A,
      hpro::TDenseMatrix &     C,
      const hpro::TTruncAcc &,
      const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    blas::prod( alpha,
                hpro::blas_mat_A< value_t >( A ),
                blas::adjoint( hpro::blas_mat_B< value_t >( A ) ),
                value_t( 1 ),
                hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TDenseMatrix &  A,
      hpro::TDenseMatrix &        C,
      const hpro::TTruncAcc &,
      const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    // C = C + α A
    blas::add( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "add" function
//

template < typename value_t,
           typename approx_t,
           typename matrix_A_t >
void
add ( const value_t            alpha,
      const matrix_A_t &       A,
      hpro::TMatrix &          C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    if      ( is_dense(   C ) ) add< value_t, approx_t >( alpha, A, *ptrcast( &C, hpro::TDenseMatrix ), acc, approx );
    else if ( is_lowrank( C ) ) add< value_t, approx_t >( alpha, A, *ptrcast( &C, hpro::TRkMatrix ),    acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TMatrix &    A,
      hpro::TMatrix &          C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    if      ( is_dense(   A ) ) add< value_t, approx_t, hpro::TDenseMatrix >( alpha, *cptrcast( &A, hpro::TDenseMatrix ), C, acc, approx );
    else if ( is_lowrank( A ) ) add< value_t, approx_t, hpro::TRkMatrix >(    alpha, *cptrcast( &A, hpro::TRkMatrix ),    C, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_ADD_HH
