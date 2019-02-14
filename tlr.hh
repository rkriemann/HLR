#ifndef __HLR_TLR_HH
#define __HLR_TLR_HH
//
// Project     : HLib
// File        : tlr.hh
// Description : TLR arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <hlib.hh>

#include "approx.hh"

namespace TLR
{

///////////////////////////////////////////////////////////////////////
//
// clustering
//
///////////////////////////////////////////////////////////////////////

std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile );

///////////////////////////////////////////////////////////////////////
//
// arithmetic
//
///////////////////////////////////////////////////////////////////////

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const HLIB::TDenseMatrix *  U,
         HLIB::TMatrix *             X )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "trsmuh( %d, %d )", U->id(), X->id() );
    
    if ( HLIB::is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, HLIB::TRkMatrix );
        auto  Y  = copy( HLIB::blas_mat_B< value_t >( RX ) );

        HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat< value_t >( U ) ), Y, value_t(0), HLIB::blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, HLIB::TDenseMatrix );
        auto  Y  = copy( HLIB::blas_mat< value_t >( DX ) );
    
        HLIB::BLAS::prod( value_t(1), Y, HLIB::blas_mat< value_t >( U ), value_t(0), HLIB::blas_mat< value_t >( DX ) );
    }// else
}

//
// compute C := C - A·B
//
template < typename value_t >
void
update ( const HLIB::TRkMatrix *  A,
         const HLIB::TRkMatrix *  B,
         HLIB::TRkMatrix *        C,
         const HLIB::TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = HLIB::BLAS::prod( value_t(1),  HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat_A< value_t >( B ) );
    auto  UT = HLIB::BLAS::prod( value_t(-1), HLIB::blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), UT },
                                                   { HLIB::blas_mat_B< value_t >( C ), HLIB::blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( value_t(1), UT, B->HLIB::blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const HLIB::TRkMatrix *     A,
         const HLIB::TDenseMatrix *  B,
         HLIB::TRkMatrix *           C,
         const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = HLIB::BLAS::prod( value_t(-1), HLIB::BLAS::adjoint( HLIB::blas_mat< value_t >( B ) ), HLIB::blas_mat_B< value_t >( A ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), HLIB::blas_mat_A< value_t >( A ) },
                                                   { HLIB::blas_mat_B< value_t >( C ), VB },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( value_t(1), UT, B->HLIB::blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const HLIB::TDenseMatrix *  A,
         const HLIB::TRkMatrix *     B,
         HLIB::TRkMatrix *           C,
         const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = HLIB::BLAS::prod( value_t(-1), HLIB::blas_mat< value_t >( A ), HLIB::blas_mat_A< value_t >( B ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { HLIB::blas_mat_A< value_t >( C ), AU },
                                                   { HLIB::blas_mat_B< value_t >( C ), HLIB::blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( value_t(1), UT, B->HLIB::blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const HLIB::TDenseMatrix *  A,
         const HLIB::TDenseMatrix *  B,
         HLIB::TRkMatrix *           C,
         const HLIB::TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = HLIB::BLAS::prod( value_t(-1), HLIB::blas_mat< value_t >( A ), HLIB::blas_mat< value_t >( B ) );

    HLIB::BLAS::prod( value_t(1), HLIB::blas_mat_A< value_t >( C ), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = LR::approx_svd< value_t >( AB, acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
update ( const HLIB::TRkMatrix *  A,
         const HLIB::TRkMatrix *  B,
         HLIB::TDenseMatrix *     C,
         const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat_A< value_t >( B ) );
    auto  UT = HLIB::BLAS::prod( value_t(1), HLIB::blas_mat_A< value_t >( A ), T );

    HLIB::BLAS::prod( value_t(-1), UT, HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( B ) ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const HLIB::TDenseMatrix *  A,
         const HLIB::TRkMatrix *     B,
         HLIB::TDenseMatrix *        C,
         const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = HLIB::BLAS::prod( value_t(1), HLIB::blas_mat< value_t >( A ), HLIB::blas_mat_A< value_t >( B ) );

    HLIB::BLAS::prod( value_t(-1), AU, HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( B ) ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const HLIB::TRkMatrix *     A,
         const HLIB::TDenseMatrix *  B,
         HLIB::TDenseMatrix *        C,
         const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat_B< value_t >( A ) ), HLIB::blas_mat< value_t >( B ) );

    HLIB::BLAS::prod( value_t(-1), HLIB::blas_mat_A< value_t >( A ), VB, value_t(1), HLIB::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const HLIB::TDenseMatrix *  A,
         const HLIB::TDenseMatrix *  B,
         HLIB::TDenseMatrix *        C,
         const HLIB::TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + A B
    HLIB::BLAS::prod( value_t(-1), HLIB::blas_mat< value_t >( A ), HLIB::blas_mat< value_t >( B ), value_t(1), HLIB::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "update" function
//

template < typename T_value,
           typename T_matrix1,
           typename T_matrix2 >
void
update ( const T_matrix1 *        A,
         const T_matrix2 *        B,
         HLIB::TMatrix *          C,
         const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   C ) ) update< T_value >( A, B, ptrcast( C, HLIB::TDenseMatrix ), acc );
    else if ( HLIB::is_lowrank( C ) ) update< T_value >( A, B, ptrcast( C, HLIB::TRkMatrix ),    acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

template < typename T_value,
           typename T_matrix1 >
void
update ( const T_matrix1 *        A,
         const HLIB::TMatrix *    B,
         HLIB::TMatrix *          C,
         const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   B ) ) update< T_value, T_matrix1, HLIB::TDenseMatrix >( A, cptrcast( B, HLIB::TDenseMatrix ), C, acc );
    else if ( HLIB::is_lowrank( B ) ) update< T_value, T_matrix1, HLIB::TRkMatrix >(    A, cptrcast( B, HLIB::TRkMatrix ),    C, acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

template < typename T_value >
void
update ( const HLIB::TMatrix *    A,
         const HLIB::TMatrix *    B,
         HLIB::TMatrix *          C,
         const HLIB::TTruncAcc &  acc )
{
    if      ( HLIB::is_dense(   A ) ) update< T_value, HLIB::TDenseMatrix >( cptrcast( A, HLIB::TDenseMatrix ), B, C, acc );
    else if ( HLIB::is_lowrank( A ) ) update< T_value, HLIB::TRkMatrix >(    cptrcast( A, HLIB::TRkMatrix ),    B, C, acc );
    else
        HERROR( HLIB::ERR_NOT_IMPL, "", "" );
}

}// namespace TLR

#endif // __HLR_TLR_HH
