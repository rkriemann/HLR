//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : functions for TLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "approx.hh"

namespace TLR
{

template < typename value_t >
void
trsmuh ( const TDenseMatrix *  U,
         TMatrix *             X )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "trsmuh( %d, %d )", U->id(), X->id() );
    
    if ( is_lowrank( X ) )
    {
        auto                  RX = ptrcast( X, TRkMatrix );
        B::Matrix< value_t >  Y( blas_mat_B< value_t >( RX ), copy_value );

        B::prod( 1.0, B::adjoint( blas_mat< value_t >( U ) ), Y, 0.0, blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto                  DX = ptrcast( X, TDenseMatrix );
        B::Matrix< value_t >  Y( blas_mat< value_t >( DX ), copy_value );
    
        B::prod( 1.0, Y, blas_mat< value_t >( U ), 0.0, blas_mat< value_t >( DX ) );
    }// else
}

template < typename value_t >
void
update ( const TRkMatrix *  A,
         const TRkMatrix *  B,
         TRkMatrix *        C,
         const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = B::prod( value_t(1),  B::adjoint( blas_mat_B< value_t >( A ) ), blas_mat_A< value_t >( B ) );
    auto  UT = B::prod( value_t(-1), blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), UT },
                                                   { blas_mat_B< value_t >( C ), blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( 1.0, UT, B->blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const TRkMatrix *     A,
         const TDenseMatrix *  B,
         TRkMatrix *           C,
         const TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = B::prod( value_t(-1), B::adjoint( blas_mat< value_t >( B ) ), blas_mat_B< value_t >( A ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), blas_mat_A< value_t >( A ) },
                                                   { blas_mat_B< value_t >( C ), VB },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( 1.0, UT, B->blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const TDenseMatrix *  A,
         const TRkMatrix *     B,
         TRkMatrix *           C,
         const TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = B::prod( value_t(-1), blas_mat< value_t >( A ), blas_mat_A< value_t >( B ) );

    auto [ U, V ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( C ), AU },
                                                   { blas_mat_B< value_t >( C ), blas_mat_B< value_t >( B ) },
                                                   acc );
        
    C->set_rank( U, V );
    // C->add_rank( 1.0, UT, B->blas_mat_B(), acc );
}

template < typename value_t >
void
update ( const TDenseMatrix *  A,
         const TDenseMatrix *  B,
         TRkMatrix *           C,
         const TTruncAcc &     acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "update( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = B::prod( value_t(-1), blas_mat< value_t >( A ), blas_mat< value_t >( B ) );

    B::prod( value_t(1), blas_mat_A< value_t >( C ), B::adjoint( blas_mat_B< value_t >( C ) ), 1.0, AB );

    auto [ U, V ] = LR::approx_svd< value_t >( AB, acc );
        
    C->set_rank( U, V );
}

template < typename value_t >
void
update ( const TRkMatrix *  A,
         const TRkMatrix *  B,
         TDenseMatrix *     C,
         const TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = B::prod( value_t(1), B::adjoint( blas_mat_B< value_t >( A ) ), blas_mat_A< value_t >( B ) );
    auto  UT = B::prod( value_t(1), blas_mat_A< value_t >( A ), T );

    B::prod( value_t(-1), UT, B::adjoint( blas_mat_B< value_t >( B ) ), value_t(1), blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const TDenseMatrix *  A,
         const TRkMatrix *     B,
         TDenseMatrix *        C,
         const TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = B::prod( value_t(1), blas_mat< value_t >( A ), blas_mat_A< value_t >( B ) );

    B::prod( value_t(-1), AU, B::adjoint( blas_mat_B< value_t >( B ) ), value_t(1), blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const TRkMatrix *     A,
         const TDenseMatrix *  B,
         TDenseMatrix *        C,
         const TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = B::prod( value_t(1), B::adjoint( blas_mat_B< value_t >( A ) ), blas_mat< value_t >( B ) );

    B::prod( value_t(-1), blas_mat_A< value_t >( A ), VB, 1.0, blas_mat< value_t >( C ) );
}

template < typename value_t >
void
update ( const TDenseMatrix *  A,
         const TDenseMatrix *  B,
         TDenseMatrix *        C,
         const TTruncAcc & )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "updated( %d, %d, %d )", A->id(), B->id(), C->id() );
    
    // C = C + A B
    B::prod( value_t(-1), blas_mat< value_t >( A ), blas_mat< value_t >( B ), value_t(1), blas_mat< value_t >( C ) );
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
        HERROR( ERR_NOT_IMPL, "", "" );
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
        HERROR( ERR_NOT_IMPL, "", "" );
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
        HERROR( ERR_NOT_IMPL, "", "" );
}

}// namespace TLR

// Local Variables:
// mode: c++
// End:
