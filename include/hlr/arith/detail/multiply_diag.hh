#ifndef __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions with diagonal matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

namespace hlr { 

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·D·B + C
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H D U(B) ] , [ V(C), V(B)^H ] )
    auto  VD  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_D, blas::mat< value_t >( D ) ) );
    auto  VDU = blas::prod( value_t(1), VD, blas::mat_U< value_t >( B, op_B ) );
    auto  UT  = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), VDU );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UT },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H D B)^H ] )
    auto  DV  = blas::prod( value_t(1),
                            blas::adjoint( blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) ),
                            blas::mat_V< value_t >( A, op_A ) );
    auto  BDV = blas::prod( alpha,
                            blas::adjoint( blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) ),
                            DV );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), BDV },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A D U(B) ] , [ V(C), V(B) ] )
    auto  DU  = blas::prod( value_t(1),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ),
                            blas::mat_U< value_t >( B, op_B ) );
    auto  ADU = blas::prod( alpha,
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            DU );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), ADU },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AD  = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );
    auto  ADB = blas::prod( alpha,
                            AD,
                            blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( value_t(1), blas::mat_U< value_t >( C ), blas::adjoint( blas::mat_V< value_t >( C ) ), value_t(1), ADB );

    auto [ U, V ] = approx( ADB, acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) (( V(A)^H D) U(B) ) V(B)^H
    auto  VD   = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_D, blas::mat< value_t >( D ) ) );
    auto  VDU  = blas::prod( value_t(1), VD, blas::mat_U< value_t >( B, op_B ) );
    auto  UVDU = blas::prod( value_t(1), blas::mat_U< value_t >( A, op_A ), VDU );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UVDU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A D U(B) ) V(B)^H
    auto  DU  = blas::prod( value_t(1),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ),
                            blas::mat_U< value_t >( B, op_B ) );
    auto  ADU = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            DU );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, ADU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H D B )
    auto  VD  = blas::prod( value_t(1),
                            blas::adjoint( blas::mat_V< value_t >( A, op_A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );
    auto  VDB = blas::prod( value_t(1),
                            VD,
                            blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), VDB, value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_D,
           const hpro::TDenseMatrix &  D,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A D B
    auto  AD  = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );

    blas::prod( alpha,
                AD,
                blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ),
                value_t(1), hpro::blas_mat< value_t >( C ) );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
