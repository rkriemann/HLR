#ifndef __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions with compressed matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

namespace hlr
{

//
// blocked x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_MULT_PRINT;

    auto  DA = A.mat_decompressed();
    auto  DB = B.mat_decompressed();
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    blas::prod( alpha,
                blas::mat_view( op_A, DA ),
                blas::mat_view( op_B, DB ),
                value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    auto  DA = A.mat_decompressed();
    auto  DB = B.mat_decompressed();
    
    blas::prod( alpha,
                blas::mat_view( op_A, DA ),
                blas::mat_view( op_B, DB ),
                value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    auto  DA = A.mat_decompressed();
    
    blas::prod( alpha,
                blas::mat_view( op_A, DA ),
                blas::mat_view( op_B, blas::mat( B ) ),
                value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    auto  DB = B.mat_decompressed();
    
    blas::prod( alpha,
                blas::mat_view( op_A, blas::mat( A ) ),
                blas::mat_view( op_B, DB ),
                value_t(1), blas::mat( C ) );
}

//
// dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  DA = A.mat_decompressed();
    auto  DB = B.mat_decompressed();
    auto  AB = blas::prod( alpha,
                           blas::mat_view( op_A, DA ),
                           blas::mat_view( op_B, DB ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( value_t(1), blas::mat_U( C ), blas::adjoint( blas::mat_V( C ) ), value_t(1), AB );

    auto [ U, V ] = approx( AB, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  DA = A.mat_decompressed();
    auto  UB = B.U_decompressed( op_B );
    auto  VB = B.V_decompressed( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  UA = A.U_decompressed( op_A );
    auto  VA = A.V_decompressed( op_A );
    auto  DB = B.mat_decompressed();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UA, VB, value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  UA = A.U_decompressed( op_A );
    auto  VA = A.V_decompressed( op_A );
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, blas::mat( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UA, VB, value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  UA = A.U_decompressed( op_A );
    auto  VA = A.V_decompressed( op_A );
    auto  UB = B.U_decompressed( op_B );
    auto  VB = B.V_decompressed( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod(      alpha, UA, T );
    auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( C.row_is(), C.col_is(), UT, VB );
        
    hlr::add< value_t >( value_t(1), *R, C, acc, approx );
}

//
// lowrank x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  UA = A.U_decompressed( op_A );
    auto  VA = A.V_decompressed( op_A );

    auto  UB = B.U_decompressed( op_B );
    auto  VB = B.V_decompressed( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod( value_t(1), UA, T );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
