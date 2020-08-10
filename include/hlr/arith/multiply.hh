#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/utils/log.hh"
#include "hlr/arith/add.hh"

namespace hlr
{

namespace hpro = HLIB;

/////////////////////////////////////////////////////////////////////////////////
//
// helper functions
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TBlockMatrix &       A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = blas::matrix< value_t >( C, A.block( i, 0, op_A )->nrows( op_A ), C.ncols() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  is_j = A_ij->col_is( op_A );
            auto  B_j  = blas::matrix< value_t >( B, is_j - A_ij->col_ofs( op_A ), blas::range::all );
            
            multiply( alpha, op_A, * A_ij, B_j, C_i );
        }// for
    }// for
}

/////////////////////////////////////////////////////////////////////////////////
//
// newer version with matrix operators (normal, transposed, adjoint)
//
/////////////////////////////////////////////////////////////////////////////////

#define HLR_MULT_PRINT   HLR_LOG( 4, hpro::to_string( "multiply( %s %d, %s %d, %s %d )", \
                                                      A.typestr().c_str(), A.id(), \
                                                      B.typestr().c_str(), B.id(), \
                                                      C.typestr().c_str(), C.id() ) )
//
// forward decl.
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx );

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                    multiply< value_t >( alpha,
                                         op_A, *A.block( i, l, op_A ),
                                         op_B, *B.block( l, j, op_B ),
                                         *C.block( i, j ), acc, approx );
            }// if       
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< hpro::TBlockMatrix >( C.row_is(), C.col_is() );

    BC->set_block_struct( A.nblock_rows( op_A ), B.nblock_cols( op_B ) );
    
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new hpro::TRkMatrix( A_il->row_is( op_A ), B_lj->col_is( op_B ),
                                                                  hpro::value_type< value_t >::value ) );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    // apply update
    hlr::add< value_t >( value_t(1), *BC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    //
    // perform block × block multiplication and compute local result
    // which is used to update C
    //

    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            std::unique_ptr< hpro::TDenseMatrix >  C_ij;
            
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( C_ij ) )
                        C_ij = std::make_unique< hpro::TDenseMatrix >( A_il->row_is( op_A ), B_lj->col_is( op_B ), hpro::value_type< value_t >::value );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                }// if
            }// if

            if ( ! is_null( C_ij ) )
                C.add_block( value_t(1), value_t(1), C_ij.get() );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UC },
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
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V< value_t >( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, hpro::adjoint( op_B ), B, VA, VC );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), VC },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = hpro::TDenseMatrix( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  DB   = hpro::blas_mat< value_t >( B );
            auto  is_j = A_ij->col_is( op_A );

            if ( op_B == hpro::apply_normal )
            {
                auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i, acc, approx );
            }// if
            else
            {
                auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i, acc, approx );
            }// else
        }// for

        C.add_block( hpro::real(1), hpro::real(1), &C_i );
    }// for
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TBlockMatrix &     C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), T );
    auto  R  = std::make_unique< hpro::TRkMatrix >( C.row_is(), C.col_is(), UT, blas::mat_V< value_t >( B, op_B ) );
        
    hlr::add< value_t >( value_t(1), *R, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TRkMatrix &        C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), T );

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
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( alpha,
                           blas::adjoint( blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) ),
                           blas::mat_V< value_t >( A, op_A ) );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), VB },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = blas::prod( alpha,
                           blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                           blas::mat_U< value_t >( B, op_B ) );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), AU },
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
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = blas::prod( alpha,
                           blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                           blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    blas::prod( value_t(1), blas::mat_U< value_t >( C ), blas::adjoint( blas::mat_V< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = approx( AB, acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TDenseMatrix &     C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod( value_t(1), blas::mat_U< value_t >( A, op_A ), T );

    blas::prod( alpha, UT, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ), blas::mat_U< value_t >( B, op_B ) );

    blas::prod( alpha, AU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), VB, value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C,
           const hpro::TTruncAcc &,
           const approx_t & )
{
    HLR_MULT_PRINT;
    
    // C = C + A B
    blas::prod( alpha,
                blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ),
                value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "multiply" function
//

template < typename value_t,
           typename approx_t,
           typename matrixA_t,
           typename matrixB_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const matrixA_t &        A,
           const hpro::matop_t      op_B,
           const matrixB_t &        B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_MULT_PRINT;

    if      ( is_blocked( C ) )
        multiply< value_t, approx_t >( alpha, op_A, A, op_B, B, * ptrcast( &C, hpro::TBlockMatrix ), acc, approx );
    else if ( is_dense(   C ) )
        multiply< value_t, approx_t >( alpha, op_A, A, op_B, B, * ptrcast( &C, hpro::TDenseMatrix ), acc, approx );
    else if ( is_lowrank( C ) )
        multiply< value_t, approx_t >( alpha, op_A, A, op_B, B, * ptrcast( &C, hpro::TRkMatrix ),    acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
}

template < typename value_t,
           typename approx_t,
           typename matrixA_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const matrixA_t &        A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if      ( is_blocked( B ) )
        multiply< value_t, approx_t, matrixA_t, hpro::TBlockMatrix >( alpha, op_A, A, op_B, * cptrcast( &B, hpro::TBlockMatrix ), C, acc, approx );
    else if ( is_dense(   B ) )
        multiply< value_t, approx_t, matrixA_t, hpro::TDenseMatrix >( alpha, op_A, A, op_B, * cptrcast( &B, hpro::TDenseMatrix ), C, acc, approx );
    else if ( is_lowrank( B ) )
        multiply< value_t, approx_t, matrixA_t, hpro::TRkMatrix >(    alpha, op_A, A, op_B, * cptrcast( &B, hpro::TRkMatrix ),    C, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + B.typestr() );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if      ( is_blocked( A ) )
        multiply< value_t, approx_t, hpro::TBlockMatrix >( alpha, op_A, * cptrcast( &A, hpro::TBlockMatrix ), op_B, B, C, acc, approx );
    else if ( is_dense(   A ) )
        multiply< value_t, approx_t, hpro::TDenseMatrix >( alpha, op_A, * cptrcast( &A, hpro::TDenseMatrix ), op_B, B, C, acc, approx );
    else if ( is_lowrank( A ) )
        multiply< value_t, approx_t, hpro::TRkMatrix >(    alpha, op_A, * cptrcast( &A, hpro::TRkMatrix ),    op_B, B, C, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

/////////////////////////////////////////////////////////////////////////////////
//
// older, simplified version
//
/////////////////////////////////////////////////////////////////////////////////

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
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A ) ), blas::mat_U< value_t >( B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U< value_t >( A ), T );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UT },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B ) },
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
    auto  VB = blas::prod( alpha, blas::adjoint( hpro::blas_mat< value_t >( B ) ), blas::mat_V< value_t >( A ) );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A ) },
                            { blas::mat_V< value_t >( C ), VB },
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
    auto  AU = blas::prod( alpha, hpro::blas_mat< value_t >( A ), blas::mat_U< value_t >( B ) );

    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), AU },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B ) },
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

    blas::prod( value_t(1), blas::mat_U< value_t >( C ), blas::adjoint( blas::mat_V< value_t >( C ) ), value_t(1), AB );

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
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A ) ), blas::mat_U< value_t >( B ) );
    auto  UT = blas::prod( value_t(1), blas::mat_U< value_t >( A ), T );

    blas::prod( alpha, UT, blas::adjoint( blas::mat_V< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
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
    auto  AU = blas::prod( value_t(1), hpro::blas_mat< value_t >( A ), blas::mat_U< value_t >( B ) );

    blas::prod( alpha, AU, blas::adjoint( blas::mat_V< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
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
    auto  VB = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A ) ), hpro::blas_mat< value_t >( B ) );

    blas::prod( alpha, blas::mat_U< value_t >( A ), VB, value_t(1), hpro::blas_mat< value_t >( C ) );
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
        HLR_ERROR( "unsupported matrix type : " + C->typestr() );
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
        HLR_ERROR( "unsupported matrix type : " + B->typestr() );
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
        HLR_ERROR( "unsupported matrix type : " + A->typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_MULTIPLY_HH
