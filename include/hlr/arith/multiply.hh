#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/add.hh"
#include "hlr/approx/svd.hh"
// #include "hlr/seq/matrix.hh" // DEBUG

namespace hlr
{

namespace detail
{

/////////////////////////////////////////////////////////////////////////////////
//
// helper functions
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TMatrix &            A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C );

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
        
        auto  C_i = blas::matrix< value_t >( C, A.block( i, 0, op_A )->row_is( op_A ) - A.row_ofs( op_A ), blas::range::all );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  B_j  = blas::matrix< value_t >( B, A_ij->col_is( op_A ) - A.col_ofs( op_A ), blas::range::all );
            
            multiply( alpha, op_A, * A_ij, B_j, C_i );
        }// for
    }// for
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TRkMatrix &          A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    switch ( op_A )
    {
        case hpro::apply_normal :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_U< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_adjoint :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( blas::mat_U< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_V< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_conjugate :
        {
            HLR_ASSERT( ! hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( blas::mat_V< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_U< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_transposed :
        {
            HLR_ASSERT( ! hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( blas::mat_U< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_V< value_t >( A ), T, value_t(1), C );
        }
        break;
    }// switch
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TDenseMatrix &       A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    blas::prod( alpha, blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ), B, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TMatrix &            A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    if      ( is_blocked( A ) ) multiply( alpha, op_A, * cptrcast( & A, hpro::TBlockMatrix ), B, C );
    else if ( is_lowrank( A ) ) multiply( alpha, op_A, * cptrcast( & A, hpro::TRkMatrix ), B, C );
    else if ( is_dense(   A ) ) multiply( alpha, op_A, * cptrcast( & A, hpro::TDenseMatrix ), B, C );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace detail

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·B + C
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
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    detail::multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), UC, blas::mat_V< value_t >( B, op_B ) );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
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

    detail::multiply< value_t >( alpha, op_A, A, UB, UC );

    std::scoped_lock  lock( C.mutex() );
    
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
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V< value_t >( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    detail::multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), blas::mat_U< value_t >( A, op_A ), VC );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
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

    detail::multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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

    std::scoped_lock  lock( C.mutex() );
    
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
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    blas::prod( alpha,
                blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ),
                value_t(1), hpro::blas_mat< value_t >( C ) );
}

//////////////////////////////////////////////////////////////////////
//
// deduction of optimal "multiply" function based on matrix type
//
//////////////////////////////////////////////////////////////////////

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
    // auto  Cc = C.copy();

    // hpro::multiply( alpha, op_A, &A, op_B, &B, value_t(1), Cc.get(), acc );

    // hpro::DBG::write( A, "A.mat", "A" );
    // hpro::DBG::write( B, "B.mat", "B" );
    // hpro::DBG::write( C, "C.mat", "C" );
    
    if ( is_blocked( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha, 
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense(   A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    // hpro::DBG::write( C,  "C1.mat", "C1" );
    // hpro::DBG::write( *Cc, "C2.mat", "C2" );
    
    // auto  DC1 = hpro::to_dense( &C );
    // auto  DC2 = hpro::to_dense( Cc.get() );

    // blas::add( value_t(-1), blas::mat< value_t >( DC1 ), blas::mat< value_t >( DC2 ) );
    // if ( blas::norm_F( blas::mat< value_t >( DC2 ) ) > 1e-14 )
    //     std::cout << hpro::to_string( "multiply( %d, %d, %d )", A.id(), B.id(), C.id() ) << ", error = " << blas::norm_F( blas::mat< value_t >( DC2 ) ) << std::endl;
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc )
{
    auto  apx = approx::SVD< value_t >();

    multiply( alpha, op_A, A, op_B, B, C, acc, apx );
}

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

//
// general function
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_D,
           const hpro::TMatrix &    D,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    // auto  TA = hlr::seq::matrix::convert_to_dense< value_t >( A );
    // auto  TD = hlr::seq::matrix::convert_to_dense< value_t >( D );
    // auto  TB = hlr::seq::matrix::convert_to_dense< value_t >( B );
    // auto  TC = hlr::seq::matrix::convert_to_dense< value_t >( C );

    // multiply< value_t, approx_t >( alpha, op_A, *TA, op_D, *TD, op_B, *TB, *TC, acc, approx );
    
    HLR_ASSERT( is_dense( D ) );

    auto  DD = cptrcast( &D, hpro::TDenseMatrix );
    
    if ( is_lowrank( A ) )
    {
        if ( is_lowrank( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_lowrank( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense(   C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_D, *DD,
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TDenseMatrix ),
                                               acc, approx );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    // auto  TT = hlr::seq::matrix::convert_to_dense< value_t >( C );

    // blas::add( value_t(-1), blas::mat< value_t >( *TC ), blas::mat< value_t >( *TT ) );

    // std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat< value_t >( *TT ) ) << std::endl;
}

}// namespace hlr

#endif // __HLR_ARITH_MULTIPLY_HH
