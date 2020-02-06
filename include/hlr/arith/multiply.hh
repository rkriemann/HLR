#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/algebra/mat_mul_core.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_add.hh>

#include "hlr/arith/approx.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/utils/log.hh"

namespace hlr { namespace arith {

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

using hlr::matrix::tiled_lrmatrix;

//////////////////////////////////////////////////////////////////////
//
// forward declarations
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::TMatrix *    A,
           const hpro::TMatrix *    B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc );

//////////////////////////////////////////////////////////////////////
//
// combinations with structured matrices
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TBlockMatrix *  A,
           const hpro::TBlockMatrix *  B,
           hpro::TBlockMatrix *        C,
           const hpro::TTruncAcc &     acc )
{
    for ( uint  i = 0; i < C->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C->nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C->block( i, j ) ) );
                
            for ( uint  l = 0; l < A->nblock_cols(); ++l )
            {
                if ( ! is_null_any( A->block( i, l ), B->block( l, j ) ) )
                    multiply< value_t >( alpha, A->block( i, l ), B->block( l, j ), C->block( i, j ), acc );
            }// if       
        }// for
    }// for
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TBlockMatrix *  A,
           const hpro::TBlockMatrix *  B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc )
{
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< hpro::TBlockMatrix >( C->row_is(), C->col_is() );

    BC->set_block_struct( A->nblock_rows(), B->nblock_cols() );
    
    for ( uint  i = 0; i < A->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            for ( uint  l = 0; l < A->nblock_cols(); ++l )
            {
                if ( ! is_null_any( A->block( i, l ), B->block( l, j ) ) )
                {
                    auto  A_il = A->block( i, l );
                    auto  B_lj = B->block( l, j );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new hpro::TRkMatrix( A_il->row_is(), B_lj->col_is(), hpro::value_type< value_t >::value ) );
                    
                    multiply< value_t >( alpha, A_il, B_lj, BC->block( i, j ), acc );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    // apply update
    hpro::add( value_t(1), BC.get(), value_t(1), C, acc );
}

#define  HLR_BLOCKED_MAT_MUL( typeA_t, typeB_t, typeC_t )           \
    template < typename value_t >                                   \
    void                                                            \
    multiply ( const value_t            alpha,                      \
               const typeA_t *          A,                          \
               const typeB_t *          B,                          \
               typeC_t *                C,                          \
               const hpro::TTruncAcc &  acc )                       \
    {                                                               \
    auto  T = hpro::multiply< value_t >( alpha,                     \
                                         hpro::apply_normal, A,     \
                                         hpro::apply_normal, B );   \
                                                                    \
    hpro::add( value_t(1), T.get(), value_t(1), C, acc );           \
}

HLR_BLOCKED_MAT_MUL( hpro::TBlockMatrix, hpro::TRkMatrix,    hpro::TBlockMatrix )
HLR_BLOCKED_MAT_MUL( hpro::TBlockMatrix, hpro::TRkMatrix,    hpro::TRkMatrix    )
HLR_BLOCKED_MAT_MUL( hpro::TRkMatrix,    hpro::TBlockMatrix, hpro::TBlockMatrix )
HLR_BLOCKED_MAT_MUL( hpro::TRkMatrix,    hpro::TBlockMatrix, hpro::TRkMatrix    )
HLR_BLOCKED_MAT_MUL( hpro::TRkMatrix,    hpro::TRkMatrix,    hpro::TBlockMatrix )

#undef HLR_BLOCKED_MAT_MUL

template < typename value_t >
void
multiply ( const value_t                alpha,
           const hpro::TBlockMatrix *   A,
           const hpro::TBlockMatrix *   B,
           tiled_lrmatrix< value_t > *  C,
           const hpro::TTruncAcc &      acc )
{
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< hpro::TBlockMatrix >( C->row_is(), C->col_is() );

    BC->set_block_struct( A->nblock_rows(), B->nblock_cols() );
    
    for ( uint  i = 0; i < A->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            for ( uint  l = 0; l < A->nblock_cols(); ++l )
            {
                if ( ! is_null_any( A->block( i, l ), B->block( l, j ) ) )
                {
                    auto  A_il = A->block( i, l );
                    auto  B_lj = B->block( l, j );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new tiled_lrmatrix< value_t >( A_il->row_is(), B_lj->col_is(),
                                                                            C->row_tile_is_map(), C->col_tile_is_map() ) );
                    
                    multiply< value_t >( alpha, A_il, B_lj, BC->block( i, j ), acc );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    // apply update
    hpro::add( value_t(1), BC.get(), value_t(1), C, acc );
}

//////////////////////////////////////////////////////////////////////
//
// low-level combinations of dense and standard low-rank
//
//////////////////////////////////////////////////////////////////////

//
// compute C := C + α A·B
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::TRkMatrix *  A,
           const hpro::TRkMatrix *  B,
           hpro::TRkMatrix *        C,
           const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat_A< value_t >( B ) );
    auto  UT = blas::prod(      alpha, hpro::blas_mat_A< value_t >( A ), T );

    auto [ U, V ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( C ), UT },
                                                    { hpro::blas_mat_B< value_t >( C ), hpro::blas_mat_B< value_t >( B ) },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TRkMatrix *     A,
           const hpro::TDenseMatrix *  B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( alpha, blas::adjoint( hpro::blas_mat< value_t >( B ) ), hpro::blas_mat_B< value_t >( A ) );

    auto [ U, V ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( C ), hpro::blas_mat_A< value_t >( A ) },
                                                    { hpro::blas_mat_B< value_t >( C ), VB },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TRkMatrix *     B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat_A< value_t >( B ) );

    auto [ U, V ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( C ), AU },
                                                    { hpro::blas_mat_B< value_t >( C ), hpro::blas_mat_B< value_t >( B ) },
                                                    acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TDenseMatrix *  B,
           hpro::TRkMatrix *           C,
           const hpro::TTruncAcc &     acc )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( B ) );

    blas::prod( value_t(1), hpro::blas_mat_A< value_t >( C ), blas::adjoint( hpro::blas_mat_B< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = hlr::approx_svd< value_t >( AB, acc );
        
    C->set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::TRkMatrix *  A,
           const hpro::TRkMatrix *  B,
           hpro::TDenseMatrix *     C,
           const hpro::TTruncAcc & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat_A< value_t >( B ) );
    auto  UT = blas::prod( value_t(1), hpro::blas_mat_A< value_t >( A ), T );

    blas::prod( alpha, UT, blas::adjoint( hpro::blas_mat_B< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TRkMatrix *     B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), hpro::blas_mat< value_t >( A ), hpro::blas_mat_A< value_t >( B ) );

    blas::prod( alpha, AU, blas::adjoint( hpro::blas_mat_B< value_t >( B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TRkMatrix *     A,
           const hpro::TDenseMatrix *  B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A ) ), hpro::blas_mat< value_t >( B ) );

    blas::prod( alpha, hpro::blas_mat_A< value_t >( A ), VB, value_t(1), hpro::blas_mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::TDenseMatrix *  A,
           const hpro::TDenseMatrix *  B,
           hpro::TDenseMatrix *        C,
           const hpro::TTruncAcc & )
{
    HLR_LOG( 4, hpro::to_string( "multiply( %d, %d, %d )", A->id(), B->id(), C->id() ) );
    
    // C = C + A B
    blas::prod( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( B ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

//////////////////////////////////////////////////////////////////////
//
// semi-automatic type deduction
//
//////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename matrix1_t,
           typename matrix2_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const matrix2_t *        B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc )
{
    if      ( is_blocked( C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, hpro::TBlockMatrix ), acc );
    else if ( is_lowrank( C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, hpro::TRkMatrix ),    acc );
    else if ( is_dense(   C ) ) multiply< value_t >( alpha, A, B, ptrcast( C, hpro::TDenseMatrix ), acc );
    else
        HLR_ERROR( "unsupported matrix type : " + C->typestr() );
}

template < typename value_t,
           typename matrix1_t >
void
multiply ( const value_t            alpha,
           const matrix1_t *        A,
           const hpro::TMatrix *    B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc )
{
    if      ( is_blocked( B ) ) multiply< value_t, matrix1_t, hpro::TBlockMatrix >( alpha, A, cptrcast( B, hpro::TBlockMatrix ), C, acc );
    else if ( is_lowrank( B ) ) multiply< value_t, matrix1_t, hpro::TRkMatrix >(    alpha, A, cptrcast( B, hpro::TRkMatrix ),    C, acc );
    else if ( is_dense(   B ) ) multiply< value_t, matrix1_t, hpro::TDenseMatrix >( alpha, A, cptrcast( B, hpro::TDenseMatrix ), C, acc );
    else
        HLR_ERROR( "unsupported matrix type : " + B->typestr() );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::TMatrix *    A,
           const hpro::TMatrix *    B,
           hpro::TMatrix *          C,
           const hpro::TTruncAcc &  acc )
{
    if      ( is_blocked( A ) ) multiply< value_t, hpro::TBlockMatrix >( alpha, cptrcast( A, hpro::TBlockMatrix ), B, C, acc );
    else if ( is_lowrank( A ) ) multiply< value_t, hpro::TRkMatrix >(    alpha, cptrcast( A, hpro::TRkMatrix ),    B, C, acc );
    else if ( is_dense(   A ) ) multiply< value_t, hpro::TDenseMatrix >( alpha, cptrcast( A, hpro::TDenseMatrix ), B, C, acc );
    else
        HLR_ERROR( "unsupported matrix type : " + B->typestr() );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::TMatrix &    A,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc )
{
    multiply< value_t >( alpha, & A, & B, & C, acc );
}

}}// namespace hlr::arith

#endif // __HLR_ARITH_MULTIPLY_HH
