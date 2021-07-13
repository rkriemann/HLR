#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/add.hh>
#include <hlr/arith/norm.hh>
#include <hlr/approx/svd.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>

// #include <hlr/matrix/convert.hh> // DEBUG
#include <hlr/seq/matrix.hh> // DEBUG

#include <hlr/arith/detail/multiply.hh>
#include <hlr/arith/detail/multiply_diag.hh>

namespace hlr
{

// to enable accuracy tests
#define HLR_MULT_TESTS  0

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·B + C
//
/////////////////////////////////////////////////////////////////////////////////

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
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;
    
    using hlr::matrix::is_uniform_lowrank;
    using hlr::matrix::uniform_lrmatrix;

    #if HLR_MULT_TESTS == 1

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    multiply( alpha, op_A, *TA, op_B, *TB, *TC );

    #endif
    
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, hpro::TBlockMatrix ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                     op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, hpro::TRkMatrix ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                               * ptrcast( &C, hpro::TRkMatrix ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                               op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, hpro::TRkMatrix ),
                                               acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
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
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if HLR_MULT_TESTS == 1
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), blas::mat< value_t >( *TC ), blas::mat< value_t >( *TT ) );

    if ( blas::norm_F( blas::mat< value_t >( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat< value_t >( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
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

//
// version for matrix types without (!) approximation
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C )
{
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;
    
    using hlr::matrix::is_uniform_lowrank;
    using hlr::matrix::uniform_lrmatrix;
    
    if ( ! is_dense( C ) )
        HLR_ERROR( "unsupported matrix type : " + C.typestr() );

    #if HLR_MULT_TESTS == 1

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    multiply( alpha, op_A, *TA, op_B, *TB, *TC );

    #endif
    
    if ( is_blocked( A ) )
    {
        if ( is_blocked( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                 op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                 op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                 op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_dense( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                 op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                 op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                 op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_dense( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                 op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_dense( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                 op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                 op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                 op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else if ( is_dense( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                 op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                 * ptrcast( &C, hpro::TDenseMatrix ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if HLR_MULT_TESTS == 1

    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), blas::mat< value_t >( *TC ), blas::mat< value_t >( *TT ) );

    if ( blas::norm_F( blas::mat< value_t >( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat< value_t >( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
}

//
// matrix multiplication with generation of result matrix
// - only supported for non-structured results
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B )
{
    std::unique_ptr< hpro::TMatrix >  C;
    
    if ( is_lowrank( A ) )
    {
        // U·V' × B = W·X'
        auto  RA = cptrcast( &A, hpro::TRkMatrix );
        auto  V  = blas::mat_V< value_t >( RA, op_A );
        auto  W  = blas::copy( blas::mat_U< value_t >( RA, op_A ) );
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), RA->rank() );

        multiply( alpha, blas::adjoint( op_B ), B, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< hpro::TRkMatrix >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_lowrank( B ) )
    {
        // A × U·V' = W·X'
        auto  RB = cptrcast( &B, hpro::TRkMatrix );
        auto  U  = blas::mat_U< value_t >( RB, op_B );
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), RB->rank() );
        auto  X  = blas::copy( blas::mat_V< value_t >( RB, op_B ) );

        multiply( alpha, op_A, A, U, W );

        return std::make_unique< hpro::TRkMatrix >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_dense_any( A, B ) )
    {
        C = std::make_unique< hpro::TDenseMatrix >( A.row_is( op_A ), B.col_is( op_B ), hpro::value_type_v< value_t > );

        multiply( alpha, op_A, A, op_B, B, *C );
    }// if
    else
        HLR_ERROR( "unsupported matrix types : " + A.typestr() + " × " + B.typestr() );

    return C;
}

//
// compute C = C + α op( A ) op( B ) with additional approximation
// by omitting sub products based on Frobenius norm of factors
//
template < typename value_t,
           typename approx_t >
void
multiply_apx ( const value_t            alpha,
               const matop_t            op_A,
               const hpro::TMatrix &    A,
               const matop_t            op_B,
               const hpro::TMatrix &    B,
               hpro::TMatrix &          C,
               const hpro::TTruncAcc &  acc,
               const approx_t &         approx,
               typename hpro::real_type< value_t >::type_t  tol )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_rows( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_apx< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx, tol );
                }// for
            }// for
        }// for
    }// if
    else
    {
        if ( is_lowrank( C ) )
        {
            //
            // look for Frobenius norm of factors and return if too small
            //

            const auto  norm_A = norm::frobenius( A );
            const auto  norm_B = norm::frobenius( B );

            if ( norm_A * norm_B < tol )
                return;
        }// if
        
        multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx );
    }// else
}

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·D·B + C
//
// with (block-) diagonal D
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t            alpha,
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
    #if HLR_MULT_TESTS == 1

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  DD = hlr::seq::matrix::copy_diag( D );
    auto  TD = hlr::matrix::convert_to_dense< value_t >( *DD );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    io::matlab::write( *TC, "C1" );
    
    multiply_diag( alpha, op_A, *TA, op_D, *TD, op_B, *TB, *TC );

    // io::matlab::write( *TA, "A" );
    // io::matlab::write( *TD, "D" );
    // io::matlab::write( *TB, "B" );
    // io::matlab::write( *TC, "C" );
    
    #endif

    if ( is_blocked( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TBlockMatrix ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                                        op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                                        op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                                        * ptrcast( &C, hpro::TRkMatrix ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                              op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                              op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                              * ptrcast( &C, hpro::TDenseMatrix ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for A : " + A.typestr() );

    
    #if HLR_MULT_TESTS == 1
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    // io::matlab::write( *TT, "T" );
    
    blas::add( value_t(-1), blas::mat< value_t >( *TC ), blas::mat< value_t >( *TT ) );

    if ( blas::norm_F( blas::mat< value_t >( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat< value_t >( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if
    
    #endif
}

template < typename value_t >
void
multiply_diag ( const value_t            alpha,
                const hpro::matop_t      op_A,
                const hpro::TMatrix &    A,
                const hpro::matop_t      op_D,
                const hpro::TMatrix &    D,
                const hpro::matop_t      op_B,
                const hpro::TMatrix &    B,
                hpro::TMatrix &          C )
{
    #if HLR_MULT_TESTS == 1

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  DD = hlr::seq::matrix::copy_diag( D );
    auto  TD = hlr::matrix::convert_to_dense< value_t >( *DD );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    multiply_diag( alpha, op_A, *TA, op_D, *TD, op_B, *TB, *TC );

    #endif
    
    HLR_ASSERT( is_dense( D ) || is_blocked( D ) );
    HLR_ASSERT( is_dense( C ) );

    if ( is_blocked( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TRkMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TRkMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_D, * cptrcast( &D, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for A : " + A.typestr() );

    
    #if HLR_MULT_TESTS == 1
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), blas::mat< value_t >( *TC ), blas::mat< value_t >( *TT ) );

    if ( blas::norm_F( blas::mat< value_t >( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat< value_t >( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
}

//
// generate and return destination matrix
// - only supported fr non-structured results
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
multiply_diag ( const value_t            alpha,
                const hpro::matop_t      op_A,
                const hpro::TMatrix &    A,
                const hpro::matop_t      op_D,
                const hpro::TMatrix &    D,
                const hpro::matop_t      op_B,
                const hpro::TMatrix &    B )
{
    if ( is_lowrank( A ) )
    {
        // U·V' × D × B = W·X' ⇒ B' × D' × V = X
        auto  RA = cptrcast( &A, hpro::TRkMatrix );
        auto  V  = blas::mat_V< value_t >( RA, op_A );
        auto  W  = blas::copy( blas::mat_U< value_t >( RA, op_A ) );
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), RA->rank() );

        multiply_diag( alpha, blas::adjoint( op_B ), B, blas::adjoint( op_D ), D, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< hpro::TRkMatrix >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_lowrank( B ) )
    {
        // A × D × U·V' = W·X' ⇒ A × D × U = W
        auto  RB = cptrcast( &B, hpro::TRkMatrix );
        auto  U  = blas::mat_U< value_t >( RB, op_B );
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), RB->rank() );
        auto  X  = blas::copy( blas::mat_V< value_t >( RB, op_B ) );

        multiply_diag( alpha, op_A, A, op_D, D, U, W );

        if ( op_B == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< hpro::TRkMatrix >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_dense_any( A, B ) )
    {
        auto  C = std::make_unique< hpro::TDenseMatrix >( A.row_is( op_A ), B.col_is( op_B ), hpro::value_type_v< value_t > );

        multiply_diag( alpha, op_A, A, op_D, D, op_B, B, *C );

        return C;
    }// if
    else
        HLR_ERROR( "unsupported matrix types : " + A.typestr() + " × " + B.typestr() );

    return std::unique_ptr< hpro::TMatrix >();
}

////////////////////////////////////////////////////////////////////////////////
//
// compute Hadamard product A = α A*B 
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply_hadamard ( const value_t            alpha,
                    hpro::TMatrix &          A,
                    const hpro::TMatrix &    B,
                    const hpro::TTruncAcc &  acc,
                    const approx_t &         approx )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = ptrcast( &A,  hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                auto  A_ij = BA->block( i, j );
                auto  B_ij = BB->block( i, j );
                
                HLR_ASSERT( ! is_null_any( A_ij, B_ij ) );
            
                multiply_hadamard< value_t >( alpha, *A_ij, *B_ij, acc, approx );
            }// for
        }// for
    }// if
    else if ( is_dense_all( A, B ) )
    {
        auto        DA     = ptrcast( &A,  hpro::TDenseMatrix );
        auto        DB     = cptrcast( &B, hpro::TDenseMatrix );
        auto        blas_A = hpro::blas_mat< value_t >( DA );
        auto        blas_B = hpro::blas_mat< value_t >( DB );
        const auto  nrows  = DA->nrows();
        const auto  ncols  = DA->ncols();

        for ( size_t  i = 0; i < nrows*ncols; ++i )
            blas_A.data()[i] *= alpha * blas_B.data()[i];
    }// if
    else if ( is_lowrank_all( A, B ) )
    {
        auto  RA = ptrcast( &A,  hpro::TRkMatrix );
        auto  RB = cptrcast( &B, hpro::TRkMatrix );

        //
        // construct product with rank rank(A)·rank(B) and fill
        // new low-rank vectors based on hadamard product
        //
        //  a_ij · b_ij = ( Σ_l u^l_i · v^l_j ) ( Σ_k w^k_i · x^k_j )
        //              = Σ_l Σ_k ( u^l_i · w^k_i ) ( v^l_j · x^k_j )
        //
        //  i.e., C = Y·Z' with y^p_i = u^l_i · w^k_i and
        //                      z^p_j = v^l_j · x^k_j
        //
        //  with p = l·rank(B)+k
        //

        auto  rank_A = RA->rank();
        auto  rank_B = RB->rank();
        auto  rank   = rank_A * rank_B;

        auto  nrows = RA->nrows();
        auto  ncols = RA->ncols();

        auto  U = blas::mat_U< value_t >( RA );
        auto  V = blas::mat_V< value_t >( RA );
        auto  W = blas::mat_U< value_t >( RB );
        auto  X = blas::mat_V< value_t >( RB );
        auto  Y = blas::matrix< value_t >( nrows, rank );
        auto  Z = blas::matrix< value_t >( ncols, rank );

        uint  p = 0;
        
        for ( uint  l = 0; l < rank_A; ++l )
        {
            auto  u_l = U.column( l );
            auto  v_l = V.column( l );
                
            for ( uint  k = 0; k < rank_B; ++k, ++p )
            {
                auto  w_k = W.column( k );
                auto  x_k = X.column( k );
                auto  y_p = Y.column( p );
                auto  z_p = Z.column( p );

                for ( size_t  i = 0; i < nrows; ++i )
                    y_p(i) = alpha * u_l(i) * w_k(i);

                for ( size_t  j = 0; j < ncols; ++j )
                    z_p(j) = v_l(j) * x_k(j);
            }// for
        }// for

        //
        // truncate Y·Z and copy back to A
        //

        auto [ Y_acc, Z_acc ] = approx( Y, Z, acc );
        
        RA->set_lrmat( std::move( Y_acc ), std::move( Z_acc ) );
    }// if
}

}// namespace hlr

#endif // __HLR_ARITH_MULTIPLY_HH
