#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/add.hh>
#include <hlr/arith/detail/norm.hh>
#include <hlr/approx/svd.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

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
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C,
           const Hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;

    using namespace hlr::matrix;

    // if (( A.id() == 68 ) && ( B.id() == 111 ) && ( C.id() == 151 ))
    //     std::cout << std::endl;
    
    // test initial data
    // A.check_data();
    // B.check_data();
    // C.check_data();
    
    // // decompress destination as it is modified anyway
    // if ( is_compressible( C ) )
    // {
    //     auto  lock = std::scoped_lock( C.mutex() );
        
    //     dynamic_cast< compressible * >( &C )->decompress();
    // }// if
    
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
                                               op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                               op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
            {
                if ( compress::is_compressible( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, lrmatrix< value_t > ),
                                                   acc, approx );
                else
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                   acc, approx );
            }// if
            else if ( is_dense( C ) )
            {
                if ( compress::is_compressible( C ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                         * ptrcast( &C, dense_matrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( B ) )
        {
            if ( compress::is_compressible( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// else
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                               acc, approx );
            else if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                               acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
        {
            if ( compress::is_compressible( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                         multiply< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                        * ptrcast( &C, lrmatrix< value_t > ),
                                                        acc, approx );
                    else
                         multiply< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// else
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( compress::is_compressible( A ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense(   C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// if
        else
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// else
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TRkMatrix< value_t > ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TRkMatrix< value_t > ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense( B ) )
        {
            if ( is_lowrank( C ) )
                multiply< value_t, approx_t >( alpha,
                                               op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                               op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                               * ptrcast( &C, Hpro::TRkMatrix< value_t > ), acc, approx );
            else if ( is_dense( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( compress::is_compressible( A ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else if ( is_uniform_lowrank( B ) )
            {
                if ( is_lowrank( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                   op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_dense( C ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                         op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// if
        else
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_lowrank( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, lrmatrix< value_t > ),
                                                       acc, approx );
                    else
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                       acc, approx );
                }// if
                else if ( is_dense( C ) )
                {
                    if ( compress::is_compressible( C ) )
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, dense_matrix< value_t > ),
                                             acc );
                    else
                        multiply< value_t >( alpha,
                                             op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                             op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                             * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else if ( is_uniform_lowrank( B ) )
            {
                if ( is_lowrank( C ) )
                    multiply< value_t, approx_t >( alpha,
                                                   op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                   op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                                   * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                   acc, approx );
                else if ( is_dense( C ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                         op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// if
                else
                {
                    if ( is_blocked( C ) )
                        multiply< value_t, approx_t >( alpha,
                                                       op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                       op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                       * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                       acc, approx );
                    else if ( is_lowrank( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, lrmatrix< value_t > ),
                                                           acc, approx );
                        else
                            multiply< value_t, approx_t >( alpha,
                                                           op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                           op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                           * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                           acc, approx );
                    }// if
                    else if ( is_dense( C ) )
                    {
                        if ( compress::is_compressible( C ) )
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, dense_matrix< value_t > ),
                                                 acc );
                        else
                            multiply< value_t >( alpha,
                                                 op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
                }// else
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    // test data in result
    // C.check_data();
    
    #if HLR_MULT_TESTS == 1
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), blas::mat( *TC ), blas::mat( *TT ) );

    if ( blas::norm_F( blas::mat( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif

    // // recompress after update
    // if ( compress::is_compressible( C ) )
    // {
    //     auto  lock = std::scoped_lock( C.mutex() );
        
    //     dynamic_cast< compressible * >( &C )->compress( acc );
    // }// if
}

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C,
           const Hpro::TTruncAcc &           acc )
{
    auto  apx = approx::SVD< value_t >();

    multiply( alpha, op_A, A, op_B, B, C, acc, apx );
}

//
// version for matrix types without (!) approximation
//
template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C )
{
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;
    
    using namespace hlr::matrix;
    
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
                                 op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                 op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else if ( is_lowrank( B ) )
        {
            if ( compress::is_compressible( B ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                     op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                 op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else if ( is_dense( B ) )
        {
            if ( compress::is_compressible( B ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                     op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( compress::is_compressible( A ) )
        {
            if ( is_blocked( B ) )
            {
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                         op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                         op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, lrmatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// if
        else
        {
            if ( is_blocked( B ) )
            {
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                         op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                         op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// else
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else if ( is_dense( B ) )
        {
            multiply< value_t >( alpha,
                                 op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                 op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                 * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( compress::is_compressible( A ) )
        {
            if ( is_blocked( B ) )
            {
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                         op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                         op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, dense_matrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// if
        else
        {
            if ( is_blocked( B ) )
            {
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                     op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                         op_B, * cptrcast( &B, lrmatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_uniform_lowrank( B ) )
            {
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else if ( is_dense( B ) )
            {
                if ( compress::is_compressible( B ) )
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                         op_B, * cptrcast( &B, dense_matrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    multiply< value_t >( alpha,
                                         op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                         op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                         * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B.typestr() );
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if HLR_MULT_TESTS == 1

    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), blas::mat( *TC ), blas::mat( *TT ) );

    if ( blas::norm_F( blas::mat( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
}

//
// matrix multiplication with generation of result matrix
// - only supported for non-structured results
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B )
{
    std::unique_ptr< Hpro::TMatrix< value_t > >  C;

    if ( is_lowrank( A ) )
    {
        //
        // U·V' × B = W·X'
        //

        uint  k = 0;
        auto  V = blas::matrix< value_t >();
        auto  W = blas::matrix< value_t >();
        
        if ( compress::is_compressible( A ) )
        {
            auto  RA = cptrcast( &A, matrix::lrmatrix< value_t > );

            V = std::move( RA->V( op_A ) );
            W = std::move( blas::copy( RA->U( op_A ) ) );
            k = RA->rank();
        }// if
        else
        {
            auto  RA = cptrcast( &A, Hpro::TRkMatrix< value_t > );

            V = std::move( blas::mat_V( RA, op_A ) );
            W = std::move( blas::copy( blas::mat_U( RA, op_A ) ) );
            k = RA->rank();
        }// else
        
        auto  X = blas::matrix< value_t >( B.ncols( op_B ), k );

        multiply( alpha, blas::adjoint( op_B ), B, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if
        
        return std::make_unique< Hpro::TRkMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_lowrank( B ) )
    {
        //
        // A × U·V' = W·X'
        //

        uint  k = 0;
        auto  U = blas::matrix< value_t >();
        auto  X = blas::matrix< value_t >();
        
        if ( compress::is_compressible( B ) )
        {
            auto  RB = cptrcast( &B, matrix::lrmatrix< value_t > );
            
            U = std::move( RB->U( op_B ) );
            X = std::move( blas::copy( RB->V( op_B ) ) );
            k = RB->rank();
        }// if
        else
        {
            auto  RB = cptrcast( &B, Hpro::TRkMatrix< value_t > );
            
            U = std::move( blas::mat_U( RB, op_B ) );
            X = std::move( blas::copy( blas::mat_V( RB, op_B ) ) );
            k = RB->rank();
        }// else
        
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), k );
        
        multiply( alpha, op_A, A, U, W );

        return std::make_unique< Hpro::TRkMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_dense_any( A, B ) )
    {
        C = std::make_unique< Hpro::TDenseMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ) );

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
multiply_apx ( const value_t                     alpha,
               const matop_t                     op_A,
               const Hpro::TMatrix< value_t > &  A,
               const matop_t                     op_B,
               const Hpro::TMatrix< value_t > &  B,
               Hpro::TMatrix< value_t > &        C,
               const Hpro::TTruncAcc &           acc,
               const approx_t &                  approx,
               Hpro::real_type_t< value_t >      tol )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );
        
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
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B,
                Hpro::TMatrix< value_t > &        C,
                const Hpro::TTruncAcc &           acc,
                const approx_t &                  approx )
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
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
    
    blas::add( value_t(-1), blas::mat( *TC ), blas::mat( *TT ) );

    if ( blas::norm_F( blas::mat( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if
    
    #endif
}

template < typename value_t >
void
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B,
                Hpro::TMatrix< value_t > &        C )
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
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
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

    blas::add( value_t(-1), blas::mat( *TC ), blas::mat( *TT ) );

    if ( blas::norm_F( blas::mat( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
}

//
// generate and return destination matrix
// - only supported fr non-structured results
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B )
{
    if ( is_lowrank( A ) )
    {
        // U·V' × D × B = W·X' ⇒ B' × D' × V = X
        auto  RA = cptrcast( &A, Hpro::TRkMatrix< value_t > );
        auto  V  = blas::mat_V( RA, op_A );
        auto  W  = blas::copy( blas::mat_U( RA, op_A ) );
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), RA->rank() );

        multiply_diag( alpha, blas::adjoint( op_B ), B, blas::adjoint( op_D ), D, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< Hpro::TRkMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_lowrank( B ) )
    {
        // A × D × U·V' = W·X' ⇒ A × D × U = W
        auto  RB = cptrcast( &B, Hpro::TRkMatrix< value_t > );
        auto  U  = blas::mat_U( RB, op_B );
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), RB->rank() );
        auto  X  = blas::copy( blas::mat_V( RB, op_B ) );

        multiply_diag( alpha, op_A, A, op_D, D, U, W );

        if ( op_B == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< Hpro::TRkMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( is_dense_any( A, B ) )
    {
        auto  C = std::make_unique< Hpro::TDenseMatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ) );

        multiply_diag( alpha, op_A, A, op_D, D, op_B, B, *C );

        return C;
    }// if
    else
        HLR_ERROR( "unsupported matrix types : " + A.typestr() + " × " + B.typestr() );

    return std::unique_ptr< Hpro::TMatrix< value_t > >();
}

////////////////////////////////////////////////////////////////////////////////
//
// compute Hadamard product A = α A*B 
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply_hadamard ( const value_t                     alpha,
                    Hpro::TMatrix< value_t > &        A,
                    const Hpro::TMatrix< value_t > &  B,
                    const Hpro::TTruncAcc &           acc,
                    const approx_t &                  approx )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = ptrcast( &A,  Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        
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
        auto        DA     = ptrcast( &A,  Hpro::TDenseMatrix< value_t > );
        auto        DB     = cptrcast( &B, Hpro::TDenseMatrix< value_t > );
        auto        blas_A = Hpro::blas_mat< value_t >( DA );
        auto        blas_B = Hpro::blas_mat< value_t >( DB );
        const auto  nrows  = DA->nrows();
        const auto  ncols  = DA->ncols();

        for ( size_t  i = 0; i < nrows*ncols; ++i )
            blas_A.data()[i] *= alpha * blas_B.data()[i];
    }// if
    else if ( is_lowrank_all( A, B ) )
    {
        auto  RA = ptrcast( &A,  Hpro::TRkMatrix< value_t > );
        auto  RB = cptrcast( &B, Hpro::TRkMatrix< value_t > );

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

        auto  U = blas::mat_U( RA );
        auto  V = blas::mat_V( RA );
        auto  W = blas::mat_U( RB );
        auto  X = blas::mat_V( RB );
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
