#ifndef __HLR_TF_ARITH_CUDA_HH
#define __HLR_TF_ARITH_CUDA_HH
//
// Project     : HLR
// Module      : tf/arith_cuda
// Description : arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <taskflow/cublasflow.hpp>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/blas.hh"

#include "hlr/tf/arith_tiled.hh"

namespace hlr { namespace tf { namespace cuda {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

namespace detail
{

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( ::tf::Subflow &                  tf,
          const value_t                    alpha,
          const matop_t                    op_M,
          const TMatrix &                  M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    // assert( ! is_null( M ) );
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, TBlockMatrix );
        const auto  row_ofs = B->row_is( op_M ).first();
        const auto  col_ofs = B->col_is( op_M ).first();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                    auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );

                    mul_vec( tf, alpha, op_M, *B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = cptrcast( &M, TDenseMatrix );

        value_t    beta      = value_t(1);
        value_t *  dev_D     = nullptr;
        value_t *  dev_x     = nullptr;
        value_t *  dev_y     = nullptr;
        value_t *  dev_alpha = nullptr;
        value_t *  dev_beta  = nullptr;

        auto  alloc_alpha = tf.emplace( [&] () { dev_alpha = ::tf::cuda_malloc_device< value_t >( 1 ); } );
        auto  alloc_beta  = tf.emplace( [&] () { dev_beta  = ::tf::cuda_malloc_device< value_t >( 1 ); } );
        auto  alloc_D     = tf.emplace( [&] () { dev_D     = ::tf::cuda_malloc_device< value_t >( D->nrows() * D->ncols() ); } );
        auto  alloc_x     = tf.emplace( [&] () { dev_x     = ::tf::cuda_malloc_device< value_t >( x.length() ); } );
        auto  alloc_y     = tf.emplace( [&] () { dev_y     = ::tf::cuda_malloc_device< value_t >( y.length() ); } );

        auto  compute     = tf.emplace(
            [&]( ::tf::cudaFlowCapturer & capturer )
            {
                auto  blas = capturer.make_capturer< ::tf::cublasFlowCapturer >();

                ::tf::cudaTask  copy_alpha = capturer.copy( dev_alpha, &alpha, 1 );
                ::tf::cudaTask  copy_beta  = capturer.copy( dev_beta,  &beta,  1 );
                ::tf::cudaTask  copy_D     = capturer.copy( dev_D, blas::mat< value_t >( D ).data(), D->nrows() * D->ncols() );
                ::tf::cudaTask  copy_x     = capturer.copy( dev_x, x.data(), x.length() );
                ::tf::cudaTask  copy_y     = capturer.copy( dev_y, y.data(), y.length() );

                // TODO: op -> CUBLAS_OP
                ::tf::cudaTask  gemv       = blas->gemv( CUBLAS_OP_N, D->nrows(), D->ncols(), dev_alpha, dev_D, D->nrows(), dev_x, 1, dev_beta, dev_y, 1 );
                
                ::tf::cudaTask  copy_res   = capturer.copy( y.data(), dev_y, y.length() );
                
                gemv.succeed( copy_alpha, copy_beta, copy_D, copy_x, copy_y ).precede( copy_res );
            } );

        auto  free_mem = tf.emplace(
            [&] ()
            {
                ::tf::cuda_free( dev_y ); 
                ::tf::cuda_free( dev_x ); 
                ::tf::cuda_free( dev_D ); 
                ::tf::cuda_free( dev_beta ); 
                ::tf::cuda_free( dev_alpha );
            } );

        compute.succeed( alloc_alpha, alloc_beta, alloc_D, alloc_x, alloc_y ).precede( free_mem );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, TRkMatrix );

        if ( op_M == apply_normal )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( blas_mat_B< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_A< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_transposed )
        {
            assert( is_complex_type< value_t >::value == false );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_adjoint )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
    }// if
    else
        assert( false );
}

}// namespace detail

template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const matop_t                    op_M,
          const TMatrix &                  M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    ::tf::Taskflow  tf;
    
    tf.emplace( [=,&M,&x,&y] ( ::tf::Subflow &  sf ) { detail::mul_vec( sf, alpha, op_M, M, x, y ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}
    
}}}// namespace hlr::tf

#endif // __HLR_TF_ARITH_HH
