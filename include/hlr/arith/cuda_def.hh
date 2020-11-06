#ifndef __HLR_ARITH_CUDA_DEF_HH
#define __HLR_ARITH_CUDA_DEF_HH
//
// Project     : HLR
// Module      : arith/cuda_def
// Description : basic linear algebra functions using cuda (wrapper definition)
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace hlr { namespace blas { namespace cuda {

// wrapper for cuda, cuBlas and cuSolver functions
#define HLR_CUDA_CHECK( func, args )                        \
    {                                                       \
        auto  result = func args ;                          \
        if ( result != cudaSuccess )                        \
            HLR_ERROR( "CUDA error in function " #func + hpro::to_string( " (error = %d)", result ) );   \
    }

#define HLR_CUBLAS_CHECK( func, args )                      \
    {                                                       \
        auto  result = func args ;                          \
        if ( result != CUBLAS_STATUS_SUCCESS )              \
            HLR_ERROR( "cuBLAS error in function " #func + hpro::to_string( " (error = %d)", result ) ); \
    }

#define HLR_CUSOLVER_CHECK( func, args )                        \
    {                                                           \
        auto  result = func args ;                              \
        if ( result != CUSOLVER_STATUS_SUCCESS )                \
            HLR_ERROR( "cuSOLVER error in function " #func + hpro::to_string( " (error = %d)", result ) );   \
    }

//
// mapping of default types to cuBLAS types
//
template < typename T > struct cuda_type                            { using  type_t = T; };
template <>             struct cuda_type< std::complex< float > >   { using  type_t = cuFloatComplex; };
template <>             struct cuda_type< std::complex< double > >  { using  type_t = cuDoubleComplex; };

template <typename T>   struct real_type                            { using  type_t = T; };
template <>             struct real_type< cuFloatComplex >          { using  type_t = float; };
template <>             struct real_type< cuDoubleComplex >         { using  type_t = double; };

template < typename T > struct cuda_type_ptr                        { using  type_t = typename cuda_type< T >::type_t *; };

//
// joined handle for cuBLAS and cuSolverDn
//
struct handle
{
    cudaStream_t        stream;
    cublasHandle_t      blas;
    cusolverDnHandle_t  solver;
};

//////////////////////////////////////////////////////////////////////
//
// allocation and transfer
//
//////////////////////////////////////////////////////////////////////

//
// device memory allocation with CUDA
//
template < typename value_t >
typename cuda_type_ptr< value_t >::type_t
device_alloc ( const size_t  n )
{
    void *  ptr = nullptr;
    
    HLR_CUDA_CHECK( cudaMalloc, ( & ptr, n * sizeof(value_t) ) );

    return typename cuda_type_ptr< value_t >::type_t( ptr );
}

template < typename value_t >
void
device_free ( value_t *  ptr )
{
    HLR_CUDA_CHECK( cudaFree, ( ptr ) )
}

//
// host to device copy
//
template < typename value_t >
void
to_device ( const matrix< value_t > &                  M_host,
            typename cuda_type_ptr< value_t >::type_t  M_dev,
            const int                                  lda_dev )
{
    HLR_CUBLAS_CHECK( cublasSetMatrix,
                      ( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                        M_host.data(), M_host.col_stride(),
                        M_dev, lda_dev ) );
}

template < typename value_t >
void
to_device_async ( handle                                     handle,
                  const matrix< value_t > &                  M_host,
                  typename cuda_type_ptr< value_t >::type_t  M_dev,
                  const int                                  lda_dev )
{
    HLR_CUBLAS_CHECK( cublasSetMatrixAsync,
                      ( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                        M_host.data(), M_host.col_stride(),
                        M_dev, lda_dev,
                        handle.stream ) );
}

template < typename value_t >
void
to_device ( const vector< value_t > &                  v_host,
            typename cuda_type_ptr< value_t >::type_t  v_dev,
            const int                                  inc_dev )
{
    HLR_CUBLAS_CHECK( cublasSetVector,
                      ( v_host.length(), sizeof(value_t),
                        v_host.data(), v_host.stride(),
                        v_dev, inc_dev ) );
}

template < typename value_t >
void
to_device ( const std::vector< value_t > &             v_host,
            typename cuda_type_ptr< value_t >::type_t  v_dev,
            const int                                  inc_dev )
{
    HLR_CUBLAS_CHECK( cublasSetVector,
                      ( v_host.size(), sizeof(value_t),
                        v_host.data(), 1,
                        v_dev, inc_dev ) );
}

//
// device to host copy
//
template < typename value_t >
void
from_device ( typename cuda_type_ptr< value_t >::type_t  M_dev,
              const int                                  lda_dev,
              matrix< value_t > &                        M_host )
{
    HLR_CUBLAS_CHECK( cublasGetMatrix,
                      ( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                        M_dev, lda_dev,
                        M_host.data(), M_host.col_stride() ) );
}

template < typename value_t >
void
from_device_async ( handle                                     handle,
                    typename cuda_type_ptr< value_t >::type_t  M_dev,
                    const int                                  lda_dev,
                    matrix< value_t > &                        M_host )
{
    HLR_CUBLAS_CHECK( cublasGetMatrixAsync,
                      ( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                        M_dev, lda_dev,
                        M_host.data(), M_host.col_stride(),
                        handle.stream ) );
}

template < typename value_t >
void
from_device ( typename cuda_type_ptr< value_t >::type_t  v_dev,
              const int                                  inc_dev,
              vector< value_t > &                        v_host )
{
    HLR_CUBLAS_CHECK( cublasGetVector,
                      ( v_host.length(), sizeof(value_t),
                        v_dev, inc_dev,
                        v_host.data(), v_host.stride() ) );
}

template < typename value_t >
void
from_device_async ( handle                                     handle,
                    typename cuda_type_ptr< value_t >::type_t  v_dev,
                    const int                                  inc_dev,
                    vector< value_t > &                        v_host )
{
    HLR_CUBLAS_CHECK( cublasGetVectorAsync,
                      ( v_host.length(), sizeof(value_t),
                        v_dev, inc_dev,
                        v_host.data(), v_host.stride(),
                        handle.stream ) );
}

template < typename value_t >
value_t
from_device ( typename cuda_type_ptr< value_t >::type_t  dev_data )
{
    value_t  data;

    HLR_CUDA_CHECK( cudaMemcpy, ( & data, dev_data, sizeof(value_t), cudaMemcpyDeviceToHost ) );

    return data;
}

template < typename value_t >
value_t
from_device_async ( handle                                     handle,
                    typename cuda_type_ptr< value_t >::type_t  dev_data )
{
    value_t  data;

    HLR_CUDA_CHECK( cudaMemcpyAsync, ( & data, dev_data, sizeof(value_t), cudaMemcpyDeviceToHost, handle.stream ) );

    return data;
}

//
// wrapper to create cuBlas compatible constants
//
template < typename value_t >
value_t
make_constant ( const typename cuda::real_type< value_t >::type_t  f )
{
    return f;
}

template <>
cuFloatComplex
make_constant< cuFloatComplex > ( const float  f )
{
    return make_cuFloatComplex( f, 0 );
}

template <>
cuDoubleComplex
make_constant< cuDoubleComplex > ( const double  f )
{
    return make_cuDoubleComplex( f, 0 );
}

// clear given memory
template < typename value_t >
void
clear ( const int  n,
        value_t *  x )
{
    HLR_CUDA_CHECK( cudaMemset, ( x, 0, sizeof(value_t) * n ) );
}

//////////////////////////////////////////////////////////////////////
//
// vector routines
//
//////////////////////////////////////////////////////////////////////

#define HLR_CUDA_COPY( type, func )                 \
    inline                                          \
    void                                            \
    copy ( handle        handle,                    \
           const int     n,                         \
           const type *  x,                         \
           const int     inc_x,                     \
           type *        y,                         \
           const int     inc_y )                    \
    {                                               \
        HLR_CUBLAS_CHECK( func, ( handle.blas, n, x, inc_x, y, inc_y ) );   \
    }

HLR_CUDA_COPY( float,           cublasScopy )
HLR_CUDA_COPY( double,          cublasDcopy )
HLR_CUDA_COPY( cuFloatComplex,  cublasCcopy )
HLR_CUDA_COPY( cuDoubleComplex, cublasZcopy )

#undef HLR_CUDA_COPY

#define HLR_CUDA_SCALE( type, func )               \
    inline                                         \
    void                                           \
    scale ( handle      handle,                    \
            const int   n,                         \
            const type  alpha,                     \
            type *      x,                         \
            const int   inc_x )                    \
    {                                              \
        HLR_CUBLAS_CHECK( func, ( handle.blas, n, & alpha, x, inc_x ) ); \
    }

HLR_CUDA_SCALE( float,           cublasSscal )
HLR_CUDA_SCALE( double,          cublasDscal )
HLR_CUDA_SCALE( cuFloatComplex,  cublasCscal )
HLR_CUDA_SCALE( cuDoubleComplex, cublasZscal )

#undef HLR_CUDA_SCALE

#define HLR_CUDA_AXPY( type, func )                 \
    inline                                          \
    void                                            \
    axpy ( handle        handle,                    \
           const int     n,                         \
           const type    alpha,                     \
           const type *  x,                         \
           const int     inc_x,                     \
           type *        y,                         \
           const int     inc_y )                    \
    {                                               \
        HLR_CUBLAS_CHECK( func, ( handle.blas, n, & alpha, x, inc_x, y, inc_y ) ); \
    }

HLR_CUDA_AXPY( float,           cublasSaxpy )
HLR_CUDA_AXPY( double,          cublasDaxpy )
HLR_CUDA_AXPY( cuFloatComplex,  cublasCaxpy )
HLR_CUDA_AXPY( cuDoubleComplex, cublasZaxpy )

#undef HLR_CUDA_COPY

#define HLR_CUDA_DOT( type, func )                   \
    type                                             \
    dot ( handle         handle,                     \
          int            n,                          \
          const type *   x,                          \
          int            incx,                       \
          const type *   y,                          \
          int            incy )                      \
    {                                                \
        type  res;                                   \
                                                     \
        HLR_CUBLAS_CHECK( func, ( handle.blas, n, x, incx, y, incy, & res ) ); \
        return res;                                  \
    }

HLR_CUDA_DOT( float,  cublasSdot )
HLR_CUDA_DOT( double, cublasDdot )

#undef HLR_CUDA_DOT

#define HLR_CUDA_NORM2( type, func )                 \
    typename real_type< type >::type_t               \
    norm_2 ( handle         handle,                  \
             int            n,                       \
             const type *   x,                       \
             int            incx )                   \
    {                                                \
        typename real_type< type >::type_t  res;     \
                                                     \
        HLR_CUBLAS_CHECK( func, ( handle.blas, n, x, incx, & res ) ); \
        return res;                                  \
    }

HLR_CUDA_NORM2( float,           cublasSnrm2 )
HLR_CUDA_NORM2( double,          cublasDnrm2 )
HLR_CUDA_NORM2( cuFloatComplex,  cublasScnrm2 )
HLR_CUDA_NORM2( cuDoubleComplex, cublasDznrm2 )

#undef HLR_CUDA_NORM2

//////////////////////////////////////////////////////////////////////
//
// multiplication routines
//
//////////////////////////////////////////////////////////////////////

//
// multiply k columns of M with diagonal matrix D,
// e.g. compute M ≔ M·D
//
template < typename value1_t,
           typename value2_t >
void
prod_diag ( handle                      handle,
            const int                   nrows,
            value1_t *                  dev_M,
            const vector< value2_t > &  D,
            const int                   k )
{
    for ( idx_t  i = 0; i < k; ++i )
    {
        auto  D_i = make_constant< value1_t >( D(i) );
            
        scale( handle, nrows, D_i, dev_M + i * nrows, 1 );
    }// for
}

//
// general matrix multiplication C ≔ α·op(A)·op(B) + β·C
//
#define HLR_CUDA_GEMM( type, func )                                     \
    inline                                                              \
    void                                                                \
    prod ( handle                   handle,                             \
           const cublasOperation_t  trans_A,                            \
           const cublasOperation_t  trans_B,                            \
           const int                nrows_C,                            \
           const int                ncols_C,                            \
           const int                nrows_A,                            \
           const type               alpha,                              \
           const type *             A,                                  \
           const int                ld_A,                               \
           const type *             B,                                  \
           const int                ld_B,                               \
           const type               beta,                               \
           type *                   C,                                  \
           const int                ld_C )                              \
    {                                                                   \
        HLR_CUBLAS_CHECK( func, ( handle.blas, trans_A, trans_B, nrows_C, ncols_C, nrows_A, \
                                  & alpha, A, ld_A, B, ld_B, & beta, C, ld_C ) ); \
    }

HLR_CUDA_GEMM( float,           cublasSgemm )
HLR_CUDA_GEMM( double,          cublasDgemm )
HLR_CUDA_GEMM( cuFloatComplex,  cublasCgemm )
HLR_CUDA_GEMM( cuDoubleComplex, cublasZgemm )

#undef HLR_CUDA_GEMM

#define HLR_CUDA_GEMM_BATCHED( type, func )                             \
    inline                                                              \
    void                                                                \
    prod_batched ( handle                   handle,                     \
                   const cublasOperation_t  trans_A,                    \
                   const cublasOperation_t  trans_B,                    \
                   const int                nrows_C,                    \
                   const int                ncols_C,                    \
                   const int                nrows_A,                    \
                   const type               alpha,                      \
                   type **                  A,                          \
                   const int                ld_A,                       \
                   type **                  B,                          \
                   const int                ld_B,                       \
                   const type               beta,                       \
                   type **                  C,                          \
                   const int                ld_C,                       \
                   const int                nbatch )                    \
    {                                                                   \
        HLR_CUBLAS_CHECK( func, ( handle.blas, trans_A, trans_B, nrows_C, ncols_C, nrows_A, \
                                  & alpha, A, ld_A, B, ld_B, & beta, C, ld_C, nbatch ) ); \
    }

HLR_CUDA_GEMM_BATCHED( float,           cublasSgemmBatched )
HLR_CUDA_GEMM_BATCHED( double,          cublasDgemmBatched )
HLR_CUDA_GEMM_BATCHED( cuFloatComplex,  cublasCgemmBatched )
HLR_CUDA_GEMM_BATCHED( cuDoubleComplex, cublasZgemmBatched )

#undef HLR_CUDA_GEMM_BATCHED

//////////////////////////////////////////////////////////////////////
//
// QR related functions
//
//////////////////////////////////////////////////////////////////////

//
// return work buffer size for geqrf/orgqr
//
#define HLR_CUDA_GEQRF_BUFFERSIZE( type, func )                         \
    inline                                                              \
    int                                                                 \
    geqrf_buffersize ( cusolverDnHandle_t  handle,                      \
                       int                 nrows,                       \
                       int                 ncols,                       \
                       type *              A,                           \
                       int                 ldA )                        \
    {                                                                   \
        int  lwork = 0;                                                 \
                                                                        \
        HLR_CUSOLVER_CHECK( func, ( handle, nrows, ncols, A, ldA, & lwork ) ); \
                                                                        \
        return  lwork;                                                  \
    }

HLR_CUDA_GEQRF_BUFFERSIZE( float,           cusolverDnSgeqrf_bufferSize )
HLR_CUDA_GEQRF_BUFFERSIZE( double,          cusolverDnDgeqrf_bufferSize )
HLR_CUDA_GEQRF_BUFFERSIZE( cuFloatComplex,  cusolverDnCgeqrf_bufferSize )
HLR_CUDA_GEQRF_BUFFERSIZE( cuDoubleComplex, cusolverDnZgeqrf_bufferSize )

#undef HLR_CUDA_GEQRF_BUFFERSIZE

#define HLR_CUDA_GEQRF( type, func )            \
    inline                                      \
    void                                        \
    geqrf ( cusolverDnHandle_t  handle,         \
            int                 nrows,          \
            int                 ncols,          \
            type *              A,              \
            int                 ldA,            \
            type *              tau,            \
            type *              work,           \
            int                 lwork,          \
            int *               info )          \
    {                                                                   \
        HLR_CUSOLVER_CHECK( func, ( handle, nrows, ncols, A, ldA, tau, work, lwork, info ) ); \
    }

HLR_CUDA_GEQRF( float,           cusolverDnSgeqrf )
HLR_CUDA_GEQRF( double,          cusolverDnDgeqrf )
HLR_CUDA_GEQRF( cuFloatComplex,  cusolverDnCgeqrf )
HLR_CUDA_GEQRF( cuDoubleComplex, cusolverDnZgeqrf )

#undef HLR_CUDA_GEQRF

#define HLR_CUDA_ORGQR_BUFFERSIZE( type, func )     \
    inline                                          \
    int                                             \
    orgqr_buffersize ( cusolverDnHandle_t  handle,  \
                       int                 nrows,   \
                       int                 ncols,   \
                       int                 k,       \
                       const type *        A,       \
                       int                 ldA,     \
                       const type *        tau )    \
    {                                               \
        int  lwork = 0;                                                 \
                                                                        \
        HLR_CUSOLVER_CHECK( func, ( handle, nrows, ncols, k, A, ldA, tau, & lwork ) ); \
                                                                        \
        return  lwork;                                                  \
    }

HLR_CUDA_ORGQR_BUFFERSIZE( float,           cusolverDnSorgqr_bufferSize )
HLR_CUDA_ORGQR_BUFFERSIZE( double,          cusolverDnDorgqr_bufferSize )
HLR_CUDA_ORGQR_BUFFERSIZE( cuFloatComplex,  cusolverDnCungqr_bufferSize )
HLR_CUDA_ORGQR_BUFFERSIZE( cuDoubleComplex, cusolverDnZungqr_bufferSize )

#undef HLR_CUDA_ORGQR_BUFFERSIZE

#define HLR_CUDA_ORGQR( type, func )            \
    inline                                      \
    void                                        \
    orgqr ( cusolverDnHandle_t  handle,         \
            int                 nrows,          \
            int                 ncols,          \
            int                 k,              \
            type *              A,              \
            int                 ldA,            \
            type *              tau,            \
            type *              work,           \
            int                 lwork,          \
            int *               info )          \
    {                                                                   \
        HLR_CUSOLVER_CHECK( func, ( handle, nrows, ncols, k, A, ldA, tau, work, lwork, info ) ); \
    }

HLR_CUDA_ORGQR( float,           cusolverDnSorgqr )
HLR_CUDA_ORGQR( double,          cusolverDnDorgqr )
HLR_CUDA_ORGQR( cuFloatComplex,  cusolverDnCungqr )
HLR_CUDA_ORGQR( cuDoubleComplex, cusolverDnZungqr )

#undef HLR_CUDA_ORGQR

//////////////////////////////////////////////////////////////////////
//
// eigenvalue related functions
//
//////////////////////////////////////////////////////////////////////

//
// gesvdj
//
#define HLR_CUDA_GESVDJ_BUFFERSIZE( type, func )     \
    inline                                           \
    int                                              \
    gesvdj_buffersize ( cusolverDnHandle_t  handle,  \
                        cusolverEigMode_t   jobz,    \
                        int                 econ,    \
                        int                 m,       \
                        int                 n,       \
                        const type *        A,       \
                        int                 lda,     \
                        const typename cuda::real_type< type >::type_t *  S, \
                        const type *        U,       \
                        int                 ldu,     \
                        const type *        V,       \
                        int                 ldv,     \
                        gesvdjInfo_t        params ) \
    {                                                \
        int  lwork = 0;                              \
                                                     \
        HLR_CUSOLVER_CHECK( func, ( handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, & lwork, params ) ); \
                                                     \
        return lwork;                                \
    }

HLR_CUDA_GESVDJ_BUFFERSIZE( float,           cusolverDnSgesvdj_bufferSize )
HLR_CUDA_GESVDJ_BUFFERSIZE( double,          cusolverDnDgesvdj_bufferSize )
HLR_CUDA_GESVDJ_BUFFERSIZE( cuFloatComplex,  cusolverDnCgesvdj_bufferSize )
HLR_CUDA_GESVDJ_BUFFERSIZE( cuDoubleComplex, cusolverDnZgesvdj_bufferSize )

#undef HLR_CUDA_GESVDJ_BUFFERSIZE

#define HLR_CUDA_GESVDJ( type, func )                                   \
    inline                                                              \
    void                                                                \
    gesvdj ( cusolverDnHandle_t  handle,                                \
             cusolverEigMode_t   jobz,                                  \
             int                 econ,                                  \
             int                 m,                                     \
             int                 n,                                     \
             type *              A,                                     \
             int                 lda,                                   \
             typename cuda::real_type< type >::type_t *  S,             \
             type *              U,                                     \
             int                 ldu,                                   \
             type *              V,                                     \
             int                 ldv,                                   \
             type *              work,                                  \
             int                 lwork,                                 \
             int *               info,                                  \
             gesvdjInfo_t        params )                               \
    {                                                                   \
        HLR_CUSOLVER_CHECK( func, ( handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params ) ); \
    }

HLR_CUDA_GESVDJ( float,           cusolverDnSgesvdj )
HLR_CUDA_GESVDJ( double,          cusolverDnDgesvdj )
HLR_CUDA_GESVDJ( cuFloatComplex,  cusolverDnCgesvdj )
HLR_CUDA_GESVDJ( cuDoubleComplex, cusolverDnZgesvdj )

#undef HLR_CUDA_GESVDJ

//
// syevj
//
#define HLR_CUDA_SYEVJ_BUFFERSIZE( type, func )                         \
    int                                                                 \
    syevj_buffersize ( handle              handle,                      \
                       cusolverEigMode_t   jobz,                        \
                       cublasFillMode_t    uplo,                        \
                       int                 n,                           \
                       const type *        A,                           \
                       int                 ldA,                         \
                       const type *        W,                           \
                       syevjInfo_t         params )                     \
    {                                                                   \
        int  lwork = 0;                                                 \
                                                                        \
        HLR_CUSOLVER_CHECK( func, ( handle.solver, jobz, uplo, n, A, ldA, W, & lwork, params ) ); \
                                                                        \
        return  lwork;                                                  \
    }

HLR_CUDA_SYEVJ_BUFFERSIZE( float,  cusolverDnSsyevj_bufferSize )
HLR_CUDA_SYEVJ_BUFFERSIZE( double, cusolverDnDsyevj_bufferSize )

#undef HLR_CUDA_SYEVJ_BUFFERSIZE

#define HLR_CUDA_SYEVJ( type, func )                                    \
    void                                                                \
    syevj ( handle              handle,                                 \
            cusolverEigMode_t   jobz,                                   \
            cublasFillMode_t    uplo,                                   \
            int                 n,                                      \
            type *              A,                                      \
            int                 ldA,                                    \
            type *              W,                                      \
            type *              work,                                   \
            int                 lwork,                                  \
            int *               info,                                   \
            syevjInfo_t         params )                                \
    {                                                                   \
        HLR_CUSOLVER_CHECK( func, ( handle.solver, jobz, uplo, n, A, ldA, W, work, lwork, info, params ) ); \
}

HLR_CUDA_SYEVJ( float,  cusolverDnSsyevj )
HLR_CUDA_SYEVJ( double, cusolverDnDsyevj )

#undef HLR_CUDA_SYEVJ

//
// syevd
//
#define HLR_CUDA_SYEVD_BUFFERSIZE( type, func )                         \
    int                                                                 \
    syevd_buffersize ( handle              handle,                      \
                       cusolverEigMode_t   jobz,                        \
                       cublasFillMode_t    uplo,                        \
                       int                 n,                           \
                       const type *        A,                           \
                       int                 ldA,                         \
                       typename real_type< type >::type_t *  W )        \
    {                                                                   \
        int  lwork = 0;                                                 \
                                                                        \
        HLR_CUSOLVER_CHECK( func, ( handle.solver, jobz, uplo, n, A, ldA, W, & lwork ) ); \
                                                                        \
        return  lwork;                                                  \
    }

HLR_CUDA_SYEVD_BUFFERSIZE( float,           cusolverDnSsyevd_bufferSize )
HLR_CUDA_SYEVD_BUFFERSIZE( double,          cusolverDnDsyevd_bufferSize )
HLR_CUDA_SYEVD_BUFFERSIZE( cuFloatComplex,  cusolverDnCheevd_bufferSize )
HLR_CUDA_SYEVD_BUFFERSIZE( cuDoubleComplex, cusolverDnZheevd_bufferSize )

#undef HLR_CUDA_SYEVD_BUFFERSIZE

#define HLR_CUDA_SYEVD( type, func )                                    \
    void                                                                \
    syevd ( handle              handle,                                 \
            cusolverEigMode_t   jobz,                                   \
            cublasFillMode_t    uplo,                                   \
            int                 n,                                      \
            type *              A,                                      \
            int                 ldA,                                    \
            typename real_type< type >::type_t *  W,                    \
            type *              work,                                   \
            int                 lwork,                                  \
            int *               info )                                  \
    {                                                                   \
        HLR_CUSOLVER_CHECK( func, ( handle.solver, jobz, uplo, n, A, ldA, W, work, lwork, info ) ); \
}

HLR_CUDA_SYEVD( float,           cusolverDnSsyevd )
HLR_CUDA_SYEVD( double,          cusolverDnDsyevd )
HLR_CUDA_SYEVD( cuFloatComplex,  cusolverDnCheevd )
HLR_CUDA_SYEVD( cuDoubleComplex, cusolverDnZheevd )

#undef HLR_CUDA_SYEVD

}}}// namespace hlr::blas::cuda

#endif // __HLR_ARITH_CUDA_DEF_HH
