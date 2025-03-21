#ifndef __HLR_ARITH_CUDA_HH
#define __HLR_ARITH_CUDA_HH
//
// Project     : HLR
// Module      : arith/cuda
// Description : basic linear algebra functions using cuda
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace blas { namespace cuda {

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

// wrapper for cuda, cuBlas and cuSolver functions
#define HLR_CUDA_CHECK( func, args )         \
    {                                        \
        auto  result = func args ;           \
        HLR_ASSERT( result == cudaSuccess ); \
    }

#define HLR_CUBLAS_CHECK( func, args )                  \
    {                                                   \
        auto  result = func args ;                      \
        HLR_ASSERT( result == CUBLAS_STATUS_SUCCESS );  \
    }

#define HLR_CUSOLVER_CHECK( func, args )                 \
    {                                                    \
        auto  result = func args ;                       \
        if ( result != CUSOLVER_STATUS_SUCCESS )         \
            HLR_ERROR( "cusolver result = " + Hpro::to_string( int(result) ) ); \
    }

//
// joined handle for cuBLAS and cuSolverDn
//
struct handle
{
    cudaStream_t        stream;
    cublasHandle_t      blas;
    cusolverDnHandle_t  solver;
};

// default handle
handle  default_handle;

//
// initialize cuBLAS/cuSolver
//
inline
void
init ()
{
    HLR_CUDA_CHECK( cudaStreamCreate, ( & default_handle.stream ) );
    
    HLR_CUBLAS_CHECK( cublasCreate,    ( & default_handle.blas ) );
    HLR_CUBLAS_CHECK( cublasSetStream, (   default_handle.blas, default_handle.stream ) );

    HLR_CUSOLVER_CHECK( cusolverDnCreate,    ( & default_handle.solver ) );
    HLR_CUSOLVER_CHECK( cusolverDnSetStream, (   default_handle.solver, default_handle.stream ) );
}

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
            int                                        lda_dev )
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
                  int                                        lda_dev )
{
    HLR_CUBLAS_CHECK( cublasSetMatrixAsync,
                      ( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                        M_host.data(), M_host.col_stride(),
                        M_dev, lda_dev,
                        handle.stream ) );
}

//
// device to host copy
//
template < typename value_t >
void
from_device ( typename cuda_type_ptr< value_t >::type_t  M_dev,
              int                                        lda_dev,
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
                    int                                        lda_dev,
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
              int                                        inc_dev,
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
                    int                                        inc_dev,
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
        func( handle.blas, n, x, inc_x, y, inc_y ); \
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
        func( handle.blas, n, & alpha, x, inc_x ); \
    }

HLR_CUDA_SCALE( float,           cublasSscal )
HLR_CUDA_SCALE( double,          cublasDscal )
HLR_CUDA_SCALE( cuFloatComplex,  cublasCscal )
HLR_CUDA_SCALE( cuDoubleComplex, cublasZscal )

#undef HLR_CUDA_SCALE

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
        func( handle.blas, trans_A, trans_B, nrows_C, ncols_C, nrows_A, \
              & alpha, A, ld_A, B, ld_B, & beta, C, ld_C );             \
    }

HLR_CUDA_GEMM( float,           cublasSgemm )
HLR_CUDA_GEMM( double,          cublasDgemm )
HLR_CUDA_GEMM( cuFloatComplex,  cublasCgemm )
HLR_CUDA_GEMM( cuDoubleComplex, cublasZgemm )

#undef HLR_CUDA_GEMM

//////////////////////////////////////////////////////////////////////
//
// QR related functions
//
//////////////////////////////////////////////////////////////////////

//
// return work buffer size for geqrf/orgqr
//
#define GEQRF_BUFFERSIZE( type, func )                                  \
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

GEQRF_BUFFERSIZE( float,           cusolverDnSgeqrf_bufferSize )
GEQRF_BUFFERSIZE( double,          cusolverDnDgeqrf_bufferSize )
GEQRF_BUFFERSIZE( cuFloatComplex,  cusolverDnCgeqrf_bufferSize )
GEQRF_BUFFERSIZE( cuDoubleComplex, cusolverDnZgeqrf_bufferSize )

#undef GEQRF_BUFFERSIZE

#define GEQRF( type, func )                     \
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

GEQRF( float,           cusolverDnSgeqrf )
GEQRF( double,          cusolverDnDgeqrf )
GEQRF( cuFloatComplex,  cusolverDnCgeqrf )
GEQRF( cuDoubleComplex, cusolverDnZgeqrf )

#undef GEQRF

#define ORGQR_BUFFERSIZE( type, func )              \
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

ORGQR_BUFFERSIZE( float,           cusolverDnSorgqr_bufferSize )
ORGQR_BUFFERSIZE( double,          cusolverDnDorgqr_bufferSize )
ORGQR_BUFFERSIZE( cuFloatComplex,  cusolverDnCungqr_bufferSize )
ORGQR_BUFFERSIZE( cuDoubleComplex, cusolverDnZungqr_bufferSize )

#undef ORGQR_BUFFERSIZE

#define ORGQR( type, func )                     \
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

ORGQR( float,           cusolverDnSorgqr )
ORGQR( double,          cusolverDnDorgqr )
ORGQR( cuFloatComplex,  cusolverDnCungqr )
ORGQR( cuDoubleComplex, cusolverDnZungqr )

#undef ORGQR

//
// compute [Q,R] = qr(M) with Q overwriting M
//
template < typename value_t >
void
qr ( handle               handle,
     matrix< value_t > &  M,
     matrix< value_t > &  R )
{
    //
    // copy/allocate data to/on device
    //

    const auto  nrows    = M.nrows();
    const auto  ncols    = M.ncols();
    auto        dev_M    = device_alloc< value_t >( nrows * ncols );
    auto        dev_tau  = device_alloc< value_t >( ncols );
    auto        dev_info = device_alloc< int >( 1 );

    to_device( M, dev_M, nrows );

    // get work buffer size
    const auto  lwork    = std::max( geqrf_buffersize( handle.solver, nrows, ncols, dev_M, nrows ),
                                     orgqr_buffersize( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau ) );
    
    auto        dev_work = device_alloc< value_t >( lwork );

    //
    // factorise and get R
    //

    int  info = 0;
    
    geqrf( handle.solver, nrows, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );

    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );

    // copy R and reset lower triangular part
    if (( R.nrows() != ncols ) || ( R.ncols() != ncols ))
        R = std::move( matrix< value_t >( nrows, ncols ) );
        
    from_device( dev_M, nrows, R );

    for ( size_t  i = 0; i < ncols; i++ )
    {
        vector< value_t >  R_i( R, range( i+1, ncols-1 ), i );

        fill( value_t(0), R_i );
    }// for
    
    //
    // compute Q
    //
    
    orgqr( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );
    
    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );

    from_device( dev_M, nrows, M );

    //
    // release device memory
    //

    device_free( dev_work );
    device_free( dev_info );
    device_free( dev_tau );
    device_free( dev_M );
}

template < typename value_t >
void
qr_dev ( handle     handle,
         const int  nrows,
         const int  ncols,
         value_t *  dev_M,
         value_t *  dev_R )
{
    //
    // allocate data on device
    //

    auto  dev_tau  = device_alloc< value_t >( ncols );
    auto  dev_info = device_alloc< int >( 1 );

    // get work buffer size
    const auto  lwork    = std::max( geqrf_buffersize( handle.solver, nrows, ncols, dev_M, nrows ),
                                     orgqr_buffersize( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau ) );
    
    auto        dev_work = device_alloc< value_t >( lwork );

    //
    // factorise and get R
    //

    int  info = 0;
    
    geqrf( handle.solver, nrows, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );

    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );

    // copy upper triangular part to R
    auto  zero = make_constant< value_t >( 0 );

    scale( handle, ncols*ncols, zero, dev_R, 1 );

    for ( int  i = 0; i < ncols; i++ )
        copy( handle, i+1, dev_M + i*nrows, 1, dev_R + i*ncols, 1 );
    
    //
    // compute Q
    //
    
    orgqr( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );
    
    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );

    //
    // release device memory
    //

    device_free( dev_work );
    device_free( dev_info );
    device_free( dev_tau );
}

template < typename value_t >
int
qr_worksize ( handle     handle,
              const int  nrows,
              const int  ncols,
              value_t *  dev_M,
              value_t *  dev_tau )
{
    return std::max( geqrf_buffersize( handle.solver, nrows, ncols, dev_M, nrows ),
                     orgqr_buffersize( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau ) );
}
    
template < typename value_t >
void
qr_dev ( handle     handle,
         const int  nrows,
         const int  ncols,
         value_t *  dev_M,
         value_t *  dev_R,
         value_t *  dev_tau,
         const int  lwork,
         value_t *  dev_work,
         int *      dev_info )
{
    //
    // factorise 
    //

    geqrf( handle.solver, nrows, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );

    //
    // copy upper triangular part to R
    //
    
    cudaMemsetAsync( dev_R, 0, ncols*ncols*sizeof(value_t), handle.stream );

    for ( int  i = 0; i < ncols; i++ )
        copy( handle, i+1, dev_M + i*nrows, 1, dev_R + i*ncols, 1 );
    
    //
    // compute Q
    //
    
    orgqr( handle.solver, nrows, ncols, ncols, dev_M, nrows, dev_tau, dev_work, lwork, dev_info );
}

//////////////////////////////////////////////////////////////////////
//
// SVD related functions
//
//////////////////////////////////////////////////////////////////////

#define GESVDJ_BUFFERSIZE( type, func )              \
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

GESVDJ_BUFFERSIZE( float,           cusolverDnSgesvdj_bufferSize )
GESVDJ_BUFFERSIZE( double,          cusolverDnDgesvdj_bufferSize )
GESVDJ_BUFFERSIZE( cuFloatComplex,  cusolverDnCgesvdj_bufferSize )
GESVDJ_BUFFERSIZE( cuDoubleComplex, cusolverDnZgesvdj_bufferSize )

#undef GESVDJ_BUFFERSIZE

#define GESVDJ( type, func )              \
    inline                                \
    void                                  \
    gesvdj ( cusolverDnHandle_t  handle,  \
             cusolverEigMode_t   jobz,    \
             int                 econ,    \
             int                 m,       \
             int                 n,       \
             type *              A,       \
             int                 lda,     \
             typename cuda::real_type< type >::type_t *  S, \
             type *              U,       \
             int                 ldu,     \
             type *              V,       \
             int                 ldv,     \
             type *              work,    \
             int                 lwork,   \
             int *               info,    \
             gesvdjInfo_t        params ) \
    {                                     \
        HLR_CUSOLVER_CHECK( func, ( handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params ) ); \
    }

GESVDJ( float,           cusolverDnSgesvdj )
GESVDJ( double,          cusolverDnDgesvdj )
GESVDJ( cuFloatComplex,  cusolverDnCgesvdj )
GESVDJ( cuDoubleComplex, cusolverDnZgesvdj )

//
// compute SVD M = U·S·V^H
//
template < typename value_t >
void
svd ( handle                                    handle,
      matrix< value_t > &                       M,
      vector< Hpro::real_type_t< value_t > > &  S,
      matrix< value_t > &                       VH )
{
    using cuda_t = typename cuda_type< value_t >::type_t;
    using real_t = typename cuda::real_type< cuda_t >::type_t;
    
    //
    // copy/allocate data to/on device
    //

    const int  nrows    = M.nrows();
    const int  ncols    = M.ncols();
    const int  minrc    = std::min( nrows, ncols );
    auto       dev_M    = device_alloc< value_t >( nrows * ncols );
    auto       dev_U    = device_alloc< value_t >( nrows * minrc );
    auto       dev_S    = device_alloc< real_t >( minrc );
    auto       dev_VH   = device_alloc< value_t >( minrc * ncols );
    auto       dev_info = device_alloc< int >( 1 );

    to_device( M, dev_M, nrows );

    // SVD parameters
    const double       tol        = 1e-7; // from accuracy?
    const int          max_sweeps = 15;
    cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    const int          econ       = 1; // economy mode
    gesvdjInfo_t       gesvdj_params;

    HLR_CUSOLVER_CHECK( cusolverDnCreateGesvdjInfo,    ( & gesvdj_params ) );
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetTolerance, ( gesvdj_params, tol ) );
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetMaxSweeps, ( gesvdj_params, max_sweeps ) );
    
    // get work buffer size
    const auto  lwork    = gesvdj_buffersize( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows,
                                              dev_S, dev_U, nrows, dev_VH, ncols, gesvdj_params );
    
    auto        dev_work = device_alloc< value_t >( lwork );

    // compute SVD
    int  info = 0;
    
    gesvdj( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows, dev_S, dev_U, nrows, dev_VH, ncols,
            dev_work, lwork, dev_info, gesvdj_params );

    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );

    if ( int(S.length()) != minrc )
        S = std::move( vector< real_t >( minrc ) );
    
    if (( int(VH.nrows()) != minrc ) || ( int(VH.ncols()) != ncols ))
        VH = std::move( matrix< real_t >( minrc, ncols ) );
    
    from_device( dev_U,  nrows, M );
    from_device( dev_S,      1, S );
    from_device( dev_VH, minrc, VH );
}

template < typename value_t >
void
svd_dev ( handle                                         handle,
          const int                                      nrows,
          const int                                      ncols,
          value_t *                                      dev_M,
          typename cuda::real_type< value_t >::type_t *  dev_S,
          value_t *                                      dev_VH )
{
    //
    // allocate data on device
    //

    auto  dev_U    = device_alloc< value_t >( nrows * ncols );
    auto  dev_info = device_alloc< int >( 1 );

    copy( handle, nrows*ncols, dev_M, 1, dev_U, 1 );
    
    // SVD parameters
    const double       tol        = 1e-7; // from accuracy?
    const int          max_sweeps = 15;
    cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    const int          econ       = 1; // economy mode
    gesvdjInfo_t       gesvdj_params;

    HLR_CUSOLVER_CHECK( cusolverDnCreateGesvdjInfo,    ( & gesvdj_params ) );
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetTolerance, ( gesvdj_params, tol ) );
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetMaxSweeps, ( gesvdj_params, max_sweeps ) );
    
    // get work buffer size
    const auto  lwork    = gesvdj_buffersize( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows,
                                              dev_S, dev_U, nrows, dev_VH, ncols, gesvdj_params );
    
    auto        dev_work = device_alloc< value_t >( lwork );

    // compute SVD
    int  info = 0;
    
    gesvdj( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows, dev_S, dev_U, nrows, dev_VH, ncols,
            dev_work, lwork, dev_info, gesvdj_params );

    info = from_device< int >( dev_info );

    HLR_ASSERT( info == 0 );
}

template < typename value_t >
int
svd_worksize ( handle                                         handle,
               const int                                      nrows,
               const int                                      ncols,
               value_t *                                      dev_M,
               value_t *                                      dev_U,
               typename cuda::real_type< value_t >::type_t *  dev_S,
               value_t *                                      dev_VH,
               gesvdjInfo_t &                                 gesvdj_params )
{
    cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    const int          econ       = 1; // economy mode
    
    return gesvdj_buffersize( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows,
                              dev_S, dev_U, nrows, dev_VH, ncols, gesvdj_params );
}

template < typename value_t >
void
svd_dev ( handle                                         handle,
          const int                                      nrows,
          const int                                      ncols,
          value_t *                                      dev_M,
          value_t *                                      dev_U,
          typename cuda::real_type< value_t >::type_t *  dev_S,
          value_t *                                      dev_VH,
          gesvdjInfo_t &                                 gesvdj_params,
          int                                            lwork,
          value_t *                                      dev_work,
          int *                                          dev_info )
{
    //
    // allocate data on device
    //

    cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    const int          econ       = 1; // economy mode

    copy( handle, nrows*ncols, dev_M, 1, dev_U, 1 );
    
    gesvdj( handle.solver, jobz, econ, nrows, ncols, dev_M, nrows, dev_S, dev_U, nrows, dev_VH, ncols,
            dev_work, lwork, dev_info, gesvdj_params );
}

//////////////////////////////////////////////////////////////////////
//
// lowrank truncation
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( handle                           handle,
      const blas::matrix< value_t > &  U,
      const blas::matrix< value_t > &  V,
      const Hpro::TTruncAcc &          acc )
{
    using  cuda_t = typename cuda_type< value_t >::type_t;
    using  real_t = typename cuda::real_type< cuda_t >::type_t;

    HLR_ASSERT( U.ncols() == V.ncols() );

    const int  nrows_U = int( U.nrows() );
    const int  nrows_V = int( V.nrows() );
    const int  inrank  = int( V.ncols() );

    //
    // don't increase rank
    //

    const int  acc_rank = int( acc.rank() );

    blas::matrix< value_t >  OU, OV;
    
    if ( inrank == 0 )
    {
        // reset matrices
        OU = std::move( blas::matrix< value_t >( nrows_U, 0 ) );
        OV = std::move( blas::matrix< value_t >( nrows_V, 0 ) );

        return { std::move( OU ), std::move( OV ) };
    }// if

    if ( inrank <= acc_rank )
    {
        OU = std::move( blas::matrix< value_t >( U, Hpro::copy_value ) );
        OV = std::move( blas::matrix< value_t >( V, Hpro::copy_value ) );

        return { std::move( OU ), std::move( OV ) };
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    int  orank = 0;
        
    if ( acc_rank >= std::min( nrows_U, nrows_V ) )
    {
        HLR_ERROR( "TO DO" );
    }// if
    else
    {
        //
        // do QR-factorisation of U and V
        //

        auto  dev_QU = device_alloc< cuda_t >( nrows_U * inrank );
        auto  dev_RU = device_alloc< cuda_t >( inrank * inrank );

        to_device< value_t >( U, dev_QU, nrows_U );
        
        qr_dev( handle, nrows_U, inrank, dev_QU, dev_RU );

        // {
        //     matrix< value_t >  QU( nrows_U, inrank );
        //     matrix< value_t >  RU( inrank, inrank );

        //     from_device( dev_QU, nrows_U, QU );
        //     from_device( dev_RU, inrank,  RU );

        //     Hpro::DBG::write( QU, "QU.mat", "QU" );
        //     Hpro::DBG::write( RU, "RU.mat", "RU" );
        // }
        
        auto  dev_QV = device_alloc< cuda_t >( nrows_V * inrank );
        auto  dev_RV = device_alloc< cuda_t >( inrank * inrank );

        to_device< value_t >( V, dev_QV, nrows_V );
        
        qr_dev( handle, nrows_V, inrank, dev_QV, dev_RV );

        // {
        //     matrix< value_t >  QV( nrows_V, inrank );
        //     matrix< value_t >  RV( inrank, inrank );

        //     from_device( dev_QV, nrows_V, QV );
        //     from_device( dev_RV, inrank,  RV );

        //     Hpro::DBG::write( QV, "QV.mat", "QV" );
        //     Hpro::DBG::write( RV, "RV.mat", "RV" );
        // }
        
        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  dev_R = device_alloc< cuda_t >( inrank * inrank );
        auto  one   = make_constant< cuda_t >( 1 );
        auto  zero  = make_constant< cuda_t >( 0 );

        prod( handle, CUBLAS_OP_N, CUBLAS_OP_C, inrank, inrank, inrank, one, dev_RU, inrank, dev_RV, inrank, zero, dev_R, inrank );
        
        // {
        //     matrix< value_t >  R( inrank, inrank );

        //     from_device( dev_R, inrank,  R );

        //     Hpro::DBG::write( R, "R.mat", "R" );
        // }
        
        //
        // SVD(R) = U S V^H
        //
            
        auto  dev_Us = dev_R;  // reuse memory
        auto  dev_Ss = device_alloc< real_t >( inrank );
        auto  dev_Vs = dev_RV; // reuse memory
            
        svd_dev< cuda_t >( handle, inrank, inrank, dev_Us, dev_Ss, dev_Vs );
        
        // {
        //     matrix< value_t >  Us( inrank, inrank );
        //     vector< real_t >   Ss( inrank );
        //     matrix< value_t >  Vs( inrank, inrank );

        //     from_device( dev_Us, inrank,  Us );
        //     from_device< real_t >( dev_Ss, 1, Ss );
        //     from_device( dev_Vs, inrank,  Vs );

        //     Hpro::DBG::write( Us, "Us.mat", "Us" );
        //     Hpro::DBG::write( Ss, "Ss.mat", "Ss" );
        //     Hpro::DBG::write( Vs, "Vs.mat", "Vs" );
        // }
        
        // determine truncated rank based on singular values
        auto  Ss = blas::vector< real_t >( inrank );

        from_device< real_t >( dev_Ss, 1, Ss );
        
        orank = int( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < inrank )
        {
            //
            // build new matrices U and V
            //

            // U := U·S
            prod_diag( handle, inrank, dev_Us, Ss, orank );

            // OU := Q_U · U
            auto  dev_OU = device_alloc< cuda_t >( nrows_U * orank );

            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_U, orank, inrank, one, dev_QU, nrows_U, dev_Us, inrank, zero, dev_OU, nrows_U );
            
            // V := Q_V · conj(V)
            auto  dev_OV = device_alloc< cuda_t >( nrows_V * orank );
            
            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_V, orank, inrank, one, dev_QV, nrows_V, dev_Vs, inrank, zero, dev_OV, nrows_V );

            // copy from device
            OU = std::move( blas::matrix< value_t >( nrows_U, orank ) );
            OV = std::move( blas::matrix< value_t >( nrows_V, orank ) );

            from_device( dev_OU, nrows_U, OU );
            from_device( dev_OV, nrows_V, OV );

            device_free( dev_OU );
            device_free( dev_OV );
        }// if
        else
        {
            OU = std::move( blas::matrix< value_t >( U, Hpro::copy_value ) );
            OV = std::move( blas::matrix< value_t >( V, Hpro::copy_value ) );
        }// else

        device_free( dev_Ss );
        device_free( dev_R  );
        device_free( dev_QV );
        device_free( dev_RV );
        device_free( dev_QU );
        device_free( dev_RU );
    }// else

    return { std::move( OU ), std::move( OV ) };
}

//
// on-device version of the above SVD based low-rank truncation
//
// on exit : dev_U and dev_V will be (normally) reallocated and
//           hold the truncated low-rank factors
//
template < typename value_t >
int
svd_dev ( handle                   handle,
          const int                nrows_U,
          const int                nrows_V,
          const int                rank,
          value_t * &              dev_U,
          value_t * &              dev_V,
          const Hpro::TTruncAcc &  acc )
{
    using  cuda_t = value_t;
    using  real_t = typename cuda::real_type< cuda_t >::type_t;

    //
    // don't increase rank
    //

    const int  in_rank   = rank;
    const int  acc_rank = int( acc.rank() );

    if ( in_rank == 0 )
        return in_rank;

    if ( in_rank <= acc_rank )
        return in_rank;

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    int  out_rank = 0;
        
    if ( std::max( in_rank, acc_rank ) >= std::min( nrows_U, nrows_V ) / 2 )
    {
        HLR_ERROR( "TO DO" );
        return in_rank;
    }// if
    else
    {
        //
        // do QR-factorisation of U and V
        //

        auto  dev_QU = device_alloc< cuda_t >( nrows_U * in_rank );
        auto  dev_RU = device_alloc< cuda_t >( in_rank * in_rank );

        copy( handle, nrows_U * in_rank, dev_U, 1, dev_QU, 1 );
        qr_dev( handle, nrows_U, in_rank, dev_QU, dev_RU );
        
        auto  dev_QV = device_alloc< cuda_t >( nrows_V * in_rank );
        auto  dev_RV = device_alloc< cuda_t >( in_rank * in_rank );

        copy( handle, nrows_V * in_rank, dev_V, 1, dev_QV, 1 );
        qr_dev( handle, nrows_V, in_rank, dev_QV, dev_RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  dev_R = device_alloc< cuda_t >( in_rank * in_rank );
        auto  one   = make_constant< cuda_t >( 1 );
        auto  zero  = make_constant< cuda_t >( 0 );

        prod( handle, CUBLAS_OP_N, CUBLAS_OP_C, in_rank, in_rank, in_rank, one, dev_RU, in_rank, dev_RV, in_rank, zero, dev_R, in_rank );
        
        //
        // SVD(R) = U S V^H
        //
            
        auto  dev_Us = dev_R;  // reuse memory
        auto  dev_Ss = device_alloc< real_t >( in_rank );
        auto  dev_Vs = dev_RV; // reuse memory
            
        svd_dev< cuda_t >( handle, in_rank, in_rank, dev_Us, dev_Ss, dev_Vs );
        
        // determine truncated rank based on singular values
        auto  Ss = blas::vector< real_t >( in_rank );

        from_device< real_t >( dev_Ss, 1, Ss );
        
        out_rank = int( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( out_rank < in_rank )
        {
            //
            // free old and build new matrices U and V
            //

            device_free( dev_U );
            device_free( dev_V );

            // U := U·S
            prod_diag( handle, in_rank, dev_Us, Ss, out_rank );

            // OU := Q_U · U
            dev_U = device_alloc< cuda_t >( nrows_U * out_rank );
            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_U, out_rank, in_rank, one, dev_QU, nrows_U, dev_Us, in_rank, zero, dev_U, nrows_U );
            
            // V := Q_V · conj(V)
            dev_V = device_alloc< cuda_t >( nrows_V * out_rank );
            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_V, out_rank, in_rank, one, dev_QV, nrows_V, dev_Vs, in_rank, zero, dev_V, nrows_V );
        }// if
        else
        {
            // adjust for return value below
            out_rank = in_rank;
        }// else

        device_free( dev_Ss );
        device_free( dev_R  );
        device_free( dev_QV );
        device_free( dev_RV );
        device_free( dev_QU );
        device_free( dev_RU );

        return  out_rank;
    }// else

}

//
// on-device version of the above SVD based low-rank truncation
//
// on exit : dev_U and dev_V will be (normally) reallocated and
//           hold the truncated low-rank factors
//
__global__
template < typename value_t >
void
svd_dev2 ( handle                   handle,
           const int                nrows_U,
           const int                nrows_V,
           const int                rank,
           value_t * &              dev_U,
           value_t * &              dev_V,
           const Hpro::TTruncAcc &  acc,
           int &                    new_rank )
{
    using  cuda_t = value_t;
    using  real_t = typename cuda::real_type< cuda_t >::type_t;

    //
    // don't increase rank
    //

    const int  in_rank   = rank;
    const int  acc_rank = int( acc.rank() );

    if ( in_rank == 0 )
    {
        new_rank = in_rank;
        return;
    }// if

    if ( in_rank <= acc_rank )
    {
        new_rank = in_rank;
        return;
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    int  out_rank = 0;
        
    if ( std::max( in_rank, acc_rank ) >= std::min( nrows_U, nrows_V ) / 2 )
    {
        HLR_ERROR( "TO DO" );

        new_rank = in_rank;
        return;
    }// if
    else
    {
        //
        // do QR-factorisation of U and V
        //

        auto  dev_QU = device_alloc< cuda_t >( nrows_U * in_rank );
        auto  dev_RU = device_alloc< cuda_t >( in_rank * in_rank );

        copy( handle, nrows_U * in_rank, dev_U, 1, dev_QU, 1 );
        qr_dev( handle, nrows_U, in_rank, dev_QU, dev_RU );
        
        auto  dev_QV = device_alloc< cuda_t >( nrows_V * in_rank );
        auto  dev_RV = device_alloc< cuda_t >( in_rank * in_rank );

        copy( handle, nrows_V * in_rank, dev_V, 1, dev_QV, 1 );
        qr_dev( handle, nrows_V, in_rank, dev_QV, dev_RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  dev_R = device_alloc< cuda_t >( in_rank * in_rank );
        auto  one   = make_constant< cuda_t >( 1 );
        auto  zero  = make_constant< cuda_t >( 0 );

        prod( handle, CUBLAS_OP_N, CUBLAS_OP_C, in_rank, in_rank, in_rank, one, dev_RU, in_rank, dev_RV, in_rank, zero, dev_R, in_rank );
        
        //
        // SVD(R) = U S V^H
        //
            
        auto  dev_Us = dev_R;  // reuse memory
        auto  dev_Ss = device_alloc< real_t >( in_rank );
        auto  dev_Vs = dev_RV; // reuse memory
            
        svd_dev< cuda_t >( handle, in_rank, in_rank, dev_Us, dev_Ss, dev_Vs );
        
        // determine truncated rank based on singular values
        auto  Ss = blas::vector< real_t >( in_rank );

        from_device< real_t >( dev_Ss, 1, Ss );
        
        out_rank = int( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( out_rank < in_rank )
        {
            //
            // free old and build new matrices U and V
            //

            device_free( dev_U );
            device_free( dev_V );

            // U := U·S
            prod_diag( handle, in_rank, dev_Us, Ss, out_rank );

            // OU := Q_U · U
            dev_U = device_alloc< cuda_t >( nrows_U * out_rank );
            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_U, out_rank, in_rank, one, dev_QU, nrows_U, dev_Us, in_rank, zero, dev_U, nrows_U );
            
            // V := Q_V · conj(V)
            dev_V = device_alloc< cuda_t >( nrows_V * out_rank );
            prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, nrows_V, out_rank, in_rank, one, dev_QV, nrows_V, dev_Vs, in_rank, zero, dev_V, nrows_V );

            new_rank = out_rank;
        }// if
        else
        {
            // adjust for return value below
            new_rank = in_rank;
        }// else

        device_free( dev_Ss );
        device_free( dev_R  );
        device_free( dev_QV );
        device_free( dev_RV );
        device_free( dev_QU );
        device_free( dev_RU );
    }// else
}

}}}// hlr::blas::cuda

#endif // __HLR_ARITH_CUDA_HH
