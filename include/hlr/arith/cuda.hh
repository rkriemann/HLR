#ifndef __HLR_ARITH_CUDA_HH
#define __HLR_ARITH_CUDA_HH
//
// Project     : HLR
// Module      : arith/cuda
// Description : basic linear algebra functions using cuda
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace blas { namespace cuda {

//
// mapping of default types to cuBLAS types
//
template < typename T > struct cuda_type                             { using  type_t = T; };
template <>             struct cuda_type< hpro::Complex< float > >   { using  type_t = cuFloatComplex; };
template <>             struct cuda_type< hpro::Complex< double > >  { using  type_t = cuDoubleComplex; };

template < typename T > struct cuda_type_ptr                         { using  type_t = typename cuda_type< T >::type_t *; };

//
// joined handle for cuBLAS and cuSolverDn
//
struct handle
{
    cublasHandle_t      blas;
    cusolverDnHandle_t  solver;
};

// default handle
handle  default_handle;

//
// initialize cuBLAS/cuSolver
//
void
init ()
{
    if ( cublasCreate( & default_handle.blas ) != CUBLAS_STATUS_SUCCESS )
        HLR_ERROR( "CUBLAS initialization failed" );

    if ( cusolverDnCreate( & default_handle.solver ) != CUSOLVER_STATUS_SUCCESS )
        HLR_ERROR( "error during cusolverDnCreate" );
}

//
// device memory allocation with CUDA
//
template < typename value_t >
typename cuda_type_ptr< value_t >::type_t
device_alloc ( const size_t  n )
{
    void *  ptr = nullptr;
    
    if ( cudaMalloc( & ptr, n * sizeof(value_t) ) != cudaSuccess )
        HLR_ERROR( "device memory allocation failed" );

    return typename cuda_type_ptr< value_t >::type_t( ptr );
}

template < typename value_t >
void
device_free ( value_t *  ptr )
{
    if ( cudaFree( ptr ) != cudaSuccess )
        HLR_ERROR( "device memory deallocation failed" );
}

//
// host to device copy
//
template < typename value_t >
void
to_device ( matrix< value_t > &                        M_host,
            typename cuda_type_ptr< value_t >::type_t  M_dev,
            int                                        lda_dev )
{
    // queue ???
    if ( cublasSetMatrix( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                          M_host.data(), M_host.col_stride(),
                          M_dev, lda_dev ) != CUBLAS_STATUS_SUCCESS )
        HLR_ERROR( "transfer to device failed" );
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
    // queue ???
    if ( cublasGetMatrix( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                          M_dev, lda_dev,
                          M_host.data(), M_host.col_stride() ) != CUBLAS_STATUS_SUCCESS )
        HLR_ERROR( "transfer to host failed" );
}

template < typename value_t >
value_t
from_device ( typename cuda_type_ptr< value_t >::type_t  dev_data )
{
    value_t  data;

    if ( cudaMemcpy( & data, dev_data, sizeof(value_t), cudaMemcpyDeviceToHost ) != cudaSuccess )
        HLR_ERROR( "transfer to host failed" );

    return data;
}

//////////////////////////////////////////////////////////////////////
//
// QR
//
//////////////////////////////////////////////////////////////////////

//
// return work buffer size for geqrf/orgqr
//
#define GEQRF_BUFFERSIZE( type, func )                                  \
    int                                                                 \
    geqrf_buffersize ( cusolverDnHandle_t  handle,                      \
                       int                 nrows,                       \
                       int                 ncols,                       \
                       type *              A,                           \
                       int                 ldA )                        \
    {                                                                   \
        int  lwork = 0;                                                 \
                                                                        \
        if ( func( handle, nrows, ncols, A, ldA, & lwork ) != CUSOLVER_STATUS_SUCCESS ) \
            HLR_ERROR( "error during cusolverDn*geqrf_bufferSize" );    \
                                                                        \
        return  lwork;                                                  \
    }

GEQRF_BUFFERSIZE( float,           cusolverDnSgeqrf_bufferSize )
GEQRF_BUFFERSIZE( double,          cusolverDnDgeqrf_bufferSize )
GEQRF_BUFFERSIZE( cuFloatComplex,  cusolverDnCgeqrf_bufferSize )
GEQRF_BUFFERSIZE( cuDoubleComplex, cusolverDnZgeqrf_bufferSize )

#undef GEQRF_BUFFERSIZE

#define GEQRF( type, func )                     \
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
        if ( func( handle, nrows, ncols, A, ldA, tau, work, lwork, info ) != CUSOLVER_STATUS_SUCCESS ) \
            HLR_ERROR( "error during cusolverDn*geqrf" ); \
    }

GEQRF( float,           cusolverDnSgeqrf )
GEQRF( double,          cusolverDnDgeqrf )
GEQRF( cuFloatComplex,  cusolverDnCgeqrf )
GEQRF( cuDoubleComplex, cusolverDnZgeqrf )

#undef GEQRF

#define ORGQR_BUFFERSIZE( type, func )              \
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
        if ( func( handle, nrows, ncols, k, A, ldA, tau, & lwork ) != CUSOLVER_STATUS_SUCCESS ) \
            HLR_ERROR( "error during cusolverDn*orgqr_bufferSize" );    \
                                                                        \
        return  lwork;                                                  \
    }

ORGQR_BUFFERSIZE( float,           cusolverDnSorgqr_bufferSize )
ORGQR_BUFFERSIZE( double,          cusolverDnDorgqr_bufferSize )
ORGQR_BUFFERSIZE( cuFloatComplex,  cusolverDnCungqr_bufferSize )
ORGQR_BUFFERSIZE( cuDoubleComplex, cusolverDnZungqr_bufferSize )

#undef ORGQR_BUFFERSIZE

#define ORGQR( type, func )                     \
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
        if ( func( handle, nrows, ncols, k, A, ldA, tau, work, lwork, info ) != CUSOLVER_STATUS_SUCCESS ) \
            HLR_ERROR( "error during cusolverDn*orgqr" ); \
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

    if ( info != 0 )
        HLR_ERROR( "error during geqrf" );

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

    if ( info != 0 )
        HLR_ERROR( "error during orgqr" );

    from_device( dev_M, nrows, M );

    //
    // release device memory
    //

    device_free( dev_work );
    device_free( dev_info );
    device_free( dev_tau );
    device_free( dev_M );
}

}}}// hlr::blas::cuda

#endif // __HLR_ARITH_CUDA_HH
