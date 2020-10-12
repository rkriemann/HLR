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

#include <boost/format.hpp>

#include <hlr/arith/blas.hh>
#include <hlr/arith/blas_eigen.hh>
#include <hlr/arith/cuda_def.hh>
#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace blas { namespace cuda {

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

//////////////////////////////////////////////////////////////////////
//
// QR related functions
//
//////////////////////////////////////////////////////////////////////

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

//
// compute SVD M = U·S·V^H
//
template < typename value_t >
void
svd ( handle               handle,
      matrix< value_t > &  M,
      vector< typename hpro::real_type< value_t >::type_t > &  S,
      matrix< value_t > &  VH )
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
      const hpro::TTruncAcc &          acc )
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
        OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
        OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );

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

        //     hpro::DBG::write( QU, "QU.mat", "QU" );
        //     hpro::DBG::write( RU, "RU.mat", "RU" );
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

        //     hpro::DBG::write( QV, "QV.mat", "QV" );
        //     hpro::DBG::write( RV, "RV.mat", "RV" );
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

        //     hpro::DBG::write( R, "R.mat", "R" );
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

        //     hpro::DBG::write( Us, "Us.mat", "Us" );
        //     hpro::DBG::write( Ss, "Ss.mat", "Ss" );
        //     hpro::DBG::write( Vs, "Vs.mat", "Vs" );
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
            OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
            OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );
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
          const hpro::TTruncAcc &  acc )
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
           const hpro::TTruncAcc &  acc,
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
        auto  Ss = vector< real_t >( in_rank );

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

//
// eigenvalues for hermitean matrices using Jacobi iteration
//
template < typename value_t >
std::pair< vector< typename hpro::real_type< value_t >::type_t >,
           matrix< value_t > >
eigen_jac ( handle                                             handle,
            matrix< value_t > &                                M,
            const typename hpro::real_type< value_t >::type_t  tol,
            const size_t                                       max_sweeps,
            eigen_stat *                                       stat = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    // square matrix assumed
    HLR_ASSERT( M.nrows() == M.ncols() );

    if ( ! is_null( stat ) )
        stat->reset();
    
    //
    // set up parameters for Jacobi
    //

    const auto   n = M.nrows();
    syevjInfo_t  syevj_params;
    
    cusolverDnCreateSyevjInfo( & syevj_params );
    cusolverDnXsyevjSetTolerance( syevj_params, tol );
    
    HLR_CUSOLVER_CHECK( cusolverDnXsyevjSetMaxSweeps, ( syevj_params, max_sweeps ) );
    
    //
    // copy/allocate data to/on device
    //
    
    auto  dev_M    = device_alloc< value_t >( n*n );
    auto  dev_W    = device_alloc< real_t >( n );
    auto  dev_info = device_alloc< int >( 1 );

    to_device( M, dev_M, n );

    // get work buffer size
    const auto  lwork    = syevj_buffersize( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dev_M, n, dev_W, syevj_params );
    auto        dev_work = device_alloc< value_t >( lwork );

    //
    // compute eigenvalues
    //

    syevj( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dev_M, n, dev_W, dev_work, lwork, dev_info, syevj_params );

    HLR_CUDA_CHECK( cudaDeviceSynchronize, () );

    int     sweeps   = 0;
    double  residual = 0;
    
    cusolverDnXsyevjGetSweeps(   handle.solver, syevj_params, & sweeps );
    cusolverDnXsyevjGetResidual( handle.solver, syevj_params, & residual );

    if ( ! is_null( stat ) )
    {
        stat->nsweeps = sweeps;
        stat->error   = residual;
    }// if
    
    vector< real_t >   E( n );
    matrix< value_t >  V( n, n );

    from_device( dev_W, 1, E );
    from_device( dev_M, n, V );

    auto  info = from_device< int >( dev_info );

    if ( ! is_null( stat ) && ( info == 0 ))
        stat->converged = true;
    
    if ( info != 0 )
        HLR_ERROR( hpro::to_string( "syevj failed (info == %d)", info ) );
    
    device_free( dev_work );
    device_free( dev_M );
    device_free( dev_W );
    device_free( dev_info );
    
    return  { std::move( E ), std::move( V ) };
}

//
// eigenvalues for hermitean matrices
//
template < typename value_t >
std::pair< vector< typename hpro::real_type< value_t >::type_t >,
           matrix< value_t > >
eigen_herm ( handle               handle,
             matrix< value_t > &  M,
             eigen_stat *         stat = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    // square matrix assumed
    HLR_ASSERT( M.nrows() == M.ncols() );

    if ( ! is_null( stat ) )
        stat->reset();
    
    //
    // copy/allocate data to/on device
    //
    
    const auto  n        = M.nrows();
    auto        dev_M    = device_alloc< value_t >( n*n );
    auto        dev_W    = device_alloc< real_t >( n );
    auto        dev_info = device_alloc< int >( 1 );

    to_device( M, dev_M, n );

    // get work buffer size
    const auto  lwork    = syevd_buffersize( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dev_M, n, dev_W );
    auto        dev_work = device_alloc< value_t >( lwork );

    //
    // compute eigenvalues
    //

    syevd( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dev_M, n, dev_W, dev_work, lwork, dev_info );

    HLR_CUDA_CHECK( cudaDeviceSynchronize, () );

    vector< real_t >   E( n );
    matrix< value_t >  V( n, n );

    from_device( dev_W, 1, E );
    from_device( dev_M, n, V );

    auto  info = from_device< int >( dev_info );

    if ( ! is_null( stat ) && ( info == 0 ))
        stat->converged = true;
    
    if ( info != 0 )
        HLR_ERROR( hpro::to_string( "syevj failed (info == %d)", info ) );
    
    device_free( dev_work );
    device_free( dev_M );
    device_free( dev_W );
    device_free( dev_info );
    
    return  { std::move( E ), std::move( V ) };
}

namespace device
{

template < typename value_t >
void
eigen_herm ( handle        handle,
             const int     n,
             value_t *     dev_M,
             typename real_type< value_t >::type_t *  dev_E,
             value_t *     dev_work,
             const int     lwork,
             int *         dev_info )
{
    //
    // compute eigenvalues
    //

    syevd( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dev_M, n, dev_E, dev_work, lwork, dev_info );

    // HLR_CUDA_CHECK( cudaDeviceSynchronize, () );
}

}// namespace device

//
// compute Q = I + α·Θ⊗M with Θ_ij = 1 / ( m_ii - m_jj )
// - implemented directly in CUDA
//
#define HMUL_THETA( type )                                          \
    void                                                            \
    hmul_theta ( const int      nrows,                              \
                 const int      ncols,                              \
                 const typename real_type< type >::type_t  alpha,   \
                 const type *   diag_M,                             \
                 const type *   M,                                  \
                 type *         Q );

HMUL_THETA( float )
HMUL_THETA( double )
HMUL_THETA( cuFloatComplex )
HMUL_THETA( cuDoubleComplex )

//
// compute eigen values of given matrix M using DPT iteration
//
//   - stop iteration if |A_i - A_i-1| < tol or i > max_it
//   - if tol == -1, then machine precision w.r.t. value_t is chosen
//   - if max_it == 0, then max_it = 100 is set
//
template < typename value_t >
std::pair< vector< value_t >,
           matrix< value_t > >
eigen_dpt ( handle                                             handle,
            matrix< value_t > &                                M,
            const typename hpro::real_type< value_t >::type_t  tol        = -1,
            const size_t                                       max_it     = 0,
            const std::string &                                error_type = "frobenius",
            const int                                          verbosity  = 0,
            eigen_stat *                                       stat       = nullptr )
{
    using  cuda_t = typename cuda_type< value_t >::type_t;
    using  real_t = typename cuda::real_type< cuda_t >::type_t;
    
    // square matrix assumed
    HLR_ASSERT( M.nrows() == M.ncols() );

    if ( ! is_null( stat ) )
        stat->reset();
    
    //
    // set up auxiliary data structures
    //
    // Θ_ij = 1 / ( M_ii - M_jj )
    // Δ    = M - diag(M)
    //

    const auto         n = M.nrows();
    vector< value_t >  diag( n );
    vector< value_t >  one( n ); // needed for Identity vector in cuda

    fill( value_t(1), one );
    
    //
    // initialize CUDA/cuBLAS
    //

    value_t *  dev_V     = device_alloc< cuda_t >( n * n );
    value_t *  dev_Delta = device_alloc< cuda_t >( n * n );;
    value_t *  dev_T     = device_alloc< cuda_t >( n * n );;
    value_t *  dev_diagM = device_alloc< cuda_t >( n );;

    // Δ = M
    to_device< value_t >( M, dev_Delta, n );
    
    // diag = diag(M)
    copy( handle, n, dev_Delta, n+1, dev_diagM, 1 );

    // Δ = M - diag(M)
    scale( handle, n, value_t(0), dev_Delta, n+1 );

    // V = I
    to_device( one, dev_V, n+1 );

    //
    // iteration
    //

    const real_t   precision = ( tol < 0
                                 ? value_t(10) * std::numeric_limits< real_t >::epsilon()
                                 : tol );
    const size_t   max_steps = ( max_it == 0 ? 100 : max_it );
    size_t         nsteps = 0;
    real_t         old_error = 0;

    do
    {
        //
        // iteration step: I - ( Θ ∗ Δ·V - V·diag(Δ·V) )
        //
        
        // Δ·V
        prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, value_t(1), dev_Delta, n, dev_V, n, value_t(0), dev_T, n );
        
        // Δ·V - V·diag(Δ·V) = T - V·diag(T) 
        // computed as T(i,:) = T(i,:) - T(i,i) · A(i,:)
        from_device< value_t >( dev_T, n+1, diag );
            
        for ( size_t  i = 0; i < n; ++i )
            axpy( handle, n, -diag(i), dev_V + i*n, 1, dev_T + i*n, 1 );

        // I - Θ ∗ Δ·V - V·diag(Δ·V) = I - Θ ∗ T
        hmul_theta( n, n, value_t(1), dev_diagM, dev_T, dev_T );

        //
        // compute error: ||V-T||_F
        //

        real_t  error = 0;
        
        if (( error_type == "frobenius" ) || ( error_type == "fro" ))
        {
            axpy( handle, n*n, value_t(-1), dev_T, 1, dev_V, 1 );
            error = norm_2( handle, n*n, dev_V, 1 );
        }// if
        else if (( error_type == "maximum" ) || ( error_type == "max" ))
        {
            throw std::runtime_error( "max error not supported" );
        }// if
        else if (( error_type == "residual" ) || ( error_type == "res" ))
        {
            throw std::runtime_error( "res error not supported" );
        }// if
        else
            throw std::runtime_error( "unknown error type" );

        //
        // test stop criterion
        //
        
        copy( handle, n*n, dev_T, 1, dev_V, 1 );

        if ( verbosity >= 1 )
        {
            std::cout << "    step " << boost::format( "%03d" ) % nsteps
                      << " : error = " << boost::format( "%.4e" ) % error;

            if ( nsteps > 0 )
                std::cout << ", reduction = " << boost::format( "%.4e" ) % ( error / old_error );
            
            std::cout << std::endl;
        }// if
        
        if ( ! is_null( stat ) )
        {
            stat->nsweeps = nsteps;
            stat->error   = error;
        }// if
        
        old_error = error;

        ++nsteps;

        if ( error < precision )
        {
            if ( ! is_null( stat ) )
                stat->converged = true;
            
            break;
        }// if

        if ( ! std::isnormal( error ) )
            break;

    } while ( nsteps < max_steps );

    //
    // eigenvalues  : diag( M + Δ·V )
    // eigenvectors : V
    //

    vector< value_t >  E( n );
    matrix< value_t >  V( n, n );

    for ( size_t  i = 0; i < n; ++i )
        E(i) = M(i,i) + dot( handle, n, dev_Delta + i, n, dev_V + i*n, 1 );

    from_device( dev_V, n, V );

    device_free( dev_diagM );
    device_free( dev_T );
    device_free( dev_Delta );
    device_free( dev_V );
    
    return { std::move( E ), std::move( V ) };
}

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_bjac ( handle                                             handle,
             matrix< value_t > &                                M,
             const size_t                                       block_size  = 128,
             const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
             const size_t                                       amax_sweeps = 0,
             const uint                                         verbosity   = 0,
             blas::eigen_stat *                                 stat        = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const auto    bs         = block_size; // for abbrv.
    const auto    nrows      = M.nrows();
    const auto    ncols      = M.ncols();
    const auto    nbrows     = nrows / bs;
    const auto    nbcols     = ncols / bs;

    HLR_ASSERT( ( nrows / bs ) * bs == nrows );
    HLR_ASSERT( ( ncols / bs ) * bs == ncols );
    
    const auto    minrc      = std::min( nrows, ncols );
    const size_t  max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15*minrc*minrc );
    const real_t  tolerance  = ( atolerance > 0 ? atolerance : real_t(100) * std::numeric_limits< real_t >::epsilon() );
    bool          converged  = false;
    uint          sweep      = 0;
    auto          dev_M      = std::vector< value_t * >( nrows * ncols );
    auto          dev_A      = device_alloc< value_t >( ( 2 * bs ) * ( 2 * bs ) );
    auto          dev_A00    = device_alloc< value_t >( bs * bs );
    auto          dev_A01    = device_alloc< value_t >( bs * bs );
    auto          dev_A10    = device_alloc< value_t >( bs * bs );
    auto          dev_A11    = device_alloc< value_t >( bs * bs );
    auto          dev_T0     = device_alloc< value_t >( bs * bs );
    auto          dev_T1     = device_alloc< value_t >( bs * bs );
    auto          dev_W      = device_alloc< value_t >( 2 * bs );
    auto          dev_V      = std::vector< value_t * >( nrows * ncols );
    auto          norms      = matrix< real_t >( nbrows, nbcols );
    auto          vone       = vector< value_t >( bs ); // needed for Identity vector in cuda
    const auto    lwork      = syevd_buffersize( handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 2*bs, dev_A, 2*bs, dev_W );
    auto          dev_work   = device_alloc< value_t >( lwork );
    auto          dev_info   = device_alloc< int >( 1 );
    auto          one        = make_constant< value_t >( 1 );
    auto          zero       = make_constant< value_t >( 0 );

    auto  dbg_A = blas::matrix< value_t >( 2*bs, 2*bs );
    auto  dbg_W = blas::vector< real_t >( 2*bs );
    
    #define  DEV_M( i, j ) dev_M[ j*nbrows+i ]
    #define  DEV_V( i, j ) dev_V[ j*nbrows+i ]
    
    //
    // allocate block wise M/V and copy M to device and initialize V
    // also determine initial norm of blocks in M
    //

    fill( value_t(1), vone );
    
    for ( size_t  i = 0; i < nbrows; ++i )
    {
        for ( size_t  j = 0; j < nbcols; ++j )
        {
            DEV_M(i,j) = device_alloc< value_t >( bs * bs );
            DEV_V(i,j) = device_alloc< value_t >( bs * bs );

            const auto  r_i  = blas::range( i*bs, (i+1)*bs-1 );
            const auto  r_j  = blas::range( j*bs, (j+1)*bs-1 );
            auto        M_ij = blas::matrix< value_t >( M, r_i, r_j );

            to_device( M_ij, DEV_M(i,j), bs );

            norms(i,j) = norm_2( handle, bs*bs, DEV_M(i,j), 1 );
            std::cout << norms(i,j) << " : " << blas::norm_F( M_ij ) << std::endl;
            
            if ( i == j )
                to_device( vone, DEV_V(i,j), bs+1 );
        }// for
    }// for
    
    //
    // block wise Jacobi
    //

    real_t  norm_off = real_t(0);

    if ( verbosity > 0 )
        std::cout << "    sweep       off        red        error" << std::endl;
    
    while ( ! converged && ( sweep < max_sweeps ))
    {
        sweep++;

        //
        // sort norm of blocks
        //

        auto    norm_idxs = std::list< std::tuple< real_t, uint, uint > >();
        real_t  norm_sum  = real_t(0);

        for ( uint  i = 0; i < nbrows-1; i++ )
        {
            for ( uint  j = i + 1; j < nbcols; j++ )
            {
                norm_sum += 2 * math::square( norms(i,j) );
                norm_idxs.push_back( { norms(i,j), i, j } ); 
            }// for
        }// for

        norm_sum = math::sqrt( norm_sum );
        
        norm_idxs.sort( [] ( const auto &  n1, const auto &  n2 )
                        {
                            // reverse order for big to small!
                            return std::get<0>( n1 ) > std::get<0>( n2 );
                        } );

        if ( verbosity > 2 )
        {
            for ( auto [ n, i, j ] : norm_idxs )
                std::cout << "        " << i << " / " << j << " = " << n << std::endl;
        }// if
        
        //
        // check norm of off-diagonal part and stop if threshold was reached
        //
        
        const auto  max_norm = std::get<0>( norm_idxs.front() );
        const auto  max_i    = std::get<1>( norm_idxs.front() );
        const auto  max_j    = std::get<2>( norm_idxs.front() );
        const auto  error    = max_norm / std::sqrt( norms( max_i, max_i ) * norms( max_j, max_j ) );

        if ( verbosity > 0 )
            std::cout << "    "
                      << boost::format( " %4d" ) % (sweep-1) << "  "
                      << boost::format( "%.4e" ) % norm_sum << "  "
                      << boost::format( "%.4e" ) % ( sweep > 1 ? norm_sum / norm_off : real_t(0) ) << "  "
                      << boost::format( "%.4e" ) % error << std::endl;

        norm_off = norm_sum;
        
        if ( error < tolerance )
            break;

        //
        // set up block index pairs by successively choosing maximal norm
        // and removing all pairs having same indices
        //
        
        std::deque< std::pair< uint, uint > >  idx_pairs;

        if ( true )
        {
            while ( ! norm_idxs.empty() )
            {
                auto  first = norm_idxs.front();

                norm_idxs.pop_front();

                const auto  i = std::get<1>( first );
                const auto  j = std::get<2>( first );
            
                idx_pairs.push_back( { i, j } );

                // remove all other entries containing i/j
                for ( auto  it = norm_idxs.begin(); it != norm_idxs.end(); )
                {
                    auto  ti = std::get<1>( *it );
                    auto  tj = std::get<2>( *it );
                
                    if (( ti == i ) || ( tj == i ) || ( ti == j ) || ( tj == j ))
                        it = norm_idxs.erase( it );
                    else
                        ++it;
                }// for
            }// while
        }// if
        else
            idx_pairs.push_back( { max_i, max_j } );
        
        //
        // diagonalize ⎛M_ii  M_ij⎞, i.e., compute
        //             ⎝M_ji  M_jj⎠
        //
        // ⎛V_ii  V_ij⎞' ⎛M_ii  M_ij⎞ ⎛V_ii  V_ij⎞ = ⎛D_ii    0 ⎞
        // ⎝V_ji  V_jj⎠  ⎝M_ji  M_jj⎠ ⎝V_ji  V_jj⎠   ⎝  0   D_jj⎠
        //
      
        const auto  npairs = idx_pairs.size();
        auto        Vs     = std::vector< blas::matrix< value_t > >( npairs );
        
        for ( size_t  idx = 0; idx < npairs; ++idx )
        {
            const auto  [ i, j ] = idx_pairs[ idx ];

            if ( verbosity > 1 )
                std::cout << i << " / " << j << std::endl;
                
            //
            // A = [ M_ii, M_ij ; M_ji, M_jj ]
            //
            
            auto  M_ii = DEV_M(i,i);
            auto  M_ij = DEV_M(i,j);
            auto  M_ji = DEV_M(j,i);
            auto  M_jj = DEV_M(j,j);

            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, M_ii + k*bs, 1, dev_A + k*2*bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, M_ij + k*bs, 1, dev_A + (bs+k)*2*bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, M_ji + k*bs, 1, dev_A + k*2*bs + bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, M_jj + k*bs, 1, dev_A + (bs+k)*2*bs + bs, 1 );

            //
            // compute eigenvalues of sub-system
            //
            
            device::eigen_herm( handle, 2*bs, dev_A, dev_W, dev_work, lwork, dev_info );

            //
            // split A
            //
            
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, dev_A + k*2*bs, 1, dev_A00 + k*bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, dev_A + (bs+k)*2*bs, 1, dev_A01 + k*bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, dev_A + k*2*bs + bs, 1, dev_A10 + k*bs, 1 );
            for ( size_t  k = 0; k < bs; ++k )
                copy( handle, bs, dev_A + (bs+k)*2*bs + bs, 1, dev_A11 + k*bs, 1 );

            //
            // apply local eigenvectors (stored in A) to M from left
            //
            
            for ( size_t  l = 0; l < nbcols; ++l )
            {
                //
                // ⎛A₀₀, A₀₁⎞' ⎛M_il⎞
                // ⎝A₁₀, A₁₁⎠  ⎝M_jl⎠
                //

                // T0 = M_il, T1 = M_jl
                copy( handle, bs * bs, DEV_M(i,l), 1, dev_T0, 1 );
                copy( handle, bs * bs, DEV_M(j,l), 1, dev_T1, 1 );
                    
                prod( handle, CUBLAS_OP_C, CUBLAS_OP_N, bs, bs, bs, one, dev_A00, bs, dev_T0, bs, zero, DEV_M(i,l), bs );
                prod( handle, CUBLAS_OP_C, CUBLAS_OP_N, bs, bs, bs, one, dev_A10, bs, dev_T1, bs,  one, DEV_M(i,l), bs );
                
                prod( handle, CUBLAS_OP_C, CUBLAS_OP_N, bs, bs, bs, one, dev_A01, bs, dev_T0, bs, zero, DEV_M(j,l), bs );
                prod( handle, CUBLAS_OP_C, CUBLAS_OP_N, bs, bs, bs, one, dev_A11, bs, dev_T1, bs,  one, DEV_M(j,l), bs );
                
                norms(i,l) = norm_2( handle, bs * bs, DEV_M(i,l), 1 );
                norms(j,l) = norm_2( handle, bs * bs, DEV_M(j,l), 1 );
            }// for

            for ( size_t  l = 0; l < nbrows; ++l )
            {
                //
                // (M_li M_lj) ⎛A₀₀, A₀₁⎞
                //             ⎝A₁₀, A₁₁⎠   
                //

                // T0 = M_li, T1 = M_lj
                copy( handle, bs * bs, DEV_M(l,i), 1, dev_T0, 1 );
                copy( handle, bs * bs, DEV_M(l,j), 1, dev_T1, 1 );
                
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T0, bs, dev_A00, bs, zero, DEV_M(l,i), bs );
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T1, bs, dev_A10, bs,  one, DEV_M(l,i), bs );
                
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T0, bs, dev_A01, bs, zero, DEV_M(l,j), bs );
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T1, bs, dev_A11, bs,  one, DEV_M(l,j), bs );
                
                norms(l,i) = norm_2( handle, bs * bs, DEV_M(l,i), 1 );
                norms(l,j) = norm_2( handle, bs * bs, DEV_M(l,j), 1 );

                //
                // (V_li V_lj) ⎛A₀₀, A₀₁⎞
                //             ⎝A₁₀, A₁₁⎠   
                //

                // T0 = V_li, T1 = V_lj
                copy( handle, bs * bs, DEV_V(l,i), 1, dev_T0, 1 );
                copy( handle, bs * bs, DEV_V(l,j), 1, dev_T1, 1 );
                
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T0, bs, dev_A00, bs, zero, DEV_V(l,i), bs );
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T1, bs, dev_A10, bs,  one, DEV_V(l,i), bs );
                
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T0, bs, dev_A01, bs, zero, DEV_V(l,j), bs );
                prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, bs, bs, bs, one, dev_T1, bs, dev_A11, bs,  one, DEV_V(l,j), bs );
            }// for
        }// for
    }// while

    if ( ! is_null( stat ) )
    {
        stat->nsweeps   = sweep;
        stat->converged = converged;
    }// if
    
    //
    // extract eigenvalues as diagonal elements of M
    //

    blas::vector< value_t >  E( minrc );
    blas::matrix< value_t >  V( nrows, ncols );

    for ( size_t  bi = 0; bi < nbrows; bi++ )
    {
        blas::vector< value_t >  Ei( bs );

        from_device( DEV_M(bi,bi), bs+1, Ei );

        for ( size_t  i = 0; i < bs; ++i )
            E( bi*bs + i ) = Ei(i);
    }// for

    for ( size_t  i = 0; i < nbrows; ++i )
    {
        for ( size_t  j = 0; j < nbcols; ++j )
        {
            const auto  r_i  = blas::range( i*bs, (i+1)*bs-1 );
            const auto  r_j  = blas::range( j*bs, (j+1)*bs-1 );
            auto        V_ij = blas::matrix< value_t >( V, r_i, r_j );

            from_device( DEV_V(i,j), bs, V_ij );
        }// for
    }// for
    
    device_free( dev_A );
    device_free( dev_W );
    device_free( dev_work );
    device_free( dev_info );
    
    for ( size_t  i = 0; i < nbrows; ++i )
    {
        for ( size_t  j = 0; j < nbcols; ++j )
        {
            device_free( DEV_M(i,j) );
            device_free( DEV_V(i,j) );
        }// for
    }// for
    
    return { std::move( E ), std::move( V ) };
}

}}}// hlr::blas::cuda

#endif // __HLR_ARITH_CUDA_HH
