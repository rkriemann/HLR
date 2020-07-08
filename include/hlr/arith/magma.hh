#ifndef __HLR_ARITH_MAGMA_HH
#define __HLR_ARITH_MAGMA_HH
//
// Project     : HLR
// Module      : arith/magma
// Description : basic linear algebra functions using magma
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cuda.h>
#include <magma_v2.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace blas { namespace magma {

//
// mapping of default types to MAGMA types
//
template < typename T > struct magma_type                            { using  type_t = T; };
template <>             struct magma_type< std::complex< float > >   { using  type_t = magmaFloatComplex; };
template <>             struct magma_type< std::complex< double > >  { using  type_t = magmaDoubleComplex; };

template < typename T > struct magma_type_ptr                            { using  type_t = T *; };
template <>             struct magma_type_ptr< std::complex< float > >   { using  type_t = magmaFloatComplex_ptr; };
template <>             struct magma_type_ptr< std::complex< double > >  { using  type_t = magmaDoubleComplex_ptr; };

//
// devices and queues
//
magma_device_t  devices[ 1 ];
magma_queue_t   queue;

//
// initialize MAGMA
//
void
init ()
{
    magma_int_t  ndevices = 0;

    magma_init();
    magma_getdevices( devices, 1, & ndevices );

    if ( ndevices == 0 )
        HLR_ERROR( "no GPU found" );

    magma_queue_create( devices[0], & queue );
}

namespace detail
{

//
// device memory allocation with MAGMA
//
template < typename value_t >
typename magma_type_ptr< value_t >::type_t
device_alloc ( const size_t  n )
{
    void *  ptr = nullptr;
    
    auto  res = magma_malloc( & ptr, n * sizeof(typename magma_type< value_t >::type_t) );

    if ( res != 0 )
        HLR_ERROR( "error during magma_alloc" );

    return typename magma_type_ptr< value_t >::type_t( ptr );
}

//
// host to device copy
//
template < typename value_t >
void
to_device ( matrix< value_t > &                         M_host,
            typename magma_type_ptr< value_t >::type_t  M_dev,
            magma_int_t                                 lda_dev,
            magma_queue_t                               queue )
{
    magma_setmatrix_internal( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                              M_host.data(), M_host.col_stride(),
                              M_dev, lda_dev, queue,
                              "to_device", __FILE__, __LINE__ );
}

//
// device to host copy
//
template < typename value_t >
void
from_device ( typename magma_type_ptr< value_t >::type_t  M_dev,
              magma_int_t                                 lda_dev,
              matrix< value_t > &                         M_host,
              magma_queue_t                               queue )
{
    magma_getmatrix_internal( M_host.nrows(), M_host.ncols(), sizeof(value_t),
                              M_dev, lda_dev,
                              M_host.data(), M_host.col_stride(),
                              queue,
                              "from_device", __FILE__, __LINE__ );
}

//
// QR factorization wrapper
//
#define HLR_MAGMA_GEQRF( type, func )               \
    inline void                                     \
    magma_geqrf ( const magma_int_t  nrows,         \
                  const magma_int_t  ncols,         \
                  type               M,             \
                  magma_int_t        ldM,           \
                  type               tau,           \
                  type               T,             \
                  magma_int_t *      info )         \
    {                                               \
        func( nrows, ncols, M, ldM, tau, T, info ); \
    }

HLR_MAGMA_GEQRF( magmaFloat_ptr,         magma_sgeqrf_gpu )
HLR_MAGMA_GEQRF( magmaDouble_ptr,        magma_dgeqrf_gpu )
HLR_MAGMA_GEQRF( magmaFloatComplex_ptr,  magma_cgeqrf_gpu )
HLR_MAGMA_GEQRF( magmaDoubleComplex_ptr, magma_zgeqrf_gpu )

template < typename  value_t >
void
geqrf ( const magma_int_t                           nrows,
        const magma_int_t                           ncols,
        typename magma_type_ptr< value_t >::type_t  M,
        magma_int_t                                 ldM,
        value_t *                                   tau,
        typename magma_type_ptr< value_t >::type_t  T )
{
    magma_int_t  info = 0;
    
    magma_geqrf( nrows, ncols, M, ldM, typename magma_type_ptr< value_t >::type_t( tau ), T, & info );
    
    if ( info != 0 )
        HLR_ERROR( "error in magma_get_geqrf_gpu" );
}

#undef HLR_MAGMA_GEQRF

//
// computation of Q
//
#define HLR_MAGMA_UNGQR( type, func )                          \
    void                                                       \
    magma_ungqr ( const magma_int_t  nrows,                    \
                  const magma_int_t  ncols,                    \
                  type               A,                        \
                  const magma_int_t  ldA,                      \
                  type               tau,                      \
                  type               T,                        \
                  const magma_int_t  nb,                       \
                  magma_int_t *      info )                    \
    {                                                          \
        func( nrows, ncols, ncols, A, ldA, tau, T, nb, info ); \
    }

HLR_MAGMA_UNGQR( magmaFloat_ptr,         magma_sorgqr_gpu )
HLR_MAGMA_UNGQR( magmaDouble_ptr,        magma_dorgqr_gpu )
HLR_MAGMA_UNGQR( magmaFloatComplex_ptr,  magma_cungqr_gpu )
HLR_MAGMA_UNGQR( magmaDoubleComplex_ptr, magma_zungqr_gpu )

template < typename  value_t >
void
ungqr ( const magma_int_t                           nrows,
        const magma_int_t                           ncols,
        typename magma_type_ptr< value_t >::type_t  A,
        const magma_int_t                           ldA,
        value_t *                                   tau,
        typename magma_type_ptr< value_t >::type_t  T,
        const magma_int_t                           nb )
{
    magma_int_t  info = 0;
    
    magma_ungqr( nrows, ncols, A, ldA, typename magma_type_ptr< value_t >::type_t( tau ), T, nb, & info );
    
    if ( info != 0 )
        HLR_ERROR( "error in magma_get_or/ungqr" );
}

#undef HLR_MAGMA_UNGQR

//
// wrapper for QR factorization with column pivoting
//
inline void
magma_geqp3 ( const magma_int_t  nrows,
              const magma_int_t  ncols,
              magmaFloat_ptr     M,
              magma_int_t        ldM,
              magma_int_t *      jpvt,
              float *            tau,
              magmaFloat_ptr     dwork,
              magma_int_t        lwork,
              float *            /* rwork */,
              magma_int_t *      info )
{
    magma_sgeqp3_gpu( nrows, ncols, M, ldM, jpvt, tau, dwork, lwork, info );
}

inline void
magma_geqp3 ( const magma_int_t  nrows,
              const magma_int_t  ncols,
              magmaDouble_ptr    M,
              magma_int_t        ldM,
              magma_int_t *      jpvt,
              double *           tau,
              magmaDouble_ptr    dwork,
              magma_int_t        lwork,
              double *           /* rwork */,
              magma_int_t *      info )
{
    magma_dgeqp3_gpu( nrows, ncols, M, ldM, jpvt, tau, dwork, lwork, info );
}

inline void
magma_geqp3 ( const magma_int_t      nrows,
              const magma_int_t      ncols,
              magmaFloatComplex_ptr  M,
              magma_int_t            ldM,
              magma_int_t *          jpvt,
              magmaFloatComplex *    tau,
              magmaFloatComplex_ptr  dwork,
              magma_int_t            lwork,
              float *                rwork,
              magma_int_t *          info )
{
    magma_cgeqp3_gpu( nrows, ncols, M, ldM, jpvt, tau, dwork, lwork, rwork, info );
}

inline void
magma_geqp3 ( const magma_int_t       nrows,
              const magma_int_t       ncols,
              magmaDoubleComplex_ptr  M,
              magma_int_t             ldM,
              magma_int_t *           jpvt,
              magmaDoubleComplex *    tau,
              magmaDoubleComplex_ptr  dwork,
              magma_int_t             lwork,
              double *                rwork,
              magma_int_t *           info )
{
    magma_zgeqp3_gpu( nrows, ncols, M, ldM, jpvt, tau, dwork, lwork, rwork, info );
}

template < typename  value_t >
void
geqp3 ( const magma_int_t                           nrows,
        const magma_int_t                           ncols,
        typename magma_type_ptr< value_t >::type_t  M,
        magma_int_t                                 ldM,
        magma_int_t *                               jpvt,
        value_t *                                   tau,
        typename magma_type_ptr< value_t >::type_t  work,
        magma_int_t                                 lwork,
        typename hpro::real_type< value_t >::type_t *  rwork )
{
    magma_int_t  info = 0;
    
    magma_geqp3( nrows, ncols, M, ldM, jpvt, typename magma_type_ptr< value_t >::type_t( tau ), work, lwork, rwork, & info );
    
    if ( info != 0 )
        HLR_ERROR( "error in magma_get_geqp3_gpu" );
}

}// namespace detail

//
// compute QR factorisation M = Q·R of the n×m matrix M
// - M will be overwritten with Q upon exit
//
template < typename value_t >
void
qr ( matrix< value_t > &  M,
     matrix< value_t > &  R )
{
    const magma_int_t  nrows = magma_int_t( M.nrows() );
    const magma_int_t  ncols = magma_int_t( M.ncols() );
    const magma_int_t  nb    = magma_get_dgeqrf_nb( nrows, ncols ); // CHECK for sgeqrf/cgeqrf/zgeqrf

    // adjust size of R
    if (( magma_int_t(R.nrows()) != ncols ) || ( magma_int_t(R.ncols()) != ncols ))
        R = std::move( matrix< value_t >( ncols, ncols ) );
    
    //
    // allocate memory and copy data
    //
    
    auto  M_dev = detail::device_alloc< value_t >( nrows * ncols );
    auto  T_dev = detail::device_alloc< value_t >( ( 2 * std::min( nrows, ncols ) + magma_roundup( ncols, 32 ) ) * nb  );

    detail::to_device( M, M_dev, nrows, queue );

    auto  tau  = vector< value_t >( ncols );
        
    detail::geqrf( nrows, ncols, M_dev, M.col_stride(), tau.data(), T_dev );
    
    //
    // copy nrows × nrows part of M to R and reset strictly lower left part
    //

    detail::from_device( M_dev, nrows, R, queue );
    
    for ( magma_int_t  i = 0; i < ncols-1; i++ )
    {
        vector< value_t >  col_i( R, range( i+1, ncols-1 ), i );

        fill( value_t(0), col_i );
    }// for
    
    //
    // compute Q
    //
    
    detail::ungqr( nrows, ncols, M_dev, M.col_stride(), tau.data(), T_dev, nb );
    detail::from_device( M_dev, nrows, M, queue );
}

//
// Compute QR factorisation with column pivoting of the matrix M.
//
template < typename value_t >
void
qrp ( matrix< value_t > &           M,
      matrix< value_t > &           R,
      std::vector< magma_int_t > &  P )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    using  hpro::blas_int_t;
    
    const magma_int_t  nrows = magma_int_t( M.nrows() );
    const magma_int_t  ncols = magma_int_t( M.ncols() );
    const magma_int_t  nb    = magma_get_dgeqp3_nb( nrows, ncols ); // CHECK for sgeqrf/cgeqrf/zgeqrf

    if ( magma_int_t( P.size() ) != ncols )
        P.resize( ncols );
    
    if (( magma_int_t( R.nrows() ) != ncols ) || ( magma_int_t( R.ncols() ) != ncols ))
        R = std::move( matrix< value_t >( ncols, ncols ) );
    
    //
    // allocate memory and copy data
    //
    
    auto         M_dev     = detail::device_alloc< value_t >( nrows * ncols );
    magma_int_t  lwork     = magma_roundup( ncols+1, 32 )*nb + magma_roundup( 2*ncols, 32 );
    auto         work_dev  = detail::device_alloc< value_t >( lwork );

    detail::to_device( M, M_dev, nrows, queue );

    //
    // compute Householder vectors and R
    //

    std::vector< value_t >  tau( std::min( nrows, ncols ) );
    std::vector< real_t >   rwork( 2*ncols );
    
    detail::geqp3( nrows, ncols, M_dev, M.col_stride(), P.data(), tau.data(), work_dev, lwork, rwork.data() );

    //
    // copy M back and proceed on CPU
    //

    detail::from_device( M_dev, nrows, M, queue );

    for ( magma_int_t  i = 0; i < ncols; i++ )
    {
        vector< value_t >  colM( M, range( 0, i ), i );
        vector< value_t >  colR( R, range( 0, i ), i );

        copy( colM, colR );
    }// for
    
    //
    // compute Q
    //

    blas_int_t              info = 0;
    std::vector< value_t >  work( 1 );
    
    orgqr< value_t >( nrows, ncols, ncols, M.data(), blas_int_t( M.col_stride() ), tau.data(), work.data(), -1, info );

    work.resize( hpro::re( work[0] ) );
    
    orgqr< value_t >( nrows, ncols, ncols, M.data(), blas_int_t( M.col_stride() ), tau.data(), work.data(), work.size(), info );

    //
    // correct indices in P (1-counted to 0-counted)
    //

    for ( magma_int_t  i = 0; i < nrows; i++ )
        --P[i];
}

}}}// hlr::blas::magma

#endif // __HLR_ARITH_MAGMA_HH
