#ifndef __HLR_UTILS_DETAIL_BLOSC_HH
#define __HLR_UTILS_DETAIL_BLOSC_HH
//
// Project     : HLR
// Module      : compress/blosc
// Description : BLOSC related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#if defined(HLR_HAS_BLOSC)

#include <cstring>
#include <blosc2.h>

#include <hlr/arith/blas.hh>
#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress { namespace blosc {

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_APLR

//
// define compression mode
//
struct config
{
    byte_t  bitrate;
};

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 0, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_aplr ( const double  eps ) { return eps_to_rate( eps ) + 1; }

//
// define various compression modes
//
inline config  get_config  ( const double  eps  ) { return config{ eps_to_rate( eps ) }; }

// holds compressed data
using  byte_t = unsigned char;
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v ) { return sizeof(zarray) + v.size(); }
inline size_t  compressed_size ( const zarray &  v ) { return v.size(); }

//
// compression function
//
template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    if ( nsize == 0 )
        return zarray();
            
    const auto      lastpos = BLOSC2_MAX_FILTERS - 1;
    blosc2_cparams  cparams = BLOSC2_CPARAMS_DEFAULTS;
    
    cparams.typesize         = sizeof( value_t );
    cparams.compcode         = BLOSC_LZ4HC;
    cparams.clevel           = 9;
    cparams.filters[0]       = BLOSC_TRUNC_PREC; // truncate precision bits
    cparams.filters_meta[0]  = config.bitrate;   // number of precision bits
    cparams.filters[lastpos] = BLOSC_BITSHUFFLE; // use bit shuffling
    cparams.nthreads         = 1;                // sequential!

    auto  cctx   = blosc2_create_cctx( cparams );
    auto  buffer = std::vector< byte_t >( nsize * sizeof(value_t) );
    auto  zsize  = blosc2_compress_ctx( cctx, data, sizeof(value_t) * nsize, buffer.data(), buffer.size() );

    blosc2_free_ctx( cctx );

    if      ( zsize == 0 ) return zarray(); // not compressed
    else if ( zsize <  0 ) { HLR_ERROR( "internal error in blosc" ); }

    auto  result = zarray( zsize );

    std::copy( buffer.begin(), buffer.begin() + zsize, result.begin() );

    return result;
}

template <>
inline
zarray
compress< std::complex< float > > ( const config &           config,
                                    std::complex< float > *  data,
                                    const size_t             dim0,
                                    const size_t             dim1,
                                    const size_t             dim2,
                                    const size_t             dim3 )
{
    if      ( dim1 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, 2 );
    else                  return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, dim3 * 2 );
}

template <>
inline
zarray
compress< std::complex< double > > ( const config &            config,
                                     std::complex< double > *  data,
                                     const size_t              dim0,
                                     const size_t              dim1,
                                     const size_t              dim2,
                                     const size_t              dim3 )
{
    if      ( dim1 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, 2 );
    else                  return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, dim3 * 2 );
}

//
// decompression function
//
template < typename value_t >
void
decompress ( const byte_t *  zdata,
             const size_t    zsize,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    const size_t    nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    blosc2_dparams  dparams = BLOSC2_DPARAMS_DEFAULTS;
    
    dparams.nthreads = 1;

    auto  dctx  = blosc2_create_dctx( dparams );
    auto  dsize = blosc2_decompress_ctx( dctx, zdata, zsize, dest, nsize * sizeof(value_t) );

    blosc2_free_ctx( dctx );
    
    if ( dsize < 0 )
    { HLR_ERROR( "internal error in blosc" ); }
    else if ( dsize < nsize * sizeof(value_t) )
    { HLR_ERROR( "insufficient decompression" ); }
}

template < typename value_t >
void
decompress ( const zarray &  buffer,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    decompress( buffer.data(), buffer.size(), dest, dim0, dim1, dim2, dim3 );
}

template <>
inline
void
decompress< std::complex< float > > ( const zarray &           zdata,
                                      std::complex< float > *  dest,
                                      const size_t             dim0,
                                      const size_t             dim1,
                                      const size_t             dim2,
                                      const size_t             dim3 )
{
    if      ( dim1 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, 2 );
    else                  decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, dim3 * 2 );
}
    
template <>
inline
void
decompress< std::complex< double > > ( const zarray &            zdata,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3 )
{
    if      ( dim1 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, 2 );
    else                  decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, dim3 * 2 );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// special version for lowrank matrices
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > &                       U,
              const blas::vector< Hpro::real_type_t< value_t > > &  S )
{
    //
    // first, determine exponent bits and mantissa bits for all
    // columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    size_t        zsize = 0;
    auto          zlist = std::vector< zarray >( k );

    for ( uint  l = 0; l < k; ++l )
    {
        auto  zconf = config{ eps_to_rate_aplr( S(l) ) };
        auto  z_i   = compress( zconf, U.data() + l * n, n );

        zsize += z_i.size();
        zlist[l] = std::move( z_i );
    }// for

    zarray  zdata( zsize + sizeof(size_t) * k );
    size_t  pos = 0;

    for ( auto &  z_i : zlist )
    {
        const size_t  s_i = z_i.size();
        
        memcpy( zdata.data() + pos, & s_i, sizeof(size_t) );
        pos += sizeof(size_t);
        
        memcpy( zdata.data() + pos, z_i.data(), s_i );
        pos += s_i;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    const size_t  n   = U.nrows();
    const uint    k   = U.ncols();
    size_t        pos = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        size_t  s_i = 0;

        memcpy( & s_i, zdata.data() + pos, sizeof(size_t) );
        pos += sizeof(size_t);
        
        decompress( zdata.data() + pos, s_i, U.data() + l*n, n );
        pos += s_i;
    }// for
}

////////////////////////////////////////////////////////////////////////////////
//
// compressed blas
//
////////////////////////////////////////////////////////////////////////////////

namespace
{

template < typename value_t >
void
mulvec ( blosc2_context *  ctx,    // BLOSC decompression context
         const void *      zdata,  // pointer to compressed data
         const int32_t     zsize,  // size of compressed data
         const int32_t     zofs,   // offset within compressed data
         const size_t      nrows,
         const size_t      ncols,
         const matop_t     op_A,
         const value_t     alpha,
         const value_t *   x,
         value_t *         y )
{
    //
    // set up read-ahead buffer size as either multiple of nrows
    // or less than nrows
    
    size_t  nbuf     = 16384; // size of read-ahead buffer
    size_t  col_step = 1;

    if ( nbuf > nrows )
    {
        // ensure full columns
        col_step = std::min( nbuf / nrows, ncols );
        nbuf     = col_step * nrows;
    }// if
    else
        nbuf = std::min( nbuf, nrows );
    
    auto  fbuf = std::vector< value_t >( nbuf );
    auto  fptr = fbuf.data();

    if ( col_step > 1 )
    {
        const size_t  ncols_buf = (ncols / col_step) * col_step; // upper limit of cols for <nbuf> step width
        
        switch ( op_A )
        {
            case  apply_normal :
            {
                size_t  pos = zofs;
                size_t  j   = 0;
                
                for ( ; j < ncols_buf; j += col_step, pos += nbuf )
                {
                    // decompress multiple columns
                    const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nbuf, fptr, nbuf * sizeof(value_t) );

                    HLR_ASSERT( dsize == nbuf * sizeof(value_t) );

                    blas::gemv( 'N', nrows, col_step, alpha, fptr, nrows, x + j, 1, value_t(1), y, 1 );
                }// for

                if ( j < ncols )
                {
                    // handle remaining columns
                    const auto  nrest_cols = ncols - j;
                    const auto  nrest      = nrest_cols * nrows;
                    const auto  dsize      = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nrest, fptr, nrest * sizeof(value_t) );
                    
                    HLR_ASSERT( dsize == nrest * sizeof(value_t) );
                    
                    blas::gemv( 'N', nrows, nrest_cols, alpha, fptr, nrows, x + j, 1, value_t(1), y, 1 );
                }// if
            }// case
            break;
        
            case  apply_adjoint :
            {
                size_t  pos = zofs;
                size_t  j   = 0;
                
                for ( ; j < ncols_buf; j += col_step, pos += nbuf )
                {
                    // decompress multiple columns
                    const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nbuf, fptr, nbuf * sizeof(value_t) );

                    HLR_ASSERT( dsize == nbuf * sizeof(value_t) );

                    blas::gemv( 'C', nrows, col_step, alpha, fptr, nrows, x, 1, value_t(1), y + j, 1 );
                }// for
                
                if ( j < ncols )
                {
                    // decompress multiple columns
                    const auto  nrest_cols = ncols - j;
                    const auto  nrest      = nrest_cols * nrows;
                    const auto  dsize      = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nrest, fptr, nrest * sizeof(value_t) );
                    
                    HLR_ASSERT( dsize == nrest * sizeof(value_t) );
                    
                    blas::gemv( 'C', nrows, nrest_cols, alpha, fptr, nrows, x, 1, value_t(1), y + j, 1 );
                }// if
            }// case
            break;

            default:
                HLR_ERROR( "TODO" );
        }// switch
    }// if
    else
    {
        const size_t  nrows_buf = (nrows / nbuf) * nbuf;         // upper limit of rows for <nbuf> step width
        
        switch ( op_A )
        {
            case  apply_normal :
            {
                size_t  pos = zofs;
            
                for ( size_t  j = 0; j < ncols; ++j )
                {
                    const auto  x_j = alpha * x[j];
                    size_t      i   = 0;

                    for ( ; i < nrows_buf; i += nbuf, pos += nbuf )
                    {
                        const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nbuf, fptr, nbuf * sizeof(value_t) );

                        HLR_ASSERT( dsize == nbuf * sizeof(value_t) );
                    
                        blas::axpy( nbuf, x_j, fptr, 1, y + i, 1 );
                        
                        // #pragma GCC ivdep
                        // for ( uint  ii = 0; ii < nbuf; ++ii )
                        //     y[i+ii] += fptr[ii] * x_j;
                    }// for

                    // handle remaining part
                    {
                        const auto  nrest = nrows - i;
                        const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nrest, fptr, nrest * sizeof(value_t) );

                        HLR_ASSERT( dsize == nrest * sizeof(value_t) );
                    
                        blas::axpy( nrest, x_j, fptr, 1, y + i, 1 );
                        
                        // #pragma GCC ivdep
                        // for ( uint  ii = 0; ii < nrest; ++ii )
                        //     y[i+ii] += fptr[ii] * x_j;
                    }// for
                }// for
            }// case
            break;
        
            case  apply_adjoint :
            {
                size_t  pos = zofs;
            
                for ( size_t  j = 0; j < ncols; ++j )
                {
                    value_t  y_j = value_t(0);
                    size_t   i   = 0;

                    for ( ; i < nrows_buf; i += nbuf, pos += nbuf )
                    {
                        const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nbuf, fptr, nbuf * sizeof(value_t) );
                        
                        HLR_ASSERT( dsize == nbuf * sizeof(value_t) );
                    
                        for ( uint  ii = 0; ii < nbuf; ++ii )
                            y_j += fptr[ii] * x[i+ii];
                    }// for

                    // handle remaining part
                    {
                        const auto  nrest = nrows - i;
                        const auto  dsize = blosc2_getitem_ctx( ctx, zdata, zsize, pos, nrest, fptr, nrest * sizeof(value_t) );
                    
                        HLR_ASSERT( dsize == nrest * sizeof(value_t) );
                    
                        for ( uint  ii = 0; ii < nrest; ++ii )
                            y_j += fptr[ii] * x[i+ii];
                    }

                    y[j] += alpha * y_j;
                }// for
            }// case
            break;

            default:
                HLR_ERROR( "TODO" );
        }// switch
    }// if
}

}// namespace anonymous

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    blosc2_dparams  dparams = BLOSC2_DPARAMS_DEFAULTS;
    
    dparams.nthreads = 1;
    
    auto  dctx = blosc2_create_dctx( dparams );

    mulvec< value_t >( dctx, zA.data(), zA.size(), 0, nrows, ncols, op_A, alpha, x, y );

    blosc2_free_ctx( dctx );
}

template < typename value_t >
void
mulvec_lr ( const size_t     nrows,
            const size_t     ncols,
            const matop_t    op_A,
            const value_t    alpha,
            const zarray &   zA,
            const value_t *  x,
            value_t *        y )
{
    blosc2_dparams  dparams = BLOSC2_DPARAMS_DEFAULTS;
    
    dparams.nthreads = 1;
    
    auto    dctx = blosc2_create_dctx( dparams );
    size_t  pos  = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                auto  s_i = * reinterpret_cast< const size_t * >( zA.data() + pos );

                pos += sizeof(size_t);
                mulvec( dctx, zA.data() + pos, s_i, 0, nrows, 1, op_A, alpha, x+l, y );
                pos += s_i;
            }// for
        }// case
        break;
        
        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                auto  s_i = * reinterpret_cast< const size_t * >( zA.data() + pos );

                pos += sizeof(size_t);
                mulvec( dctx, zA.data() + pos, s_i, 0, nrows, 1, op_A, alpha, x, y+l );
                pos += s_i;
            }// for
        }// case
        break;
    }// switch

    blosc2_free_ctx( dctx );
}

}}}// namespace hlr::compress::blosc

#endif // HLR_HAS_BLOSC

#endif // __HLR_UTILS_DETAIL_BLOSC_HH
