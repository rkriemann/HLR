#ifndef __HLR_UTILS_DETAIL_BLOSC_HH
#define __HLR_UTILS_DETAIL_BLOSC_HH
//
// Project     : HLR
// Module      : utils/detail/blosc
// Description : BLOSC related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HLR_HAS_BLOSC)

#include <cstring>
#include <blosc2.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace compress { namespace blosc {

using byte_t = uint8_t;

//
// define compression mode
//
struct config
{
    byte_t  bitrate;
};

//
// convert precision to bitrate
//
inline
byte_t
eps_to_rate ( const double eps )
{
    return byte_t( std::ceil( std::abs( std::log2( eps ) ) ) );
}

inline
byte_t
tol_to_rate ( const double  tol )
{
    return byte_t( std::max< double >( 1, -std::log2( tol ) + 1 ) );
}

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
    auto  dsize = blosc2_decompress_ctx( dctx, zdata, zsize, dest, nsize * sizeof(double) );

    blosc2_free_ctx( dctx );
    
    if ( dsize < 0 )
    {  HLR_ERROR( "internal error in blosc" ); }
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
        auto  zconf = config{ tol_to_rate( S(l) ) };
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

}}}// namespace hlr::compress::blosc

#endif // HLR_HAS_BLOSC

#endif // __HLR_UTILS_DETAIL_BLOSC_HH
