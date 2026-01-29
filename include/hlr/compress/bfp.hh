#ifndef __HLR_UTILS_DETAIL_BFP_HH
#define __HLR_UTILS_DETAIL_BFP_HH
//
// Project     : HLR
// Module      : compress/bfp
// Description : functions for block floating point representation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2026. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/compress/byte_n.hh>
#include <hlr/compress/ztypes.hh>
#include <hlr/compress/bitstream.hh>

//
// options
//

// #define HLR_BFP_BUFFERED_MVM // (disabled by default as it seems slower)
// #define HLR_BFP_BITSTREAM

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_VALR

////////////////////////////////////////////////////////////
//
// compression using block floating point representation
//
// - use common scale
// - #bits defined by dynamic range and precision
// - scale input values to fit into [-imax,imax] with imax = 2^(#bits-1)
// - #bits rounded up to next multiple of 8
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bfp {

using byte_t = uint8_t;

//
// compression config
//
template < typename real_t >
struct Zconf
{};

template <>
struct Zconf< float >
{
    constexpr static uint8_t   scale_ofs  = 4;
    constexpr static uint8_t   header_ofs = 8;
    constexpr static float     minval     = 1e-30;

    using  bs_storage_t = uint32_t;
};
    
template <>
struct Zconf< double >
{
    constexpr static uint8_t   scale_ofs  = 4;
    constexpr static uint8_t   header_ofs = 12;
    constexpr static double    minval     = 1e-50;

    using  bs_storage_t = uint64_t;
};

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline uint8_t  eps_to_rate      ( const double  eps ) { return std::max< uint8_t >( 0, std::ceil( -std::log2( eps ) ) ); }
inline uint8_t  eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ); }

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

// return actual memory size of compressed data
inline size_t  byte_size       ( const zarray &  v ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v ) { return v.size(); }

// return compression configuration for desired accuracy eps
inline config  get_config ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

////////////////////////////////////////////////////////////////////////////////
//
// helper functions
//
////////////////////////////////////////////////////////////////////////////////

//
// compute min/max non-zero(!) values of given data
//
template < typename value_t >
// __attribute__ ((target ("default")))
std::pair< Hpro::real_type_t< value_t >,   // min
           Hpro::real_type_t< value_t > >  // max
nzmin_max ( const value_t *  data,
            const size_t     nsize )
{
    using  real_t = Hpro::real_type_t< value_t >;

    auto  vmin = FPinfo< real_t >::maximum;
    auto  vmax = real_t(0);

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i < Zconf< real_t >::minval ? FPinfo< real_t >::maximum : d_i );

        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_ASSERT( vmin > real_t(0) );

    return { vmin, vmax };
}

//
// return number of bits needed to represent given dynamic range
//
inline
uint8_t
nexpbits ( const auto  drange )
{
    return uint8_t( std::max< decltype( drange ) >( 0, std::ceil( std::log2( std::log2( drange ) ) ) ) );
}

//
// return number of bits for given dynamic range and precision
//
inline
uint8_t
nzbits ( const auto  drange,
         const uint  bitrate )
{
    #if defined(HLR_BFP_BITSTREAM)
    return 1 + nexpbits( drange ) + 1 + bitrate;
    #else
    return byte_pad( 1 + nexpbits( drange ) + bitrate );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

//
// compress data as float
//
template < typename storage_t >
void
compress ( const float *  data,
           const size_t   nsize,
           byte_t *       zdata,
           float          scale,
           const uint8_t  nbits )
{
    using  value_t   = float;
    using  integer_t = Zconf< value_t >::bs_storage_t;
    
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    #if defined(HLR_BFP_BITSTREAM)
    const uint8_t        nbyte = nbits / 8;
    const integer_t      imask = ( integer_t(1) << nbits ) - integer_t(1);  // mask to extract nbyte integer value (also maximal unsigned integer value)
    const value_t        imax  = imask / 2;                                 // maximal signed integer value
    #else
    constexpr uint8_t    nbyte = sizeof(storage_t);
    constexpr auto       imax  = value_t( ( integer_t(1) << ( 8*nbyte - 1 ) ) - 1 );
    constexpr integer_t  imask = ( 0xFFFFFFFF >> 8 * ( sizeof(integer_t) - nbyte ) );
    #endif

    // adjust scaling for integer max
    scale = scale * imax;

    HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    // store number of bits and scaling factor for decompression
    #if defined(HLR_BFP_BITSTREAM)
    zdata[0] = nbits;
    #else
    zdata[0] = nbyte;
    #endif
    
    memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    zdata += Zconf< value_t >::header_ofs;
    
    //
    // in case of 4 byte, just copy data
    //

    if ( nbyte == sizeof(value_t) )
    {
        std::copy( data, data + nsize, reinterpret_cast< value_t * >( zdata ) );
        return;
    }// if
    
    //
    // compress data
    //

    #if defined(HLR_BFP_BITSTREAM)
    const size_t  bssize = pad_bs< integer_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< integer_t >( zdata, bssize );
    #else
    auto          zptr   = reinterpret_cast< storage_t * >( zdata );
    #endif

    #if defined(HLR_BFP_BITSTREAM)
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        bs.write_bits( integer_t( data[i] * scale + imax ) & imask, nbits );
        
    #else

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        zptr[i] = integer_t( data[i] * scale + imax ) & imask;

    #endif
}

template < typename storage_t >
void
decompress ( float *         data,
             const size_t    nsize,
             const byte_t *  zdata )
{
    using  value_t   = float;
    using  integer_t = Zconf< value_t >::bs_storage_t;
    
    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    #if defined(HLR_BFP_BITSTREAM)
    const auto  nbits = zdata[0];
    const auto  nbyte = nbits / 8;
    #else
    const auto  nbyte = zdata[0];
    #endif
    
    HLR_ASSERT( nbyte <= sizeof(value_t) );

    if ( nbyte == 0 )
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            data[i] = value_t(0);

        return;
    }// if
    else if ( nbyte == sizeof(value_t) )
    {
        // just retrieve data
        std::copy( reinterpret_cast< const value_t * >( zdata + Zconf< value_t >::header_ofs ),
                   reinterpret_cast< const value_t * >( zdata + Zconf< value_t >::header_ofs ) + nsize,
                   data );
        return;
    }// if
         
    value_t  scale;

    // get scaling factor
    memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    scale = 1.0 / scale; // for multiplication below

    zdata += Zconf< value_t >::header_ofs;
    
    //
    // decompress data
    //
        
    #if defined(HLR_BFP_BITSTREAM)

    const integer_t  imax   = integer_t(1) << (nbits-1);        // maximal signed integer value
    const size_t     bssize = pad_bs< integer_t >( byte_pad( nsize * nbits ) / 8 );
    auto             bs     = bitstream< integer_t >( const_cast< byte_t * >( zdata ), bssize );
    
    #else
    
    constexpr auto  imax = value_t( ( integer_t(1) << ( 8*sizeof(storage_t) - 1 ) ) - 1 );
    auto            zptr = reinterpret_cast< const storage_t * >( zdata );
    
    #endif

    #if defined(HLR_BFP_BITSTREAM)
        
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = ( value_t( integer_t( bs.read_bits( nbits ) ) ) - imax ) * scale;
        
    #else
        
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = ( value_t( integer_t(zptr[i]) ) - imax ) * scale;
        
    #endif
}

//
// compress data needing more than 32 bits
//
template < typename storage_t >
void
compress ( const double *  data,  // points to actual start of buffer
           const size_t    nsize,
           byte_t *        zdata,
           double          scale,
           const uint8_t   nbits )
{
    using  value_t   = double;
    using  integer_t = Zconf< value_t >::bs_storage_t;
   
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    #if defined(HLR_BFP_BITSTREAM)
    const uint8_t        nbyte = nbits / 8;
    const integer_t      imask = ( integer_t(1) << nbits ) - integer_t(1);  // mask to extract nbyte integer value (also maximal unsigned integer value)
    const value_t        imax  = imask / 2;                                 // maximal signed integer value
    #else
    constexpr uint8_t    nbyte = sizeof(storage_t);
    constexpr auto       imax  = value_t( ( integer_t(1) << ( 8*nbyte - 1 ) ) - 1 );
    constexpr integer_t  imask = ( 0xFFFFFFFFFFFFFFFF >> 8 * ( sizeof(integer_t) - nbyte ) );
    #endif

    // adjust scaling for integer max
    scale = scale * imax;

    HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    // store number of bits and scaling factor for decompression
    #if defined(HLR_BFP_BITSTREAM)
    zdata[0] = nbits;
    #else
    zdata[0] = nbyte;
    #endif
    
    memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    zdata += Zconf< value_t >::header_ofs;
    
    //
    // in case of 8 byte, just copy data
    //

    if ( nbyte == sizeof(value_t) )
    {
        std::copy( data, data + nsize, reinterpret_cast< value_t * >( zdata ) );
        return;
    }// if
    
    //
    // compress data
    //

    #if defined(HLR_BFP_BITSTREAM)
    const size_t  bssize = pad_bs< integer_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< integer_t >( zdata, bssize );
    #else
    auto          zptr   = reinterpret_cast< storage_t * >( zdata );
    #endif

    #if defined(HLR_BFP_BITSTREAM)
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        bs.write_bits( integer_t( data[i] * scale + imax ) & imask, nbits );
        
    #else

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        zptr[i] = integer_t( data[i] * scale + imax ) & imask;

    #endif
}

template < typename storage_t >
void
decompress ( double *        data,
             const size_t    nsize,
             const byte_t *  zdata )
{
    using  value_t   = double;
    using  integer_t = Zconf< value_t >::bs_storage_t;
    
    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    #if defined(HLR_BFP_BITSTREAM)
    const auto  nbits = zdata[0];
    const auto  nbyte = nbits / 8;
    #else
    const auto  nbyte = zdata[0];
    #endif
    
    HLR_ASSERT( nbyte <= sizeof(value_t) );

    if ( nbyte == 0 )
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            data[i] = value_t(0);

        return;
    }// if
    else if ( nbyte == sizeof(value_t) )
    {
        // just retrieve data 
        std::copy( reinterpret_cast< const value_t * >( zdata + Zconf< value_t >::header_ofs ),
                   reinterpret_cast< const value_t * >( zdata + Zconf< value_t >::header_ofs ) + nsize,
                   data );
        return;
    }// if
         
    value_t  scale;

    // get scaling factor
    memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    scale = 1.0 / scale; // for multiplication below

    zdata += Zconf< value_t >::header_ofs;
    
    //
    // decompress data
    //
        
    #if defined(HLR_BFP_BITSTREAM)

    const integer_t  imax   = integer_t(1) << (nbits-1);        // maximal signed integer value
    const size_t     bssize = pad_bs< integer_t >( byte_pad( nsize * nbits ) / 8 );
    auto             bs     = bitstream< integer_t >( const_cast< byte_t * >( zdata ), bssize );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = ( value_t( integer_t( bs.read_bits( nbits ) ) ) - imax ) * scale;
    
    #else
    
    constexpr auto  imax  = value_t( ( integer_t(1) << ( 8*sizeof(storage_t) - 1 ) ) - 1 );
    auto            zptr   = reinterpret_cast< const storage_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = ( value_t( integer_t(zptr[i]) ) - imax ) * scale;

    #endif
}

//
// auxiliary functions to avoid compiler warning due to too large bit shifts
//
inline
void
call_compress ( const float *  data,
                const size_t   nsize,
                byte_t *       zdata,
                float          scale,
                const uint8_t  nbyte )
{
    switch ( nbyte )
    {
        case  1 : compress< byte1_t >( data, nsize, zdata, scale, 0 ); break;
        case  2 : compress< byte2_t >( data, nsize, zdata, scale, 0 ); break;
        case  3 : compress< byte3_t >( data, nsize, zdata, scale, 0 ); break;
        case  4 : compress< byte4_t >( data, nsize, zdata, scale, 0 ); break;
        default : HLR_ERROR( "invalid storage size" );
    }// switch
}

inline
void
call_compress ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata,
                double          scale,
                const uint8_t   nbyte )
{
    switch ( nbyte )
    {
        case  1 : compress< byte1_t >( data, nsize, zdata, scale, 0 ); break;
        case  2 : compress< byte2_t >( data, nsize, zdata, scale, 0 ); break;
        case  3 : compress< byte3_t >( data, nsize, zdata, scale, 0 ); break;
        case  4 : compress< byte4_t >( data, nsize, zdata, scale, 0 ); break;
        case  5 : compress< byte5_t >( data, nsize, zdata, scale, 0 ); break;
        case  6 : compress< byte6_t >( data, nsize, zdata, scale, 0 ); break;
        case  7 : compress< byte7_t >( data, nsize, zdata, scale, 0 ); break;
        case  8 : compress< byte8_t >( data, nsize, zdata, scale, 0 ); break;
        default : HLR_ERROR( "invalid storage size" );
    }// switch
}

inline
void
call_decompress ( float *         data,
                  const size_t    nsize,
                  const byte_t *  zdata,
                  const uint8_t   nbyte )
{
    switch ( nbyte )
    {
        case  1 : decompress< byte1_t >( data, nsize, zdata ); break;
        case  2 : decompress< byte2_t >( data, nsize, zdata ); break;
        case  3 : decompress< byte3_t >( data, nsize, zdata ); break;
        case  4 : decompress< byte4_t >( data, nsize, zdata ); break;
        default : HLR_ERROR( "invalid storage size" );
    }// switch
}

inline
void
call_decompress ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata,
                  const uint8_t   nbyte )
{
    switch ( nbyte )
    {
        case  1 : decompress< byte1_t >( data, nsize, zdata ); break;
        case  2 : decompress< byte2_t >( data, nsize, zdata ); break;
        case  3 : decompress< byte3_t >( data, nsize, zdata ); break;
        case  4 : decompress< byte4_t >( data, nsize, zdata ); break;
        case  5 : decompress< byte5_t >( data, nsize, zdata ); break;
        case  6 : decompress< byte6_t >( data, nsize, zdata ); break;
        case  7 : decompress< byte7_t >( data, nsize, zdata ); break;
        case  8 : decompress< byte8_t >( data, nsize, zdata ); break;
        default : HLR_ERROR( "invalid storage size" );
    }// switch
}

////////////////////////////////////////////////////////////////////////////////
//
// compression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;

    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    // determine min/max value (> 0!)
    const auto  [ vmin, vmax ] = nzmin_max( data, nsize );
    
    if ( vmax == real_t(0) )
    {
        //
        // in case of zero data, return special data
        //

        auto  zdata = std::vector< byte_t >( 2 );
        
        zdata[0] = 0;
        zdata[1] = 0;

        return zdata;
    }// if

    constexpr auto  nmaxbits = sizeof(real_t) * 8;
    const auto      scale    = real_t(1) / vmax;                                             // scale all values v_i such that |v_i| >= 1
    const auto      nbits    = std::min< uint >( nmaxbits, nzbits( vmax / vmin, config.bitrate ) );  // rounded up total no. of bits per value

    HLR_DBG_ASSERT( std::isfinite( scale ) );
    HLR_ASSERT( nbits <= sizeof(real_t) * 8 );

    #if defined(HLR_BFP_BITSTREAM)
    const size_t    nbytes   = pad_bs< integer_t >( byte_pad( nsize * nbits ) / 8 );
    #else
    const size_t    nbyte    = nbits / 8;
    const size_t    nbytes   = nsize * nbyte;
    #endif
    
    auto            zdata    = std::vector< byte_t >( Zconf< real_t >::header_ofs + nbytes ); // array storing compressed data

    #if defined(HLR_BFP_BITSTREAM)

    compress< byte_t >( data, nsize, zdata.data(), scale, nbits );
    
    #else

    call_compress( data, nsize, zdata.data(), scale, nbyte );

    #endif

    // // DEBUG
    // {
    //     std::vector< double >  tmp( nsize );

    //     switch ( nbyte )
    //     {
    //         case  1 : decompress< byte1_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  2 : decompress< byte2_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  3 : decompress< byte3_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  4 : decompress< byte4_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  5 : decompress< byte5_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  6 : decompress< byte6_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  7 : decompress< byte7_t >( tmp.data(), nsize, zdata.data() ); break;
    //         case  8 : decompress< byte8_t >( tmp.data(), nsize, zdata.data() ); break;
    //         default : HLR_ERROR( "invalid storage size" );
    //     }// switch

    //     double  err = 0;
    //     double  nrm = 0;

    //     for ( size_t  i = 0; i < nsize; ++i )
    //     {
    //         const auto  d_i = data[i] - tmp[i];

    //         if ( std::abs( d_i ) > 1e-8 )
    //             std::cout << i << " : " << data[i] << " / " << tmp[i] << " / " << d_i << std::endl;
            
    //         err += d_i * d_i;
    //         nrm += data[i] * data[i];
    //     }// for

    //     std::cout << std::sqrt( err ) << " / " << std::sqrt( err ) / std::sqrt( nrm ) << std::endl;
    // }
    // // DEBUG
    
    return zdata;
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

////////////////////////////////////////////////////////////////////////////////
//
// decompression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    #if defined(HLR_BFP_BITSTREAM)

    decompress< byte_t >( dest, nsize, zdata.data() );

    #else

    const auto  nbyte = zdata[0];

    call_decompress( dest, nsize, zdata.data(), nbyte );
    
    #endif
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
    using  real_t = Hpro::real_type_t< value_t >;
    
    constexpr real_t   fp_maximum  = FPinfo< real_t >::maximum;
    constexpr size_t   header_size = Zconf< real_t >::header_ofs; // sizeof(real_t) + 2;
    constexpr uint8_t  nmaxbits    = sizeof(real_t) * 8;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t    n = U.nrows();
    const uint32_t  k = U.ncols();
    auto            m = std::vector< uint8_t >( k );
    auto            s = std::vector< real_t >( k );
    size_t          zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        HLR_DBG_ASSERT( U.row_stride() == 1 );

        const auto  [ vmin, vmax ] = nzmin_max( U.ptr(0,l), n );
        const auto  nbits          = std::min( nmaxbits, nzbits( vmax / vmin, eps_to_rate_valr( S(l) ) ) );
        
        #if defined(HLR_BFP_BITSTREAM)
        const size_t  nbytes = pad_bs< integer_t >( byte_pad( n * nbits ) / 8 );
        #else
        const size_t  nbytes = ( n * nbits ) / 8;
        #endif

        s[l] = real_t(1) / vmax;
        m[l] = nbits;

        HLR_ASSERT( std::isfinite( s[l] ) );

        zsize += header_size + nbytes;
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const real_t   scale  = s[l];
        const uint8_t  nbits  = m[l];

        #if defined(HLR_BFP_BITSTREAM)
        
        const size_t   nbytes = pad_bs< integer_t >( byte_pad( n * nbits ) / 8 );
        
        compress< byte_t >( U.ptr(0,l), n, zdata.data() + pos, scale, nbits );

        #else

        const uint8_t  nbyte  = nbits / 8;
        const size_t   nbytes = nbyte * n;

        call_compress( U.ptr(0,l), n, zdata.data() + pos, scale, nbyte );
        
        #endif

        // // DEBUG
        // {
        //     auto  tmp = std::vector< value_t >( n );
            
        //     decompress( tmp.data(), n, zdata.data() + pos, nbyte );
            
        //     double  err = 0;
        //     double  nrm = 0;

        //     for ( size_t  i = 0; i < n; ++i )
        //     {
        //         const auto  U_i = U.ptr(0,l)[i];
        //         const auto  d_i = U_i - tmp[i];
            
        //         err += d_i * d_i;
        //         nrm += U_i * U_i;
        //     }// for

        //     std::cout << std::sqrt( err ) << " / " << std::sqrt( err ) / std::sqrt( nrm ) << std::endl;
        // }
        // // DEBUG
        
        pos += header_size + nbytes;
    }// for
    
    return zdata;
}

template <>
inline
zarray
compress_lr< std::complex< float > > ( const blas::matrix< std::complex< float > > &  U,
                                       const blas::vector< float > &                  S )
{
    HLR_ERROR( "TODO" );
}
                                     
template <>
inline
zarray
compress_lr< std::complex< double > > ( const blas::matrix< std::complex< double > > &  U,
                                        const blas::vector< double > &                  S )
{
    HLR_ERROR( "TODO" );
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const size_t      n           = U.nrows();
    const uint32_t    k           = U.ncols();
    size_t            pos         = 0;
    constexpr size_t  header_size = Zconf< real_t >::header_ofs;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        #if defined(HLR_BFP_BITSTREAM)
        
        const uint8_t  nbits = zdata[ pos ];
        
        decompress< byte_t >( U.data() + l * n, n, zdata.data() + pos );

        pos += header_size + pad_bs< integer_t >( byte_pad( nbits * n ) / 8 );
        
        #else

        const uint8_t  nbyte = zdata[ pos ];
        
        call_decompress( U.data() + l * n, n, zdata.data() + pos, nbyte );
        
        pos += header_size + nbyte * n;
        
        #endif
    }// for
}

template <>
inline
void
decompress_lr< std::complex< float > > ( const zarray &                           zdata,
                                         blas::matrix< std::complex< float > > &  U )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
decompress_lr< std::complex< double > > ( const zarray &                            zdata,
                                          blas::matrix< std::complex< double > > &  U )
{
    HLR_ERROR( "TODO" );
}

//
// compressed blas
//

namespace
{

template < typename value_t,
           typename storage_t >
void
mulvec ( const size_t                        nrows,
         const size_t                        ncols,
         const matop_t                       op_A,
         const value_t                       alpha,
         const Hpro::real_type_t< value_t >  zscale,
         const storage_t *                   zA,
         const value_t *                     x,
         value_t *                           y,
         const uint8_t                       nbits )
{
    using  integer_t = Zconf< value_t >::bs_storage_t;
    
    #if defined(HLR_BFP_BITSTREAM)
    
    const auto      imax   = value_t( integer_t(1) << (nbits-1) );        // maximal signed integer value
    const size_t    bssize = pad_bs< integer_t >( byte_pad( nrows * ncols * nbits ) / 8 );
    auto            bs     = bitstream< integer_t >( const_cast< byte_t * >( zA ), bssize );
    
    #else
    
    constexpr auto  imax   = value_t( ( uint64_t(1) << ( 8*sizeof(storage_t) - 1 ) ) - 1 );
    
    #endif
    
    const auto      scale  = alpha * zscale;
    
    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = scale * x[j];
                
                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                {
                    #if defined(HLR_BFP_BITSTREAM)
                    y[i] += ( double( bs.read_bits( nbits ) ) - imax ) * x_j;
                    #else
                    y[i] += ( double( zA[pos] ) - imax ) * x_j;
                    #endif
                }// for
            }// for
        }// case
        break;
        
        case  apply_adjoint :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                value_t  y_j = value_t(0);
                
                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                {
                    #if defined(HLR_BFP_BITSTREAM)
                    y_j += ( double( bs.read_bits( nbits ) ) - imax ) * x[i];
                    #else
                    y_j += ( double( zA[pos] ) - imax ) * x[i];
                    #endif
                }// for

                y[j] += scale * y_j;
            }// for
        }// case
        break;

        default:
            HLR_ERROR( "TODO" );
    }// switch
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
    using  real_t = Hpro::real_type_t< value_t >;

    const real_t      scale    = real_t(1) / ( * ( reinterpret_cast< const real_t * >( zA.data() + Zconf< real_t >::scale_ofs ) ) );
    constexpr size_t  data_ofs = Zconf< real_t >::header_ofs;

    #if defined(HLR_BFP_BITSTREAM)

    const uint8_t     nbits    = zA[0];

    mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte_t * >( zA.data() + data_ofs ), x, y, nbits );
    
    #else
    
    const uint8_t     nbyte    = zA[0];
    
    switch ( nbyte )
    {
        case  1 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  2 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  3 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  4 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  5 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  6 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  7 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        case  8 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + data_ofs ), x, y, 0 ); break;
        default : HLR_ERROR( "unsupported byte size" );
    }// switch

    #endif
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
    using  real_t    = Hpro::real_type_t< value_t >;
    using  integer_t = Zconf< value_t >::bs_storage_t;

    constexpr size_t  scale_ofs = Zconf< real_t >::scale_ofs;
    constexpr size_t  data_ofs  = Zconf< real_t >::header_ofs;
    auto              zdata     = zA.data();

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const real_t   scale = real_t(1) / ( * ( reinterpret_cast< const real_t * >( zdata + scale_ofs ) ) );
                
                #if defined(HLR_BFP_BITSTREAM)

                const uint8_t  nbits  = zdata[0];
                const auto     nbytes = pad_bs< integer_t >( byte_pad( nrows * nbits ) / 8 );
                
                mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zdata + data_ofs ), x+l, y, nbits );
    
                #else

                const auto  nbyte  = zdata[0];
                const auto  nbytes = nbyte * nrows;
                
                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zdata + data_ofs ), x+l, y, 0 ); break;
                    default : HLR_ERROR( "unsupported byte size" );
                }// switch

                #endif

                zdata += data_ofs + nbytes;
            }// for
        }// case
        break;
        
        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const real_t   scale = real_t(1) / ( * ( reinterpret_cast< const real_t * >( zdata + scale_ofs ) ) );
                
                #if defined(HLR_BFP_BITSTREAM)

                const uint8_t  nbits  = zdata[0];
                const auto     nbytes = pad_bs< integer_t >( byte_pad( nrows * nbits ) / 8 );
                
                mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zdata + data_ofs ), x, y+l, nbits );

                #else

                const auto  nbyte  = zdata[0];
                const auto  nbytes = nbyte * nrows;
                
                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zdata + data_ofs ), x, y+l, 0 ); break;
                    default : HLR_ERROR( "unsupported byte size" );
                }// switch

                #endif

                zdata += data_ofs + nbytes;
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::bfp

#endif // __HLR_UTILS_DETAIL_BFP_HH
