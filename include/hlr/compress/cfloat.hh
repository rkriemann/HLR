#ifndef __HLR_UTILS_DETAIL_CFLOAT_HH
#define __HLR_UTILS_DETAIL_CFLOAT_HH
//
// Project     : HLR
// Module      : compress/cfloat
// Description : compression functions based on universal::cfloat
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdint>
#include <limits>

#if defined(HLR_HAS_UNIVERSAL)

#include <universal/number/cfloat/cfloat.hpp>
#include <hlr/compress/byte_n.hh>

////////////////////////////////////////////////////////////
//
// compression using adaptive float representation
//
// - exponent size based on exponent range of input
// - scale input D such that |d_i| ≥ 1
// - mantissa size depends on precision and is rounded
//   up to next byte size for more efficient memory I/O
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace cfloat {

using byte_t = uint8_t;

constexpr float   fp32_infinity = std::numeric_limits< float >::infinity();
constexpr double  fp64_infinity = std::numeric_limits< double >::infinity();

template< class T > inline constexpr T  fp_infinity = std::numeric_limits< T >::infinity();
template< class T > inline constexpr T  fp_maximum  = std::numeric_limits< T >::max();

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 0, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_aplr ( const double  eps ) { return eps_to_rate( eps ); }

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

// return actual memory size of compressed data
inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + v.size(); }

inline size_t  compressed_size ( const zarray &  v   ) { return v.size(); }

// return compression configuration for desired accuracy eps
inline config  get_config      ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           int      bitsize,
           int      expsize,
           typename storage_t >
void
to_cfloat ( byte_t *         cptr,
            const value_t *  data,
            const size_t     nsize,
            const value_t    scale )
{
    if constexpr ( expsize == 1 )
    {
        using  cfloat_t = sw::universal::cfloat< bitsize, expsize, storage_t, true, true, false >;

        auto  ptr = reinterpret_cast< cfloat_t * >( cptr );
    
        for ( size_t  i = 0; i < nsize; ++i )
            ptr[i] = cfloat_t( scale * data[i] );
    }// if
    else
    {
        using  cfloat_t = sw::universal::cfloat< bitsize, expsize, storage_t, true, false, false >;

        auto  ptr = reinterpret_cast< cfloat_t * >( cptr );
    
        for ( size_t  i = 0; i < nsize; ++i )
            ptr[i] = cfloat_t( scale * data[i] );
    }// else
}

template < typename value_t,
           int      bitsize,
           int      expsize,
           typename storage_t >
void
from_cfloat ( const byte_t *  cptr,
              value_t *       data,
              const size_t    nsize,
              const value_t   scale )
{
    if constexpr ( expsize == 1 )
    {
        using  cfloat_t = sw::universal::cfloat< bitsize, expsize, storage_t, true, true, false >;

        auto  ptr = reinterpret_cast< const cfloat_t * >( cptr );
    
        for ( size_t  i = 0; i < nsize; ++i )
            data[i] = value_t( ptr[i] ) / scale;
    }// if
    else
    {
        using  cfloat_t = sw::universal::cfloat< bitsize, expsize, storage_t, true, false, false >;

        auto  ptr = reinterpret_cast< const cfloat_t * >( cptr );
    
        for ( size_t  i = 0; i < nsize; ++i )
            data[i] = value_t( ptr[i] ) / scale;
    }// else
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
           const size_t     dim3 = 0 );

template <>
inline
zarray
compress< float > ( const config &   config,
                    float *          data,
                    const size_t     dim0,
                    const size_t     dim1,
                    const size_t     dim2,
                    const size_t     dim3 )
{
    using  value_t = float;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // look for min/max value (> 0!)
    //
    
    value_t  vmin = fp32_infinity;
    value_t  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == value_t(0) ? fp32_infinity : d_i );

        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > value_t(0) );
    
    
    const value_t   scale      = 1.0 / vmax;                                                                   // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits   = std::max< value_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  nbits      = byte_pad( 1 + exp_bits + config.bitrate );                                    // total no. of bits per value
    const uint32_t  nbyte      = nbits / 8;
    const uint32_t  prec_bits  = nbits - 1 - exp_bits;                                                         // actual number of precision bits
    const auto      ofs        = 1 + 1 + sizeof(value_t);                                                      // offset in zdata for real data
    auto            zdata      = std::vector< byte_t >();                                                      // array storing compressed data

    HLR_ASSERT( nbits <= 32 );

    zdata.resize( ofs + nsize * nbyte );
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata.data() + 2, & scale, sizeof(value_t) );
    
    switch ( nbits )
    {
        case  8:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 8, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 8, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 8, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 8, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 8, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 8, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 16:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 16, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 16, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 16, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 16, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 16, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 16, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 16, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 16, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 24:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 24, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 24, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 24, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 24, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 24, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 24, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 24, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 24, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 32:
        {
            using  storage_t = uint32_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 32, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 32, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 32, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 32, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 32, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 32, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 32, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 32, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        default : HLR_ERROR( "TODO" );
    }// switch

    return zdata;
}

template <>
inline
zarray
compress< double > ( const config &   config,
                     double *         data,
                     const size_t     dim0,
                     const size_t     dim1,
                     const size_t     dim2,
                     const size_t     dim3 )
{
    using value_t = double;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // look for min/max value (> 0!)
    //
    
    value_t  vmin = fp64_infinity;
    value_t  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == value_t(0) ? fp64_infinity : d_i );
            
        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > value_t(0) );
    
    
    const value_t   scale     = 1.0 / vmax;                                                                   // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits  = std::max< value_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  nbits     = byte_pad( 1 + exp_bits + config.bitrate );                                    // total no. of bits per value
    const uint32_t  nbyte     = nbits / 8;
    const uint32_t  prec_bits = nbits - 1 - exp_bits;                                                         // actual number of precision bits
    const auto      ofs       = 1 + 1 + sizeof(value_t);                                                      // offset in zdata for real data
    auto            zdata     = std::vector< byte_t >();                                                      // array storing compressed data

    HLR_ASSERT( nbits <= 64 );

    zdata.resize( ofs + nsize * nbyte );
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata.data() + 2, & scale, sizeof(value_t) );

    switch ( nbits )
    {
        case  8:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 8, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 8, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 8, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 8, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 8, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 8, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 16:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 16, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 16, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 16, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 16, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 16, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 16, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 16, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 16, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 24:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 24, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 24, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 24, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 24, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 24, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 24, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 24, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 24, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 32:
        {
            using  storage_t = uint32_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 32, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 32, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 32, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 32, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 32, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 32, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 32, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 32, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 40:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 40, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 40, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 40, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 40, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 40, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 40, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 40, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 40, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 48:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 48, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 48, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 48, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 48, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 48, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 48, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 48, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 48, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 56:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 56, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 56, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 56, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 56, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 56, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 56, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 56, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 56, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 64:
        {
            using  storage_t = uint64_t;
            
            switch ( exp_bits )
            {
                case 1  : to_cfloat< value_t, 64, 1, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 2  : to_cfloat< value_t, 64, 2, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 3  : to_cfloat< value_t, 64, 3, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 4  : to_cfloat< value_t, 64, 4, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 5  : to_cfloat< value_t, 64, 5, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 6  : to_cfloat< value_t, 64, 6, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 7  : to_cfloat< value_t, 64, 7, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                case 8  : to_cfloat< value_t, 64, 8, storage_t >( zdata.data() + ofs, data, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        default : HLR_ERROR( "TODO" );
    }// switch

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
decompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 );

template <>
inline
void
decompress< float > ( const zarray &  zdata,
                      float *         dest,
                      const size_t    dim0,
                      const size_t    dim1,
                      const size_t    dim2,
                      const size_t    dim3 )
{
    using value_t = float;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    auto            scale     = value_t(0);
    const auto      ofs       = 1 + 1 + sizeof(value_t);

    memcpy( & scale, zdata.data() + 2, sizeof(value_t) );
    
    switch ( nbits )
    {
        case  8:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 8, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 8, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 8, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 8, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 8, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 8, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 16:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 16, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 16, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 16, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 16, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 16, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 16, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 16, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 16, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 24:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 24, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 24, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 24, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 24, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 24, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 24, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 24, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 24, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 32:
        {
            using  storage_t = uint32_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 32, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 32, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 32, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 32, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 32, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 32, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 32, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 32, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        default : HLR_ERROR( "TODO" );
    }// switch
}

template <>
inline
void
decompress< double > ( const zarray &  zdata,
                       double *        dest,
                       const size_t    dim0,
                       const size_t    dim1,
                       const size_t    dim2,
                       const size_t    dim3 )
{
    using value_t = double;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    auto            scale     = value_t(0);
    const auto      ofs       = 1 + 1 + sizeof(value_t);

    memcpy( & scale, zdata.data() + 2, sizeof(value_t) );
    
    switch ( nbits )
    {
        case  8:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 8, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 8, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 8, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 8, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 8, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 8, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 16:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 16, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 16, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 16, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 16, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 16, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 16, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 16, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 16, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 24:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 24, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 24, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 24, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 24, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 24, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 24, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 24, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 24, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 32:
        {
            using  storage_t = uint32_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 32, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 32, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 32, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 32, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 32, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 32, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 32, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 32, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 40:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 40, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 40, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 40, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 40, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 40, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 40, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 40, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 40, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 48:
        {
            using  storage_t = uint16_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 48, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 48, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 48, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 48, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 48, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 48, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 48, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 48, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 56:
        {
            using  storage_t = uint8_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 56, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 56, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 56, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 56, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 56, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 56, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 56, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 56, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        case 64:
        {
            using  storage_t = uint64_t;
            
            switch ( exp_bits )
            {
                case 1  : from_cfloat< value_t, 64, 1, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 2  : from_cfloat< value_t, 64, 2, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 3  : from_cfloat< value_t, 64, 3, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 4  : from_cfloat< value_t, 64, 4, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 5  : from_cfloat< value_t, 64, 5, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 6  : from_cfloat< value_t, 64, 6, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 7  : from_cfloat< value_t, 64, 7, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                case 8  : from_cfloat< value_t, 64, 8, storage_t >( zdata.data() + ofs, dest, nsize, scale ); break;
                default : HLR_ERROR( "TODO" );
            }// switch
        }
        break;

        default : HLR_ERROR( "TODO" );
    }// switch
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
    
    constexpr size_t  header_size = 2 * sizeof(real_t);
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n = U.nrows();
    const size_t  k = U.ncols();
    auto          m = std::vector< uint32_t >( k );
    auto          e = std::vector< uint32_t >( k );
    auto          s = std::vector< real_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        auto  vmin = fp_maximum< real_t >;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == real_t(0) ? fp_maximum< real_t > : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = 1.0 / vmax;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = eps_to_rate_aplr( S(l) );

        HLR_ASSERT( std::isfinite( s[l] ) );
        
        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += header_size + n * byte_pad( nbits ) / 8;
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
    auto    U_ptr = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = byte_pad( 1 + exp_bits + prec_bits ); // number of bits per value

        zdata[pos]   = exp_bits;
        zdata[pos+1] = prec_bits;
        memcpy( zdata.data() + pos + 2, & scale, sizeof(value_t) );
        
        pos += header_size;
        
        switch ( nbits )
        {
            case  8:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 8, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 8, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 8, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 8, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 8, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 8, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 16:
            {
                using  storage_t = uint16_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 16, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 16, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 16, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 16, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 16, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 16, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 16, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 16, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 24:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 24, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 24, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 24, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 24, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 24, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 24, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 24, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 24, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 32:
            {
                using  storage_t = uint32_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 32, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 32, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 32, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 32, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 32, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 32, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 32, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 32, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 40:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 40, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 40, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 40, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 40, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 40, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 40, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 40, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 40, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 48:
            {
                using  storage_t = uint16_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 48, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 48, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 48, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 48, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 48, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 48, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 48, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 48, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 56:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 56, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 56, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 56, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 56, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 56, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 56, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 56, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 56, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 64:
            {
                using  storage_t = uint64_t;
            
                switch ( exp_bits )
                {
                    case 1  : to_cfloat< value_t, 64, 1, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 2  : to_cfloat< value_t, 64, 2, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 3  : to_cfloat< value_t, 64, 3, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 4  : to_cfloat< value_t, 64, 4, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 5  : to_cfloat< value_t, 64, 5, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 6  : to_cfloat< value_t, 64, 6, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 7  : to_cfloat< value_t, 64, 7, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    case 8  : to_cfloat< value_t, 64, 8, storage_t >( zdata.data() + pos, U.data() + l*n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            default : HLR_ERROR( "TODO" );
        }// switch
        
        pos += n * byte_pad( nbits ) / 8;
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

    constexpr size_t  header_size = 2 * sizeof(real_t);
    
    const size_t    n   = U.nrows();
    const uint32_t  k   = U.ncols();
    size_t          pos = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = byte_pad( 1 + exp_bits + prec_bits );
        value_t         scale     = 0;

        memcpy( & scale, zdata.data() + pos + 2, sizeof(value_t) );
        pos += header_size;
        
        switch ( nbits )
        {
            case  8:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 8, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 8, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 8, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 8, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 8, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 8, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 16:
            {
                using  storage_t = uint16_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 16, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 16, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 16, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 16, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 16, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 16, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 16, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 16, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 24:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 24, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 24, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 24, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 24, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 24, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 24, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 24, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 24, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 32:
            {
                using  storage_t = uint32_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 32, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 32, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 32, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 32, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 32, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 32, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 32, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 32, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 40:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 40, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 40, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 40, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 40, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 40, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 40, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 40, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 40, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 48:
            {
                using  storage_t = uint16_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 48, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 48, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 48, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 48, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 48, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 48, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 48, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 48, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 56:
            {
                using  storage_t = uint8_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 56, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 56, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 56, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 56, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 56, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 56, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 56, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 56, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            case 64:
            {
                using  storage_t = uint64_t;
            
                switch ( exp_bits )
                {
                    case 1  : from_cfloat< value_t, 64, 1, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 2  : from_cfloat< value_t, 64, 2, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 3  : from_cfloat< value_t, 64, 3, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 4  : from_cfloat< value_t, 64, 4, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 5  : from_cfloat< value_t, 64, 5, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 6  : from_cfloat< value_t, 64, 6, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 7  : from_cfloat< value_t, 64, 7, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    case 8  : from_cfloat< value_t, 64, 8, storage_t >( zdata.data() + pos, U.data() + l * n, n, scale ); break;
                    default : HLR_ERROR( "TODO" );
                }// switch
            }
            break;

            default : HLR_ERROR( "TODO" );
        }// switch
        
        pos += n * byte_pad( nbits ) / 8;
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

}}}// namespace hlr::compress::cfloat

#endif // HLR_HAS_UNIVERSAL

#endif // __HLR_UTILS_DETAIL_CFLOAT_HH
