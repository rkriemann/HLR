#ifndef __HLR_UTILS_DETAIL_POSITS_HH
#define __HLR_UTILS_DETAIL_POSITS_HH
//
// Project     : HLR
// Module      : compress/posits
// Description : posits related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#if defined(HLR_HAS_UNIVERSAL)

#include <cstdint>

#include <universal/number/posit/posit.hpp>

#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress { namespace posits {

using byte_t = uint8_t;

// fixed number of exponent bits
constexpr uint8_t  ES = 2;

//
// return bitrate for given accuracy
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 0, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_aplr ( const double  eps ) { return eps_to_rate( eps ); }

//
// compression configuration
// (just precision bitsize)
//
struct config
{
    uint  bitsize;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v ) { return sizeof(zarray) + v.size(); }
inline size_t  compressed_size ( const zarray &  v ) { return v.size(); }

inline config  get_config ( const double  eps  ) { return config{ eps_to_rate( eps ) }; }

//////////////////////////////////////////////////////////////////////
//
// 0 = bitwise storage / 1 = byte wise storage 
//
//////////////////////////////////////////////////////////////////////

#if 0

// //
// // convert given array <data> into posits and store results in <cptr>
// //
// template < typename value_t, int nbits, typename storage_t >
// struct convert
// {
//     static constexpr uint64_t  mask = ( 1ul << nbits ) - 1ul;
    
//     static void
//     to_posit ( byte_t *         cptr,
//                const value_t *  data,
//                const size_t     nsize,
//                const value_t    scale )
//     {
//         using  posit_t = sw::universal::posit< nbits, ES >;
        
//         auto  ptr = reinterpret_cast< storage_t * >( cptr );
        
//         for ( size_t  i = 0; i < nsize; ++i )
//         {
//             auto  p    = posit_t( data[i] * scale );
//             auto  bits = p.get();

//             ptr[i] = bits.to_ullong() & mask;
//         }// if
//     }// for

//     static void
//     from_posit ( const byte_t *  cptr,
//                  value_t *       data,
//                  const size_t    nsize,
//                  const value_t   scale )
//     {
//         using  posit_t    = sw::universal::posit< nbits, ES >;
//         using  bitblock_t = sw::universal::bitblock< nbits >;

//         auto  ptr = reinterpret_cast< const storage_t * >( cptr );
        
//         for ( size_t  i = 0; i < nsize; ++i )
//         {
//             auto        raw  = ptr[i];
//             // bitblock_t  bits;
//             posit_t     p;
            
//             // bits = uint64_t(raw);
//             // p.setbits( bits );

//             p.setbits( raw );

//             data[i] = value_t( p ) / scale;
//         }// for
//     }
// };

// //
// // compression function
// //
// template < typename value_t >
// zarray
// compress ( const config &   config,
//            value_t *        data,
//            const size_t     dim0,
//            const size_t     dim1 = 0,
//            const size_t     dim2 = 0,
//            const size_t     dim3 = 0 )
// {
//     const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

//     constexpr float   fp32_infinity = std::numeric_limits< float >::infinity();
//     constexpr double  fp64_infinity = std::numeric_limits< double >::infinity();
    
//     //
//     // look for min/max value (> 0!)
//     //
    
//     value_t  vmin = fp64_infinity;
//     value_t  vmax = 0;

//     for ( size_t  i = 0; i < nsize; ++i )
//     {
//         const auto  d_i = std::abs( data[i] );
//         const auto  val = ( d_i == value_t(0) ? value_t(fp64_infinity) : d_i );

//         vmin = std::min( vmin, val );
//         vmax = std::max( vmax, d_i );
//     }// for

//     HLR_DBG_ASSERT( vmin > value_t(0) );
    
//     const value_t  scale = 1.0 / vmax;
//     const auto     nbits = 1 + ES + config.bitsize; // sign bit + exponent bits
//     const auto     pbits = byte_pad( nbits );
//     const auto     ofs   = 1 + sizeof(value_t);
//     zarray         zdata( ofs + nsize * pbits / 8 );

//     zdata[0] = nbits;
//     memcpy( zdata.data() + 1, & scale, sizeof(value_t) );
    
//     switch ( nbits )
//     {
//         case  3: { convert< value_t,  3, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  4: { convert< value_t,  4, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  5: { convert< value_t,  5, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  6: { convert< value_t,  6, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  7: { convert< value_t,  7, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  8: { convert< value_t,  8, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case  9: { convert< value_t,  9, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 10: { convert< value_t, 10, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 11: { convert< value_t, 11, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 12: { convert< value_t, 12, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 13: { convert< value_t, 13, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 14: { convert< value_t, 14, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 15: { convert< value_t, 15, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 16: { convert< value_t, 16, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 17: { convert< value_t, 17, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 18: { convert< value_t, 18, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 19: { convert< value_t, 19, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 20: { convert< value_t, 20, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 21: { convert< value_t, 21, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 22: { convert< value_t, 22, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 23: { convert< value_t, 23, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 24: { convert< value_t, 24, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 25: { convert< value_t, 25, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 26: { convert< value_t, 26, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 27: { convert< value_t, 27, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 28: { convert< value_t, 28, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 29: { convert< value_t, 29, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 30: { convert< value_t, 30, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 31: { convert< value_t, 31, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 32: { convert< value_t, 32, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 33: { convert< value_t, 33, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 34: { convert< value_t, 34, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 35: { convert< value_t, 35, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 36: { convert< value_t, 36, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 37: { convert< value_t, 37, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 38: { convert< value_t, 38, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 39: { convert< value_t, 39, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 40: { convert< value_t, 40, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 48: { convert< value_t, 48, byte6_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         case 56: { convert< value_t, 56, byte7_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
//         // case 64: { convert< value_t, 64, uint64_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;

//         default:
//             HLR_ERROR( "unsupported bitsize " + Hpro::to_string( config.bitsize ) );
//     }// switch

//     return zdata;
// }

// template <>
// inline
// zarray
// compress< std::complex< float > > ( const config &           config,
//                                     std::complex< float > *  data,
//                                     const size_t             dim0,
//                                     const size_t             dim1,
//                                     const size_t             dim2,
//                                     const size_t             dim3 )
// {
//     if      ( dim1 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, 2, 0, 0 );
//     else if ( dim2 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, 2, 0 );
//     else if ( dim3 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, 2 );
//     else                  return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, dim3 * 2 );
// }

// template <>
// inline
// zarray
// compress< std::complex< double > > ( const config &            config,
//                                      std::complex< double > *  data,
//                                      const size_t              dim0,
//                                      const size_t              dim1,
//                                      const size_t              dim2,
//                                      const size_t              dim3 )
// {
//     if      ( dim1 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, 2, 0, 0 );
//     else if ( dim2 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, 2, 0 );
//     else if ( dim3 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, 2 );
//     else                  return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, dim3 * 2 );
// }

// //
// // decompression function
// //
// template < typename value_t >
// void
// decompress ( const zarray &  zdata,
//              value_t *       dest,
//              const size_t    dim0,
//              const size_t    dim1 = 0,
//              const size_t    dim2 = 0,
//              const size_t    dim3 = 0 )
// {
//     const size_t  nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
//     const auto    bitsize = zdata[0];
//     auto          scale   = value_t(0);
//     const auto    ofs     = 1 + sizeof(value_t);

//     memcpy( & scale, zdata.data() + 1, sizeof(value_t) );
    
//     switch ( bitsize )
//     {
//         case  3: { convert< value_t,  3, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  4: { convert< value_t,  4, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  5: { convert< value_t,  5, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  6: { convert< value_t,  6, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  7: { convert< value_t,  7, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  8: { convert< value_t,  8, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case  9: { convert< value_t,  9, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 10: { convert< value_t, 10, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 11: { convert< value_t, 11, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 12: { convert< value_t, 12, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 13: { convert< value_t, 13, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 14: { convert< value_t, 14, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 15: { convert< value_t, 15, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 16: { convert< value_t, 16, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 17: { convert< value_t, 17, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 18: { convert< value_t, 18, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 19: { convert< value_t, 19, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 20: { convert< value_t, 20, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 21: { convert< value_t, 21, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 22: { convert< value_t, 22, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 23: { convert< value_t, 23, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 24: { convert< value_t, 24, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 25: { convert< value_t, 25, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 26: { convert< value_t, 26, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 27: { convert< value_t, 27, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 28: { convert< value_t, 28, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 29: { convert< value_t, 29, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 30: { convert< value_t, 30, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 31: { convert< value_t, 31, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 32: { convert< value_t, 32, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 33: { convert< value_t, 33, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 34: { convert< value_t, 34, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 35: { convert< value_t, 35, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 36: { convert< value_t, 36, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 37: { convert< value_t, 37, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 38: { convert< value_t, 38, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 39: { convert< value_t, 39, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 40: { convert< value_t, 40, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 41: { convert< value_t, 41, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 42: { convert< value_t, 42, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 43: { convert< value_t, 43, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 44: { convert< value_t, 44, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 45: { convert< value_t, 45, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 46: { convert< value_t, 46, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 47: { convert< value_t, 47, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 48: { convert< value_t, 48, byte6_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 49: { convert< value_t, 49, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 50: { convert< value_t, 50, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 51: { convert< value_t, 51, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 52: { convert< value_t, 52, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 53: { convert< value_t, 53, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 54: { convert< value_t, 54, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 55: { convert< value_t, 55, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         case 56: { convert< value_t, 56, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
//         // case 64: { convert< value_t, 64, uint64_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;

//         default:
//             HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
//     }// switch
// }

// template <>
// inline
// void
// decompress< std::complex< float > > ( const zarray &           zdata,
//                                       std::complex< float > *  dest,
//                                       const size_t             dim0,
//                                       const size_t             dim1,
//                                       const size_t             dim2,
//                                       const size_t             dim3 )
// {
//     if      ( dim1 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, 2, 0, 0 );
//     else if ( dim2 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, 2, 0 );
//     else if ( dim3 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, 2 );
//     else                  decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, dim3 * 2 );
// }
    
// template <>
// inline
// void
// decompress< std::complex< double > > ( const zarray &            zdata,
//                                        std::complex< double > *  dest,
//                                        const size_t              dim0,
//                                        const size_t              dim1,
//                                        const size_t              dim2,
//                                        const size_t              dim3 )
// {
//     if      ( dim1 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, 2, 0, 0 );
//     else if ( dim2 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, 2, 0 );
//     else if ( dim3 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, 2 );
//     else                  decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, dim3 * 2 );
// }

#else

//
// convert given array <data> into posits and store results in <cptr>
//
template < typename value_t, int nbits >
struct convert
{
    static constexpr uint64_t  mask = ( 1ul << nbits ) - 1ul;
    
    static void
    to_posit ( byte_t *         ptr,
               const value_t *  data,
               const size_t     nsize,
               const value_t    scale )
    {
        using  posit_t = sw::universal::posit< nbits, ES >;
        
        uint32_t  bpos = 0; // start bit position in current byte
        size_t    pos  = 0; // byte position in <ptr>
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            auto      p     = posit_t( data[i] * scale );
            auto      zval  = p.get().to_ullong();
            uint32_t  sbits = 0; // number of already stored bits of zval
            
            do
            {
                const uint32_t  crest = 8 - bpos;       // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;  // remaining bits in zval
                const byte_t    zbyte = zval & 0xff;    // lowest byte of zval
                    
                // HLR_DBG_ASSERT( pos < zsize );
                    
                ptr[pos]  |= (zbyte << bpos);
                zval     >>= crest;
                sbits     += crest;
            
                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );
        }// if
    }// for

    static void
    from_posit ( const byte_t *  ptr,
                 value_t *       data,
                 const size_t    nsize,
                 const value_t   scale )
    {
        using  posit_t    = sw::universal::posit< nbits, ES >;
        using  bitblock_t = sw::universal::bitblock< nbits >;

        size_t    count = 0;
        uint32_t  bpos = 0; // start bit position in current byte
        size_t    pos  = 0; // byte position in <ptr>
        
        do
        {
            uint64_t  zval  = 0;
            uint32_t  sbits = 0;  // already read bits of zval
            
            do
            {
                // HLR_DBG_ASSERT( pos < zdata );
        
                const uint32_t  crest = 8 - bpos;                               // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;                          // remaining bits to read for zval
                const byte_t    zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                const byte_t    data  = (ptr[pos] >> bpos) & zmask;             // part of zval in current byte
                
                zval  |= (uint64_t(data) << sbits); // lowest to highest bit in zdata
                sbits += crest;

                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );

            // bitblock_t  bits;
            posit_t     p;
            
            // bits = zval;
            // p.setbits( bits );
            p.setbits( zval );

            data[count] = value_t( p ) / scale;
            
        } while ( ++count < nsize );
    }
};

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

    constexpr auto  fp_infinity = std::numeric_limits< value_t >::infinity();
    
    //
    // look for max value
    //
    
    value_t  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
        vmax = std::max( vmax, std::abs( data[i] ) );

    const value_t  scale    = 1.0 / vmax;
    const auto     nbits    = 1 + ES + config.bitsize;  // sign bit + exponent bits
    const auto     ntotbits = byte_pad( nsize * nbits );
    const auto     nbytes   = ntotbits / 8;
    const auto     ofs      = 1 + sizeof(value_t);
    zarray         zdata( ofs + nbytes );

    zdata[0] = nbits;
    memcpy( zdata.data() + 1, & scale, sizeof(value_t) );
    
    switch ( nbits )
    {
        case  6: { convert< value_t,  6 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case  7: { convert< value_t,  7 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case  8: { convert< value_t,  8 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case  9: { convert< value_t,  9 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 10: { convert< value_t, 10 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 11: { convert< value_t, 11 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 12: { convert< value_t, 12 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 13: { convert< value_t, 13 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 14: { convert< value_t, 14 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 15: { convert< value_t, 15 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 16: { convert< value_t, 16 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 17: { convert< value_t, 17 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 18: { convert< value_t, 18 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 19: { convert< value_t, 19 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 20: { convert< value_t, 20 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 21: { convert< value_t, 21 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 22: { convert< value_t, 22 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 23: { convert< value_t, 23 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 24: { convert< value_t, 24 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 25: { convert< value_t, 25 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 26: { convert< value_t, 26 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 27: { convert< value_t, 27 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 28: { convert< value_t, 28 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 29: { convert< value_t, 29 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 30: { convert< value_t, 30 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 31: { convert< value_t, 31 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 32: { convert< value_t, 32 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 33: { convert< value_t, 33 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 34: { convert< value_t, 34 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 35: { convert< value_t, 35 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 36: { convert< value_t, 36 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 37: { convert< value_t, 37 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 38: { convert< value_t, 38 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 39: { convert< value_t, 39 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 40: { convert< value_t, 40 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 41: { convert< value_t, 41 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 42: { convert< value_t, 42 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 43: { convert< value_t, 43 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 44: { convert< value_t, 44 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 45: { convert< value_t, 45 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 46: { convert< value_t, 46 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 47: { convert< value_t, 47 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 48: { convert< value_t, 48 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 49: { convert< value_t, 49 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 50: { convert< value_t, 50 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 51: { convert< value_t, 51 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 52: { convert< value_t, 52 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 53: { convert< value_t, 53 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 54: { convert< value_t, 54 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 55: { convert< value_t, 55 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 56: { convert< value_t, 56 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        // case 64: { convert< value_t, 64 >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( nbits ) );
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

//
// decompression function
//
template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    const size_t  nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    bitsize = zdata[0];
    auto          scale   = value_t(0);
    const auto    ofs     = 1 + sizeof(value_t);

    memcpy( & scale, zdata.data() + 1, sizeof(value_t) );
    
    switch ( bitsize )
    {
        case  6: { convert< value_t,  6 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case  7: { convert< value_t,  7 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case  8: { convert< value_t,  8 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case  9: { convert< value_t,  9 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 10: { convert< value_t, 10 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 11: { convert< value_t, 11 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 12: { convert< value_t, 12 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 13: { convert< value_t, 13 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 14: { convert< value_t, 14 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 15: { convert< value_t, 15 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 16: { convert< value_t, 16 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 17: { convert< value_t, 17 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 18: { convert< value_t, 18 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 19: { convert< value_t, 19 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 20: { convert< value_t, 20 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 21: { convert< value_t, 21 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 22: { convert< value_t, 22 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 23: { convert< value_t, 23 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 24: { convert< value_t, 24 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 25: { convert< value_t, 25 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 26: { convert< value_t, 26 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 27: { convert< value_t, 27 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 28: { convert< value_t, 28 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 29: { convert< value_t, 29 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 30: { convert< value_t, 30 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 31: { convert< value_t, 31 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 32: { convert< value_t, 32 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 33: { convert< value_t, 33 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 34: { convert< value_t, 34 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 35: { convert< value_t, 35 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 36: { convert< value_t, 36 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 37: { convert< value_t, 37 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 38: { convert< value_t, 38 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 39: { convert< value_t, 39 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 40: { convert< value_t, 40 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 41: { convert< value_t, 41 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 42: { convert< value_t, 42 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 43: { convert< value_t, 43 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 44: { convert< value_t, 44 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 45: { convert< value_t, 45 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 46: { convert< value_t, 46 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 47: { convert< value_t, 47 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 48: { convert< value_t, 48 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 49: { convert< value_t, 49 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 50: { convert< value_t, 50 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 51: { convert< value_t, 51 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 52: { convert< value_t, 52 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 53: { convert< value_t, 53 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 54: { convert< value_t, 54 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 55: { convert< value_t, 55 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 56: { convert< value_t, 56 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        // case 64: { convert< value_t, 64 >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
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

#endif

//////////////////////////////////////////////////////////////////////////////////////
//
// special version for lowrank matrices
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > &                 U,
              const blas::vector< real_type_t< value_t > > &  S )
{
    using  real_t = real_type_t< value_t >;
    
    //
    // first, determine mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          b     = std::vector< uint8_t >( k );
    auto          s     = std::vector< real_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        real_t  vmax = 0;

        for ( size_t  i = 0; i < n; ++i )
            vmax = std::max( vmax, std::abs( U(i,l) ) );

        s[l] = real_t(1) / vmax;
        b[l] = 1 + ES + eps_to_rate_aplr( S(l) ); // sign + exponent bits

        zsize += 1;                          // for nbits
        zsize += sizeof(real_t);             // for scaling factor
        zsize += byte_pad( n * b[l] ) / 8;   // for data
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto  nbits = b[l];
        const auto  scale = s[l];
        const auto  ofs   = pos + 1 + sizeof(real_t);

        zdata[pos] = nbits;
        memcpy( zdata.data() + pos + 1, & scale, sizeof(real_t) );
        
        switch ( nbits )
        {
            case  3: { convert< value_t,  3 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  4: { convert< value_t,  4 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  5: { convert< value_t,  5 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  6: { convert< value_t,  6 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  7: { convert< value_t,  7 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  8: { convert< value_t,  8 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  9: { convert< value_t,  9 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 10: { convert< value_t, 10 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 11: { convert< value_t, 11 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 12: { convert< value_t, 12 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 13: { convert< value_t, 13 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 14: { convert< value_t, 14 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 15: { convert< value_t, 15 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 16: { convert< value_t, 16 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 17: { convert< value_t, 17 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 18: { convert< value_t, 18 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 19: { convert< value_t, 19 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 20: { convert< value_t, 20 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 21: { convert< value_t, 21 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 22: { convert< value_t, 22 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 23: { convert< value_t, 23 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 24: { convert< value_t, 24 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 25: { convert< value_t, 25 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 26: { convert< value_t, 26 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 27: { convert< value_t, 27 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 28: { convert< value_t, 28 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 29: { convert< value_t, 29 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 30: { convert< value_t, 30 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 31: { convert< value_t, 31 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 32: { convert< value_t, 32 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 33: { convert< value_t, 33 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 34: { convert< value_t, 34 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 35: { convert< value_t, 35 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 36: { convert< value_t, 36 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 37: { convert< value_t, 37 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 38: { convert< value_t, 38 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 39: { convert< value_t, 39 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 40: { convert< value_t, 40 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 41: { convert< value_t, 41 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 42: { convert< value_t, 42 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 43: { convert< value_t, 43 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 44: { convert< value_t, 44 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 45: { convert< value_t, 45 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 46: { convert< value_t, 46 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 47: { convert< value_t, 47 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 48: { convert< value_t, 48 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 49: { convert< value_t, 49 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 50: { convert< value_t, 50 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 51: { convert< value_t, 51 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 52: { convert< value_t, 52 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 53: { convert< value_t, 53 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 54: { convert< value_t, 54 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 55: { convert< value_t, 55 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 56: { convert< value_t, 56 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
                // case 64: { convert< value_t, 64 >::to_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;

            default:
                HLR_ERROR( "unsupported bitsize " + Hpro::to_string( nbits ) );
        }// switch
        
        pos = ofs + byte_pad( n * nbits ) / 8;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    using  real_t = real_type_t< value_t >;

    const size_t    n   = U.nrows();
    const uint32_t  k   = U.ncols();
    size_t          pos = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read bitsize and scaling factor and decompress data
        //
    
        const uint8_t  nbits = zdata[ pos ];
        auto           scale = real_t(0);
        const auto     ofs   = pos + 1 + sizeof(real_t);

        memcpy( & scale, zdata.data() + pos + 1, sizeof(value_t) );

        switch ( nbits )
        {
            case  3: { convert< value_t,  3 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  4: { convert< value_t,  4 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  5: { convert< value_t,  5 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  6: { convert< value_t,  6 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  7: { convert< value_t,  7 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  8: { convert< value_t,  8 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case  9: { convert< value_t,  9 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 10: { convert< value_t, 10 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 11: { convert< value_t, 11 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 12: { convert< value_t, 12 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 13: { convert< value_t, 13 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 14: { convert< value_t, 14 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 15: { convert< value_t, 15 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 16: { convert< value_t, 16 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 17: { convert< value_t, 17 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 18: { convert< value_t, 18 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 19: { convert< value_t, 19 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 20: { convert< value_t, 20 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 21: { convert< value_t, 21 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 22: { convert< value_t, 22 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 23: { convert< value_t, 23 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 24: { convert< value_t, 24 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 25: { convert< value_t, 25 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 26: { convert< value_t, 26 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 27: { convert< value_t, 27 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 28: { convert< value_t, 28 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 29: { convert< value_t, 29 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 30: { convert< value_t, 30 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 31: { convert< value_t, 31 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 32: { convert< value_t, 32 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 33: { convert< value_t, 33 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 34: { convert< value_t, 34 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 35: { convert< value_t, 35 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 36: { convert< value_t, 36 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 37: { convert< value_t, 37 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 38: { convert< value_t, 38 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 39: { convert< value_t, 39 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 40: { convert< value_t, 40 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 41: { convert< value_t, 41 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 42: { convert< value_t, 42 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 43: { convert< value_t, 43 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 44: { convert< value_t, 44 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 45: { convert< value_t, 45 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 46: { convert< value_t, 46 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 47: { convert< value_t, 47 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 48: { convert< value_t, 48 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 49: { convert< value_t, 49 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 50: { convert< value_t, 50 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 51: { convert< value_t, 51 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 52: { convert< value_t, 52 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 53: { convert< value_t, 53 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 54: { convert< value_t, 54 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 55: { convert< value_t, 55 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
            case 56: { convert< value_t, 56 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;
                // case 64: { convert< value_t, 64 >::from_posit( zdata.data() + ofs, U.data() + l*n, n, scale ); } break;

            default:
                HLR_ERROR( "unsupported bitsize " + Hpro::to_string( nbits ) );
        }// switch
        
        pos = ofs + byte_pad( nbits * n ) / 8;
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

// namespace detail
// {
//
// //
// // some basic blas functions
// //
// template < size_t nbits,
//            size_t es >
// inline
// void
// mulvec ( const size_t           nrows,
//          const size_t           ncols,
//          const Hpro::matop_t    op_A,
//          const double           dalpha,
//          const byte_t *         A_ptr,
//          const double *         x_ptr,
//          const double           beta,
//          double *               y_ptr )
// {
//     using  posit_t = sw::universal::posit< nbits, es >;

//     auto           A     = reinterpret_cast< const posit_t * >( A_ptr );
//     const posit_t  alpha = dalpha;

//     if ( op_A == Hpro::apply_normal )
//     {
//         auto  y = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             y[i] = beta * y_ptr[i];

//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             const posit_t  x_j = x_ptr[j];
            
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y[i] += alpha * A[j*nrows+i] * x_j;
//         }// for

//         for ( size_t  i = 0; i < nrows; ++i )
//             y_ptr[i] = double( y[i] );
//     }// if
//     else if ( op_A == Hpro::apply_transposed )
//     {
//         auto  x = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             x[i] = x_ptr[i];
        
//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             posit_t  y_j = beta * y_ptr[j];
        
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y_j += alpha * A[j*nrows+i] * x[i];

//             y_ptr[j] = double( y_j );
//         }// for
//     }// if
//     else if ( op_A == Hpro::apply_adjoint )
//     {
//         auto  x = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             x[i] = x_ptr[i];
        
//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             posit_t  y_j = beta * y_ptr[j];
        
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y_j += alpha * A[j*nrows+i] * x[i];

//             y_ptr[j] = double( y_j );
//         }// for
//     }// if
//     else
//         HLR_ERROR( "TODO" );
// }

// }// namespace detail

// template < typename value_t >
// void
// mulvec ( const size_t         nrows,
//          const size_t         ncols,
//          const Hpro::matop_t  op_A,
//          const value_t        alpha,
//          const zarray &       A,
//          const value_t *      x,
//          const value_t        beta,
//          value_t *            y );

// template <>
// inline
// void
// mulvec< double > ( const size_t         nrows,
//                    const size_t         ncols,
//                    const Hpro::matop_t  op_A,
//                    const double         alpha,
//                    const zarray &       A,
//                    const double *       x,
//                    const double         beta,
//                    double *             y )
// {
//     const auto  bitsize = A[0];

//     switch ( bitsize )
//     {
//         case  8: detail::mulvec<  8, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 10: detail::mulvec< 10, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 12: detail::mulvec< 12, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 14: detail::mulvec< 14, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 16: detail::mulvec< 16, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 18: detail::mulvec< 18, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 20: detail::mulvec< 20, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 22: detail::mulvec< 22, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 24: detail::mulvec< 24, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 26: detail::mulvec< 26, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 28: detail::mulvec< 28, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 30: detail::mulvec< 30, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 32: detail::mulvec< 32, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 34: detail::mulvec< 34, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 36: detail::mulvec< 36, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 40: detail::mulvec< 40, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 44: detail::mulvec< 44, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 54: detail::mulvec< 54, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 64: detail::mulvec< 64, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;

//         default:
//             HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
//     }// switch
// }
    
}}}// namespace hlr::compress::posits

#endif // HLR_HAS_UNIVERSAL

#endif // __HLR_UTILS_DETAIL_POSITS_HH
