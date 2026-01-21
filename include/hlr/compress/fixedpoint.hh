#ifndef __HLR_UTILS_DETAIL_FIXEDPOINT_HH
#define __HLR_UTILS_DETAIL_FIXEDPOINT_HH
//
// Project     : HLR
// Module      : compress/fixedpoint
// Description : functions for fixed point representation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2026. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/compress/byte_n.hh>
#include <hlr/compress/ztypes.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_VALR
// #define HLR_FIXEDPOINT_BUFFERED_MVM // (disabled by default as it seems slower)

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

namespace hlr { namespace compress { namespace fixedpoint {

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
};
    
template <>
struct Zconf< double >
{
    constexpr static uint8_t   scale_ofs   = 4;
    constexpr static uint8_t   header_ofs  = 12;
};

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ) + 3; }
inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ); }

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
Hpro::real_type_t< value_t >
max ( const value_t *  data,
      const size_t     nsize )
{
    using  real_t = Hpro::real_type_t< value_t >;

    auto  vmax = real_t(0);

    for ( size_t  i = 0; i < nsize; ++i )
    {
        vmax = std::max( vmax, std::abs( data[i] ) );
    }// for

    return vmax;
}

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

//
// compress data as float
//
inline
void
compress ( const float *  data,
           const size_t   nsize,
           byte_t *       zdata,
           float          scale,
           const uint8_t  nbits )
{
    // using value_t = float;
    
    // const uint8_t   nbits    = 1 + exp_bits + prec_bits;
    // const uint8_t   nbyte    = nbits / 8;
    // const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;                  // bit mask for exponent
    // const uint8_t   prec_ofs = FP32::mant_bits - prec_bits;
    // const uint32_t  zero_val = Zconf< value_t >::zero_val & (( 1 << nbits) - 1 );
        
    // //
    // // store header (exponent bits, precision bits and scaling factor)
    // //
        
    // zdata[0] = exp_bits;
    // zdata[1] = prec_bits;
    // memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    // scale = 1.f / scale;

    // HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    // //
    // // compress data in "vectorized" form
    // //
        
    // constexpr size_t  nbuf   = 64;
    // const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    // uint8_t           zero[ nbuf ];                   // mark zero entries
    // uint8_t           sign[ nbuf ];                   // holds sign per entry
    // float             fbuf[ nbuf ];                   // holds rescaled value
    // uint32_t          ibuf[ nbuf ];                   // holds value in compressed format
    // size_t            pos = Zconf< value_t >::header_ofs;
    // size_t            i   = 0;
        
    // for ( ; i < nbsize; i += nbuf )
    // {
    //     //
    //     // Use absolute value and scale v_i and add 1 such that v_i >= 2.
    //     // With this, highest exponent bit is 1 and we only need to store
    //     // lowest <exp_bits> exponent bits
    //     //
            
    //     // scale/shift data to [2,...]
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //     {
    //         const auto  val  = data[i+j];
    //         const auto  aval = std::abs( val );

    //         zero[j] = ( aval < Zconf< value_t >::minval ); // avoid denormalized values
    //         sign[j] = ( aval != val );
    //         fbuf[j] = aval * scale + 1.f;

    //         HLR_DBG_ASSERT( zero[j] || ( fbuf[j] >= float(2) ));
    //         HLR_DBG_ASSERT( zero[j] || std::isfinite( fbuf[j] ));
    //     }// for

    //     // convert to compressed format
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //     {
    //         const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & fbuf[j] ) );
    //         const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & FP32::exp_mask; // extract exponent
    //         const uint32_t  smant = ( isval & FP32::mant_mask );                  // and mantissa
    //         const uint32_t  zexp  = sexp & exp_mask;                             // extract needed exponent
    //         const uint32_t  zmant = smant >> prec_ofs;                           // and precision bits
                
    //         ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
    //     }// for

    //     // correct zeroes
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //         if ( zero[j] )
    //             ibuf[j] = zero_val;

    //     // write to destination buffer
    //     switch ( nbyte )
    //     {
    //         case  4 : { auto ptr = reinterpret_cast< byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
    //         case  3 : { auto ptr = reinterpret_cast< byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
    //         case  2 : { auto ptr = reinterpret_cast< byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = uint16_t( ibuf[j] & 0xffff ); } break;
    //         case  1 : { auto ptr = & zdata[pos];                                  for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = byte_t( ibuf[j] & 0xff   ); } break;
    //         default : HLR_ERROR( "unsupported storage size" );
    //     }// switch

    //     pos += nbyte * nbuf;
    // }// for

    // // handle remaining values
    // for ( ; i < nsize; ++i )
    // {
    //     const float  val  = data[i];
    //     uint32_t     zval = zero_val;

    //     if ( std::abs( val ) >= Zconf< value_t >::minval )
    //     {
    //         const bool      zsign = ( val < 0 );
    //         const float     sval  = std::abs(val) * scale + float(1);
            
    //         HLR_DBG_ASSERT( std::isfinite( sval ) && ( sval >= float(2) ));
            
    //         const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & sval ) );
    //         const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & FP32::exp_mask;
    //         const uint32_t  smant = ( isval & FP32::mant_mask );
    //         const uint32_t  zexp  = sexp & exp_mask;
    //         const uint32_t  zmant = smant >> prec_ofs;

    //         zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

    //         HLR_DBG_ASSERT( zval != zero_val );
    //     }// if
        
    //     switch ( nbyte )
    //     {
    //         case  4 : zdata[pos+3] = ( zval & 0xff000000 ) >> 24;
    //         case  3 : zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
    //         case  2 : zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
    //         case  1 : zdata[pos]   = ( zval & 0x000000ff ); break;
    //         default : HLR_ERROR( "unsupported storage size" );
    //     }// switch

    //     pos += nbyte;
    // }// for
}

inline
void
decompress ( float *         data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint8_t   nbyte )
{
    // using  value_t = float;
    
    // const uint8_t   nbits      = 1 + exp_bits + prec_bits;
    // const uint8_t   nbyte      = nbits / 8;
    // const uint32_t  prec_mask  = ( 1 << prec_bits ) - 1;
    // const uint8_t   prec_ofs   = FP32::mant_bits - prec_bits;
    // const uint32_t  exp_mask   = ( 1 << exp_bits ) - 1;
    // const uint8_t   sign_shift = exp_bits + prec_bits;
    // const uint32_t  zero_val   = Zconf< value_t >::zero_val & (( 1 << nbits) - 1 );
    // float           scale;

    // // get scaling factor
    // memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    // //
    // // decompress in "vectorised" form
    // //
        
    // constexpr size_t  nbuf   = 64;
    // const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    // uint8_t           zero[ nbuf ]; // mark zero entries
    // uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    // float             fbuf[ nbuf ]; // holds uncompressed values
    // size_t            pos = Zconf< value_t >::header_ofs;
    // size_t            i   = 0;

    // for ( ; i < nbsize; i += nbuf )
    // {
    //     // read data
    //     switch ( nbyte )
    //     {
    //         case  4 : { auto ptr = reinterpret_cast< const byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
    //         case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
    //         case  2 : { auto ptr = reinterpret_cast< const byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
    //         case  1 : { auto ptr = & zdata[pos];                                        for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
    //         default : HLR_ERROR( "unsupported storage size" );
    //     }// switch

    //     // convert from compressed format
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //     {
    //         const auto  zval = ibuf[j];

    //         zero[j] = ( zval == zero_val );
            
    //         const uint32_t  mant  = zval & prec_mask;
    //         const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
    //         const uint32_t  sign  = ( zval >> sign_shift ) << FP32::sign_bit;
    //         fp32int_t       fival = { .u = ((exp | Zconf< value_t >::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs) };
            
    //         fival.f  = ( fival.f - 1.f ) * scale;
    //         fival.u |= sign;
    //         fbuf[j]  = fival.f;

    //         HLR_DBG_ASSERT( zero[j] || std::isfinite( fbuf[j] ) );
    //     }// for

    //     // correct zeroes
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //         if ( zero[j] )
    //             fbuf[j] = value_t(0);

    //     // copy values
    //     for ( size_t  j = 0; j < nbuf; ++j )
    //         data[i+j] = fbuf[j];
            
    //     pos += nbyte * nbuf;
    // }// for

    // // handle remaining values
    // for ( ; i < nsize; ++i )
    // {
    //     uint32_t  zval = 0;
            
    //     switch ( nbyte )
    //     {
    //         case  4 : zval |= zdata[pos+3] << 24;
    //         case  3 : zval |= zdata[pos+2] << 16;
    //         case  2 : zval |= zdata[pos+1] << 8;
    //         case  1 : zval |= zdata[pos]; break;
    //         default : HLR_ERROR( "unsupported storage size" );
    //     }// switch

    //     if ( zval == zero_val )
    //         data[i] = 0;
    //     else
    //     {
    //         const uint32_t  mant  = zval & prec_mask;
    //         const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
    //         const uint32_t  sign  = ( zval >> sign_shift ) << FP32::sign_bit;
    //         fp32int_t       fival = { .u = ((exp | Zconf< value_t >::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs) };

    //         fival.f  = ( fival.f - 1.f ) * scale;
    //         fival.u |= sign;
    //         data[i]  = fival.f;

    //         HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    //     }// else

    //     pos += nbyte;
    // }// for
}

//
// compress data needing more than 32 bits
//
inline
void
compress ( const double *  data,  // points to actual start of buffer
           const size_t    nsize,
           byte_t *        zdata,
           double          scale,
           const uint8_t   nbits )
{
    using  value_t = double;
    
    const uint8_t  nbyte = nbits / 8;
        
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    const uint64_t  imask = ( 1ul << nbits ) - 1ul;    // mask to extract nbyte integer value (also maximal unsigned integer value)
    const double    imax  = imask / 2;             // maximal signed integer value

    // adjust scaling for integer max
    scale = scale * imax;
    
    zdata[0] = nbyte;
    memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    zdata += Zconf< value_t >::header_ofs;
    
    //
    // in case of 8 byte, just copy data
    //

    if ( nbyte == 8 )
    {
        std::copy( data, data + nsize, reinterpret_cast< double * >( zdata ) );
        return;
    }// if
    
    //
    // compress data in "vectorized" form
    //

    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            i   = 0;
        
    for ( ; i < nbsize; i += nbuf )
    {
        //
        // Use absolute value and scale v_i and add 1 such that v_i >= 2.
        // With this, highest exponent bit is 1 and we only need to store
        // lowest <exp_bits> exponent bits
        //
            
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            // - scale to [-1,1] and then to [-imax,imax]  (signed int)
            // - shift to [0,2*imax]    (unsigned int)  // TODO: needed???
            // - extract "nbyte" bytes
            ibuf[j] = uint64_t( data[i+j] * scale + imax ) & imask;
        }// for

        // write to destination buffer
        switch ( nbyte )
        {
            case  1 : { auto ptr = zdata + i;                                  for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< byte2_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< byte3_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  4 : { auto ptr = reinterpret_cast< byte4_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  5 : { auto ptr = reinterpret_cast< byte5_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< byte6_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< byte7_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< byte8_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            default : HLR_ERROR( "invalid storage size" );
        }// switch
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const auto  zval = uint64_t( data[i] * scale + imax ) & imask;

        switch ( nbyte )
        {
            case  1 : { auto ptr = zdata + i;                                  *ptr = zval; } break;
            case  2 : { auto ptr = reinterpret_cast< byte2_t * >( zdata ) + i; *ptr = zval; } break;
            case  3 : { auto ptr = reinterpret_cast< byte3_t * >( zdata ) + i; *ptr = zval; } break;
            case  4 : { auto ptr = reinterpret_cast< byte4_t * >( zdata ) + i; *ptr = zval; } break;
            case  5 : { auto ptr = reinterpret_cast< byte5_t * >( zdata ) + i; *ptr = zval; } break;
            case  6 : { auto ptr = reinterpret_cast< byte6_t * >( zdata ) + i; *ptr = zval; } break;
            case  7 : { auto ptr = reinterpret_cast< byte7_t * >( zdata ) + i; *ptr = zval; } break;
            case  8 : { auto ptr = reinterpret_cast< byte8_t * >( zdata ) + i; *ptr = zval; } break;
            default : HLR_ERROR( "invalid storage size" );
        }// switch
    }// for
}

inline
void
decompress ( double *        data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint8_t   nbyte )
{
    using  value_t = double;
    
    // just retrieve data for nbyte == 8
    if ( nbyte == 8 )
    {
        std::copy( reinterpret_cast< const double * >( zdata + Zconf< value_t >::header_ofs ),
                   reinterpret_cast< const double * >( zdata + Zconf< value_t >::header_ofs ) + nsize,
                   data );
        return;
    }// if
         
    double  scale;

    // get scaling factor
    memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    scale = 1.0 / scale; // for multiplication below

    zdata += Zconf< value_t >::header_ofs;
    
    //
    // decompress in "vectorised" form
    //
        
    const uint64_t    imax   = 1ul << (8*nbyte-1);        // maximal signed integer value
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            i      = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        switch ( nbyte )
        {
            case  1 : { auto ptr = zdata + i;                                        for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< const byte2_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  4 : { auto ptr = reinterpret_cast< const byte4_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  5 : { auto ptr = reinterpret_cast< const byte5_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< const byte6_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< const byte7_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< const byte8_t * >( zdata ) + i; for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            default : HLR_ERROR( "invalid storage size" );
        }// switch
            
        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            data[i+j] = ( double(ibuf[j]) - imax ) * scale;
        }// for
    }// for
    
    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint64_t  zval = 0;
            
        switch ( nbyte )
        {
            case  1 : { auto ptr = zdata + i;                                        zval = *ptr; } break;
            case  2 : { auto ptr = reinterpret_cast< const byte2_t * >( zdata ) + i; zval = *ptr; } break;
            case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( zdata ) + i; zval = *ptr; } break;
            case  4 : { auto ptr = reinterpret_cast< const byte4_t * >( zdata ) + i; zval = *ptr; } break;
            case  5 : { auto ptr = reinterpret_cast< const byte5_t * >( zdata ) + i; zval = *ptr; } break;
            case  6 : { auto ptr = reinterpret_cast< const byte6_t * >( zdata ) + i; zval = *ptr; } break;
            case  7 : { auto ptr = reinterpret_cast< const byte7_t * >( zdata ) + i; zval = *ptr; } break;
            case  8 : { auto ptr = reinterpret_cast< const byte8_t * >( zdata ) + i; zval = *ptr; } break;
            default : HLR_ERROR( "invalid storage size" );
        }// switch

        data[i] = ( double(zval) - imax ) * scale;
    }// for
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
    const auto  vmax = max( data, nsize );
    
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
    
    const auto          scale         = real_t(1) / vmax;                                              // scale all values v_i such that |v_i| >= 1
    constexpr uint32_t  max_mant_bits = FPinfo< real_t >::mant_bits;
    uint8_t             prec_bits     = std::min< uint32_t >( max_mant_bits, config.bitrate );         // number of precision bits due to config
    const uint8_t       nbits         = byte_pad( 1 + prec_bits );                                     // rounded up total no. of bits per value
    const uint8_t       nbyte         = nbits / 8;
    auto                zdata         = std::vector< byte_t >( Zconf< real_t >::header_ofs + nsize * nbyte ); // array storing compressed data

    // adjust precision (or exponent bits)
    prec_bits = nbits - 1;

    if ( prec_bits > max_mant_bits )
    {
        const auto  diff = prec_bits - max_mant_bits;
            
        prec_bits = max_mant_bits;
    }// if
            
    HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    HLR_ASSERT( nbits     <= sizeof(real_t) * 8 );
    HLR_ASSERT( prec_bits <= max_mant_bits );

    compress( data, nsize, zdata.data(), scale, nbits );

    // // DEBUG
    // {
    //     std::vector< double >  tmp( nsize );

    //     decompress( tmp.data(), nsize, zdata.data(), nbyte );

    //     double  err = 0;
    //     double  nrm = 0;

    //     for ( size_t  i = 0; i < nsize; ++i )
    //     {
    //         const auto  d_i = data[i] - tmp[i];
            
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

    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    const uint8_t  nbyte  = zdata[0];
    
    // HLR_ASSERT( 1 + exp_bits + prec_bits <= sizeof(value_t) * 8 );
    // HLR_ASSERT( prec_bits <= FPinfo< real_t >::mant_bits );

    if ( nbyte == 0 )
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            dest[i] = value_t(0);
    }// if
    else
        decompress( dest, nsize, zdata.data(), nbyte );
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
    
    constexpr real_t  fp_maximum  = FPinfo< real_t >::maximum;
    constexpr size_t  header_size = Zconf< real_t >::header_ofs; // sizeof(real_t) + 2;
    
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

        const auto  vmax = max( U.ptr(0,l), n );

        s[l] = real_t(1) / vmax;

        HLR_ASSERT( std::isfinite( s[l] ) );

        const auto  nbits = byte_pad( 1 + eps_to_rate_valr( S(l) ) );
        const auto  nbyte = nbits / 8;

        m[l] = nbits;

        zsize += header_size + n * nbyte;
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const real_t   scale = s[l];
        const uint8_t  nbits = m[l];
        const uint8_t  nbyte = nbits / 8;

        compress( U.ptr(0,l), n, zdata.data() + pos, scale, nbits );

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
        
        pos += header_size + n*nbyte;
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
    // using  real_t = double;
    
    // constexpr real_t  fp_maximum  = FPinfo< real_t >::maximum;
    // constexpr size_t  header_size = Zconf< real_t >::header_ofs; // sizeof(real_t) + 2;
    
    // //
    // // first, determine exponent bits and mantissa bits for all columns
    // //

    // const size_t  n     = U.nrows();
    // const size_t  k     = U.ncols();
    // const size_t  n2    = 2 * n;
    // auto          m     = std::vector< uint32_t >( k );
    // auto          s     = std::vector< real_t >( k );
    // size_t        zsize = 0;

    // for ( uint32_t  l = 0; l < k; ++l )
    // {
    //     auto  vmin = fp_maximum;
    //     auto  vmax = real_t(0);

    //     for ( size_t  i = 0; i < n; ++i )
    //     {
    //         const auto  u_il   = U(i,l);
    //         const auto  u_re   = std::abs( std::real( u_il ) );
    //         const auto  u_im   = std::abs( std::imag( u_il ) );
    //         const auto  val_re = ( u_re == real_t(0) ? fp_maximum : u_re );
    //         const auto  val_im = ( u_im == real_t(0) ? fp_maximum : u_im );
            
    //         vmin = std::min( vmin, std::min( val_re, val_im ) );
    //         vmax = std::max( vmax, std::max( u_re, u_im ) );
    //     }// for

    //     s[l] = vmin;
    //     e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

    //     HLR_DBG_ASSERT( std::isfinite( s[l] ) );

    //     const auto  nprecbits = eps_to_rate_valr( S(l) );
    //     const auto  nbits     = 1 + e[l] + nprecbits;

    //     // increase mantissa bits such that sum is multiple of 8
    //     m[l] = nprecbits + ( byte_pad( nbits ) - nbits );

    //     const size_t  npbits = 1 + e[l] + m[l]; // number of bits per value
    //     const size_t  npbyte = npbits / 8;
        
    //     zsize += header_size + n2 * npbyte; // sizeof(real_t) + 1 + 1 + n2 * npbyte; // twice because real+imag
    // }// for

    // //
    // // convert each column to compressed form
    // //

    // auto            zdata = std::vector< byte_t >( zsize );
    // size_t          pos   = 0;
    // const real_t *  U_ptr = reinterpret_cast< const real_t * >( U.data() );
        
    // for ( uint32_t  l = 0; l < k; ++l )
    // {
    //     const uint32_t  exp_bits  = e[l];
    //     const uint32_t  prec_bits = m[l];
    //     const real_t    scale     = s[l];
    //     const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value
    //     const size_t    nbyte     = nbits / 8;

    //     compress( U_ptr + l * n2, n2, zdata.data() + pos, scale, exp_bits, prec_bits );
    //     pos += header_size + n2*nbyte;
    // }// for

    // return zdata;
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
    
        const uint8_t  nbyte = zdata[ pos ];

        decompress( U.data() + l * n, n, zdata.data() + pos, nbyte );
        pos += header_size + nbyte * n;
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
    // using  real_t = double;
    
    // const size_t      n           = U.nrows();
    // const uint32_t    k           = U.ncols();
    // size_t            pos         = 0;
    // constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    // real_t *          U_ptr       = reinterpret_cast< real_t * >( U.data() );
    // const size_t      n2          = 2 * n;

    // for ( uint32_t  l = 0; l < k; ++l )
    // {
    //     //
    //     // read compression header (scaling, exponent and precision bits)
    //     // and decompress data
    //     //
    
    //     const uint32_t  exp_bits  = zdata[ pos ];
    //     const uint32_t  prec_bits = zdata[ pos+1 ];
    //     const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    //     const uint32_t  nbyte     = nbits / 8;

    //     decompress( U_ptr + l * n2, n2, zdata.data() + pos, prec_bits );
    //     pos += header_size + nbyte * n2;
    // }// for
}

//
// compressed blas
//

namespace
{

template < typename value_t,
           typename accessor_t >
void
mulvec ( const size_t        nrows,
         const size_t        ncols,
         const matop_t       op_A,
         const value_t       alpha,
         const accessor_t &  zA,
         const value_t *     x,
         value_t *           y )
{
    auto  iter_A = zA.begin();
        
    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = alpha * x[j];
                
                for ( size_t  i = 0; i < nrows; ++i, ++iter_A )
                    y[i] += *iter_A * x_j;
            }// for
        }// case
        break;
        
        case  apply_adjoint :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                auto  y_j = value_t(0);
                
                for ( size_t  i = 0; i < nrows; ++i, ++iter_A )
                    y_j += *iter_A * x[i];

                y[j] += alpha * y_j;
            }// for
        }// case
        break;

        default:
            HLR_ERROR( "TODO" );
    }// switch
}

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
         const uint8_t                       nbyte )
{
    const double      imax  = 1ul << (8*nbyte-1);        // maximal signed integer value
    const auto        scale = alpha * zscale;

    #if defined(HLR_FIXEDPOINT_BUFFERED_MVM)
    constexpr size_t  max_nbuf = 64;
    const size_t      nbuf     = std::min< size_t >( max_nbuf, nrows );
    const size_t      nrowsbuf = ( nrows > nbuf ? nrows - nrows % nbuf : nrows );
    value_t           fcache[nbuf];
    #endif

    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = scale * x[j];
                size_t      i   = 0;
                
                #if defined(HLR_FIXEDPOINT_BUFFERED_MVM)
                
                for ( ; i < nrowsbuf; i += nbuf, pos += nbuf )
                {
                    #pragma GCC ivdep
                    for ( uint  ii = 0; ii < nbuf; ++ii )
                    {
                        const double  A_ij = double(zA[pos+ii]);

                        fcache[ii] = fival.f;
                    }// for

                    #pragma GCC ivdep
                    for ( uint  ii = 0; ii < nbuf; ++ii )
                        y[i+ii] += fcache[ii] * x_j;
                }// for

                #endif
                
                for ( ; i < nrows; ++i, pos++ )
                {
                    auto  A_ij = double( zA[pos] ) - imax;

                    y[i] += A_ij * x_j;
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
                size_t   i   = 0;
                
                #if defined(HLR_FIXEDPOINT_BUFFERED_MVM)

                for ( ; i < nrowsbuf; i += nbuf, pos += nbuf )
                {
                    #pragma GCC ivdep
                    for ( uint  ii = 0; ii < nbuf; ++ii )
                    {
                        const uint64_t  z_ij  = zA[pos+ii];
                        const uint64_t  sign  = (z_ij >> sign_shift) << FP64::sign_bit;
                        fp64int_t       fival = { .u = ( ( z_ij & zmask ) << prec_shift ) | Zconf< value_t >::exp_highbit };
                        
                        fival.f  = ( fival.f - 1.0 );
                        fival.u |= sign;

                        fcache[ii] = fival.f;
                    }// for
                        
                    for ( uint  ii = 0; ii < nbuf; ++ii )
                        y_j += fcache[ii] * x[i+ii];
                }// for

                #endif
                
                for ( ; i < nrows; ++i, pos++ )
                {
                    const auto  A_ij = double( zA[pos] ) - imax;

                    y_j += A_ij * x[i];
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

    const uint8_t      nbyte     = zA[0];
    real_t             scale     = * ( reinterpret_cast< const real_t * >( zA.data() + Zconf< real_t >::scale_ofs ) );
    constexpr size_t   data_ofs  = Zconf< real_t >::header_ofs;

    scale = real_t(1) / scale;
    
    switch ( nbyte )
    {
        case  1 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  2 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  3 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  4 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  5 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  6 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  7 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        case  8 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + data_ofs ), x, y, nbyte ); break;
        default :
            HLR_ERROR( "unsupported byte size" );
    }// switch
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
    using  real_t = Hpro::real_type_t< value_t >;

    constexpr size_t  scale_ofs = Zconf< real_t >::scale_ofs;
    constexpr size_t  data_ofs  = Zconf< real_t >::header_ofs;
    size_t            pos       = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = zA[pos];
                const real_t   scale = real_t(1) / ( * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) ) );
                
                HLR_ASSERT( pos + data_ofs + nrows * nbyte <= zA.size() );

                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + pos + data_ofs ), x+l, y, nbyte ); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                pos += data_ofs + nbyte * nrows;
            }// for
        }// case
        break;
        
        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = zA[pos];
                const real_t   scale = real_t(1) / ( * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) ) );
                
                HLR_ASSERT( pos + data_ofs + nrows * nbyte <= zA.size() );
                
                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + pos + data_ofs ), x, y+l, nbyte ); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                pos += data_ofs + nbyte * nrows;
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::fixedpoint

#endif // __HLR_UTILS_DETAIL_FIXEDPOINT_HH
