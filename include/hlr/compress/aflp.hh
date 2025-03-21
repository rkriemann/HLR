#ifndef __HLR_UTILS_DETAIL_AFLP_HH
#define __HLR_UTILS_DETAIL_AFLP_HH
//
// Project     : HLR
// Module      : compress/aflp
// Description : functions for adaptive padded floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/compress/byte_n.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_APLR

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

namespace hlr { namespace compress { namespace aflp {

using byte_t = uint8_t;

//
// shared float/int types
//
union fp32int_t
{
    uint32_t u;
    float    f;
};

union fp64int_t
{
    uint64_t  u;
    double    f;
};

//
// floating point data
//
template < typename real_t >
struct FP_info
{};

template <>
struct FP_info< float >
{
    constexpr static uint8_t   scale_ofs   = 4;
    constexpr static uint8_t   header_ofs  = 8;
    
    constexpr static uint8_t   mant_bits   = 23;
    constexpr static uint8_t   exp_bits    = 8;
    constexpr static uint8_t   sign_bit    = 31;
    constexpr static uint32_t  exp_highbit = 0b10000000;

    constexpr static uint32_t  exp_mask    = ((1u <<  exp_bits) - 1);
    constexpr static uint32_t  mant_mask   = ((1u << mant_bits) - 1);

    constexpr static uint32_t  zero_val    = 0xffffffff;
    constexpr static float     maximum     = std::numeric_limits< float >::max();
};
    
template <>
struct FP_info< double >
{
    constexpr static uint8_t   scale_ofs   = 4;
    constexpr static uint8_t   header_ofs  = 12;

    constexpr static uint8_t   mant_bits   = 52;
    constexpr static uint8_t   exp_bits    = 11;
    constexpr static uint8_t   sign_bit    = 63;
    constexpr static uint64_t  exp_highbit = 0b10000000000;

    constexpr static uint64_t  exp_mask    = ((1ul <<  exp_bits) - 1);
    constexpr static uint64_t  mant_mask   = ((1ul << mant_bits) - 1);

    constexpr static uint64_t  zero_val    = 0xffffffffffffffff;
    constexpr static double    maximum     = std::numeric_limits< double >::max();
};

using FP32 = FP_info< float >;
using FP64 = FP_info< double >;

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ) + 1; }

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
           const float    scale,
           const uint8_t  exp_bits,
           const uint8_t  prec_bits )
{
    using value_t = float;
    
    const uint8_t   nbits    = 1 + exp_bits + prec_bits;
    const uint8_t   nbyte    = nbits / 8;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;                  // bit mask for exponent
    const uint8_t   prec_ofs = FP32::mant_bits - prec_bits;
    const uint32_t  zero_val = FP32::zero_val & (( 1 << nbits) - 1 );
        
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + FP32::scale_ofs, & scale, sizeof(scale) );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint8_t           zero[ nbuf ];                   // mark zero entries
    uint8_t           sign[ nbuf ];                   // holds sign per entry
    float             fbuf[ nbuf ];                   // holds rescaled value
    uint32_t          ibuf[ nbuf ];                   // holds value in compressed format
    size_t            pos = FP32::header_ofs;
    size_t            i   = 0;
        
    for ( ; i < nbsize; i += nbuf )
    {
        //
        // Use absolute value and scale v_i and add 1 such that v_i >= 2.
        // With this, highest exponent bit is 1 and we only need to store
        // lowest <exp_bits> exponent bits
        //
            
        // scale/shift data to [2,...]
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto  val  = data[i+j];
            const auto  aval = std::abs( val );

            zero[j] = ( aval == float(0) );
            sign[j] = ( aval != val );
            fbuf[j] = aval / scale + 1.f;

            HLR_DBG_ASSERT( fbuf[j] >= float(2) );
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & fbuf[j] ) );
            const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & FP32::exp_mask; // extract exponent
            const uint32_t  smant = ( isval & FP32::mant_mask );                  // and mantissa
            const uint32_t  zexp  = sexp & exp_mask;                             // extract needed exponent
            const uint32_t  zmant = smant >> prec_ofs;                           // and precision bits
                
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write to destination buffer
        switch ( nbyte )
        {
            case  4 : { auto ptr = reinterpret_cast< byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = uint16_t( ibuf[j] & 0xffff ); } break;
            case  1 : { auto ptr = & zdata[pos];                                  for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = byte_t( ibuf[j] & 0xff   ); } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        pos += nbyte * nbuf;
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const float  val  = data[i];
        uint32_t     zval = zero_val;

        if ( std::abs( val ) != float(0) )
        {
            const bool      zsign = ( val < 0 );
            const float     sval  = std::abs(val) / scale + 1.f;
            
            HLR_DBG_ASSERT( sval >= float(2) );
            
            const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & sval ) );
            const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & FP32::exp_mask;
            const uint32_t  smant = ( isval & FP32::mant_mask );
            const uint32_t  zexp  = sexp & exp_mask;
            const uint32_t  zmant = smant >> prec_ofs;

            zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        switch ( nbyte )
        {
            case  4 : zdata[pos+3] = ( zval & 0xff000000 ) >> 24;
            case  3 : zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
            case  2 : zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
            case  1 : zdata[pos]   = ( zval & 0x000000ff ); break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        pos += nbyte;
    }// for
}

inline
void
decompress ( float *         data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint8_t   exp_bits,
             const uint8_t   prec_bits )
{
    using  value_t = float;
    
    const uint8_t   nbits      = 1 + exp_bits + prec_bits;
    const uint8_t   nbyte      = nbits / 8;
    const uint32_t  prec_mask  = ( 1 << prec_bits ) - 1;
    const uint8_t   prec_ofs   = FP32::mant_bits - prec_bits;
    const uint32_t  exp_mask   = ( 1 << exp_bits ) - 1;
    const uint8_t   sign_shift = exp_bits + prec_bits;
    const uint32_t  zero_val   = FP32::zero_val & (( 1 << nbits) - 1 );
    float           scale;

    // get scaling factor
    memcpy( & scale, zdata + FP32::scale_ofs, sizeof(scale) );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint8_t           zero[ nbuf ]; // mark zero entries
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    float             fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos = FP32::header_ofs;
    size_t            i   = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        switch ( nbyte )
        {
            case  4 : { auto ptr = reinterpret_cast< const byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< const byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  1 : { auto ptr = & zdata[pos];                                        for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      zval  = ibuf[j];

            zero[j] = ( zval == zero_val );
            
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const uint32_t  sign  = ( zval >> sign_shift ) << FP32::sign_bit;
            fp32int_t       fival = { ((exp | FP32::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs) };

            fival.f  = ( fival.f - 1.f ) * scale;
            fival.u |= sign;
            fbuf[j]  = fival.f;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                fbuf[j] = value_t(0);

        // copy values
        for ( size_t  j = 0; j < nbuf; ++j )
            data[i+j] = fbuf[j];
            
        pos += nbyte * nbuf;
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint32_t  zval = 0;
            
        switch ( nbyte )
        {
            case  4 : zval |= zdata[pos+3] << 24;
            case  3 : zval |= zdata[pos+2] << 16;
            case  2 : zval |= zdata[pos+1] << 8;
            case  1 : zval |= zdata[pos]; break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        if ( zval == zero_val )
            data[i] = 0;
        else
        {
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const uint32_t  sign  = ( zval >> sign_shift ) << FP32::sign_bit;
            fp32int_t       fival = { ((exp | FP32::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs) };

            fival.f  = ( fival.f - 1.f ) * scale;
            fival.u |= sign;
            data[i]  = fival.f;
        }// else

        pos += nbyte;
    }// for
}

//
// compress data needing more than 32 bits
//
inline
void
compress ( const double *  data,  // points to actual start of buffer
           const size_t    nsize,
           byte_t *        zdata,
           const double    scale,
           const uint8_t   exp_bits,
           const uint8_t   prec_bits )
{
    const uint8_t   nbits    = 1 + exp_bits + prec_bits;
    const uint8_t   nbyte    = nbits / 8;
    const uint64_t  exp_mask = ( 1u << exp_bits ) - 1;                 // bit mask for exponent
    const uint8_t   prec_ofs = FP64::mant_bits - prec_bits;
    const uint64_t  zero_val = FP64::zero_val & (( 1ul << nbits) - 1 );
        
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + FP64::scale_ofs, & scale, sizeof(scale) );

    //
    // in case of 8 byte, just copy data
    //

    if ( nbyte == 8 )
    {
        std::copy( data, data + nsize, reinterpret_cast< double * >( zdata + FP64::header_ofs ) );
        return;
    }// if
    
    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint8_t           zero[ nbuf ]; // mark zero entries
    uint8_t           sign[ nbuf ]; // holds sign per entry
    double            fbuf[ nbuf ]; // holds rescaled value
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos = FP64::header_ofs;
    size_t            i   = 0;
        
    for ( ; i < nbsize; i += nbuf )
    {
        //
        // Use absolute value and scale v_i and add 1 such that v_i >= 2.
        // With this, highest exponent bit is 1 and we only need to store
        // lowest <exp_bits> exponent bits
        //
            
        // scale/shift data to [2,...]
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto  val  = data[i+j];
            const auto  aval = std::abs( val );

            zero[j] = ( aval == double(0) );
            sign[j] = ( aval != val );
            fbuf[j] = aval / scale + 1.0;

            HLR_DBG_ASSERT( zero[j] || ( fbuf[j] >= double(2) ));
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & fbuf[j] ) );
            const uint64_t  sexp  = ( isval >> FP64::mant_bits ) & FP64::exp_mask;
            const uint64_t  smant = ( isval & FP64::mant_mask );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;
                
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write to destination buffer
        switch ( nbyte )
        {
            case  1 : { auto ptr = & zdata[pos];                                  for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  4 : { auto ptr = reinterpret_cast< byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  5 : { auto ptr = reinterpret_cast< byte5_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< byte6_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< byte7_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< byte8_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        pos += nbyte * nbuf;
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const double  val  = data[i];
        uint64_t      zval = zero_val;
            
        if ( std::abs( val ) != double(0) )
        {
            const bool      zsign = ( val < 0 );
            const double    sval  = std::abs(val) / scale + 1.0;
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & sval ) );
            const uint64_t  sexp  = ( isval >> FP64::mant_bits ) & FP64::exp_mask;
            const uint64_t  smant = ( isval & FP64::mant_mask );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;

            zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        switch ( nbyte )
        {
            case  8 : zdata[pos+7] = ( zval & 0xff00000000000000 ) >> 56;
            case  7 : zdata[pos+6] = ( zval & 0x00ff000000000000 ) >> 48;
            case  6 : zdata[pos+5] = ( zval & 0x0000ff0000000000 ) >> 40;
            case  5 : zdata[pos+4] = ( zval & 0x000000ff00000000 ) >> 32;
            case  4 : zdata[pos+3] = ( zval & 0x00000000ff000000 ) >> 24;
            case  3 : zdata[pos+2] = ( zval & 0x0000000000ff0000 ) >> 16;
            case  2 : zdata[pos+1] = ( zval & 0x000000000000ff00 ) >> 8;
            case  1 : zdata[pos]   = ( zval & 0x00000000000000ff ); break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch
            
        pos += nbyte;
    }// for
}

inline
void
decompress ( double *        data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint8_t   exp_bits,
             const uint8_t   prec_bits )
{
    const uint8_t   nbits      = 1 + exp_bits + prec_bits;
    const uint8_t   nbyte      = nbits / 8;
    const uint64_t  prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint8_t   prec_ofs   = FP64::mant_bits - prec_bits;
    const uint64_t  exp_mask   = ( 1ul << exp_bits  ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint64_t  zero_val   = FP64::zero_val & (( 1ul << nbits) - 1 );
    double          scale;

    // just retrieve data for nbyte == 8
    if ( nbyte == 8 )
    {
        std::copy( reinterpret_cast< const double * >( zdata + FP64::header_ofs ),
                   reinterpret_cast< const double * >( zdata + FP64::header_ofs ) + nsize,
                   data );
        return;
    }// if
         
    // get scaling factor
    memcpy( & scale, zdata + FP64::scale_ofs, sizeof(scale) );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    uint8_t           zero[ nbuf ]; // mark zero entries
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    double            fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos = FP64::header_ofs;
    size_t            i   = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        switch ( nbyte )
        {
            case  1 : { auto ptr = & zdata[pos];                                        for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< const byte2_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  4 : { auto ptr = reinterpret_cast< const byte4_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  5 : { auto ptr = reinterpret_cast< const byte5_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< const byte6_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< const byte7_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< const byte8_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch
            
        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      zval  = ibuf[j];

            zero[j] = ( zval == zero_val );
            
            const uint64_t  mant  = zval & prec_mask;
            const uint64_t  exp   = (zval >> prec_bits) & exp_mask;
            const uint64_t  sign  = (zval >> sign_shift) << FP64::sign_bit;
            fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };

            fival.f  = ( fival.f - 1.0 ) * scale;
            fival.u |= sign;
            fbuf[j]  = fival.f;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                fbuf[j] = double(0);

        // copy values
        for ( size_t  j = 0; j < nbuf; ++j )
            data[i+j] = fbuf[j];
            
        pos += nbyte * nbuf;
    }// for
    
    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint64_t  zval = 0;
            
        switch ( nbyte )
        {
            case  8 : zval |= uint64_t(zdata[pos+7]) << 56;
            case  7 : zval |= uint64_t(zdata[pos+6]) << 48;
            case  6 : zval |= uint64_t(zdata[pos+5]) << 40;
            case  5 : zval |= uint64_t(zdata[pos+4]) << 32;
            case  4 : zval |= uint64_t(zdata[pos+3]) << 24;
            case  3 : zval |= uint64_t(zdata[pos+2]) << 16;
            case  2 : zval |= uint64_t(zdata[pos+1]) << 8;
            case  1 : zval |= uint64_t(zdata[pos]); break;
            default : HLR_ERROR( "unsupported byte size" );
        }// switch

        if ( zval == zero_val )
            data[i] = 0;
        else
        {
            const uint64_t  mant  = zval & prec_mask;
            const uint64_t  exp   = (zval >> prec_bits) & exp_mask;
            const uint64_t  sign  = (zval >> sign_shift) << FP64::sign_bit;
            fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };
            
            fival.f  = ( fival.f - 1.0 ) * scale;
            fival.u |= sign;
            data[i]  = fival.f;
        }// else

        pos += nbyte;
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

    constexpr uint8_t  max_mant_bits = FP_info< real_t >::mant_bits;
    const size_t       nsize         = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // look for min/max value (> 0!)
    //
    
    auto  vmin = FP_info< real_t >::maximum;
    auto  vmax = real_t(0);

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == real_t(0) ? FP_info< real_t >::maximum : d_i );

        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_ASSERT( vmin > real_t(0) );
    
    if ( vmin == FP_info< real_t >::maximum )
    {
        //
        // in case of zero data, return special data
        //

        auto  zdata = std::vector< byte_t >( 2 );
        
        zdata[0] = 0;
        zdata[1] = 0;

        return zdata;
    }// if
    
    const auto     scale     = vmin;                                                                         // scale all values v_i such that |v_i| >= 1
    uint8_t        exp_bits  = std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );  // no. of bits needed to represent exponent
    uint8_t        prec_bits = std::min< uint32_t >( max_mant_bits, config.bitrate );                        // number of precision bits due to config
    const uint8_t  nbits     = byte_pad( 1 + exp_bits + prec_bits );                                         // rounded up total no. of bits per value
    const uint8_t  nbyte     = nbits / 8;
    auto           zdata     = std::vector< byte_t >( FP_info< real_t >::header_ofs + nsize * nbyte );       // array storing compressed data

    // adjust precision (or exponent bits)
    prec_bits = nbits - 1 - exp_bits;

    if ( prec_bits > max_mant_bits )
    {
        const auto  diff = prec_bits - max_mant_bits;
            
        exp_bits += prec_bits - max_mant_bits;
        prec_bits = max_mant_bits;
    }// if
            
    HLR_DBG_ASSERT( std::isfinite( scale ) );
    
    HLR_ASSERT( nbits     <= sizeof(real_t) * 8 );
    HLR_ASSERT( prec_bits <= max_mant_bits );

    compress( data, nsize, zdata.data(), scale, exp_bits, prec_bits );

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
    
    const uint8_t  exp_bits  = zdata[0];
    const uint8_t  prec_bits = zdata[1];
    
    HLR_ASSERT( 1 + exp_bits + prec_bits <= sizeof(value_t) * 8 );
    HLR_ASSERT( prec_bits <= FP_info< real_t >::mant_bits );

    if (( exp_bits == 0 ) && ( prec_bits == 0 ))
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            dest[i] = value_t(0);
    }// if
    else
        decompress( dest, nsize, zdata.data(), exp_bits, prec_bits );
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
    
    constexpr real_t  fp_maximum  = FP_info< real_t >::maximum;
    constexpr size_t  header_size = FP_info< real_t >::header_ofs; // sizeof(real_t) + 2;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t    n = U.nrows();
    const uint32_t  k = U.ncols();
    auto            m = std::vector< uint8_t >( k );
    auto            e = std::vector< uint8_t >( k );
    auto            s = std::vector< real_t >( k );
    size_t          zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        auto  vmin = fp_maximum;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == real_t(0) ? fp_maximum : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = vmin;
        e[l] = uint8_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

        HLR_ASSERT( std::isfinite( s[l] ) );

        const auto  nprecbits = eps_to_rate_valr( S(l) );
        const auto  nbits     = 1 + e[l] + nprecbits;

        // increase mantissa bits such that sum is multiple of 8
        m[l] = nprecbits + ( byte_pad( nbits ) - nbits );

        const size_t  npbits = 1 + e[l] + m[l]; // number of bits per value
        const size_t  npbyte = npbits / 8;
        
        zsize += header_size + n * npbyte; // sizeof(real_t) + 1 + 1 + n * npbyte;
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint8_t  exp_bits  = e[l];
        const uint8_t  prec_bits = m[l];
        const real_t   scale     = s[l];
        const uint8_t  nbits     = 1 + exp_bits + prec_bits; // number of bits per value
        const uint8_t  nbyte     = nbits / 8;

        compress( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
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
    using  real_t = double;
    
    constexpr real_t  fp_maximum  = FP_info< real_t >::maximum;
    constexpr size_t  header_size = FP_info< real_t >::header_ofs; // sizeof(real_t) + 2;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    const size_t  n2    = 2 * n;
    auto          m     = std::vector< uint32_t >( k );
    auto          e     = std::vector< uint32_t >( k );
    auto          s     = std::vector< real_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        auto  vmin = fp_maximum;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il   = U(i,l);
            const auto  u_re   = std::abs( std::real( u_il ) );
            const auto  u_im   = std::abs( std::imag( u_il ) );
            const auto  val_re = ( u_re == real_t(0) ? fp_maximum : u_re );
            const auto  val_im = ( u_im == real_t(0) ? fp_maximum : u_im );
            
            vmin = std::min( vmin, std::min( val_re, val_im ) );
            vmax = std::max( vmax, std::max( u_re, u_im ) );
        }// for

        s[l] = vmin;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

        HLR_DBG_ASSERT( std::isfinite( s[l] ) );

        const auto  nprecbits = eps_to_rate_valr( S(l) );
        const auto  nbits     = 1 + e[l] + nprecbits;

        // increase mantissa bits such that sum is multiple of 8
        m[l] = nprecbits + ( byte_pad( nbits ) - nbits );

        const size_t  npbits = 1 + e[l] + m[l]; // number of bits per value
        const size_t  npbyte = npbits / 8;
        
        zsize += header_size + n2 * npbyte; // sizeof(real_t) + 1 + 1 + n2 * npbyte; // twice because real+imag
    }// for

    //
    // convert each column to compressed form
    //

    auto            zdata = std::vector< byte_t >( zsize );
    size_t          pos   = 0;
    const real_t *  U_ptr = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value
        const size_t    nbyte     = nbits / 8;

        compress( U_ptr + l * n2, n2, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += header_size + n2*nbyte;
    }// for

    return zdata;
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
    constexpr size_t  header_size = FP_info< real_t >::header_ofs;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint8_t  exp_bits  = zdata[ pos ];
        const uint8_t  prec_bits = zdata[ pos+1 ];
        const uint8_t  nbits     = 1 + exp_bits + prec_bits;
        const uint8_t  nbyte     = nbits / 8;

        decompress( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
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
    using  real_t = double;
    
    const size_t      n           = U.nrows();
    const uint32_t    k           = U.ncols();
    size_t            pos         = 0;
    constexpr size_t  header_size = FP_info< real_t >::header_ofs;
    real_t *          U_ptr       = reinterpret_cast< real_t * >( U.data() );
    const size_t      n2          = 2 * n;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;
        const uint32_t  nbyte     = nbits / 8;

        decompress( U_ptr + l * n2, n2, zdata.data() + pos, exp_bits, prec_bits );
        pos += header_size + nbyte * n2;
    }// for
}

//
// compressed blas
//

template < typename value_t >
struct accessor
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const uint8_t      nexp_bits;
    const uint8_t      nprec_bits;
    const uint8_t      nbits;
    const uint8_t      nbyte;
    const uint64_t     prec_mask;
    const uint8_t      prec_ofs;
    const uint64_t     exp_mask;
    const uint32_t     sign_shift;
    const uint64_t     zero_val;
    const real_t       scale;
    const byte_t *     zptr;
    
    accessor ( const zarray &  zdata )
            : nexp_bits(  zdata[0] )
            , nprec_bits( zdata[1] )
            , scale( * reinterpret_cast< const real_t * >( zdata + FP_info< real_t >::scale_ofs ) )
            , nbits( 1 + nexp_bits + nprec_bits )
            , nbyte( nbits / 8 )
            , prec_mask( ( 1ul << nprec_bits ) - 1 )
            , prec_ofs( FP64::mant_bits - nprec_bits )
            , exp_mask( ( 1ul << nexp_bits  ) - 1 )
            , sign_shift( nexp_bits + nprec_bits )
            , zero_val( FP64::zero_val & (( 1ul << nbits) - 1 ) )
            , zptr( zdata.data() + FP_info< real_t >::header_ofs )
    {}

    value_t  operator () ( const size_t  i ) const
    {
        uint64_t  zval = 0;

        switch ( nbyte )
        {
            case  1 : zval = * ( reinterpret_cast< const byte1_t * >( zptr ) + i ); break;
            case  2 : zval = * ( reinterpret_cast< const byte2_t * >( zptr ) + i ); break;
            case  3 : zval = * ( reinterpret_cast< const byte3_t * >( zptr ) + i ); break;
            case  4 : zval = * ( reinterpret_cast< const byte4_t * >( zptr ) + i ); break;
            case  5 : zval = * ( reinterpret_cast< const byte5_t * >( zptr ) + i ); break;
            case  6 : zval = * ( reinterpret_cast< const byte6_t * >( zptr ) + i ); break;
            case  7 : zval = * ( reinterpret_cast< const byte7_t * >( zptr ) + i ); break;
            case  8 : zval = * ( reinterpret_cast< const byte8_t * >( zptr ) + i ); break;
            default : HLR_ERROR( "unsupported byte size" );
        }// switch
        
        if ( zval == zero_val )
            return value_t(0);
                    
        const uint64_t  mant  = zval & prec_mask;
        const uint64_t  exp   = (zval >> nprec_bits) & exp_mask;
        const uint64_t  sign  = (zval >> sign_shift) << FP64::sign_bit;
        fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };

        fival.f  = scale * ( fival.f - 1.0 );
        fival.u |= sign;

        return fival.f;
    }

    struct const_iterator
    {
        const accessor *  acc;
        const byte_t *    zptr;

        const_iterator ( const accessor *  aacc,
                         const byte_t *    aptr )
                : acc( aacc )
                , zptr( aptr )
        {}

        const_iterator & operator ++ () { zptr += acc->nbyte; return *this; }

        value_t
        operator * ()
        {
            uint64_t  zval = 0;

            switch ( acc->nbyte )
            {
                case  1 : zval = * ( reinterpret_cast< const byte1_t * >( zptr ) ); break;
                case  2 : zval = * ( reinterpret_cast< const byte2_t * >( zptr ) ); break;
                case  3 : zval = * ( reinterpret_cast< const byte3_t * >( zptr ) ); break;
                case  4 : zval = * ( reinterpret_cast< const byte4_t * >( zptr ) ); break;
                case  5 : zval = * ( reinterpret_cast< const byte5_t * >( zptr ) ); break;
                case  6 : zval = * ( reinterpret_cast< const byte6_t * >( zptr ) ); break;
                case  7 : zval = * ( reinterpret_cast< const byte7_t * >( zptr ) ); break;
                case  8 : zval = * ( reinterpret_cast< const byte8_t * >( zptr ) ); break;
                default : HLR_ERROR( "unsupported byte size" );
            }// switch
            
            if ( zval == acc->zero_val )
                return value_t(0);
            
            const uint64_t  mant  = zval & acc->prec_mask;
            const uint64_t  exp   = (zval >> acc->nprec_bits) & acc->exp_mask;
            const uint64_t  sign  = (zval >> acc->sign_shift) << FP64::sign_bit;
            fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << acc->prec_ofs) };
            
            fival.f  = ( fival.f - 1.0 ) * acc->scale;
            fival.u |= sign;
            
            return fival.f;
        }
    };

    const_iterator begin () const  { return const_iterator( this, zptr ); }
};

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
         const uint8_t                       exp_bits,
         const uint8_t                       prec_bits )
{
    const uint8_t     nbits      = 1 + exp_bits + prec_bits;
    const uint64_t    prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint8_t     prec_ofs   = FP64::mant_bits - prec_bits;
    const uint64_t    exp_mask   = ( 1ul << exp_bits  ) - 1;
    const uint32_t    sign_shift = exp_bits + prec_bits;
    const uint64_t    zero_val   = FP64::zero_val & (( 1ul << nbits) - 1 );
    const auto        scale      = alpha * zscale;
    // constexpr size_t  nbuf       = 1;
    // const size_t      nrows_buf  = (nrows / nbuf) * nbuf; // for <nbuf> step width 
    // value_t           fcache[nbuf];

    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = scale * x[j];
                fp64int_t   fival;
                
                // size_t      i   = 0;

                // for ( ; i < nrows_buf; i += nbuf, pos += nbuf )
                // {
                //     #pragma GCC ivdep
                //     for ( uint  ii = 0; ii < nbuf; ++ii )
                //     {
                //         const uint64_t  z_ij  = zA[pos+ii];
                //         const uint64_t  mant  = z_ij & prec_mask;
                //         const uint64_t  exp   = (z_ij >> prec_bits) & exp_mask;
                //         const uint64_t  sign  = (z_ij >> sign_shift) << FP64::sign_bit;
                //         fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };
                        
                //         fival.f  = ( fival.f - 1.0 );
                //         fival.u |= sign;

                //         fcache[ii] = fival.f;
                //     }// for

                //     #pragma GCC ivdep
                //     for ( uint  ii = 0; ii < nbuf; ++ii )
                //         y[i+ii] += fcache[ii] * x_j;
                // }// for

                for ( size_t  i = 0 ; i < nrows; ++i, pos++ )
                {
                    const auto  z_ij = uint64_t( zA[pos] );

                    if ( z_ij != zero_val )
                    {
                        const uint64_t  mant = z_ij & prec_mask;
                        const uint64_t  exp  = (z_ij >> prec_bits) & exp_mask;
                        const uint64_t  sign = (z_ij >> sign_shift) << FP64::sign_bit;

                        fival.u  = ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);
                        fival.f  = ( fival.f - 1.0 );
                        fival.u |= sign;
                    }// if
                    else
                        fival.f = double(0);
                    
                    y[i] += fival.f * x_j;
                }// for
            }// for
        }// case
        break;
        
        case  apply_adjoint :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                value_t    y_j = value_t(0);
                fp64int_t  fival;
                // size_t   i   = 0;

                // for ( ; i < nrows_buf; i += nbuf, pos += nbuf )
                // {
                //     #pragma GCC ivdep
                //     for ( uint  ii = 0; ii < nbuf; ++ii )
                //     {
                //         const uint64_t  z_ij  = zA[pos+ii];
                //         const uint64_t  mant  = z_ij & prec_mask;
                //         const uint64_t  exp   = (z_ij >> prec_bits) & exp_mask;
                //         const uint64_t  sign  = (z_ij >> sign_shift) << FP64::sign_bit;
                //         fp64int_t       fival = { ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };
                        
                //         fival.f  = ( fival.f - 1.0 );
                //         fival.u |= sign;

                //         fcache[ii] = fival.f;
                //     }// for
                        
                //     for ( uint  ii = 0; ii < nbuf; ++ii )
                //         y_j += fcache[ii] * x[i+ii];
                // }// for

                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                {
                    const auto  z_ij = uint64_t( zA[pos] );

                    if ( z_ij != zero_val )
                    {
                        const uint64_t  mant = z_ij & prec_mask;
                        const uint64_t  exp  = (z_ij >> prec_bits) & exp_mask;
                        const uint64_t  sign = (z_ij >> sign_shift) << FP64::sign_bit;
                        
                        fival.u  = ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);
                        fival.f  = ( fival.f - 1.0 );
                        fival.u |= sign;
                    }// if
                    else
                        fival.f = double(0);
                    
                    y_j += fival.f * x[i];
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
    #if 0
    mulvec( nrows, ncols, op_A, alpha, accessor< value_t >( zA ), x, y );
    #else
    using  real_t = Hpro::real_type_t< value_t >;

    const uint8_t      exp_bits  = zA[0];
    const uint8_t      prec_bits = zA[1];
    const uint8_t      nbits     = 1 + exp_bits + prec_bits;
    const uint8_t      nbyte     = nbits / 8;
    real_t             scale     = * ( reinterpret_cast< const real_t * >( zA.data() + FP_info< real_t >::scale_ofs ) );
    constexpr size_t   data_ofs  = FP_info< real_t >::header_ofs;
    
    switch ( nbyte )
    {
        case  1 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  2 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  3 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  4 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  5 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  6 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  7 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        case  8 : mulvec( nrows, ncols, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + data_ofs ), x, y, exp_bits, prec_bits ); break;
        default :
            HLR_ERROR( "unsupported byte size" );
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
    using  real_t = Hpro::real_type_t< value_t >;

    constexpr size_t  scale_ofs = FP_info< real_t >::scale_ofs;
    constexpr size_t  data_ofs  = FP_info< real_t >::header_ofs;
    size_t            pos       = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  exp_bits  = zA[pos];
                const uint8_t  prec_bits = zA[pos+1];
                const uint8_t  nbyte     = ( 1 + exp_bits + prec_bits ) / 8;
                const real_t   scale     = * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) );
        
                HLR_ASSERT( pos + data_ofs + nrows * nbyte <= zA.size() );

                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + pos + data_ofs ), x+l, y, exp_bits, prec_bits ); break;
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
                const uint8_t  exp_bits  = zA[pos];
                const uint8_t  prec_bits = zA[pos+1];
                const uint8_t  nbyte     = ( 1 + exp_bits + prec_bits ) / 8;
                const real_t   scale     = * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) );

                HLR_ASSERT( pos + data_ofs + nrows * nbyte <= zA.size() );
                
                switch ( nbyte )
                {
                    case  1 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte1_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  2 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte2_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte3_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte4_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte5_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte6_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte7_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, scale, reinterpret_cast< const byte8_t * >( zA.data() + pos + data_ofs ), x, y+l, exp_bits, prec_bits ); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                pos += data_ofs + nbyte * nrows;
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::aflp

#endif // __HLR_UTILS_DETAIL_AFLP_HH
