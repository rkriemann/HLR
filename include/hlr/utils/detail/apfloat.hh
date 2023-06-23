#ifndef __HLR_UTILS_DETAIL_APFLOAT_HH
#define __HLR_UTILS_DETAIL_APFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/apfloat
// Description : functions for adaptive padded floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/utils/detail/byte_n.hh>

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

namespace hlr { namespace compress { namespace apfloat {

using byte_t = uint8_t;

constexpr byte_t    fp32_mant_bits   = 23;
constexpr byte_t    fp32_exp_bits    = 8;
constexpr byte_t    fp32_sign_bit    = 31;
constexpr uint64_t  fp32_exp_highbit = 0b10000000;
constexpr uint32_t  fp32_zero_val    = 0xffffffff;
constexpr float     fp32_infinity    = std::numeric_limits< float >::infinity();

constexpr uint32_t  fp64_mant_bits   = 52;
constexpr uint32_t  fp64_exp_bits    = 11;
constexpr uint32_t  fp64_sign_bit    = 63;
constexpr uint64_t  fp64_exp_highbit = 0b10000000000;
constexpr uint64_t  fp64_zero_val    = 0xffffffffffffffff;
constexpr double    fp64_infinity    = std::numeric_limits< double >::infinity();

inline
byte_t
eps_to_rate ( const double eps )
{
    // |d_i - ~d_i| ≤ 2^(-m) ≤ ε with m = remaining mantissa length
    return std::max< double >( 1, std::ceil( -std::log2( eps ) ) );
}

inline
uint32_t
tol_to_rate ( const double  tol )
{
    return uint32_t( std::max< double >( 1, -std::log2( tol ) ) ) + 1;
}

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

// return actual memory size of compressed data
inline size_t  byte_size  ( const zarray &  v   ) { return sizeof(v) + v.size(); }

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
template < typename value_t >
void
compress_fp32 ( const value_t *  data,
                const size_t     nsize,
                byte_t *         zdata,
                const float      scale,
                const uint32_t   exp_bits,
                const uint32_t   prec_bits )
{
    constexpr uint32_t  fp32_exp_mask  = ((1u << fp32_exp_bits)  - 1);
    constexpr uint32_t  fp32_mant_mask = ((1u << fp32_mant_bits) - 1);
    const uint32_t      nbits          = 1 + exp_bits + prec_bits;
    const uint32_t      nbyte          = nbits / 8;
    const uint32_t      exp_mask       = ( 1 << exp_bits ) - 1;                  // bit mask for exponent
    const uint32_t      prec_ofs       = fp32_mant_bits - prec_bits;
    const uint32_t      zero_val       = fp32_zero_val & (( 1 << nbits) - 1 );
        
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + 2, & scale, 4 );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ];                   // mark zero entries
    bool              sign[ nbuf ];                   // holds sign per entry
    float             fbuf[ nbuf ];                   // holds rescaled value
    uint32_t          ibuf[ nbuf ];                   // holds value in compressed format
    size_t            pos = 6;
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
            const float  val  = data[i+j];
            const auto   aval = std::abs( val );

            zero[j] = ( aval == float(0) );
            sign[j] = ( aval != val );
            fbuf[j] = std::max( scale * aval + 1, 2.f ); // prevent rounding issues when converting from fp64

            HLR_DBG_ASSERT( fbuf[j] >= float(2) );
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & fbuf[j] ) );
            const uint32_t  sexp  = ( isval >> fp32_mant_bits ) & fp32_exp_mask; // extract exponent
            const uint32_t  smant = ( isval & fp32_mant_mask );                  // and mantissa
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
            case  4 : { auto ptr = reinterpret_cast< uint32_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< byte3_t *  >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< uint16_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = uint16_t( ibuf[j] & 0xffff ); } break;
            case  1 : { auto ptr = & zdata[pos];                                   for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = byte_t( ibuf[j] & 0xff   ); } break;
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
            const float     sval  = std::max( scale * std::abs(val) + 1, 2.f );
            const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & sval ) );
            const uint32_t  sexp  = ( isval >> fp32_mant_bits ) & fp32_exp_mask;
            const uint32_t  smant = ( isval & fp32_mant_mask );
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

template < typename value_t >
void
decompress_fp32 ( value_t *        data,
                  const size_t     nsize,
                  const byte_t *   zdata,
                  const uint32_t   exp_bits,
                  const uint32_t   prec_bits )
{
    const uint32_t  nbits      = 1 + exp_bits + prec_bits;
    const uint32_t  nbyte      = nbits / 8;
    const uint32_t  prec_mask  = ( 1 << prec_bits ) - 1;
    const uint32_t  prec_ofs   = fp32_mant_bits - prec_bits;
    const uint32_t  exp_mask   = ( 1 << exp_bits ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint32_t  zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );
    float           scale;

    // get scaling factor
    memcpy( & scale, zdata, 4 );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    float             fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos = 4;
    size_t            i   = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        switch ( nbyte )
        {
            case  4 : { auto ptr = reinterpret_cast< const uint32_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  3 : { auto ptr = reinterpret_cast< const byte3_t *  >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  2 : { auto ptr = reinterpret_cast< const uint16_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  1 : { auto ptr = & zdata[pos];                                         for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      zval  = ibuf[j];
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const bool      sign  = zval >> sign_shift;
            const uint32_t  irval = (uint32_t(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint32_t(mant) << prec_ofs);

            zero[j] = ( zval == zero_val );
            fbuf[j] = value_t( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
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
            const bool      sign  = zval >> sign_shift;
            const uint32_t  irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
                
            data[i] = value_t( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
        }// else

        pos += nbyte;
    }// for
}

//
// compress data needing more than 32 bits
//
inline
void
compress_fp64 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata,
                const double    scale,
                const uint32_t  exp_bits,
                const uint32_t  prec_bits )
{
    constexpr uint64_t  fp64_exp_mask  = ((1ul << fp64_exp_bits)  - 1);
    constexpr uint64_t  fp64_mant_mask = ((1ul << fp64_mant_bits) - 1);
    const uint32_t      nbits          = 1 + exp_bits + prec_bits;
    const uint32_t      nbyte          = nbits / 8;
    const uint64_t      exp_mask       = ( 1u << exp_bits ) - 1;                 // bit mask for exponent
    const uint32_t      prec_ofs       = fp64_mant_bits - prec_bits;
    const uint64_t      zero_val       = fp64_zero_val & (( 1ul << nbits) - 1 );
        
    //
    // store header (exponent bits, precision bits and scaling factor)
    //
        
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + 2, & scale, 8 );

    //
    // in case of 8 byte, just copy data
    //

    if ( nbyte == 8 )
    {
        std::copy( data, data + nsize, reinterpret_cast< double * >( zdata + 10 ) );
        return;
    }// if
    
    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    bool              sign[ nbuf ]; // holds sign per entry
    double            fbuf[ nbuf ]; // holds rescaled value
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos = 10;
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
            fbuf[j] = scale * aval + 1;

            HLR_DBG_ASSERT( fbuf[j] >= double(2) );
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & fbuf[j] ) );
            const uint64_t  sexp  = ( isval >> fp64_mant_bits ) & fp64_exp_mask;
            const uint64_t  smant = ( isval & fp64_mant_mask );
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
            case  4 : { auto ptr = reinterpret_cast< uint32_t * >(    & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = uint32_t(ibuf[j]); } break;
            case  5 : { auto ptr = reinterpret_cast< byte5_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< byte6_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< byte7_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< uint64_t * >(   & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch

        pos += nbyte * nbuf;
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const double  val  = data[i];
        uint64_t         zval = zero_val;
            
        if ( std::abs( val ) != double(0) )
        {
            const bool      zsign = ( val < 0 );
            const double    sval  = scale * std::abs(val) + 1;
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & sval ) );
            const uint64_t  sexp  = ( isval >> fp64_mant_bits ) & fp64_exp_mask;
            const uint64_t  smant = ( isval & fp64_mant_mask );
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
            case  4 : break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch
            
        zdata[pos+3] = ( zval & 0x00000000ff000000 ) >> 24;
        zdata[pos+2] = ( zval & 0x0000000000ff0000 ) >> 16;
        zdata[pos+1] = ( zval & 0x000000000000ff00 ) >> 8;
        zdata[pos]   = ( zval & 0x00000000000000ff );

        pos += nbyte;
    }// for
}

inline
void
decompress_fp64 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata,
                  const uint32_t  exp_bits,
                  const uint32_t  prec_bits )
{
    const uint32_t  nbits      = 1 + exp_bits + prec_bits;
    const uint32_t  nbyte      = nbits / 8;
    const uint64_t  prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint32_t  prec_ofs   = fp64_mant_bits - prec_bits;
    const uint64_t  exp_mask   = ( 1ul << exp_bits  ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint64_t  zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );
    double          scale;

    // just retrieve data for nbyte == 8
    if ( nbyte == 8 )
    {
        std::copy( reinterpret_cast< const double * >( zdata + 8 ),
                   reinterpret_cast< const double * >( zdata + 8 ) + nsize,
                   data );
        return;
    }// if
         
    // get scaling factor
    memcpy( & scale, zdata, 8 );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    double            fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos = 8;
    size_t            i   = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        switch ( nbyte )
        {
            case  4 : { auto ptr = reinterpret_cast< const uint32_t * >(    & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  5 : { auto ptr = reinterpret_cast< const byte5_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  6 : { auto ptr = reinterpret_cast< const byte6_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  7 : { auto ptr = reinterpret_cast< const byte7_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            case  8 : { auto ptr = reinterpret_cast< const uint64_t * >(   & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
            default : HLR_ERROR( "unsupported storage size" );
        }// switch
            
        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      zval  = ibuf[j];
            const uint64_t  mant  = zval & prec_mask;
            const uint64_t  exp   = (zval >> prec_bits) & exp_mask;
            const bool      sign  = zval >> sign_shift;
            const uint64_t  irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);

            zero[j] = ( zval == zero_val );
            fbuf[j] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
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
            const bool      sign  = zval >> sign_shift;
            const uint64_t  irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
            const double    rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

            data[i] = rval;
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // look for min/max value (> 0!)
    //
    
    float  vmin = fp32_infinity;
    float  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == double(0) ? fp32_infinity : d_i );

        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > float(0) );
    
    
    const float     scale      = 1.0 / vmin;                                                                 // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits   = std::max< float >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  nbits      = byte_pad( 1 + exp_bits + config.bitrate );                                  // total no. of bits per value
    const uint32_t  nbyte      = nbits / 8;
    const uint32_t  prec_bits  = nbits - 1 - exp_bits;                                                       // actual number of precision bits
    auto            zdata      = std::vector< byte_t >( 4 + 1 + 1 + nsize * nbyte );                         // array storing compressed data

    HLR_ASSERT( nbits     <= 32 );
    HLR_ASSERT( prec_bits <= fp32_mant_bits );

    compress_fp32( data, nsize, zdata.data(), scale, exp_bits, prec_bits );

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // look for min/max value (> 0!)
    //
    
    double  vmin = fp64_infinity;
    double  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == double(0) ? fp64_infinity : d_i );
            
        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > double(0) );
    
    
    const double    scale      = 1.0 / vmin;                                                                  // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits   = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  nbits      = byte_pad( 1 + exp_bits + config.bitrate );                                   // total no. of bits per value
    const uint32_t  nbyte      = nbits / 8;
    const uint32_t  prec_bits  = nbits - 1 - exp_bits;                                                        // actual number of precision bits
    auto            zdata      = std::vector< byte_t >();                                                     // array storing compressed data

    HLR_ASSERT( nbits <= 64 );

    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
        zdata.resize( 4 + 1 + 1 + nsize * nbyte );
        compress_fp32( data, nsize, zdata.data(), scale, exp_bits, prec_bits );
    }// if
    else
    {
        HLR_DBG_ASSERT( nbyte >= 4 );
        
        zdata.resize( 8 + 1 + 1 + nsize * nbyte );
        compress_fp64( data, nsize, zdata.data(), scale, exp_bits, prec_bits );
    }// else

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    
    decompress_fp32( dest, nsize, zdata.data() + 2, exp_bits, prec_bits );
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    // and then the compressed data
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    
    if (( nbits <= 32 ) && ( prec_bits <= fp32_mant_bits ))
    {
        decompress_fp32( dest, nsize, zdata.data() + 2, exp_bits, prec_bits );
    }// if
    else
    {
        decompress_fp64( dest, nsize, zdata.data() + 2, exp_bits, prec_bits );
    }// if
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
              const blas::vector< Hpro::real_type_t< value_t > > &  S );

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U );

template <>
inline
zarray
compress_lr< float > ( const blas::matrix< float > &  U,
                       const blas::vector< float > &  S )
{
    using  real_t = float;
    
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
        auto  vmin = fp32_infinity;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == real_t(0) ? fp32_infinity : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

        const auto  nprecbits = tol_to_rate( S(l) );
        const auto  nbits     = 1 + e[l] + nprecbits;

        // increase mantissa bits such that sum is multiple of 8
        m[l] = nprecbits + ( byte_pad( nbits ) - nbits );

        const size_t  npbits = 1 + e[l] + m[l]; // number of bits per value
        const size_t  npbyte = npbits / 8;
        
        zsize += sizeof(float)  + 1 + 1 + n * npbyte;
    }// for

    // for ( uint32_t  l = 0; l < k; ++l )
    //     std::cout << e[l] << '/' << m[l] << ", ";
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value
        const size_t    nbyte     = nbits / 8;

        compress_fp32( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += 6 + n*nbyte;
    }// for

    return zdata;
}

template <>
inline
zarray
compress_lr< double > ( const blas::matrix< double > &  U,
                        const blas::vector< double > &  S )
{
    using  real_t = double;
    
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
        auto  vmin = fp64_infinity;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == real_t(0) ? fp64_infinity : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

        const auto  nprecbits = tol_to_rate( S(l) );
        const auto  nbits     = 1 + e[l] + nprecbits;

        // increase mantissa bits such that sum is multiple of 8
        m[l] = nprecbits + ( byte_pad( nbits ) - nbits );

        const size_t  npbits = 1 + e[l] + m[l]; // number of bits per value
        const size_t  npbyte = npbits / 8;
        
        if (( m[l] <= 23 ) && ( nbits <= 32 ))
            zsize += sizeof(float)  + 1 + 1 + n * npbyte;
        else
            zsize += sizeof(double) + 1 + 1 + n * npbyte;
    }// for

    // for ( uint32_t  l = 0; l < k; ++l )
    //     std::cout << e[l] << '/' << m[l] << std::endl;
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value
        const size_t    nbyte     = nbits / 8;

        if (( prec_bits <= fp32_mant_bits ) && ( nbyte <= 4 ))
        {
            compress_fp32( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
            pos += 6 + n*nbyte;
        }// if
        else
        {
            compress_fp64( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
            pos += 10 + n*nbyte;
        }// else
    }// for

    return zdata;
}

template <>
inline
void
decompress_lr< float > ( const zarray &           zdata,
                         blas::matrix< float > &  U )
{
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
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;
        const uint32_t  nbyte     = nbits / 8;

        decompress_fp32( U.data() + l * n, n, zdata.data() + pos + 2, exp_bits, prec_bits );
        pos += 6 + nbyte * n;
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
decompress_lr< double > ( const zarray &            zdata,
                          blas::matrix< double > &  U )
{
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
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;
        const uint32_t  nbyte     = nbits / 8;

        pos += 2;
        
        // std::cout << exp_bits << '/' << prec_bits << std::endl;
        
        if (( prec_bits <= 23 ) && ( nbits <= 32 ))
        {
            decompress_fp32( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
            pos += 4 + nbyte * n;
        }// if
        else
        {
            decompress_fp64( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
            pos += 8 + nbyte * n;
        }// else
    }// for
}

template <>
inline
void
decompress_lr< std::complex< double > > ( const zarray &                            zdata,
                                          blas::matrix< std::complex< double > > &  U )
{
    HLR_ERROR( "TODO" );
}

}}}// namespace hlr::compress::apfloat

#endif // __HLR_UTILS_DETAIL_APFLOAT_HH
