#ifndef __HLR_UTILS_DETAIL_AFL_HH
#define __HLR_UTILS_DETAIL_AFL_HH
//
// Project     : HLR
// Module      : utils/detail/afl
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/arith/blas.hh>

////////////////////////////////////////////////////////////
//
// compression using adaptive float representation
//
// - exponent size based on exponent range of input
// - scale input D such that |d_i| ≥ 1
// - mantissa size depends on precision
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace afl {

using byte_t = uint8_t;

constexpr uint32_t  fp32_mant_bits   = 23;
constexpr uint32_t  fp32_exp_bits    = 8;
constexpr uint32_t  fp32_sign_bit    = 31;
constexpr uint64_t  fp32_exp_highbit = 1 << (fp32_exp_bits-1);
constexpr uint32_t  fp32_zero_val    = 0xffffffff;
constexpr float     fp32_infinity    = std::numeric_limits< float >::infinity();

constexpr uint32_t  fp64_mant_bits   = 52;
constexpr uint32_t  fp64_exp_bits    = 11;
constexpr uint32_t  fp64_sign_bit    = 63;
constexpr uint64_t  fp64_exp_highbit = 1 << (fp64_exp_bits-1);
constexpr uint64_t  fp64_zero_val    = 0xffffffffffffffff;
constexpr double    fp64_infinity    = std::numeric_limits< double >::infinity();

// return byte padded value of <n>
inline size_t byte_pad ( size_t  n )
{
    return ( n % 8 != 0 ) ? n + (8 - n%8) : n;
}
    
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

inline size_t  byte_size  ( const zarray &  v   ) { return sizeof(v) + v.size(); }
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
    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    const uint32_t  nbits    = 1 + exp_bits + prec_bits;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;                  // bit mask for exponent
    const uint32_t  prec_ofs = fp32_mant_bits - prec_bits;
    const uint32_t  zero_val = fp32_zero_val & (( 1 << nbits) - 1 );
        
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + 2, & scale, 4 );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    bool              sign[ nbuf ]; // holds sign per entry
    float             fbuf[ nbuf ]; // holds rescaled value
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos  = 6;
    uint32_t          bpos = 0; // start bit position in current byte
    size_t            i    = 0;

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
            const uint32_t  sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1); // extract exponent
            const uint32_t  smant = ( isval & ((1u << fp32_mant_bits) - 1) );                  // and mantissa
            const uint32_t  zexp  = sexp & exp_mask;    // extract needed exponent
            const uint32_t  zmant = smant >> prec_ofs;  // and precision bits
                
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write into data buffer
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            auto  zval  = ibuf[j];
            uint32_t  sbits = 0; // number of already stored bits of zval
            
            do
            {
                const uint32_t  crest = 8 - bpos;       // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;  // remaining bits in zval
                const byte_t    zbyte = zval & 0xff;    // lowest byte of zval
                    
                // HLR_DBG_ASSERT( pos < zsize );
                    
                zdata[pos] |= (zbyte << bpos);
                zval      >>= crest;
                sbits      += crest;
                    
                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );
        }// for
    }// for
        
    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const float  val  = data[i];
        uint32_t     zval = zero_val;
            
        if ( std::abs( val ) != float(0) )
        {
            const bool      zsign = ( val < 0 );
            const float     sval  = std::max( scale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
            const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & sval ) );
            const uint32_t  sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
            const uint32_t  smant = ( isval & ((1u << fp32_mant_bits) - 1) );
            const uint32_t  zexp  = sexp & exp_mask;
            const uint32_t  zmant = smant >> prec_ofs;
                
            zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        uint32_t  sbits = 0; // number of already stored bits of zval
            
        do
        {
            const uint32_t  crest = 8 - bpos;       // remaining bits in current byte
            const uint32_t  zrest = nbits - sbits;  // remaining bits in zval
            const byte_t    zbyte = zval & 0xff;    // lowest byte of zval
                
            // HLR_DBG_ASSERT( pos < zsize );
                
            zdata[pos] |= (zbyte << bpos);
            zval      >>= crest;
            sbits      += crest;
                
            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );
    }// for
}

template < typename value_t >
void
decompress_fp32 ( value_t *       data,
                  const size_t    nsize,
                  const byte_t *  zdata,
                  const uint32_t  exp_bits,
                  const uint32_t  prec_bits )
{
    const uint32_t  nbits      = 1 + exp_bits + prec_bits;
    const uint32_t  prec_mask  = ( 1 << prec_bits ) - 1;
    const uint32_t  prec_ofs   = fp32_mant_bits - prec_bits;
    const uint32_t  exp_mask   = ( 1 << exp_bits ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint32_t  zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );
    float           scale;

    // read scaling factor
    memcpy( & scale, zdata, 4 );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    float             fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos  = 4;
    uint32_t          bpos = 0;
    size_t            i    = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            uint32_t  zval  = 0;
            uint32_t  sbits = 0;  // already read bits of zval
            
            do
            {
                // HLR_DBG_ASSERT( pos < zdata );
        
                const uint32_t  crest = 8 - bpos;                               // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;                          // remaining bits to read for zval
                const byte_t    zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                const byte_t    data  = (zdata[pos] >> bpos) & zmask;           // part of zval in current byte
                
                zval  |= (uint32_t(data) << sbits); // lowest to highest bit in zdata
                sbits += crest;

                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );

            ibuf[j] = zval;
        }// for
            
        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      zval  = ibuf[j];
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const bool      sign  = zval >> sign_shift;
            const uint32_t  irval = (uint32_t(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint32_t(mant) << prec_ofs);

            zero[j] = ( zval == zero_val );
            fbuf[j] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                fbuf[j] = double(0);

        // copy values
        for ( size_t  j = 0; j < nbuf; ++j )
            data[i+j] = fbuf[j];
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint32_t  zval  = 0;
        uint32_t  sbits = 0;
            
        do
        {
            // HLR_DBG_ASSERT( pos < zdata );
        
            const uint32_t  crest = 8 - bpos;
            const uint32_t  zrest = nbits - sbits;
            const byte_t    zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
            const byte_t    data  = (zdata[pos] >> bpos) & zmask;
                
            zval  |= (uint32_t(data) << sbits);
            sbits += crest;

            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );

        if ( zval == zero_val )
            data[i] = 0;
        else
        {
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const bool      sign  = zval >> sign_shift;
            const uint32_t  irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
            const float     rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;
                
            data[i] = double( rval );
        }// else
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
    const uint32_t  nbits    = 1 + exp_bits + prec_bits;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;
    const uint32_t  prec_ofs = fp64_mant_bits - prec_bits;
    const uint32_t  zero_val = fp32_zero_val & (( 1 << nbits) - 1 );

    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + 2, & scale, 8 );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    bool              sign[ nbuf ]; // holds sign per entry
    double            fbuf[ nbuf ]; // holds rescaled value
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos  = 10;
    uint32_t          bpos = 0; // start bit position in current byte
    size_t            i    = 0;

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
            const double  val  = data[i+j];
            const auto    aval = std::abs( val );

            zero[j] = ( aval == double(0) );
            sign[j] = ( aval != val );
            fbuf[j] = scale * aval + 1;

            HLR_DBG_ASSERT( fbuf[j] >= double(2) );
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & fbuf[j] ) );
            const uint64_t  sexp  = ( isval >> fp64_mant_bits ) & ((1ul << fp64_exp_bits) - 1);
            const uint64_t  smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;
                
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write into data buffer
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            auto  zval  = ibuf[j];
            uint32_t  sbits = 0; // number of already stored bits of zval
            
            do
            {
                const uint32_t  crest = 8 - bpos;       // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;  // remaining bits in zval
                const byte_t    zbyte = zval & 0xff;    // lowest byte of zval
                    
                // HLR_DBG_ASSERT( pos < zsize );
                    
                zdata[pos] |= (zbyte << bpos);
                zval      >>= crest;
                sbits      += crest;
            
                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );
        }// for
    }// for

    // handle remaining data
    for ( ; i < nsize; ++i )
    {
        const double  val  = data[i];
        uint64_t      zval = zero_val;
            
        if ( std::abs( val ) != double(0) )
        {
            const bool      zsign = ( val < 0 );
            const double    sval  = scale * std::abs(val) + 1;
            const uint64_t  isval = (*reinterpret_cast< const uint64_t * >( & sval ) );
            const uint64_t  sexp  = ( isval >> fp64_mant_bits ) & ((1ul << fp64_exp_bits) - 1);
            const uint64_t  smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;

            zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        uint32_t  sbits = 0;
            
        do
        {
            const uint32_t  crest = 8 - bpos;       // remaining bits in current byte
            const uint32_t  zrest = nbits - sbits;  // remaining bits in zval
            const byte_t    zbyte = zval & 0xff;    // lowest byte of zval

            // HLR_DBG_ASSERT( pos < zsize );
        
            zdata[pos] |= (zbyte << bpos);
            zval      >>= crest;
            sbits      += crest;
            
            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );
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
    const uint64_t  prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint32_t  prec_ofs   = fp64_mant_bits - prec_bits;
    const uint64_t  exp_mask   = ( 1ul << exp_bits ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint64_t  zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );
    double          scale;

    // read scaling factor
    memcpy( & scale, zdata, 8 );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    double            fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos  = 8;
    uint32_t          bpos = 0;                          // bit position in current byte
    size_t            i    = 0;

    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            uint64_t  zval  = 0;
            uint32_t  sbits = 0;  // already read bits of zval
            
            do
            {
                // HLR_DBG_ASSERT( pos < zdata );
        
                const uint32_t  crest = 8 - bpos;                               // remaining bits in current byte
                const uint32_t  zrest = nbits - sbits;                          // remaining bits to read for zval
                const byte_t    zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                const byte_t    data  = (zdata[pos] >> bpos) & zmask;           // part of zval in current byte
                
                zval  |= (uint64_t(data) << sbits); // lowest to highest bit in zdata
                sbits += crest;

                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );

            ibuf[j] = zval;
        }// for

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
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint64_t  zval  = 0;
        uint32_t  sbits = 0;
            
        do
        {
            // HLR_DBG_ASSERT( pos < zdata );
        
            const uint32_t  crest = 8 - bpos;
            const uint32_t  zrest = nbits - sbits;
            const byte_t    zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
            const byte_t    data  = (zdata[pos] >> bpos) & zmask;
                
            zval  |= (uint64_t(data) << sbits);
            sbits += crest;

            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );

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
    const uint32_t  prec_bits  = std::min< uint32_t >( fp32_mant_bits, config.bitrate );                         // total no. of bits per value
    const size_t    nbits      = 1 + exp_bits + prec_bits;                                                   // number of bits per value
    auto            zdata      = std::vector< byte_t >( 4 + 1 + 1 + byte_pad( nsize * nbits ) / 8 );

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
    
    const double    scale     = 1.0 / vmin;                                                                  // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits  = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  prec_bits = std::min< uint32_t >( fp64_mant_bits, config.bitrate );                          // total no. of bits per value
    const size_t    nbits     = 1 + exp_bits + prec_bits;                                                    // number of bits per value
    auto            zdata     = std::vector< byte_t >();                                                     // array storing compressed data

    if (( prec_bits <= 23 ) && ( nbits <= 32 ))
    {
        zdata.resize( 4 + 1 + 1 + byte_pad( nsize * nbits ) / 8 );
        compress_fp32( data, nsize, zdata.data(), scale, exp_bits, prec_bits );
    }// if
    else
    {
        zdata.resize( 8 + 1 + 1 + byte_pad( nsize * nbits ) / 8 );
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
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    
    HLR_ASSERT( nbits     <= 32 );
    HLR_ASSERT( prec_bits <= fp32_mant_bits );

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

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          m     = std::vector< uint32_t >( k );
    auto          e     = std::vector< uint32_t >( k );
    auto          s     = std::vector< real_t >( k );
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
        e[l] = uint32_t( std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = std::min( fp32_mant_bits, tol_to_rate( S(l) ) );

        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += 4 + 1 + 1 + byte_pad( n * nbits ) / 8;
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
        const size_t    nbits     = 1 + exp_bits + prec_bits;

        compress_fp32( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += 6 + byte_pad( n * nbits ) / 8;
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
    // first, determine exponent bits and mantissa bits for all
    // columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          m     = std::vector< uint32_t >( k );
    auto          e     = std::vector< uint32_t >( k );
    auto          s     = std::vector< real_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        double  vmin = fp64_infinity;
        double  vmax = 0;

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  d_i = std::abs( U(i,l) );
            const auto  val = ( d_i == double(0) ? fp64_infinity : d_i );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, d_i );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint32_t( std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = std::min( fp64_mant_bits, tol_to_rate( S(l) ) );

        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        if (( m[l] <= 23 ) && ( nbits <= 32 )) zsize += 4; // handle in FP32
        else                                   zsize += 8; // handle in FP64
        
        zsize += 1 + 1 + byte_pad( n * nbits ) / 8;
    }// for

    // for ( uint32_t  l = 0; l < k; ++l )
    //   std::cout << e[l] << '/' << m[l] << ", ";
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
        const size_t    nbits     = 1 + exp_bits + prec_bits;

        if (( prec_bits <= fp32_mant_bits ) && ( nbits <= 32 ))
        {
            compress_fp32( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
            pos += 6 + byte_pad( n * nbits ) / 8;
        }// if
        else
        {
            compress_fp64( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
            pos += 10 + byte_pad( n * nbits ) / 8;
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

        decompress_fp32( U.data() + l * n, n, zdata.data() + pos + 2, exp_bits, prec_bits );
        pos += 6 + byte_pad( nbits * n ) / 8;
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
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;

        pos += 2;
        
        if (( prec_bits <= 23 ) && ( nbits <= 32 ))
        {
            decompress_fp32( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
            pos += 4 + byte_pad( nbits * n ) / 8;
        }// if
        else
        {
            decompress_fp64( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
            pos += 8 + byte_pad( nbits * n ) / 8;
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

}}}// namespace hlr::compress::afl

#endif // __HLR_UTILS_DETAIL_AFL_HH
