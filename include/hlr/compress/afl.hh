#ifndef __HLR_UTILS_DETAIL_AFL_HH
#define __HLR_UTILS_DETAIL_AFL_HH
//
// Project     : HLR
// Module      : compress/afl
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/arith/blas.hh>
#include <hlr/compress/byte_n.hh>

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

template < typename real_t >
struct fp_info
{};

template <>
struct fp_info< float >
{
    constexpr static uint32_t  mant_bits   = 23;
    constexpr static uint32_t  exp_bits    = 8;
    constexpr static uint32_t  sign_bit    = 31;
    constexpr static uint64_t  exp_highbit = 1 << (exp_bits-1);

    constexpr static uint32_t  zero_val    = 0xffffffff;
    constexpr static float     maximum     = std::numeric_limits< float >::max();
};
    
template <>
struct fp_info< double >
{
    constexpr static uint32_t  mant_bits   = 52;
    constexpr static uint32_t  exp_bits    = 11;
    constexpr static uint32_t  sign_bit    = 63;
    constexpr static uint64_t  exp_highbit = 1 << (exp_bits-1);

    constexpr static uint64_t  zero_val    = 0xffffffffffffffff;
    constexpr static float     maximum     = std::numeric_limits< float >::max();
};

using FP32 = fp_info< float >;
using FP64 = fp_info< double >;

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_aplr ( const double  eps ) { return eps_to_rate( eps ) + 1; }

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v   ) { return v.size(); }
inline config  get_config      ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

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
compress ( const float *   data,
           const size_t    nsize,
           byte_t *        zdata,
           const float     scale,
           const uint32_t  exp_bits,
           const uint32_t  prec_bits )
{
    using value_t = float;
    
    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    const uint32_t  nbits    = 1 + exp_bits + prec_bits;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;                  // bit mask for exponent
    const uint32_t  prec_ofs = FP32::mant_bits - prec_bits;
    const uint32_t  zero_val = FP32::zero_val & (( 1 << nbits) - 1 );
        
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
            const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & ((1u << FP32::exp_bits) - 1); // extract exponent
            const uint32_t  smant = ( isval & ((1u << FP32::mant_bits) - 1) );                  // and mantissa
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
            const uint32_t  sexp  = ( isval >> FP32::mant_bits ) & ((1u << FP32::exp_bits) - 1);
            const uint32_t  smant = ( isval & ((1u << FP32::mant_bits) - 1) );
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

inline
void
decompress ( float *         data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint32_t  exp_bits,
             const uint32_t  prec_bits )
{
    using  value_t = float;
    
    const uint32_t  nbits      = 1 + exp_bits + prec_bits;
    const uint32_t  prec_mask  = ( 1 << prec_bits ) - 1;
    const uint32_t  prec_ofs   = FP32::mant_bits - prec_bits;
    const uint32_t  exp_mask   = ( 1 << exp_bits ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint32_t  zero_val   = FP32::zero_val & (( 1 << nbits) - 1 );
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
            const uint32_t  irval = (uint32_t(exp | FP32::exp_highbit) << FP32::mant_bits) | (uint32_t(mant) << prec_ofs);

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
            const uint32_t  irval = ((exp | FP32::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs);
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
compress ( const double *  data,
           const size_t    nsize,
           byte_t *        zdata,
           const double    scale,
           const uint32_t  exp_bits,
           const uint32_t  prec_bits )
{
    const uint32_t  nbits    = 1 + exp_bits + prec_bits;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;
    const uint32_t  prec_ofs = FP64::mant_bits - prec_bits;
    const uint32_t  zero_val = FP32::zero_val & (( 1 << nbits) - 1 );

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

    // size_t    nexp     = 0;
    // uint64_t  zexp_old = -1;
        
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
            const uint64_t  sexp  = ( isval >> FP64::mant_bits ) & ((1ul << FP64::exp_bits) - 1);
            const uint64_t  smant = ( isval & ((1ul << FP64::mant_bits) - 1) );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;
                
            // if ( zexp != zexp_old )
            // {
            //     zexp_old = zexp;
            //     nexp++;
            // }// if
            
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write into data buffer
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            auto      zval  = ibuf[j];
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
            const uint64_t  sexp  = ( isval >> FP64::mant_bits ) & ((1ul << FP64::exp_bits) - 1);
            const uint64_t  smant = ( isval & ((1ul << FP64::mant_bits) - 1) );
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;

            // if ( zexp != zexp_old )
            // {
            //     zexp_old = zexp;
            //     nexp++;
            // }// if
            
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

    // std::cout << nsize << " / " << nexp << " | " << ( nbits * nsize ) / 8 << " / " << (( nsize - nexp ) * exp_bits ) / 8 << std::endl;
}

inline
void
decompress ( double *        data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint32_t  exp_bits,
             const uint32_t  prec_bits )
{
    const uint32_t  nbits      = 1 + exp_bits + prec_bits;
    const uint64_t  prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint32_t  prec_ofs   = FP64::mant_bits - prec_bits;
    const uint64_t  exp_mask   = ( 1ul << exp_bits ) - 1;
    const uint32_t  sign_shift = exp_bits + prec_bits;
    const uint64_t  zero_val   = FP64::zero_val & (( 1ul << nbits) - 1 );
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
            const uint64_t  irval = ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);

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
            const uint64_t  irval = ((exp | FP64::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);
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
           const size_t     dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    constexpr real_t  fp_maximum = fp_info< real_t >::maximum;

    //
    // look for min/max value (> 0!)
    //
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    auto          vmin  = fp_maximum;
    auto          vmax  = real_t(0);

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == real_t(0) ? fp_maximum : d_i );
            
        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_ASSERT( vmin > real_t(0) );

    if ( vmin == fp_maximum )
    {
        //
        // in case of zero data, return special data
        //

        auto  zdata = std::vector< byte_t >( 2 );
        
        zdata[0] = 0;
        zdata[1] = 0;

        return zdata;
    }// if
    
    const auto      scale     = real_t(1) / vmin;                                                            // scale all values v_i such that |v_i| >= 1
    const uint32_t  exp_bits  = std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ); // no. of bits needed to represent exponent
    const uint32_t  prec_bits = std::min< uint32_t >( fp_info< real_t >::mant_bits, config.bitrate );        // total no. of bits per value
    const size_t    nbits     = 1 + exp_bits + prec_bits;                                                    // number of bits per value
    auto            zdata     = std::vector< byte_t >( sizeof(real_t) + 1 + 1 + byte_pad( nsize * nbits ) / 8 );

    HLR_ASSERT( std::isfinite( scale ) );
    
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
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    
    HLR_ASSERT( nbits     <= sizeof(value_t) * 8 );
    HLR_ASSERT( prec_bits <= fp_info< real_t >::mant_bits );

    if (( exp_bits == 0 ) && ( prec_bits == 0 ))
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            dest[i] = value_t(0);
    }// if
    else
        decompress( dest, nsize, zdata.data() + 2, exp_bits, prec_bits );
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
    
    constexpr real_t  fp_maximum = fp_info< real_t >::maximum;
    
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
        auto  vmin = fp_maximum;
        auto  vmax = real_t(0);

        for ( size_t  i = 0; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == real_t(0) ? fp_maximum : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = eps_to_rate_aplr( S(l) );

        HLR_ASSERT( std::isfinite( s[l] ) );
        
        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += sizeof(real_t) + 1 + 1 + byte_pad( n * nbits ) / 8;
    }// for

    // for ( uint32_t  l = 0; l < k; ++l )
    //     std::cout << e[l] << '/' << m[l] << std::endl;
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto              zdata       = std::vector< byte_t >( zsize );
    size_t            pos         = 0;
    constexpr size_t  header_size = sizeof(real_t) + 2;
    const real_t *    U_ptr       = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += header_size + byte_pad( n * nbits ) / 8;
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
    
    constexpr real_t  fp_maximum = fp_info< real_t >::maximum;
    
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

        s[l] = real_t(1) / vmin;
        e[l] = uint32_t( std::max< real_t >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = eps_to_rate_aplr( S(l) );

        HLR_ASSERT( std::isfinite( s[l] ) );

        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += sizeof(real_t) + 1 + 1 + byte_pad( n2 * nbits ) / 8; // twice because real+imag
    }// for

    //
    // convert each column to compressed form
    //

    auto              zdata       = std::vector< byte_t >( zsize );
    size_t            pos         = 0;
    constexpr size_t  header_size = sizeof(real_t) + 2;
    const real_t *    U_ptr       = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  exp_bits  = e[l];
        const uint32_t  prec_bits = m[l];
        const real_t    scale     = s[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U_ptr + l * n2, n2, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += header_size + byte_pad( n2 * nbits ) / 8;
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
    constexpr size_t  header_size = sizeof(real_t) + 2;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;

        decompress( U.data() + l * n, n, zdata.data() + pos + 2, exp_bits, prec_bits );
        pos += header_size + byte_pad( nbits * n ) / 8;
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
    constexpr size_t  header_size = sizeof(real_t) + 2;
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

        decompress( U_ptr + l * n2, n2, zdata.data() + pos + 2, exp_bits, prec_bits );
        pos += header_size + byte_pad( nbits * n2 ) / 8;
    }// for
}

}}}// namespace hlr::compress::afl

#endif // __HLR_UTILS_DETAIL_AFL_HH
