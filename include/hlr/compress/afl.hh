#ifndef __HLR_UTILS_DETAIL_AFL_HH
#define __HLR_UTILS_DETAIL_AFL_HH
//
// Project     : HLR
// Module      : compress/afl
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/arith/blas.hh>
#include <hlr/compress/byte_n.hh>
#include <hlr/compress/ztypes.hh>
#include <hlr/compress/bitstream.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_VALR

// enable/disable rounding
#define HLR_AFL_ROUND

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

//
// compression config
//
template < typename real_t >
struct Zconf
{};

template <>
struct Zconf< float >
{
    constexpr static uint8_t   scale_ofs   = 2;
    constexpr static uint8_t   header_ofs  = 6;
    
    constexpr static uint64_t  exp_highbit = 1 << (FPinfo< float >::exp_bits-1);

    constexpr static uint32_t  zero_val    = 0xffffffff;
    constexpr static float     minval      = 1e-30;

    using  bs_storage_t = uint32_t;
};
    
template <>
struct Zconf< double >
{
    constexpr static uint8_t   scale_ofs   = 2;
    constexpr static uint8_t   header_ofs  = 10;
    
    constexpr static uint64_t  exp_highbit = 1 << (FPinfo< double >::exp_bits-1);

    constexpr static uint64_t  zero_val    = 0xffffffffffffffff;
    constexpr static double    minval      = 1e-50;

    using  bs_storage_t = uint64_t;
};

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ); }

struct config
{
    byte_t  bitrate;
};

inline
std::ostream &
operator << ( std::ostream &  os, const config &  conf )
{
    return os << "rate " << conf.bitrate;
}

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v   ) { return v.size(); }
inline config  get_config      ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

////////////////////////////////////////////////////////////////////////////////
//
// helper functions
//
////////////////////////////////////////////////////////////////////////////////

//
// compute min/max non-zero(!) values of given data
//
template < typename value_t >
std::pair< Hpro::real_type_t< value_t >,   // min
           Hpro::real_type_t< value_t > >  // max
nzmin_max ( const value_t *  data,
            const size_t     nsize )
{
    using  real_t = Hpro::real_type_t< value_t >;

    auto  vmin = FPinfo< real_t >::maximum;
    auto  vmax = real_t(0);

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
// SIMD specializations
//

#if defined(__AVX512F__)

template <>
inline
std::pair< float, float >
nzmin_max ( const float *  data,
            const size_t   nsize )
{
    constexpr size_t  simd_size = 16;

    // TODO: handle non-multiple sizes
    HLR_DBG_ASSERT( nsize % simd_size == 0 );
    
    const auto  vzero = _mm512_set1_ps( FPinfo< float >::minval );
    auto        vMIN  = _mm512_set1_ps( FPinfo< float >::maximum );
    auto        vMAX  = _mm512_setzero_ps();
        
    for ( size_t  i = 0; i < nsize; i += simd_size )
    {
        __m512     vd   = _mm512_abs_ps( _mm512_loadu_ps( data + i ) );
        __mmask16  mask = _mm512_cmp_ps_mask( vd, vzero, _CMP_GE_OQ );

        vMAX = _mm512_max_ps( vMAX, vd );
        vMIN = _mm512_mask_min_ps( vMIN, mask, vMIN, vd );
    }// for

    return { _mm512_reduce_min_ps( vMIN ), _mm512_reduce_max_ps( vMAX ) };
}

template <>
inline
std::pair< double, double >
nzmin_max ( const double *  data,
            const size_t    nsize )
{
    constexpr size_t  simd_size = 8;

    // TODO: handle non-multiple sizes
    HLR_DBG_ASSERT( nsize % simd_size == 0 );
    
    const auto  vzero = _mm512_set1_ps( FPinfo< double >::minval );
    auto        vMIN  = _mm512_set1_pd( FPinfo< double >::maximum );
    auto        vMAX  = _mm512_setzero_pd();
        
    for ( size_t  i = 0; i < nsize; i += simd_size )
    {
        const auto  vd   = _mm512_abs_pd( _mm512_loadu_pd( data + i ) );
        const auto  mask = _mm512_cmp_pd_mask( vd, vzero, _CMP_GE_OQ );

        vMAX = _mm512_max_pd( vMAX, vd );
        vMIN = _mm512_mask_min_pd( vMIN, mask, vMIN, vd );
    }// for

    return { _mm512_reduce_min_pd( vMIN ), _mm512_reduce_max_pd( vMAX ) };
}

#endif

//
// return number of bits needed to represent given dynamic range
//
constexpr
uint8_t
nexpbits ( const auto  drange )
{
    return uint8_t( std::max< decltype( drange ) >( 0, std::ceil( std::log2( std::log2( drange ) ) ) ) );
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
    const uint32_t  zero_val = Zconf< value_t >::zero_val & (( 1 << nbits) - 1 );
        
    #if defined(HLR_AFL_ROUND)
    const uint32_t  round_shift = prec_bits - 1;
    #endif

    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    bool              sign[ nbuf ]; // holds sign per entry
    fp32int_t         fbuf[ nbuf ]; // holds rescaled value
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos  = Zconf< value_t >::header_ofs;
    size_t            i    = 0;

    const size_t      bssize = pad_bs< uint32_t >( byte_pad( nsize * nbits ) / 8 );
    auto              bs     = bitstream< uint32_t >( zdata + pos, bssize );
    
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

            zero[j]   = ( aval < Zconf< value_t >::minval );
            sign[j]   = ( aval != val );
            fbuf[j].f = aval / scale + 1;

            HLR_DBG_ASSERT( zero[j] || ( fbuf[j].f >= float(2) ));
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      ival  = fbuf[j].u;
            const uint32_t  sexp  = ( ival >> FP32::mant_bits ) & FP32::exp_mask; // extract exponent
            const uint32_t  smant = ( ival & FP32::mant_mask );                   // and mantissa
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
            bs.write_bits( ibuf[j], nbits );
    }// for
        
    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        const float  val  = data[i];
        uint32_t     zval = zero_val;
            
        if ( std::abs( val ) >= Zconf< value_t >::minval )
        {
            const auto       ival  = fbuf[i].u;
            const uint32_t   sexp  = ( ival >> FP32::mant_bits ) & FP32::exp_mask;
            const uint32_t   smant = ( ival & FP32::mant_mask );
            #if defined(HLR_AFL_ROUND)
            const uint32_t   srest = ( ival >> round_shift ) & 1;
            #endif
            const uint32_t   zexp  = sexp & exp_mask;
            const uint32_t   zmant = smant >> prec_ofs;
                
            zval = (((sign[i] << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        bs.write_bits( zval, nbits );
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
    const uint32_t  zero_val   = Zconf< value_t >::zero_val & (( 1 << nbits) - 1 );
    float           scale;

    // read scaling factor
    memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint32_t          ibuf[ nbuf ]; // holds value in compressed format
    float             fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos  = Zconf< value_t >::header_ofs;
    size_t            i    = 0;

    const size_t      bssize = pad_bs< uint32_t >( byte_pad( nsize * nbits ) / 8 );
    auto              bs     = bitstream< uint32_t >( const_cast< byte_t * >( zdata ) + pos, bssize );
    
    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        for ( size_t  j = 0; j < nbuf; ++j )
            ibuf[j] = bs.read_bits( nbits );
            
        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto       zval  = ibuf[j];
            const uint32_t   mant  = zval & prec_mask;
            const uint32_t   exp   = (zval >> prec_bits) & exp_mask;
            const bool       sign  = zval >> sign_shift;
            const fp32int_t  fival{ .u = (uint32_t(exp | Zconf< value_t >::exp_highbit) << FP32::mant_bits) | (uint32_t(mant) << prec_ofs) };

            zero[j] = ( zval == zero_val );
            fbuf[j] = (sign ? -1 : 1 ) * ( fival.f - 1 ) * scale;
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                fbuf[j] = float(0);

        // copy values
        for ( size_t  j = 0; j < nbuf; ++j )
            data[i+j] = fbuf[j];
    }// for

    // handle remaining values
    for ( ; i < nsize; ++i )
    {
        uint32_t  zval  = bs.read_bits( nbits );
            
        if ( zval == zero_val )
            data[i] = 0;
        else
        {
            const uint32_t  mant  = zval & prec_mask;
            const uint32_t  exp   = (zval >> prec_bits) & exp_mask;
            const bool      sign  = zval >> sign_shift;
            const uint32_t  irval = ((exp | Zconf< value_t >::exp_highbit) << FP32::mant_bits) | (mant << prec_ofs);
            const float     rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) * scale;
                
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
    using  value_t = double;
    
    const uint32_t  nbits    = 1 + exp_bits + prec_bits;
    const uint32_t  exp_mask = ( 1 << exp_bits ) - 1;
    const uint32_t  prec_ofs = FP64::mant_bits - prec_bits;
    const uint32_t  zero_val = Zconf< value_t >::zero_val & (( 1 << nbits) - 1 );

    #if defined(HLR_AFL_ROUND)
    const uint32_t  round_shift = prec_bits - 1;
    #endif

    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata + Zconf< value_t >::scale_ofs, & scale, sizeof(scale) );

    //
    // compress data in "vectorized" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    bool              sign[ nbuf ]; // holds sign per entry
    fp64int_t         fbuf[ nbuf ]; // holds rescaled value
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    size_t            pos  = Zconf< value_t >::header_ofs;
    size_t            i    = 0;

    const size_t      bssize = pad_bs< uint64_t >( byte_pad( nsize * nbits ) / 8 );
    auto              bs     = bitstream< uint64_t >( zdata + pos, bssize );
    
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

            zero[j]   = ( aval < Zconf< value_t >::minval );
            sign[j]   = ( aval != val );
            fbuf[j].f = aval / scale + 1;

            HLR_DBG_ASSERT( zero[j] || ( fbuf[j].f >= double(2) ));
        }// for

        // convert to compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto      ival  = fbuf[j].u;
            const uint64_t  sexp  = ( ival >> FP64::mant_bits ) & FP64::exp_mask;
            const uint64_t  smant = ( ival & FP64::mant_mask );
            #if defined(HLR_AFL_ROUND)
            const uint64_t  srest = ( ival >> round_shift ) & 1;
            #endif
            const uint64_t  zexp  = sexp & exp_mask;
            const uint64_t  zmant = smant >> prec_ofs;
            
            #if defined(HLR_AFL_ROUND)
            ibuf[j] = ( (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant ) + srest;
            #else
            ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
            #endif
        }// for

        // correct zeroes
        for ( size_t  j = 0; j < nbuf; ++j )
            if ( zero[j] )
                ibuf[j] = zero_val;

        // write into data buffer
        for ( size_t  j = 0; j < nbuf; ++j )
            bs.write_bits( ibuf[j], nbits );
    }// for

    // handle remaining data
    for ( ; i < nsize; ++i )
    {
        const double  val  = data[i];
        uint64_t      zval = zero_val;
            
        if ( std::abs( val ) >= Zconf< value_t >::minval )
        {
            const bool       zsign = ( val < 0 );
            const fp64int_t  fival = { .f = std::abs(val) / scale + 1 };
            const uint64_t   sexp  = ( fival.u >> FP64::mant_bits ) & FP64::exp_mask;
            const uint64_t   smant = ( fival.u & FP64::mant_mask );
            #if defined(HLR_AFL_ROUND)
            const uint64_t   srest = ( fival.u >> round_shift ) & 1;
            #endif
            const uint64_t   zexp  = sexp & exp_mask;
            const uint64_t   zmant = smant >> prec_ofs;
            
            #if defined(HLR_AFL_ROUND)
            zval  = ( (((zsign << exp_bits) | zexp) << prec_bits) | zmant ) + srest;
            #else
            zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;
            #endif

            HLR_DBG_ASSERT( zval != zero_val );
        }// if
        
        bs.write_bits( zval, nbits );
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
    using  value_t = double;
    
    const uint32_t  nbits       = 1 + exp_bits + prec_bits;
    const uint64_t  prec_mask   = ( 1ul << prec_bits ) - 1;
    const uint32_t  prec_ofs    = FP64::mant_bits - prec_bits;
    const uint64_t  exp_mask    = ( 1ul << exp_bits ) - 1;
    const uint32_t  sign_shift  = exp_bits + prec_bits;
    const uint64_t  zero_val    = Zconf< value_t >::zero_val & (( 1ul << nbits) - 1 );
    double          scale;

    // read scaling factor
    memcpy( & scale, zdata + Zconf< value_t >::scale_ofs, sizeof(scale) );

    //
    // decompress in "vectorised" form
    //
        
    constexpr size_t  nbuf   = 64;
    const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
    bool              zero[ nbuf ]; // mark zero entries
    uint64_t          ibuf[ nbuf ]; // holds value in compressed format
    double            fbuf[ nbuf ]; // holds uncompressed values
    size_t            pos  = Zconf< value_t >::header_ofs;
    size_t            i    = 0;

    const size_t      bssize = pad_bs< uint64_t >( byte_pad( nsize * nbits ) / 8 );
    auto              bs     = bitstream< uint64_t >( const_cast< byte_t * >( zdata ) + pos, bssize );
    
    for ( ; i < nbsize; i += nbuf )
    {
        // read data
        for ( size_t  j = 0; j < nbuf; ++j )
            ibuf[j] = bs.read_bits( nbits );

        // convert from compressed format
        for ( size_t  j = 0; j < nbuf; ++j )
        {
            const auto       zval  = ibuf[j];
            const uint64_t   mant  = zval & prec_mask;
            const uint64_t   exp   = (zval >> prec_bits) & exp_mask;
            const bool       sign  = zval >> sign_shift;
            const fp64int_t  fival = { .u = ((exp | Zconf< value_t >::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };

            zero[j] = ( zval == zero_val );
            fbuf[j] = (sign ? -1 : 1 ) * ( fival.f - 1 ) * scale;
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
        uint64_t  zval = bs.read_bits( nbits );
        
        if ( zval == zero_val )
            data[i] = 0;
        else
        {
            const uint64_t   mant  = zval & prec_mask;
            const uint64_t   exp   = (zval >> prec_bits) & exp_mask;
            const bool       sign  = zval >> sign_shift;
            const fp64int_t  fival = { .u = ((exp | Zconf< value_t >::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs) };

            data[i] = (sign ? -1 : 1 ) * ( fival.f - 1 ) * scale;
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
    using  real_t       = Hpro::real_type_t< value_t >;
    using  bs_storage_t = typename Zconf< value_t >::bs_storage_t;
    
    constexpr real_t  fp_maximum = FPinfo< real_t >::maximum;

    //
    // look for min/max value (> 0!)
    //
    
    const size_t  nsize        = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto  [ vmin, vmax ] = nzmin_max( data, nsize );

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
    
    const uint32_t  exp_bits  = nexpbits( vmax / vmin );                                              // no. of bits needed to represent exponent
    const uint32_t  prec_bits = std::min< uint32_t >( FPinfo< real_t >::mant_bits, config.bitrate ); // total no. of bits per value
    const size_t    nbits     = 1 + exp_bits + prec_bits;                                             // number of bits per value
    const auto      scale     = vmin;                                                                 // scale all values v_i such that |v_i| >= 1
    auto            zdata     = std::vector< byte_t >( Zconf< real_t >::header_ofs + pad_bs< bs_storage_t >( byte_pad( nsize * nbits ) / 8 ) );

    // std::cout << prec_bits << std::endl;
    
    HLR_ASSERT( nbits <= sizeof(value_t) * 8 );
    HLR_ASSERT( std::isfinite( scale ) );
    
    compress( data, nsize, zdata.data(), scale, exp_bits, prec_bits );

    // // DEBUG
    // {
    //     std::vector< double >  tmp( nsize );

    //     decompress( tmp.data(), nsize, zdata.data(), exp_bits, prec_bits );

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
    //
    
    const uint32_t  exp_bits  = zdata[0];
    const uint32_t  prec_bits = zdata[1];
    const uint32_t  nbits     = 1 + exp_bits + prec_bits;
    
    HLR_ASSERT( nbits     <= sizeof(value_t) * 8 );
    HLR_ASSERT( prec_bits <= FPinfo< real_t >::mant_bits );

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
    using  real_t       = Hpro::real_type_t< value_t >;
    using  bs_storage_t = typename Zconf< value_t >::bs_storage_t;
    
    constexpr real_t  fp_maximum  = FPinfo< real_t >::maximum;
    constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    
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

        s[l] = vmin;
        e[l] = nexpbits( vmax / vmin );
        m[l] = std::min< uint32_t >( FPinfo< real_t >::mant_bits, eps_to_rate_valr( S(l) ) );

        HLR_ASSERT( std::isfinite( s[l] ) );
        
        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += header_size + pad_bs< bs_storage_t >( byte_pad( n * nbits ) / 8 );
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
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U.data() + l*n, n, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += header_size + pad_bs< bs_storage_t >( byte_pad( n * nbits ) / 8 );
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
    using  real_t       = double;
    using  bs_storage_t = typename Zconf< real_t >::bs_storage_t;
    
    constexpr real_t  fp_maximum  = FPinfo< real_t >::maximum;
    constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    
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
        e[l] = nexpbits( vmax / vmin );
        m[l] = std::min< uint32_t >( FPinfo< real_t >::mant_bits, eps_to_rate_valr( S(l) ) );

        HLR_ASSERT( std::isfinite( s[l] ) );

        const size_t  nbits = 1 + e[l] + m[l]; // number of bits per value
        
        zsize += header_size + pad_bs< bs_storage_t >( byte_pad( n2 * nbits ) / 8 ); // twice because real+imag
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
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U_ptr + l * n2, n2, zdata.data() + pos, scale, exp_bits, prec_bits );
        pos += header_size + pad_bs< bs_storage_t >( byte_pad( n2 * nbits ) / 8 );
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    using  real_t       = Hpro::real_type_t< value_t >;
    using  bs_storage_t = typename Zconf< real_t >::bs_storage_t;

    constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    
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

        decompress( U.data() + l * n, n, zdata.data() + pos, exp_bits, prec_bits );
        pos += header_size + pad_bs< bs_storage_t >( byte_pad( nbits * n ) / 8 );
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
    using  real_t       = double;
    using  bs_storage_t = typename Zconf< real_t >::bs_storage_t;

    constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    
    const size_t    n     = U.nrows();
    const uint32_t  k     = U.ncols();
    size_t          pos   = 0;
    auto            U_ptr = reinterpret_cast< real_t * >( U.data() );
    const size_t    n2    = 2 * n;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;

        decompress( U_ptr + l * n2, n2, zdata.data() + pos, exp_bits, prec_bits );
        pos += header_size + pad_bs< bs_storage_t >( byte_pad( nbits * n2 ) / 8 );
    }// for
}

//
// compressed blas
//

namespace
{

template < typename value_t >
void
mulvec ( const size_t                        nrows,
         const size_t                        ncols,
         const matop_t                       op_A,
         const value_t                       alpha,
         const Hpro::real_type_t< value_t >  zscale,
         const byte_t *                      zA,
         const value_t *                     x,
         value_t *                           y,
         const uint8_t                       exp_bits,
         const uint8_t                       prec_bits )
{
    using  bs_storage_t = typename Zconf< value_t >::bs_storage_t;
    
    const uint8_t     nbits      = 1 + exp_bits + prec_bits;
    const uint64_t    prec_mask  = ( 1ul << prec_bits ) - 1;
    const uint8_t     prec_ofs   = FP64::mant_bits - prec_bits;
    const uint64_t    exp_mask   = ( 1ul << exp_bits  ) - 1;
    const uint32_t    sign_shift = exp_bits + prec_bits;
    const uint64_t    zero_val   = Zconf< value_t >::zero_val & (( 1ul << nbits) - 1 );
    const auto        scale      = alpha * zscale;

    const size_t      bssize = pad_bs< bs_storage_t >( byte_pad( nrows * ncols * nbits ) / 8 );
    auto              bs     = bitstream< bs_storage_t >( const_cast< byte_t * >( zA ), bssize );
    
    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = scale * x[j];
                fp64int_t   fival;
                
                for ( size_t  i = 0 ; i < nrows; ++i )
                {
                    const auto  z_ij = bs.read_bits( nbits );

                    if ( z_ij != zero_val )
                    {
                        const uint64_t  mant = z_ij & prec_mask;
                        const uint64_t  exp  = (z_ij >> prec_bits) & exp_mask;
                        const uint64_t  sign = (z_ij >> sign_shift) << FP64::sign_bit;

                        fival.u  = ((exp | Zconf< value_t >::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);
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
            for ( size_t  j = 0; j < ncols; ++j )
            {
                value_t    y_j = value_t(0);
                fp64int_t  fival;

                for ( size_t  i = 0; i < nrows; ++i )
                {
                    const auto  z_ij = bs.read_bits( nbits );

                    if ( z_ij != zero_val )
                    {
                        const uint64_t  mant = z_ij & prec_mask;
                        const uint64_t  exp  = (z_ij >> prec_bits) & exp_mask;
                        const uint64_t  sign = (z_ij >> sign_shift) << FP64::sign_bit;
                        
                        fival.u  = ((exp | Zconf< value_t >::exp_highbit) << FP64::mant_bits) | (mant << prec_ofs);
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
    using  real_t = Hpro::real_type_t< value_t >;

    const uint8_t      exp_bits  = zA[0];
    const uint8_t      prec_bits = zA[1];
    const uint8_t      nbits     = 1 + exp_bits + prec_bits;
    real_t             scale     = * ( reinterpret_cast< const real_t * >( zA.data() + Zconf< real_t >::scale_ofs ) );
    constexpr size_t   data_ofs  = Zconf< real_t >::header_ofs;
    
    mulvec( nrows, ncols, op_A, alpha, scale, zA.data() + data_ofs, x, y, exp_bits, prec_bits );
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
    using  real_t       = Hpro::real_type_t< value_t >;
    using  bs_storage_t = typename Zconf< value_t >::bs_storage_t;

    constexpr size_t  scale_ofs   = Zconf< real_t >::scale_ofs;
    constexpr size_t  header_size = Zconf< real_t >::header_ofs;
    size_t            pos         = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  exp_bits  = zA[pos];
                const uint8_t  prec_bits = zA[pos+1];
                const uint8_t  nbits     = 1 + exp_bits + prec_bits;
                const size_t   nbyte     = pad_bs< bs_storage_t >( byte_pad( nrows * nbits ) / 8 );
                const real_t   scale     = * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) );
        
                mulvec( nrows, 1, op_A, alpha, scale, zA.data() + pos + header_size, x+l, y, exp_bits, prec_bits );

                pos += header_size + nbyte;

                HLR_ASSERT( pos <= zA.size() );
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
                const uint8_t  nbits     = 1 + exp_bits + prec_bits;
                const size_t   nbyte     = pad_bs< bs_storage_t >( byte_pad( nrows * nbits ) / 8 );
                const real_t   scale     = * ( reinterpret_cast< const real_t * >( zA.data() + pos + scale_ofs ) );
        
                mulvec( nrows, 1, op_A, alpha, scale, zA.data() + pos + header_size, x, y+l, exp_bits, prec_bits );

                pos += header_size + nbyte;

                HLR_ASSERT( pos <= zA.size() );
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::afl

#endif // __HLR_UTILS_DETAIL_AFL_HH
