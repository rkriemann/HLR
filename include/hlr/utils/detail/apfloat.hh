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
#include <limits>

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

using byte_t = unsigned char;

constexpr byte_t  fp32_mant_bits   = 23;
constexpr byte_t  fp32_exp_bits    = 8;
constexpr byte_t  fp32_sign_bit    = 31;
constexpr ulong   fp32_exp_highbit = 0b10000000;
constexpr uint    fp32_zero_val    = 0xffffffff;
constexpr float   fp32_infinity    = std::numeric_limits< float >::infinity();

constexpr uint    fp64_mant_bits   = 52;
constexpr uint    fp64_exp_bits    = 11;
constexpr uint    fp64_sign_bit    = 63;
constexpr ulong   fp64_exp_highbit = 0b10000000000;
constexpr ulong   fp64_zero_val    = 0xffffffffffffffff;
constexpr double  fp64_infinity    = std::numeric_limits< double >::infinity();

// return byte padded value of <n>
inline size_t byte_pad ( size_t  n )
{
    return ( n % 8 != 0 ) ? n + (8 - n%8) : n;
}
    
inline
byte_t
eps_to_rate ( const double eps )
{
    #if defined(HLR_COMPRESS_RATE_ARITH)

    if      ( eps >= 1e-2  ) return 12;
    else if ( eps >= 1e-3  ) return 14;
    else if ( eps >= 1e-4  ) return 18;
    else if ( eps >= 1e-5  ) return 22;
    else if ( eps >= 1e-6  ) return 24;
    else if ( eps >= 1e-7  ) return 28;
    else if ( eps >= 1e-8  ) return 32;
    else if ( eps >= 1e-9  ) return 34;
    else if ( eps >= 1e-10 ) return 38;
    else if ( eps >= 1e-12 ) return 38;
    else if ( eps >= 1e-14 ) return 42;
    else                     return 64;

    #else

    if      ( eps >= 1e-2  ) return 10;
    else if ( eps >= 1e-3  ) return 12;
    else if ( eps >= 1e-4  ) return 14;
    else if ( eps >= 1e-5  ) return 16;
    else if ( eps >= 1e-6  ) return 20;
    else if ( eps >= 1e-7  ) return 22;
    else if ( eps >= 1e-8  ) return 26;
    else if ( eps >= 1e-9  ) return 30;
    else if ( eps >= 1e-10 ) return 34;
    else if ( eps >= 1e-12 ) return 38;
    else if ( eps >= 1e-14 ) return 42;
    else                     return 64;

    #endif
}

inline
uint
tol_to_rate ( const double  tol )
{
    return uint( std::max< double >( 1, -std::log2( tol ) ) ) + 1;
}

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

// return actual memory size of compressed data
inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }

// return compression configuration for desired accuracy eps
inline config  get_config ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

// for optimized 3-byte assignment
struct  byte3_t
{
    byte_t  data[3];
    
    void operator = ( const uint  n )
    {
        data[0] = (n & 0x0000ff);
        data[1] = (n & 0x00ff00) >> 8;
        data[2] = (n & 0xff0000) >> 16;
    }

    operator uint () const
    {
        return ( data[2] << 16 ) | ( data[1] << 8 ) | data[0];
    }
};

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

    // look for min/max value
    float  vmin = std::abs( data[0] );
    float  vmax = std::abs( data[0] );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        vmin = std::min( vmin, std::abs( data[i] ) );
        vmax = std::max( vmax, std::abs( data[i] ) );
    }// for

    HLR_DBG_ASSERT( vmin > float(0) );
    
    // scale all values v_i such that we have |v_i| >= 1
    const float   scale      = 1.0 / vmin;
    // number of bits needed to represent exponent values
    const uint    exp_bits   = std::max< float >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const uint    exp_mask   = ( 1 << exp_bits ) - 1;

    // round up to next multiple of 8
    const uint    nbits_min  = 1 + exp_bits + config.bitrate;                          // minimal number of bits per value for precision
    const uint    nbits      = ( nbits_min / 8 ) * 8 + ( nbits_min % 8 != 0 ? 8 : 0 ); // actual number of bits per value
    const uint    nbyte      = nbits / 8;

    const uint    prec_bits  = nbits - 1 - exp_bits;
    const uint    prec_mask  = ( 1 << prec_bits ) - 1;
    const uint    prec_ofs   = fp32_mant_bits - prec_bits;
    
    const size_t  zsize      = 4 + 1 + 1 + nsize * nbyte;
    auto          zdata      = std::vector< byte_t >( zsize );

    HLR_ASSERT( nbits     <= 32 );
    HLR_ASSERT( prec_bits <= fp32_mant_bits );
    
    // first, store scaling factor
    memcpy( zdata.data(), & scale, 4 );

    // then store number of exponents bits
    zdata[4] = exp_bits;
            
    // and precision bits
    zdata[5] = prec_bits;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const float   val   = data[i];
        const bool    zsign = ( val < 0 );

        //
        // Use absolute value and scale v_i and add 1 such that v_i >= 2.
        // With this, highest exponent bit is 1 and we only need to store
        // lowest <exp_bits> exponent bits
        //
        
        const float   sval  = scale * std::abs(val) + 1;
        const uint    isval = (*reinterpret_cast< const uint * >( & sval ) );
        const uint    sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
        const uint    smant = ( isval & ((1u << fp32_mant_bits) - 1) );

        // exponent and mantissa reduced to stored size
        const uint    zexp  = sexp & exp_mask;
        const uint    zmant = smant >> prec_ofs;
        const uint    zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

        // // DEBUG
        // {
        //     const byte_t  fp32_sign_bit  = 31;
        //     const byte_t  sign_shift = exp_bits + prec_bits;
            
        //     const uint   mant  = zval & prec_mask;
        //     const uint   exp   = (zval >> prec_bits) & exp_mask;
        //     const bool   sign  = zval >> sign_shift;

        //     const uint   rexp  = exp | 0b10000000; // re-add leading bit
        //     const uint   rmant = mant << prec_ofs;
        //     const uint   irval = (rexp << fp32_mant_bits) | rmant;
        //     const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

        //     std::cout << i << " : " << val << " / " << rval << " / " << std::abs( (val - rval) / val ) << std::endl;
        // }
        
        //
        // copy zval into data buffer
        //

        const size_t  pos = 6 + i * nbyte;
        
        switch ( nbyte )
        {
            case  4 : zdata[pos+3] = ( zval & 0xff000000 ) >> 24;
            case  3 : zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
            case  2 : zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
            case  1 : zdata[pos]   = ( zval & 0x000000ff ); break;
            default :
                HLR_ERROR( "unsupported storage size" );
        }// switch
    }// for

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

    for ( size_t  i = 1; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == double(0) ? fp64_infinity : d_i );
            
        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > double(0) );
    
    // scale all values v_i such that we have |v_i| >= 1
    const double  scale      = 1.0 / vmin;
    // number of bits needed to represent exponent values
    const uint    exp_bits   = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const ulong   exp_mask   = ( 1 << exp_bits ) - 1;
    // number of bits/bytes per value
    const uint    nbits      = byte_pad( 1 + exp_bits + config.bitrate ); // number of bits per value
    const uint    nbyte      = nbits / 8;
    // actual number of precision bits
    const uint    prec_bits  = nbits - 1 - exp_bits;

    HLR_ASSERT( nbits     <= 64 );
    HLR_ASSERT( prec_bits <= fp64_mant_bits );

    // array storing compressed data
    auto  zdata = std::vector< byte_t >();

    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
        const size_t  zsize = 4 + 1 + 1 + nsize * nbyte;

        zdata.resize( zsize );
        
        //
        // store header (exponent bits, precision bits and scaling factor)
        //
        
        const float  fmin     = vmin;
        const uint   prec_ofs = fp32_mant_bits - prec_bits;
        const float  fscale   = scale;
        const uint   zero_val = fp32_zero_val & (( 1 << nbits) - 1 );

        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & fscale, 4 );

        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        bool              sign[ nbuf ]; // holds sign per entry
        float             fbuf[ nbuf ]; // holds rescaled value
        uint              ibuf[ nbuf ]; // holds value in compressed format
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
                fbuf[j] = std::max( fscale * aval + 1, 2.f ); // prevent rounding issues when converting from fp64
            }// for

            // convert to compressed format
            for ( size_t  j = 0; j < nbuf; ++j )
            {
                const uint  isval = (*reinterpret_cast< const uint * >( & fbuf[j] ) );
                const uint  sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1); // extract exponent
                const uint  smant = ( isval & ((1u << fp32_mant_bits) - 1) );                  // and mantissa
                const uint  zexp  = sexp & exp_mask;    // extract needed exponent
                const uint  zmant = smant >> prec_ofs;  // and precision bits
                
                ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
            }// for

            // correct zeroes
            for ( size_t  j = 0; j < nbuf; ++j )
                if ( zero[j] )
                    ibuf[j] = zero_val;

            // write to destination buffer
            switch ( nbyte )
            {
                case  4 : { auto ptr = reinterpret_cast< uint * >(    & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
                case  3 : { auto ptr = reinterpret_cast< byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ibuf[j]; } break;
                case  2 : { auto ptr = reinterpret_cast< ushort * >(  & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = ushort( ibuf[j] & 0xffff ); } break;
                case  1 : { auto ptr = & zdata[pos];                                  for ( size_t  j = 0; j < nbuf; ++j ) ptr[j] = byte_t( ibuf[j] & 0xff   ); } break;
                default :
                    HLR_ERROR( "unsupported storage size" );
            }// switch

            pos += nbyte * nbuf;
        }// for

        // handle remaining values
        for ( ; i < nsize; ++i )
        {
            const float  val  = data[i];
            uint         zval = zero_val;

            if ( std::abs( val ) >= fmin )
            {
                const bool    zsign = ( val < 0 );
                const float   sval  = std::max( fscale * std::abs(val) + 1, 2.f );
                const uint    isval = (*reinterpret_cast< const uint * >( & sval ) );
                const uint    sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                const uint    smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                const uint    zexp  = sexp & exp_mask;
                const uint    zmant = smant >> prec_ofs;

                zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
            switch ( nbyte )
            {
                case  4 : zdata[pos+3] = ( zval & 0xff000000 ) >> 24;
                case  3 : zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
                case  2 : zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
                case  1 : zdata[pos]   = ( zval & 0x000000ff ); break;
                default :
                    HLR_ERROR( "unsupported storage size" );
            }// switch

            pos += nbyte;
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        const size_t  zsize = 8 + 1 + 1 + nsize * nbyte;

        zdata.resize( zsize );
        
        //
        // store header (exponent bits, precision bits and scaling factor)
        //
        
        const uint   prec_ofs = fp64_mant_bits - prec_bits;
        const uint   zero_val = 0xffffffff;

        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & scale, 8 );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  val  = data[i];
            uint          zval = zero_val;
            
            if ( std::abs( val ) >= vmin )
            {
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
                const bool    zsign = ( val < 0 );
                const double  sval  = scale * std::abs(val) + 1;
                const ulong   isval = (*reinterpret_cast< const ulong * >( & sval ) );
                const ulong   sexp  = ( isval >> fp64_mant_bits ) & ((1u << fp64_exp_bits) - 1);
                const ulong   smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
                
                // exponent and mantissa reduced to stored size
                const uint    zexp  = sexp & exp_mask;
                const uint    zmant = smant >> prec_ofs;

                zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
            //
            // copy zval into data buffer
            //

            const size_t  pos = 10 + i * nbyte;
        
            zdata[pos]   = ( zval & 0x000000ff );
            zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
            zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
            zdata[pos+3] = ( zval & 0xff000000 ) >> 24;
        }// for
    }// if
    else
    {
        HLR_DBG_ASSERT( nbyte >= 4 );
        
        const size_t  zsize = 8 + 1 + 1 + nsize * nbyte;

        zdata.resize( zsize );
        
        //
        // store header (exponent bits, precision bits and scaling factor)
        //
        
        const uint   prec_ofs = fp64_mant_bits - prec_bits;
        const ulong  zero_val = fp64_zero_val & (( 1ul << nbits) - 1 );

        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & scale, 8 );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  val  = data[i];
            ulong         zval = zero_val;
            
            if ( std::abs( val ) >= vmin )
            {
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
                const bool    zsign = ( val < 0 );
                const double  sval  = scale * std::abs(val) + 1;
                const ulong   isval = (*reinterpret_cast< const ulong * >( & sval ) );
                const ulong   sexp  = ( isval >> fp64_mant_bits ) & ((1u << fp64_exp_bits) - 1);
                const ulong   smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
                
                // exponent and mantissa reduced to stored size
                const ulong   zexp  = sexp & exp_mask;
                const ulong   zmant = smant >> prec_ofs;

                zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
            //
            // copy zval into data buffer
            //

            const size_t  pos = 10 + i * nbyte;
        
            switch ( nbyte )
            {
                case  8 : zdata[pos+7] = ( zval & 0xff00000000000000 ) >> 56;
                case  7 : zdata[pos+6] = ( zval & 0x00ff000000000000 ) >> 48;
                case  6 : zdata[pos+5] = ( zval & 0x0000ff0000000000 ) >> 40;
                case  5 : zdata[pos+4] = ( zval & 0x000000ff00000000 ) >> 32;
                case  4 : break;
                default :
                    HLR_ERROR( "unsupported storage size" );
            }// switch
            
            zdata[pos+3] = ( zval & 0x00000000ff000000 ) >> 24;
            zdata[pos+2] = ( zval & 0x0000000000ff0000 ) >> 16;
            zdata[pos+1] = ( zval & 0x000000000000ff00 ) >> 8;
            zdata[pos]   = ( zval & 0x00000000000000ff );
        }// for
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
    if ( dim1 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, 2 );
    else
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, dim3 * 2 );
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
    if ( dim1 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, 2 );
    else
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, dim3 * 2 );
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
    
    float       scale;
    const uint  exp_bits  = zdata[4];
    const uint  prec_bits = zdata[5];

    memcpy( & scale, zdata.data(), 4 );

    //
    // read compressed data
    //

    const uint  nbits       = 1 + exp_bits + prec_bits;
    const uint  nbyte       = nbits / 8;
    const uint  prec_mask   = ( 1 << prec_bits ) - 1;
    const uint  prec_ofs    = fp32_mant_bits - prec_bits;
    const uint  exp_mask    = ( 1 << exp_bits ) - 1;
    const uint  exp_highbit = 0b10000000;
    const uint  sign_shift  = exp_bits + prec_bits;

    // number of values to read before decoding
    constexpr size_t  nchunk = 32;
    const size_t      ncsize = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
    size_t            i      = 0;

    for ( ; i < ncsize; i += nchunk )
    {
        uint  zval_buf[ nchunk ];
        
        //
        // read next values into local buffer
        //

        if ( nbyte == 1 )
        {
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const size_t  pos  = 6 + (i+lpos)*nbyte;

                zval_buf[lpos] = zdata[pos];
            }// for
        }// if
        else if ( nbyte == 2 )
        {
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const size_t  pos  = 6 + (i+lpos)*nbyte;

                zval_buf[lpos] = ( (zdata[pos+1] << 8) |
                                   (zdata[pos]       ) );
            }// for
        }// if
        else if ( nbyte == 3 )
        {
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const size_t  pos  = 6 + (i+lpos)*nbyte;

                zval_buf[lpos] = ( (zdata[pos+2] << 16) |
                                   (zdata[pos+1] <<  8) |
                                   (zdata[pos]        ) );
            }// for
        }// if
        else if ( nbyte == 4 )
        {
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const size_t  pos  = 6 + (i+lpos)*nbyte;

                zval_buf[lpos] = ( (zdata[pos+3] << 24) |
                                   (zdata[pos+2] << 16) |
                                   (zdata[pos+1] <<  8) |
                                   (zdata[pos]        ) );
            }// for
        }// if

        //
        // convert all values
        //

        for ( uint  lpos = 0; lpos < nchunk; ++lpos )
        {
            const uint   zval  = zval_buf[lpos];
            const uint   mant  = zval & prec_mask;
            const uint   exp   = (zval >> prec_bits) & exp_mask;
            const bool   sign  = zval >> sign_shift;

            const uint   rexp  = exp | exp_highbit; // re-add leading bit
            const uint   rmant = mant << prec_ofs;
            const uint   irval = (rexp << fp32_mant_bits) | rmant;
            const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

            dest[i+lpos] = double( rval );
        }// for
    }// for

    for ( ; i < nsize; ++i )
    {
        uint          zval = 0;
        const size_t  pos  = 6 + i*nbyte;
            
        switch ( nbyte )
        {
            case  4 : zval |= zdata[pos+3] << 24;
            case  3 : zval |= zdata[pos+2] << 16;
            case  2 : zval |= zdata[pos+1] << 8;
            case  1 : zval |= zdata[pos]; break;
            default :
                HLR_ERROR( "unsupported storage size" );
        }// switch

        const uint   mant  = zval & prec_mask;
        const uint   exp   = (zval >> prec_bits) & exp_mask;
        const bool   sign  = zval >> sign_shift;
        const uint   irval = ((exp | exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
        const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

        dest[i] = double( rval );
    }// for
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
    //
    
    const uint  exp_bits  = zdata[0];
    const uint  prec_bits = zdata[1];

    //
    // read compressed data
    //
    
    const  uint  nbits = 1 + exp_bits + prec_bits;
    const  uint  nbyte = nbits / 8;
    
    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
        const uint  prec_mask   = ( 1 << prec_bits ) - 1;
        const uint  prec_ofs    = fp32_mant_bits - prec_bits;
        const uint  exp_mask    = ( 1 << exp_bits ) - 1;
        const uint  sign_shift  = exp_bits + prec_bits;
        const uint  zero_val    = fp32_zero_val & (( 1 << nbits) - 1 );
        float       scale;

        // get scaling factor
        memcpy( & scale, zdata.data() + 2, 4 );

        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        uint              ibuf[ nbuf ]; // holds value in compressed format
        float             fbuf[ nbuf ]; // holds uncompressed values
        size_t            pos = 6;
        size_t            i   = 0;

        for ( ; i < nbsize; i += nbuf )
        {
            // read data
            switch ( nbyte )
            {
                case  4 : { auto ptr = reinterpret_cast< const uint *    >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
                case  3 : { auto ptr = reinterpret_cast< const byte3_t * >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
                case  2 : { auto ptr = reinterpret_cast< const ushort *  >( & zdata[pos] ); for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
                case  1 : { auto ptr = & zdata[pos];                                        for ( size_t  j = 0; j < nbuf; ++j ) ibuf[j] = ptr[j]; } break;
                default :
                    HLR_ERROR( "unsupported storage size" );
            }// switch

            // convert from compressed format
            for ( size_t  j = 0; j < nbuf; ++j )
            {
                const auto  zval  = ibuf[j];
                const uint  mant  = zval & prec_mask;
                const uint  exp   = (zval >> prec_bits) & exp_mask;
                const bool  sign  = zval >> sign_shift;
                const uint  irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);

                zero[j] = ( zval == zero_val );
                fbuf[j] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
            }// for

            // correct zeroes
            for ( size_t  j = 0; j < nbuf; ++j )
                if ( zero[j] )
                    fbuf[j] = double(0);

            // copy values
            for ( size_t  j = 0; j < nbuf; ++j )
                dest[i+j] = fbuf[j];
            
            pos += nbyte * nbuf;
        }// for

        // handle remaining values
        for ( ; i < nsize; ++i )
        {
            uint  zval = 0;
            
            switch ( nbyte )
            {
                case  4 : zval |= zdata[pos+3] << 24;
                case  3 : zval |= zdata[pos+2] << 16;
                case  2 : zval |= zdata[pos+1] << 8;
                case  1 : zval |= zdata[pos]; break;
                default : HLR_ERROR( "unsupported storage size" );
            }// switch

            if ( zval == zero_val )
                dest[i] = 0;
            else
            {
                const uint  mant  = zval & prec_mask;
                const uint  exp   = (zval >> prec_bits) & exp_mask;
                const bool  sign  = zval >> sign_shift;
                const uint  irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
                
                dest[i] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
            }// else

            pos += nbyte;
        }// for
    }// if
    else
    {
        const uint        prec_ofs   = fp64_mant_bits - prec_bits;
        const uint        sign_shift = exp_bits + prec_bits;
        const ulong       zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );
        constexpr size_t  nbuf     = 32;
        size_t            i          = 0;
        double            scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        HLR_ASSERT( nbyte >= 4 );

        if ( nbyte == 4 )
        {
            // number of values to read before decoding
            const uint    prec_mask = ( 1u << prec_bits ) - 1;
            const uint    exp_mask  = ( 1u << exp_bits  ) - 1;
            const size_t  nbsize    = (nsize / nbuf) * nbuf;  // largest multiple of <nbuf> below <nsize>

            for ( ; i < nbsize; i += nbuf )
            {
                //
                // read next values into local buffer
                //

                uint          ibuf[ nbuf ];
                const size_t  pos = 10 + i*nbyte;
                    
                std::copy( zdata.data() + pos, zdata.data() + pos + 4*nbuf, reinterpret_cast< byte_t * >( ibuf ) );

                //
                // convert all values
                //

                for ( uint  j = 0; j < nbuf; ++j )
                {
                    const uint  zval = ibuf[j];

                    if ( zval == uint(zero_val) )
                        dest[i+j] = 0;
                    else
                    {
                        const uint   mant  = zval & prec_mask;
                        const uint   exp   = (zval >> prec_bits) & exp_mask;
                        const bool   sign  = zval >> sign_shift;
                        const ulong  irval = (ulong(exp | fp64_exp_highbit) << fp64_mant_bits) | (ulong(mant) << prec_ofs);
                        
                        dest[i+j] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                    }// else
                }// for
            }// for
        }// if
        else
        {
            const size_t  nbsize    = (nsize / nbuf) * nbuf;  // largest multiple of <nbuf> below <nsize>
            const ulong   prec_mask = ( 1ul << prec_bits ) - 1;
            const ulong   exp_mask  = ( 1ul << exp_bits  ) - 1;

            for ( ; i < nbsize; i += nbuf )
            {
                //
                // read next values into local buffer
                //

                ulong  ibuf[ nbuf ];
        
                if ( nbyte == 5 )
                {
                    for ( uint  j = 0; j < nbuf; ++j )
                    {
                        const size_t  pos  = 10 + (i+j)*nbyte;

                        ibuf[j] = ( (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if
                else if ( nbyte == 6 )
                {
                    for ( uint  j = 0; j < nbuf; ++j )
                    {
                        const size_t  pos  = 10 + (i+j)*nbyte;

                        ibuf[j] = ( (ulong(zdata[pos+5]) << 40) |
                                           (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if
                else if ( nbyte == 7 )
                {
                    for ( uint  j = 0; j < nbuf; ++j )
                    {
                        const size_t  pos  = 10 + (i+j)*nbyte;

                        ibuf[j] = ( (ulong(zdata[pos+6]) << 48) |
                                           (ulong(zdata[pos+5]) << 40) |
                                           (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if
                else if ( nbyte == 8 )
                {
                    const size_t  pos = 10 + i*nbyte;
                    
                    std::copy( zdata.data() + pos, zdata.data() + pos + 8*nbuf, reinterpret_cast< byte_t * >( ibuf ) );
                }// if

                //
                // convert all values
                //

                for ( uint  j = 0; j < nbuf; ++j )
                {
                    const ulong  zval = ibuf[j];

                    if ( zval == zero_val )
                        dest[i+j] = 0;
                    else
                    {
                        const ulong   mant  = zval & prec_mask;
                        const ulong   exp   = (zval >> prec_bits) & exp_mask;
                        const bool    sign  = zval >> sign_shift;
                        const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
                        
                        dest[i+j] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                    }// else
                }// for
            }// for
        }// else
    
        const ulong  prec_mask = ( 1ul << prec_bits ) - 1;
        const ulong  exp_mask  = ( 1ul << exp_bits  ) - 1;
        
        for ( ; i < nsize; ++i )
        {
            ulong         zval = 0;
            const size_t  pos  = 10 + i*nbyte;
            
            switch ( nbyte )
            {
                case  8 : zval |= ulong(zdata[pos+7]) << 56;
                case  7 : zval |= ulong(zdata[pos+6]) << 48;
                case  6 : zval |= ulong(zdata[pos+5]) << 40;
                case  5 : zval |= ulong(zdata[pos+4]) << 32;
                case  4 : zval |= ulong(zdata[pos+3]) << 24;
                case  3 : zval |= ulong(zdata[pos+2]) << 16;
                case  2 : zval |= ulong(zdata[pos+1]) << 8;
                case  1 : zval |= ulong(zdata[pos]); break;
                default :
                    HLR_ERROR( "unsupported byte size" );
            }// switch

            if ( zval == zero_val )
                dest[i] = 0;
            else
            {
                const ulong   mant  = zval & prec_mask;
                const ulong   exp   = (zval >> prec_bits) & exp_mask;
                const bool    sign  = zval >> sign_shift;
                const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
                const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

                dest[i] = rval;
            }// else
        }// for
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
    if ( dim1 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, 2 );
    else
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, dim3 * 2 );
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
    if ( dim1 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, 2 );
    else
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, dim3 * 2 );
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
compress_lr< double > ( const blas::matrix< double > &  U,
                        const blas::vector< double > &  S )
{
    using  real_t = double;
    
    //
    // first, determine exponent bits and mantissa bits for all
    // columns
    //

    const size_t  n = U.nrows();
    const size_t  k = U.ncols();
    auto          m = std::vector< uint >( k );
    auto          e = std::vector< uint >( k );
    auto          s = std::vector< real_t >( k );
    size_t        zsize = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        auto  vmin = fp64_infinity;
        auto  vmax = real_t(0);

        for ( size_t  i = 1; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            const auto  val  = ( u_il == double(0) ? fp64_infinity : u_il );
            
            vmin = std::min( vmin, val );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint( std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );

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

    // for ( uint  l = 0; l < k; ++l )
    //     std::cout << e[l] << '/' << m[l] << ", ";
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint  l = 0; l < k; ++l )
    {
        const uint    exp_bits  = e[l];
        const uint    exp_mask  = ( 1 << exp_bits ) - 1;
        const uint    prec_bits = m[l];
        const real_t  scale     = s[l];
        const size_t  nbits     = 1 + exp_bits + prec_bits; // number of bits per value
        const size_t  nbyte     = nbits / 8;

        if (( prec_bits <= fp32_mant_bits ) && ( nbyte <= 4 ))
        {
            //
            // store header (exponents and precision bits and scaling factor)
            //
    
            const float  fscale = scale;
        
            zdata[pos]   = exp_bits;
            zdata[pos+1] = prec_bits;
            memcpy( zdata.data() + pos + 2, & fscale, 4 );

            pos += 6;
            
            //
            // store data
            //
        
            const uint  prec_ofs = fp32_mant_bits - prec_bits;
            const uint  zero_val = fp32_zero_val & (( 1 << nbits) - 1 );

            #if 1
            constexpr size_t  nbuf = 16;
            bool              zero[ nbuf ];
            bool              sign[ nbuf ];
            float             fbuf[ nbuf ];
            uint              ibuf[ nbuf ];
            size_t            i = 0;
            
            for ( ; i < n; i += nbuf )
            {
                // scale data to [2,...]
                for ( size_t  j = 0; j < nbuf; ++j )
                {
                    const float  val  = U(i+j,l);
                    const auto   aval = std::abs( val );

                    zero[j] = ( aval == float(0) );
                    sign[j] = ( aval != val );
                    fbuf[j] = std::max( fscale * aval + 1, 2.f ); // prevent rounding issues when converting from fp64
                }// for

                // convert to compressed format
                for ( size_t  j = 0; j < nbuf; ++j )
                {
                    const uint   isval = (*reinterpret_cast< const uint * >( & fbuf[j] ) );
                    const uint   sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                    const uint   smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                
                    // exponent and mantissa reduced to stored size
                    const uint   zexp  = sexp & exp_mask;
                    const uint   zmant = smant >> prec_ofs;
                
                    ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;
                }// for

                // correct zeroes
                for ( size_t  j = 0; j < nbuf; ++j )
                    if ( zero[j] )
                        ibuf[j] = zero_val;

                // write to destination buffer
                switch ( nbyte )
                {
                    case  4 :
                    {
                        auto  ptr = reinterpret_cast< uint * >( & zdata[pos] );
                        
                        for ( size_t  j = 0; j < nbuf; ++j )
                            ptr[j] = ibuf[j];
                    }
                    break;

                    case  3 :
                    {
                        for ( size_t  j = 0; j < nbuf; ++j )
                        {
                            const auto  zval = ibuf[j];
                            
                            zdata[pos+3*j+2] = ( zval & 0x00ff0000 ) >> 16;
                            zdata[pos+3*j+1] = ( zval & 0x0000ff00 ) >> 8;
                            zdata[pos+3*j]   = ( zval & 0x000000ff );
                        }// for
                    }
                    break;

                    case  2 :
                    {
                        auto  ptr = reinterpret_cast< ushort * >( & zdata[pos] );
                        
                        for ( size_t  j = 0; j < nbuf; ++j )
                            ptr[j] = ushort( ibuf[j] & 0xffff );
                    }
                    break;

                    case  1 :
                    {
                        for ( size_t  j = 0; j < nbuf; ++j )
                            zdata[pos+j] = ( ibuf[j] & 0xff );
                    }
                    break;
                    
                    default :
                        HLR_ERROR( "unsupported storage size" );
                }// switch

                pos += nbyte * nbuf;
            }// for
            
            #endif
            
            for ( ; i < n; ++i )
            {
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
                const float  val  = U(i,l);
                uint         zval = zero_val;
            
                if ( std::abs( val ) != float(0) )
                {
                    const bool   zsign = ( val < 0 );
                    const float  sval  = std::max( fscale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
                    const uint   isval = (*reinterpret_cast< const uint * >( & sval ) );
                    const uint   sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                    const uint   smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                
                    // exponent and mantissa reduced to stored size
                    const uint   zexp  = sexp & exp_mask;
                    const uint   zmant = smant >> prec_ofs;
                
                    zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                    HLR_DBG_ASSERT( zval != zero_val );
                }// if
        
                //
                // copy zval into data buffer
                //

                switch ( nbyte )
                {
                    case  4 :
                        *reinterpret_cast< uint * >( & zdata[pos] ) = zval;
                        break;
                    case  3 :
                        zdata[pos+2] = ( zval & 0x00ff0000 ) >> 16;
                        zdata[pos+1] = ( zval & 0x0000ff00 ) >> 8;
                        zdata[pos]   = ( zval & 0x000000ff );
                        break;
                    case  2 :
                        *reinterpret_cast< ushort * >( & zdata[pos] ) = ushort(zval & 0xffff);
                        break;
                    case  1 :
                        zdata[pos]   = ( zval & 0x000000ff );
                        break;
                    default :
                        HLR_ERROR( "unsupported storage size" );
                }// switch

                pos += nbyte;
            }// for
        }// if
        else
        {
            //
            // store header (exponents and precision bits and scaling factor)
            //
    
            zdata[pos]   = exp_bits;
            zdata[pos+1] = prec_bits;
            memcpy( zdata.data() + pos + 2, & scale, 8 );

            pos += 10;
            
            //
            // store data
            //
        
            const uint   prec_ofs = fp64_mant_bits - prec_bits;
            const ulong  zero_val = fp64_zero_val & (( 1ul << nbits) - 1 );

            for ( size_t  i = 0; i < n; ++i )
            {
                const double  val  = U(i,l);
                ulong         zval = zero_val;
            
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
                if ( std::abs( val ) != double(0) )
                {
                    const bool    zsign = ( val < 0 );
                    const double  sval  = scale * std::abs(val) + 1;
                    const ulong   isval = (*reinterpret_cast< const ulong * >( & sval ) );
                    const ulong   sexp  = ( isval >> fp64_mant_bits ) & ((1ul << fp64_exp_bits) - 1);
                    const ulong   smant = ( isval & ((1ul << fp64_mant_bits) - 1) );

                    // exponent and mantissa reduced to stored size
                    const ulong   zexp  = sexp & exp_mask;
                    const ulong   zmant = smant >> prec_ofs;

                    zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                    HLR_DBG_ASSERT( zval != zero_val );
                }// if
        
                //
                // copy zval into data buffer
                //

                switch ( nbyte )
                {
                    case  8 : zdata[pos+7] = ( zval & 0xff00000000000000 ) >> 56;
                    case  7 : zdata[pos+6] = ( zval & 0x00ff000000000000 ) >> 48;
                    case  6 : zdata[pos+5] = ( zval & 0x0000ff0000000000 ) >> 40;
                    case  5 : zdata[pos+4] = ( zval & 0x000000ff00000000 ) >> 32; break;
                    default :
                        HLR_ERROR( "unsupported storage size" );
                }// switch

                // *reinterpret_cast< uint * >( zdata.data() + pos ) = uint( zval & 0x00000000ffffffff );
                zdata[pos+3] = ( zval & 0x00000000ff000000 ) >> 24;
                zdata[pos+2] = ( zval & 0x0000000000ff0000 ) >> 16;
                zdata[pos+1] = ( zval & 0x000000000000ff00 ) >> 8;
                zdata[pos]   = ( zval & 0x00000000000000ff );

                pos += nbyte;
            }// for
        }// else
    }// for

    return zdata;
}

template <>
inline
void
decompress_lr< double > ( const zarray &            zdata,
                          blas::matrix< double > &  U )
{
    const size_t  n   = U.nrows();
    const uint    k   = U.ncols();
    size_t        pos = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        //
    
        const uint  exp_bits  = zdata[ pos ];
        const uint  prec_bits = zdata[ pos+1 ];
        const uint  nbits     = 1 + exp_bits + prec_bits;
        const uint  nbyte     = nbits / 8;

        pos += 2;
        
        if (( prec_bits <= 23 ) && ( nbits <= 32 ))
        {
            //
            // read scaling factor
            //
        
            float  scale;

            memcpy( & scale, zdata.data() + pos, 4 );
            pos += 4;
            
            //
            // read and convert compressed data
            //
    
            const uint  prec_mask  = ( 1 << prec_bits ) - 1;
            const uint  prec_ofs   = fp32_mant_bits - prec_bits;
            const uint  exp_mask   = ( 1 << exp_bits ) - 1;
            const uint  sign_shift = exp_bits + prec_bits;
            const uint  zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );

            for ( size_t  i = 0; i < n; ++i )
            {
                uint  zval = 0;
                
                switch ( nbyte )
                {
                    case  4 : zval |= zdata[pos+3] << 24;
                    case  3 : zval |= zdata[pos+2] << 16;
                    case  2 : zval |= zdata[pos+1] << 8;
                    case  1 : zval |= zdata[pos]; break;
                    default :
                        HLR_ERROR( "unsupported storage size" );
                }// switch
                
                if ( zval == zero_val )
                    U(i,l) = 0;
                else
                {
                    const uint  mant  = zval & prec_mask;
                    const uint  exp   = (zval >> prec_bits) & exp_mask;
                    const bool  sign  = zval >> sign_shift;
                    const uint  irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);
                    
                    U(i,l) = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
                }// else

                pos += nbyte;
            }// for
        }// if
        else
        {
            //
            // read scaling factor
            //
        
            double  scale;

            memcpy( & scale, zdata.data() + pos, 8 );
            pos += 8;
            
            //
            // read and convert compressed data
            //
    
            const ulong  prec_mask  = ( 1ul << prec_bits ) - 1;
            const uint   prec_ofs   = fp64_mant_bits - prec_bits;
            const ulong  exp_mask   = ( 1ul << exp_bits ) - 1;
            const uint   sign_shift = exp_bits + prec_bits;
            const ulong  zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );

            for ( size_t  i = 0; i < n; ++i )
            {
                ulong  zval  = 0;

                switch ( nbyte )
                {
                    case  8 : zval |= ulong(zdata[pos+7]) << 56;
                    case  7 : zval |= ulong(zdata[pos+6]) << 48;
                    case  6 : zval |= ulong(zdata[pos+5]) << 40;
                    case  5 : zval |= ulong(zdata[pos+4]) << 32;
                    case  4 : zval |= ulong(zdata[pos+3]) << 24;
                    case  3 : zval |= ulong(zdata[pos+2]) << 16;
                    case  2 : zval |= ulong(zdata[pos+1]) << 8;
                    case  1 : zval |= ulong(zdata[pos]); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                if ( zval == zero_val )
                    U(i,l) = 0;
                else
                {
                    const ulong   mant  = zval & prec_mask;
                    const ulong   exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
                    const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

                    U(i,l) = rval;
                }// else

                pos += nbyte;
            }// for
        }// else
    }// for
}

}}}// namespace hlr::compress::apfloat

#endif // __HLR_UTILS_DETAIL_APFLOAT_HH
