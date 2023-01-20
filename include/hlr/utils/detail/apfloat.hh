#ifndef __HLR_UTILS_DETAIL_APFLOAT_HH
#define __HLR_UTILS_DETAIL_APFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/apfloat
// Description : functions for adaptive padded floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <cstring>

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

constexpr uint    fp64_mant_bits   = 52;
constexpr uint    fp64_exp_bits    = 11;
constexpr uint    fp64_sign_bit    = 63;
constexpr ulong   fp64_exp_highbit = 0b10000000000;
constexpr ulong   fp64_zero_val    = 0xffffffffffffffff;

// define for testing for zero values
// #define APFLOAT_CHECK_ZERO

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

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }
inline config  get_config ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

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
    
    #if defined(APFLOAT_CHECK_ZERO)
    
    double  vmin = 0;
    double  vmax = 0;

    {
        size_t  i = 0;

        for ( ; i < nsize; ++i )
        {
            const auto  di = std::abs( data[i] );

            if ( di > double(0) )
            {
                vmin = di;
                vmax = di;
                break;
            }// if
        }// for
        
        for ( ; i < nsize; ++i )
        {
            const auto  di = std::abs( data[i] );

            if ( di > double(0) )
            {
                vmin = std::min( vmin, di );
                vmax = std::max( vmax, di );
            }// if
        }// for
    }

    #else

    double  vmin = std::abs( data[0] );
    double  vmax = std::abs( data[0] );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        vmin = std::min( vmin, std::abs( data[i] ) );
        vmax = std::max( vmax, std::abs( data[i] ) );
    }// for

    #endif

    HLR_DBG_ASSERT( vmin > double(0) );
    
    // scale all values v_i such that we have |v_i| >= 1
    const double  scale      = 1.0 / vmin;
    
    // number of bits needed to represent exponent values
    const uint    exp_bits   = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const ulong   exp_mask   = ( 1 << exp_bits ) - 1;

    // round up to next multiple of 8
    const uint    nbits_min  = 1 + exp_bits + config.bitrate;                          // minimal number of bits per value for precision
    const uint    nbits      = ( nbits_min / 8 ) * 8 + ( nbits_min % 8 != 0 ? 8 : 0 ); // actual number of bits per value
    const uint    nbyte      = nbits / 8;

    const uint    prec_bits  = nbits - 1 - exp_bits;
    
    const size_t  zsize      = 8 + 1 + 1 + nsize * nbyte;
    auto          zdata      = std::vector< byte_t >( zsize );

    HLR_ASSERT( nbits     <= 64 );
    HLR_ASSERT( prec_bits <= fp64_mant_bits );

    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
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

        for ( size_t  i = 0; i < nsize; ++i )
        {
            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            const float  val  = data[i];
            uint         zval = zero_val;

            #if defined(APFLOAT_CHECK_ZERO)
            if ( std::abs( val ) >= fmin )
            #endif
            {
                const bool    zsign = ( val < 0 );

                const float   sval  = std::max( fscale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
                const uint    isval = (*reinterpret_cast< const uint * >( & sval ) );
                const uint    sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                const uint    smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                
                // exponent and mantissa reduced to stored size
                const uint    zexp  = sexp & exp_mask;
                const uint    zmant = smant >> prec_ofs;

                zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
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
    }// if
    else if ( nbyte == 4 )
    {
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
            
            #if defined(APFLOAT_CHECK_ZERO)
            if ( std::abs( val ) >= vmin )
            #endif
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
        HLR_DBG_ASSERT( nbyte > 4 );
        
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
            
            #if defined(APFLOAT_CHECK_ZERO)
            if ( std::abs( val ) >= vmin )
            #endif
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
                case  5 : zdata[pos+4] = ( zval & 0x000000ff00000000 ) >> 32; break;
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
    HLR_ERROR( "TODO" );
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
    HLR_ERROR( "TODO" );
}

template < typename value_t >
void
decompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 );

template <>
inline
void
decompress< float > ( const zarray &  zdata,
                      float *         dest,
                      const size_t    dim0,
                      const size_t    dim1,
                      const size_t    dim2,
                      const size_t    dim3,
                      const size_t    dim4 )
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
                       const size_t    dim3,
                       const size_t    dim4 )
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

        memcpy( & scale, zdata.data() + 2, 4 );

        // number of values to read before decoding
        constexpr size_t  nchunk = 64;
        const size_t      ncsize = nsize - nsize % nchunk;  // largest multiple of <nchunk> below <nsize>
        size_t            i      = 0;

        if ( nbyte == 1 )
        {
            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                byte_t        zval_buf[ nchunk ];
                const size_t  pos = 6 + i*nbyte;
                    
                std::copy( zdata.data() + pos, zdata.data() + pos + nchunk, reinterpret_cast< byte_t * >( zval_buf ) );

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const byte_t  zval  = zval_buf[lpos];

                    #if defined(APFLOAT_CHECK_ZERO)
                    if ( zval == zero_val )
                        dest[i+lpos] = 0;
                    else
                    #endif
                    {
                        const byte_t  mant  = zval & prec_mask;
                        const byte_t  exp   = (zval >> prec_bits) & exp_mask;
                        const bool    sign  = zval >> sign_shift;
                        const uint    irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);
                        
                        dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
                    }// else
                }// for
            }// for
        }// if
        else if ( nbyte == 2 )
        {
            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                ushort        zval_buf[ nchunk ];
                const size_t  pos = 6 + i*nbyte;
                    
                std::copy( zdata.data() + pos, zdata.data() + pos + 2*nchunk, reinterpret_cast< byte_t * >( zval_buf ) );

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const ushort  zval = zval_buf[lpos];

                    #if defined(APFLOAT_CHECK_ZERO)
                    if ( zval == zero_val )
                        dest[i+lpos] = 0;
                    else
                    #endif
                    {
                        const ushort  mant  = zval & prec_mask;
                        const ushort  exp   = (zval >> prec_bits) & exp_mask;
                        const bool    sign  = zval >> sign_shift;
                        const uint    irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);
                        
                        dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
                    }// else
                }// for
            }// for
        }// if
        else
        {
            uint  zval_buf[ nchunk ];
        
            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                if ( nbyte == 3 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 6 + (i+lpos)*nbyte;

                        zval_buf[lpos] = (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
                    }// for
                }// if
                else if ( nbyte == 4 )
                {
                    const size_t  pos = 6 + i*nbyte;
                    
                    std::copy( zdata.data() + pos, zdata.data() + pos + 4*nchunk, reinterpret_cast< byte_t * >( zval_buf ) );
                }// if

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const uint  zval  = zval_buf[lpos];

                    #if defined(APFLOAT_CHECK_ZERO)
                    if ( zval == zero_val )
                        dest[i+lpos] = 0;
                    else
                    #endif
                    {
                        const uint  mant  = zval & prec_mask;
                        const uint  exp   = (zval >> prec_bits) & exp_mask;
                        const bool  sign  = zval >> sign_shift;
                        const uint  irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);

                        dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
                    }// else
                }// for
            }// for
        }// else
        
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

            #if defined(APFLOAT_CHECK_ZERO)
            if ( zval == zero_val )
                dest[i] = 0;
            else
            #endif
            {
                const uint   mant  = zval & prec_mask;
                const uint   exp   = (zval >> prec_bits) & exp_mask;
                const bool   sign  = zval >> sign_shift;
                const uint   irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
                const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;
                
                dest[i] = double( rval );
            }// else
        }// for
    }// if
    else
    {
        const uint        prec_ofs   = fp64_mant_bits - prec_bits;
        const uint        sign_shift = exp_bits + prec_bits;
        const ulong       zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );
        constexpr size_t  nchunk     = 32;
        size_t            i          = 0;
        double            scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        if ( nbyte <= 3 )
        {
            HLR_ERROR( "nbyte == 1 .. 3" );
        }// if
        else if ( nbyte == 4 )
        {
            // number of values to read before decoding
            const uint    prec_mask = ( 1u << prec_bits ) - 1;
            const uint    exp_mask  = ( 1u << exp_bits  ) - 1;
            const size_t  ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>

            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                uint          zval_buf[ nchunk ];
                const size_t  pos = 10 + i*nbyte;
                    
                std::copy( zdata.data() + pos, zdata.data() + pos + 4*nchunk, reinterpret_cast< byte_t * >( zval_buf ) );

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const uint  zval = zval_buf[lpos];

                    #if defined(APFLOAT_CHECK_ZERO)
                    if ( zval == uint(zero_val) )
                        dest[i+lpos] = 0;
                    else
                    #endif
                    {
                        const uint   mant  = zval & prec_mask;
                        const uint   exp   = (zval >> prec_bits) & exp_mask;
                        const bool   sign  = zval >> sign_shift;
                        const ulong  irval = (ulong(exp | fp64_exp_highbit) << fp64_mant_bits) | (ulong(mant) << prec_ofs);
                        
                        dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                    }// else
                }// for
            }// for
        }// if
        else
        {
            const size_t  ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
            const ulong   prec_mask = ( 1ul << prec_bits ) - 1;
            const ulong   exp_mask  = ( 1ul << exp_bits  ) - 1;

            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                ulong  zval_buf[ nchunk ];
        
                if ( nbyte == 5 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = ( (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if
                else if ( nbyte == 6 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = ( (ulong(zdata[pos+5]) << 40) |
                                           (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if
                else if ( nbyte == 7 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = ( (ulong(zdata[pos+6]) << 48) |
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
                    
                    std::copy( zdata.data() + pos, zdata.data() + pos + 8*nchunk, reinterpret_cast< byte_t * >( zval_buf ) );
                }// if

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const ulong  zval = zval_buf[lpos];

                    #if defined(APFLOAT_CHECK_ZERO)
                    if ( zval == zero_val )
                        dest[i+lpos] = 0;
                    else
                    #endif
                    {
                        const ulong   mant  = zval & prec_mask;
                        const ulong   exp   = (zval >> prec_bits) & exp_mask;
                        const bool    sign  = zval >> sign_shift;
                        const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
                        
                        dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
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

            #if defined(APFLOAT_CHECK_ZERO)
            if ( zval == zero_val )
                dest[i] = 0;
            else
            #endif
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
                                      const size_t             dim3,
                                      const size_t             dim4 )
{
    HLR_ERROR( "TODO" );
}
    
template <>
inline
void
decompress< std::complex< double > > ( const zarray &            zdata,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    HLR_ERROR( "TODO" );
}
    
}}}// namespace hlr::compress::apfloat

#endif // __HLR_UTILS_DETAIL_APFLOAT_HH
