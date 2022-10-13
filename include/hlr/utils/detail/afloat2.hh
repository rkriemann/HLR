#ifndef __HLR_UTILS_DETAIL_AFLOAT_HH
#define __HLR_UTILS_DETAIL_AFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/afloat
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <chrono>

////////////////////////////////////////////////////////////
//
// compression using adaptive float representation
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace afloat {

// timing
using  my_clock = std::chrono::high_resolution_clock;

extern double  t_load;
extern double  t_decode;

using byte_t = unsigned char;

constexpr byte_t  fp32_mant_bits   = 23;
constexpr byte_t  fp32_exp_bits    = 8;
constexpr byte_t  fp32_sign_bit    = 31;
constexpr ulong   fp32_exp_highbit = 0b10000000;

constexpr uint    fp64_mant_bits   = 52;
constexpr uint    fp64_exp_bits    = 11;
constexpr uint    fp64_sign_bit    = 63;
constexpr ulong   fp64_exp_highbit = 0b10000000000;

inline
byte_t
eps_to_rate ( const double eps )
{
    if      ( eps >= 1e-2  ) return 10;
    else if ( eps >= 1e-3  ) return 12;
    else if ( eps >= 1e-4  ) return 14;
    else if ( eps >= 1e-5  ) return 16;
    else if ( eps >= 1e-6  ) return 20;
    else if ( eps >= 1e-7  ) return 22;
    else if ( eps >= 1e-8  ) return 24;
    else if ( eps >= 1e-9  ) return 28;
    else if ( eps >= 1e-10 ) return 32;
    else if ( eps >= 1e-12 ) return 44;
    else if ( eps >= 1e-14 ) return 54;
    else                     return 64;
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

    // scale all values v_i such that we have |v_i| >= 1
    const float   scale      = 1.0 / vmin;
    // number of bits needed to represent exponent values
    const byte_t  exp_bits   = std::max< float >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const uint    exp_mask   = ( 1 << exp_bits ) - 1;

    // round up to next multiple of 8
    const uint    nbits_min  = 1 + exp_bits + config.bitrate;                          // minimal number of bits per value for precision
    const uint    nbits      = ( nbits_min / 8 ) * 8 + ( nbits_min % 8 != 0 ? 8 : 0 ); // actual number of bits per value
    const uint    nbyte      = nbits / 8;

    const byte_t  prec_bits  = nbits - 1 - exp_bits;
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

    // look for min/max value
    double  vmin = std::abs( data[0] );
    double  vmax = std::abs( data[0] );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        vmin = std::min( vmin, std::abs( data[i] ) );
        vmax = std::max( vmax, std::abs( data[i] ) );
    }// for

    // scale all values v_i such that we have |v_i| >= 1
    const double  scale      = 1.0 / vmin;
    // number of bits needed to represent exponent values
    const byte_t  exp_bits   = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const ulong   exp_mask   = ( 1 << exp_bits ) - 1;

    // round up to next multiple of 8
    const uint    nbits_min  = 1 + exp_bits + config.bitrate;                          // minimal number of bits per value for precision
    const uint    nbits      = ( nbits_min / 8 ) * 8 + ( nbits_min % 8 != 0 ? 8 : 0 ); // actual number of bits per value
    const uint    nbyte      = nbits / 8;

    const byte_t  prec_bits  = nbits - 1 - exp_bits;
    
    const size_t  zsize      = 8 + 1 + 1 + nsize * nbyte;
    auto          zdata      = std::vector< byte_t >( zsize );

    HLR_ASSERT( nbits     <= 64 );
    HLR_ASSERT( prec_bits <= fp64_mant_bits );
    
    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
        //
        // store header (exponent bits, precision bits and scaling factor)
        //
        
        const uint   prec_ofs = fp32_mant_bits - prec_bits;
        const float  fscale   = scale;

        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & fscale, 4 );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const float   val   = data[i];
            const bool    zsign = ( val < 0 );

            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            const float   sval  = std::max( fscale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
            const uint    isval = (*reinterpret_cast< const uint * >( & sval ) );
            const uint    sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
            const uint    smant = ( isval & ((1u << fp32_mant_bits) - 1) );

            // exponent and mantissa reduced to stored size
            const uint    zexp  = sexp & exp_mask;
            const uint    zmant = smant >> prec_ofs;
            const uint    zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            // // DEBUG
            // {
            //     const byte_t  fp32_sign_bit = 31;
            //     const byte_t  sign_shift    = exp_bits + prec_bits;
            //     const uint    prec_mask     = ( 1u << prec_bits ) - 1;
            
            //     const uint   mant  = zval & prec_mask;
            //     const uint   exp   = (zval >> prec_bits) & exp_mask;
            //     const bool   sign  = zval >> sign_shift;

            //     const uint   rexp  = exp | 0b10000000; // re-add leading bit
            //     const uint   rmant = mant << prec_ofs;
            //     const uint   irval = (rexp << fp32_mant_bits) | rmant;
            //     const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

            //     if ( std::abs( (val - rval) / val ) > 1e-2 )
            //         std::cout << i << " : "
            //                   << val << " / "
            //                   << sval << " / "
            //                   << isval << " / "
            //                   << rval << " / "
            //                   << irval << " / "
            //                   << std::abs( (val - rval) / val ) << std::endl;
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
    }// if
    else
    {
        //
        // store header (exponent bits, precision bits and scaling factor)
        //
        
        const uint  prec_ofs = fp64_mant_bits - prec_bits;

        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & scale, 8 );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  val   = data[i];
            const bool    zsign = ( val < 0 );

            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            const double  sval  = scale * std::abs(val) + 1;
            const ulong   isval = (*reinterpret_cast< const ulong * >( & sval ) );
            const ulong   sexp  = ( isval >> fp64_mant_bits ) & ((1u << fp64_exp_bits) - 1);
            const ulong   smant = ( isval & ((1ul << fp64_mant_bits) - 1) );

            // exponent and mantissa reduced to stored size
            const ulong   zexp  = sexp & exp_mask;
            const ulong   zmant = smant >> prec_ofs;
            const ulong   zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            // // DEBUG
            // {
            //     const uint    fp64_sign_bit = 63;
            //     const uint    sign_shift    = exp_bits + prec_bits;
            //     const ulong   prec_mask     = ( 1ul << prec_bits ) - 1;
            
            //     const ulong   mant  = zval & prec_mask;
            //     const ulong   exp   = (zval >> prec_bits) & exp_mask;
            //     const bool    sign  = zval >> sign_shift;

            //     const ulong   rexp  = exp | 0b10000000000; // re-add leading bit
            //     const ulong   rmant = mant << prec_ofs;
            //     const ulong   irval = (rexp << fp64_mant_bits) | rmant;
            //     const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

            //     if ( std::abs( (val - rval) / val ) > 1e-2 )
            //         std::cout << i << " : "
            //                   << val << " / "
            //                   << sval << " / "
            //                   << isval << " / "
            //                   << rval << " / "
            //                   << irval << " / "
            //                   << std::abs( (val - rval) / val ) << std::endl;
            // }
        
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
                case  4 : zdata[pos+3] = ( zval & 0x00000000ff000000 ) >> 24;
                case  3 : zdata[pos+2] = ( zval & 0x0000000000ff0000 ) >> 16;
                case  2 : zdata[pos+1] = ( zval & 0x000000000000ff00 ) >> 8;
                case  1 : zdata[pos]   = ( zval & 0x00000000000000ff ); break;
                default :
                    HLR_ERROR( "unsupported storage size" );
            }// switch
        }// for
    }// else

    return zdata;
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
    
    float         scale;
    const byte_t  exp_bits  = zdata[4];
    const byte_t  prec_bits = zdata[5];

    memcpy( & scale, zdata.data(), 4 );

    //
    // read compressed data
    //

    const uint    nbits       = 1 + exp_bits + prec_bits;
    const uint    nbyte       = nbits / 8;
    const uint    prec_mask   = ( 1 << prec_bits ) - 1;
    const uint    prec_ofs    = fp32_mant_bits - prec_bits;
    const uint    exp_mask    = ( 1 << exp_bits ) - 1;
    const uint    exp_highbit = 0b10000000;
    const byte_t  sign_shift  = exp_bits + prec_bits;

    // number of values to read before decoding
    constexpr size_t  nchunk = 32;
    const size_t      ncsize = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
    size_t            i      = 0;

    for ( ; i < ncsize; i += nchunk )
    {
        uint  zval_buf[ nchunk ];
        
        //
        // read next 8 values into local buffer
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
        // convert all 8 values
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
    
    const byte_t  exp_bits  = zdata[0];
    const byte_t  prec_bits = zdata[1];

    //
    // read compressed data
    //
    
    const  uint  nbits = 1 + exp_bits + prec_bits;
    const  uint  nbyte = nbits / 8;
    
    if (( nbyte <= 4 ) && ( prec_bits <= fp32_mant_bits ))
    {
        const uint    prec_mask   = ( 1 << prec_bits ) - 1;
        const uint    prec_ofs    = fp32_mant_bits - prec_bits;
        const uint    exp_mask    = ( 1 << exp_bits ) - 1;
        const byte_t  sign_shift  = exp_bits + prec_bits;
        size_t        i           = 0;
        float         scale;

        memcpy( & scale, zdata.data() + 2, 4 );

        // number of values to read before decoding
        constexpr size_t  nchunk = 64;
        const size_t      ncsize = nsize - nsize % nchunk;  // largest multiple of <nchunk> below <nsize>

        if ( nbyte == 1 )
        {
            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                byte_t  zval_buf[ nchunk ];
        
                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    zval_buf[lpos] = zdata[6 + (i+lpos)*nbyte];

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const byte_t  zval  = zval_buf[lpos];
                    const byte_t  mant  = zval & prec_mask;
                    const byte_t  exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const uint    irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);

                    dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
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

                ushort  zval_buf[ nchunk ];
        
                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const size_t  pos  = 6 + (i+lpos)*nbyte;

                    zval_buf[lpos] = (zdata[pos+1] << 8) | zdata[pos];
                }// for

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const ushort  zval  = zval_buf[lpos];
                    const ushort  mant  = zval & prec_mask;
                    const ushort  exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const uint    irval = (uint(exp | fp32_exp_highbit) << fp32_mant_bits) | (uint(mant) << prec_ofs);

                    dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
                }// for
            }// for
        }// if
        else
        {
            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                uint  zval_buf[ nchunk ];
        
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
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 6 + (i+lpos)*nbyte;

                        zval_buf[lpos] = (zdata[pos+3] << 24) | (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
                    }// for
                }// if

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const uint  zval  = zval_buf[lpos];
                    const uint  mant  = zval & prec_mask;
                    const uint  exp   = (zval >> prec_bits) & exp_mask;
                    const bool  sign  = zval >> sign_shift;
                    const uint  irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);

                    dest[i+lpos] = double( (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale );
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

            const uint   mant  = zval & prec_mask;
            const uint   exp   = (zval >> prec_bits) & exp_mask;
            const bool   sign  = zval >> sign_shift;
            const uint   irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
            const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

            dest[i] = double( rval );
        }// for
    }// if
    else
    {
        const  uint   prec_ofs   = fp64_mant_bits - prec_bits;
        const  uint   sign_shift = exp_bits + prec_bits;
        size_t        i          = 0;
        double        scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        if ( nbyte == 1 )
        {
            // number of values to read before decoding
            constexpr size_t  nchunk    = 32;
            size_t            ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
            const byte_t      prec_mask = ( 1 << prec_bits ) - 1;
            const byte_t      exp_mask  = ( 1 << exp_bits  ) - 1;

            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //
            
                byte_t  zval_buf[ nchunk ];
        
                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    zval_buf[lpos] = zdata[10 + (i+lpos)*nbyte];

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const byte_t  zval  = zval_buf[lpos];
                    const byte_t  mant  = zval & prec_mask;
                    const byte_t  exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const ulong   irval = (ulong(exp | fp64_exp_highbit) << fp64_mant_bits) | (ulong(mant) << prec_ofs);

                    dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                }// for
            }// for
        }// if
        else if ( nbyte == 2 )
        {
            // number of values to read before decoding
            constexpr size_t  nchunk    = 32;
            const size_t      ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
            const ushort      prec_mask = ( 1 << prec_bits ) - 1;
            const ushort      exp_mask  = ( 1 << exp_bits  ) - 1;

            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                ushort  zval_buf[ nchunk ];
        
                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const size_t  pos  = 10 + (i+lpos)*nbyte;
                
                    zval_buf[lpos] = (zdata[pos+1] << 8) | zdata[pos];
                }// for

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const ushort  zval  = zval_buf[lpos];
                    const ushort  mant  = zval & prec_mask;
                    const ushort  exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const ulong   irval = (ulong(exp | fp64_exp_highbit) << fp64_mant_bits) | (ulong(mant) << prec_ofs);

                    dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                }// for
            }// for
        }// if
        else if (( nbyte == 3 ) || ( nbyte == 4 ))
        {
            // number of values to read before decoding
            constexpr size_t  nchunk    = 32;
            const size_t      ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
            const uint        prec_mask = ( 1u << prec_bits ) - 1;
            const uint        exp_mask  = ( 1u << exp_bits  ) - 1;

            for ( ; i < ncsize; i += nchunk )
            {
                //
                // read next values into local buffer
                //

                uint  zval_buf[ nchunk ];
        
                if ( nbyte == 3 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
                    }// for
                }// if
                else // if ( nbyte == 4 )
                {
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = (zdata[pos+3] << 24) | (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
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
                    const ulong  irval = (ulong(exp | fp64_exp_highbit) << fp64_mant_bits) | (ulong(mant) << prec_ofs);

                    dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                }// for
            }// for
        }// if
        else
        {
            constexpr size_t  nchunk    = 32;
            const size_t      ncsize    = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
            const ulong       prec_mask = ( 1ul << prec_bits ) - 1;
            const ulong       exp_mask  = ( 1ul << exp_bits  ) - 1;

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
                    for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                    {
                        const size_t  pos  = 10 + (i+lpos)*nbyte;

                        zval_buf[lpos] = ( (ulong(zdata[pos+7]) << 56) |
                                           (ulong(zdata[pos+6]) << 48) |
                                           (ulong(zdata[pos+5]) << 40) |
                                           (ulong(zdata[pos+4]) << 32) |
                                           (ulong(zdata[pos+3]) << 24) |
                                           (ulong(zdata[pos+2]) << 16) |
                                           (ulong(zdata[pos+1]) <<  8) |
                                           (ulong(zdata[pos])        ) );
                    }// for
                }// if

                //
                // convert all values
                //

                for ( uint  lpos = 0; lpos < nchunk; ++lpos )
                {
                    const ulong   zval  = zval_buf[lpos];
                    const ulong   mant  = zval & prec_mask;
                    const ulong   exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);

                    dest[i+lpos] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
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
                    HLR_ERROR( "???" );
            }// switch

            const ulong   mant  = zval & prec_mask;
            const ulong   exp   = (zval >> prec_bits) & exp_mask;
            const bool    sign  = zval >> sign_shift;
            const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
            const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

            dest[i] = rval;
        }// for
    }// if
    
    // // number of values to read before decoding
    // constexpr size_t  nchunk = 32;
    
    // size_t  ncsize     = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
    // size_t  i          = 0;

    // for ( ; i < ncsize; i += nchunk )
    // {
    //     ulong  zval_buf[ nchunk ];
        
    //     //
    //     // read next 8 values into local buffer
    //     //

    //     // auto  tic = my_clock::now();

    //     if ( nbyte == 1 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = zdata[pos];
    //         }// for
    //     }// if
    //     else if ( nbyte == 2 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = (zdata[pos+1] << 8) | zdata[pos];
    //         }// for
    //     }// if
    //     else if ( nbyte == 3 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
    //         }// for
    //     }// if
    //     else if ( nbyte == 4 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = (ulong(zdata[pos+3]) << 24) | (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
    //         }// for
    //     }// if
    //     else if ( nbyte == 5 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = ( (ulong(zdata[pos+4]) << 32) |
    //                                (ulong(zdata[pos+3]) << 24) |
    //                                (ulong(zdata[pos+2]) << 16) |
    //                                (ulong(zdata[pos+1]) <<  8) |
    //                                (ulong(zdata[pos])        ) );
    //         }// for
    //     }// if
    //     else if ( nbyte == 6 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = ( (ulong(zdata[pos+5]) << 40) |
    //                                (ulong(zdata[pos+4]) << 32) |
    //                                (ulong(zdata[pos+3]) << 24) |
    //                                (ulong(zdata[pos+2]) << 16) |
    //                                (ulong(zdata[pos+1]) <<  8) |
    //                                (ulong(zdata[pos])        ) );
    //         }// for
    //     }// if
    //     else if ( nbyte == 7 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = ( (ulong(zdata[pos+6]) << 48) |
    //                                (ulong(zdata[pos+5]) << 40) |
    //                                (ulong(zdata[pos+4]) << 32) |
    //                                (ulong(zdata[pos+3]) << 24) |
    //                                (ulong(zdata[pos+2]) << 16) |
    //                                (ulong(zdata[pos+1]) <<  8) |
    //                                (ulong(zdata[pos])        ) );
    //         }// for
    //     }// if
    //     else if ( nbyte == 8 )
    //     {
    //         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //         {
    //             const size_t  pos  = 10 + (i+lpos)*nbyte;

    //             zval_buf[lpos] = ( (ulong(zdata[pos+7]) << 56) |
    //                                (ulong(zdata[pos+6]) << 48) |
    //                                (ulong(zdata[pos+5]) << 40) |
    //                                (ulong(zdata[pos+4]) << 32) |
    //                                (ulong(zdata[pos+3]) << 24) |
    //                                (ulong(zdata[pos+2]) << 16) |
    //                                (ulong(zdata[pos+1]) <<  8) |
    //                                (ulong(zdata[pos])        ) );
    //         }// for
    //     }// if

    //     // auto  toc   = my_clock::now();
    //     // auto  since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

    //     // t_load += since;
        
    //     //
    //     // convert all 8 values
    //     //

    //     // tic = my_clock::now();
        
    //     for ( uint  lpos = 0; lpos < nchunk; ++lpos )
    //     {
    //         const ulong   zval  = zval_buf[lpos];
    //         const ulong   mant  = zval & prec_mask;
    //         const ulong   exp   = (zval >> prec_bits) & exp_mask;
    //         const bool    sign  = zval >> sign_shift;

    //         const ulong   rexp  = exp | fp64_exp_highbit; // re-add leading bit
    //         const ulong   rmant = mant << prec_ofs;
    //         const ulong   irval = (rexp << fp64_mant_bits) | rmant;
    //         const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

    //         dest[i+lpos] = rval;
    //     }// for

    //     // toc   = my_clock::now();
    //     // since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

    //     // t_decode += since;
    // }// for

    // for ( ; i < nsize; ++i )
    // {
    //     ulong         zval = 0;
    //     const size_t  pos  = 10 + i*nbyte;
            
    //     switch ( nbyte )
    //     {
    //         case  8 : zval |= ulong(zdata[pos+7]) << 56;
    //         case  7 : zval |= ulong(zdata[pos+6]) << 48;
    //         case  6 : zval |= ulong(zdata[pos+5]) << 40;
    //         case  5 : zval |= ulong(zdata[pos+4]) << 32;
    //         case  4 : zval |= ulong(zdata[pos+3]) << 24;
    //         case  3 : zval |= ulong(zdata[pos+2]) << 16;
    //         case  2 : zval |= ulong(zdata[pos+1]) << 8;
    //         case  1 : zval |= ulong(zdata[pos]); break;
    //         default :
    //             HLR_ERROR( "???" );
    //     }// switch

    //     const ulong   mant  = zval & prec_mask;
    //     const ulong   exp   = (zval >> prec_bits) & exp_mask;
    //     const bool    sign  = zval >> sign_shift;
    //     const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
    //     const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

    //     dest[i] = rval;
    // }// for
}

// template <>
// inline
// void
// decompress< double > ( const zarray &  zdata,
//                        double *        dest,
//                        const size_t    dim0,
//                        const size_t    dim1,
//                        const size_t    dim2,
//                        const size_t    dim3,
//                        const size_t    dim4 )
// {
//     const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

//     //
//     // read compression header (scaling, exponent and precision bits)
//     //
    
//     float   scale;
//     byte_t  exp_bits;
//     byte_t  prec_bits;

//     // extract scaling factor
//     memcpy( & scale, zdata.data(), 4 );

//     // and exponent bits
//     memcpy( & exp_bits, zdata.data() + 4, 1 );

//     // and precision bits
//     memcpy( & prec_bits, zdata.data() + 5, 1 );

//     //
//     // read compressed data
//     //
    
//     const uint    nbits       = 1 + exp_bits + prec_bits;
//     const uint    nbyte       = nbits / 8;
//     const uint    prec_mask   = ( 1 << prec_bits ) - 1;
//     const uint    prec_ofs    = fp32_mant_bits - prec_bits;
//     const uint    exp_mask    = ( 1 << exp_bits ) - 1;
//     const uint    exp_highbit = 0b10000000;
//     const byte_t  sign_shift  = exp_bits + prec_bits;

//     // number of values to read before decoding
//     constexpr size_t  nchunk = 32;
    
//     size_t  ncsize     = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
//     size_t  i          = 0;

//     for ( ; i < ncsize; i += nchunk )
//     {
//         uint  zval_buf[ nchunk ];
        
//         //
//         // read next 8 values into local buffer
//         //

//         // auto  tic = my_clock::now();

//         if ( nbyte == 1 )
//         {
//             for ( uint  lpos = 0; lpos < nchunk; ++lpos )
//             {
//                 const size_t  pos  = 6 + (i+lpos)*nbyte;

//                 zval_buf[lpos] = zdata[pos];
//             }// for
//         }// if
//         else if ( nbyte == 2 )
//         {
//             for ( uint  lpos = 0; lpos < nchunk; ++lpos )
//             {
//                 const size_t  pos  = 6 + (i+lpos)*nbyte;

//                 zval_buf[lpos] = (zdata[pos+1] << 8) | zdata[pos];
//             }// for
//         }// if
//         else if ( nbyte == 3 )
//         {
//             for ( uint  lpos = 0; lpos < nchunk; ++lpos )
//             {
//                 const size_t  pos  = 6 + (i+lpos)*nbyte;

//                 zval_buf[lpos] = (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
//             }// for
//         }// if
//         else if ( nbyte == 4 )
//         {
//             for ( uint  lpos = 0; lpos < nchunk; ++lpos )
//             {
//                 const size_t  pos  = 6 + (i+lpos)*nbyte;

//                 zval_buf[lpos] = (zdata[pos+3] << 24) | (zdata[pos+2] << 16) | (zdata[pos+1] << 8) | zdata[pos];
//             }// for
//         }// if

//         // auto  toc   = my_clock::now();
//         // auto  since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

//         // t_load += since;
        
//         //
//         // convert all 8 values
//         //

//         // tic = my_clock::now();
        
//         for ( uint  lpos = 0; lpos < nchunk; ++lpos )
//         {
//             const uint   zval  = zval_buf[lpos];
//             const uint   mant  = zval & prec_mask;
//             const uint   exp   = (zval >> prec_bits) & exp_mask;
//             const bool   sign  = zval >> sign_shift;

//             const uint   rexp  = exp | exp_highbit; // re-add leading bit
//             const uint   rmant = mant << prec_ofs;
//             const uint   irval = (rexp << fp32_mant_bits) | rmant;
//             const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

//             dest[i+lpos] = double( rval );
//         }// for

//         // toc   = my_clock::now();
//         // since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

//         // t_decode += since;
//     }// for

//     for ( ; i < nsize; ++i )
//     {
//         uint          zval = 0;
//         const size_t  pos  = 6 + i*nbyte;
            
//         switch ( nbyte )
//         {
//             case  4 : zval |= zdata[pos+3] << 24;
//             case  3 : zval |= zdata[pos+2] << 16;
//             case  2 : zval |= zdata[pos+1] << 8;
//             case  1 : zval |= zdata[pos]; break;
//             default :
//                 HLR_ERROR( "???" );
//         }// switch

//         const uint   mant  = zval & prec_mask;
//         const uint   exp   = (zval >> prec_bits) & exp_mask;
//         const bool   sign  = zval >> sign_shift;
//         const uint   irval = ((exp | exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
//         const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

//         dest[i] = double( rval );
//     }// for
// }

//
// memory accessor
//
struct mem_accessor
{
    mem_accessor ( const double  /* eps */ )
    {}
    
    template < typename value_t >
    zarray
    encode ( value_t *        data,
             const size_t     dim0,
             const size_t     dim1 = 0,
             const size_t     dim2 = 0,
             const size_t     dim3 = 0 )
    {
        return compress( config(), data, dim0, dim1, dim2, dim3 );
    }
    
    template < typename value_t >
    void
    decode ( const zarray &  buffer,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
    {
        decompress( buffer, dest, dim0, dim1, dim2, dim3 );
    }
    
    size_t
    byte_size ( const zarray &  v )
    {
        return afloat::byte_size( v );
    }
    
private:

    mem_accessor ();
};
    
}}}// namespace hlr::compress::afloat

#endif // __HLR_UTILS_DETAIL_AFLOAT_HH
