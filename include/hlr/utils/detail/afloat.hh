#ifndef __HLR_UTILS_DETAIL_AFLOAT_HH
#define __HLR_UTILS_DETAIL_AFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/afloat
// Description : functions for adaptive floating points
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
// - mantissa size depends on precision
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace afloat {

using byte_t = unsigned char;

constexpr uint   fp32_mant_bits   = 23;
constexpr uint   fp32_exp_bits    = 8;
constexpr uint   fp32_sign_bit    = 31;
constexpr ulong  fp32_exp_highbit = 1 << (fp32_exp_bits-1);
constexpr uint   fp32_zero_val    = 0xffffffff;

constexpr uint   fp64_mant_bits   = 52;
constexpr uint   fp64_exp_bits    = 11;
constexpr uint   fp64_sign_bit    = 63;
constexpr ulong  fp64_exp_highbit = 1 << (fp64_exp_bits-1);
constexpr ulong  fp64_zero_val    = 0xffffffffffffffff;

inline
byte_t
eps_to_rate ( const double eps )
{
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
}

// inline
// byte_t
// eps_to_rate ( const double eps )
// {
//     if      ( eps >= 1e-2  ) return 10;
//     else if ( eps >= 1e-3  ) return 12;
//     else if ( eps >= 1e-4  ) return 14;
//     else if ( eps >= 1e-5  ) return 16;
//     else if ( eps >= 1e-6  ) return 20;
//     else if ( eps >= 1e-7  ) return 22;
//     else if ( eps >= 1e-8  ) return 26;
//     else if ( eps >= 1e-9  ) return 30;
//     else if ( eps >= 1e-10 ) return 34;
//     else if ( eps >= 1e-12 ) return 38;
//     else if ( eps >= 1e-14 ) return 42;
//     else                     return 64;
// }

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

    //
    // look for min/max value (> 0!)
    //
    
    float  vmin = 0;
    float  vmax = 0;

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

    HLR_DBG_ASSERT( vmin > double(0) );
    
    // scale all values v_i such that we have |v_i| >= 1
    const float   scale     = 1.0 / vmin;
    
    // number of bits needed to represent exponent values
    const uint    exp_bits  = std::max< float >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const uint    exp_mask  = ( 1 << exp_bits ) - 1;
    const uint    prec_bits = config.bitrate;
    const uint    prec_mask = ( 1 << prec_bits ) - 1;
    const uint    prec_ofs  = fp32_mant_bits - prec_bits;

    const size_t  nbits      = 1 + exp_bits + prec_bits; // number of bits per value
    const size_t  n_tot_bits = nsize * nbits;            // number of bits for all values
    const size_t  zsize      = 4 + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
    auto          zdata      = std::vector< byte_t >( zsize );

    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    zdata[0] = exp_bits;
    zdata[1] = prec_bits;
    memcpy( zdata.data() + 2, & scale, 4 );

    //
    // store data
    //

    size_t  pos  = 6; // data starts after scaling factor, exponent bits and precision bits
    byte_t  bpos = 0; // start bit position in current byte
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const float  val  = data[i];
        uint         zval = fp32_zero_val;

        if ( std::abs( val ) >= vmin )
        {
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

            zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

            HLR_DBG_ASSERT( zval != fp32_zero_val );
            
            // // DEBUG
            // {
            //     const byte_t  sign_shift = exp_bits + prec_bits;
            
            //     const uint   mant  = zval & prec_mask;
            //     const uint   exp   = (zval >> prec_bits) & exp_mask;
            //     const bool   sign  = zval >> sign_shift;
            
            //     const uint   rexp  = exp | fb32_exp_highbit; // re-add leading bit
            //     const uint   rmant = mant << prec_ofs;
            //     const uint   irval = (rexp << fp32_mant_bits) | rmant;
            //     const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;
            
            //     std::cout << i << " : " << val << " / " << rval << " / " << std::abs( (val - rval) / val ) << std::endl;
            // }
        }// if
        
        //
        // copy bits of zval into data buffer
        //
        
        byte_t  sbits = 0; // number of already stored bits of zval
            
        do
        {
            const byte_t  crest = 8 - bpos;       // remaining bits in current byte
            const byte_t  zrest = nbits - sbits;  // remaining bits in zval
            const byte_t  zbyte = zval & 0xff;    // lowest byte of zval
            
            HLR_DBG_ASSERT( pos < zsize );
        
            zdata[pos] |= (zbyte << bpos);
            zval      >>= crest;
            sbits      += crest;
            
            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );
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

    HLR_DBG_ASSERT( vmin > double(0) );
    
    // scale all values v_i such that we have |v_i| >= 1
    const double  scale     = 1.0 / vmin;

    // number of bits needed to represent exponent values
    const uint    exp_bits  = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const uint    exp_mask  = ( 1 << exp_bits ) - 1;
    const uint    prec_bits = config.bitrate;

    const size_t  nbits      = 1 + exp_bits + prec_bits; // number of bits per value
    const size_t  n_tot_bits = nsize * nbits;            // number of bits for all values

    if (( prec_bits <= 23 ) && ( nbits <= 32 ))
    {
        const size_t  zsize = 4 + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
        auto          zdata = std::vector< byte_t >( zsize );
        
        //
        // store header (exponents and precision bits and scaling factor)
        //
    
        const float  fscale = scale;
        
        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & fscale, 4 );

        //
        // store data
        //
        
        
        const float  fmin     = vmin;
        const uint   prec_ofs = fp32_mant_bits - prec_bits;
        size_t       pos      = 6; // data starts after scaling factor, exponent bits and precision bits
        uint         bpos     = 0; // start bit position in current byte

        for ( size_t  i = 0; i < nsize; ++i )
        {
            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            const float  val  = data[i];
            uint         zval = fp32_zero_val;
            
            if ( std::abs( val ) >= fmin )
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

                HLR_DBG_ASSERT( zval != fp32_zero_val );
            }// if
        
            //
            // copy bits of zval into data buffer
            //
        
            uint  sbits = 0; // number of already stored bits of zval
            
            do
            {
                const uint    crest = 8 - bpos;       // remaining bits in current byte
                const uint    zrest = nbits - sbits;  // remaining bits in zval
                const byte_t  zbyte = zval & 0xff;    // lowest byte of zval
                
                HLR_DBG_ASSERT( pos < zsize );
                
                zdata[pos] |= (zbyte << bpos);
                zval      >>= crest;
                sbits      += crest;
                
                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );
        }// for
        
        return zdata;
    }// if
    else
    {
        const size_t  zsize = 8 + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
        auto          zdata = std::vector< byte_t >( zsize );
        
        //
        // store header (exponents and precision bits and scaling factor)
        //
    
        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & scale, 8 );

        //
        // store data
        //
        
        const uint  prec_ofs = fp64_mant_bits - prec_bits;
        size_t      pos      = 10; // data starts after scaling factor, exponent bits and precision bits
        uint        bpos     = 0;  // start bit position in current byte

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  val  = data[i];
            ulong         zval = fp64_zero_val;
            
            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            if ( std::abs( val ) >= vmin )
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

                HLR_DBG_ASSERT( zval != fp64_zero_val );
                
                // // DEBUG
                // {
                //     const byte_t  sign_shift = exp_bits + prec_bits;
                //     const ulong   prec_mask  = ( 1ul << prec_bits ) - 1;
            
                //     const ulong   mant  = zval & prec_mask;
                //     const ulong   exp   = (zval >> prec_bits) & exp_mask;
                //     const bool    sign  = zval >> sign_shift;

                //     const ulong   rexp  = exp | fp64_exp_highbit; // re-add leading bit
                //     const ulong   rmant = mant << prec_ofs;
                //     const ulong   irval = (rexp << fp64_mant_bits) | rmant;
                //     const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

                //     std::cout << i << " : " << val << " / " << rval << " / " << std::abs( (val - rval) / val ) << std::endl;
                // }
            }// if
        
            //
            // copy bits of zval into data buffer
            //
        
            uint  sbits = 0; // number of already stored bits of zval
            
            do
            {
                const uint    crest = 8 - bpos;       // remaining bits in current byte
                const uint    zrest = nbits - sbits;  // remaining bits in zval
                const byte_t  zbyte = zval & 0xff;    // lowest byte of zval

                HLR_DBG_ASSERT( pos < zsize );
        
                zdata[pos] |= (zbyte << bpos);
                zval      >>= crest;
                sbits      += crest;
            
                if ( crest <= zrest )
                {
                    ++pos;
                    bpos = 0;
                }// if
                else
                {
                    bpos += zrest;
                }// else
            } while ( sbits < nbits );
        }// for

        return zdata;
    }// else
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
    
    const uint  exp_bits  = zdata[0];
    const uint  prec_bits = zdata[1];
    float       scale;

    memcpy( & scale, zdata.data() + 2, 4 );

    //
    // read compressed data
    //
    
    const uint  nbits       = 1 + exp_bits + prec_bits;
    const uint  prec_mask   = ( 1 << prec_bits ) - 1;
    const uint  prec_ofs    = fp32_mant_bits - prec_bits;
    const uint  exp_mask    = ( 1 << exp_bits ) - 1;
    const uint  sign_shift  = exp_bits + prec_bits;

    size_t  pos    = 6;
    uint    bpos   = 0; // bit position in current byte

    for ( size_t  i = 0; i < nsize; ++i )
    {
        uint  zval  = 0;
        uint  sbits = 0;
            
        do
        {
            HLR_DBG_ASSERT( pos < zdata.size() );
        
            const uint    crest = 8 - bpos;
            const uint    zrest = nbits - sbits;
            const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
            const byte_t  data  = (zdata[pos] >> bpos) & zmask;
                
            zval  |= (uint(data) << sbits);
            sbits += crest;

            if ( crest <= zrest )
            {
                ++pos;
                bpos = 0;
            }// if
            else
                bpos += zrest;
        } while ( sbits < nbits );

        if ( zval == fp32_zero_val )
            dest[i] = 0;
        else
        {
            const uint   mant  = zval & prec_mask;
            const uint   exp   = (zval >> prec_bits) & exp_mask;
            const bool   sign  = zval >> sign_shift;
            const uint   irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
            const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

            dest[i] = rval;
        }// else
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
    const uint  nbits     = 1 + exp_bits + prec_bits;

    if (( prec_bits <= 23 ) && ( nbits <= 32 ))
    {
        //
        // read scaling factor
        //
        
        float  scale;

        memcpy( & scale, zdata.data() + 2, 4 );

        //
        // read compressed data
        //
    
        const uint  prec_mask  = ( 1 << prec_bits ) - 1;
        const uint  prec_ofs   = fp32_mant_bits - prec_bits;
        const uint  exp_mask   = ( 1 << exp_bits ) - 1;
        const uint  sign_shift = exp_bits + prec_bits;

        size_t  pos    = 6;
        uint    bpos   = 0; // bit position in current byte

        for ( size_t  i = 0; i < nsize; ++i )
        {
            uint  zval  = 0;
            uint  sbits = 0;
            
            do
            {
                HLR_DBG_ASSERT( pos < zdata.size() );
        
                const uint    crest = 8 - bpos;
                const uint    zrest = nbits - sbits;
                const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
                const byte_t  data  = (zdata[pos] >> bpos) & zmask;
                
                zval  |= (uint(data) << sbits);
                sbits += crest;

                if ( crest <= zrest )
                {
                    ++pos;
                    bpos = 0;
                }// if
                else
                    bpos += zrest;
            } while ( sbits < nbits );

            if ( zval == fp32_zero_val )
                dest[i] = 0;
            else
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
        //
        // read scaling factor
        //
        
        double  scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        //
        // read compressed data
        //
    
        const ulong  prec_mask  = ( 1ul << prec_bits ) - 1;
        const uint   prec_ofs   = fp64_mant_bits - prec_bits;
        const ulong  exp_mask   = ( 1ul << exp_bits ) - 1;
        const uint   sign_shift = exp_bits + prec_bits;

        size_t  pos    = 10;
        uint    bpos   = 0;  // bit position in current byte

        for ( size_t  i = 0; i < nsize; ++i )
        {
            ulong  zval  = 0;
            uint   sbits = 0;
            
            do
            {
                HLR_DBG_ASSERT( pos < zdata.size() );
        
                const uint    crest = 8 - bpos;
                const uint    zrest = nbits - sbits;
                const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
                const byte_t  data  = (zdata[pos] >> bpos) & zmask;
                
                zval  |= (ulong(data) << sbits);
                sbits += crest;

                if ( crest <= zrest )
                {
                    ++pos;
                    bpos = 0;
                }// if
                else
                    bpos += zrest;
            } while ( sbits < nbits );

            if ( zval == fp64_zero_val )
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
    }// else
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
    
}}}// namespace hlr::compress::afloat

#endif // __HLR_UTILS_DETAIL_AFLOAT_HH
