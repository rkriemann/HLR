#ifndef __HLR_UTILS_DETAIL_AFLOAT_HH
#define __HLR_UTILS_DETAIL_AFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/afloat
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstring>

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

// define for testing for zero values
// #define AFLOAT_CHECK_ZERO

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
    const uint    zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );
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
        uint         zval = zero_val;

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

            HLR_DBG_ASSERT( zval != zero_val );
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
    
    #if defined(AFLOAT_CHECK_ZERO)
    
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
        const uint   zero_val = fp32_zero_val & (( 1 << nbits) - 1 );
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
            uint         zval = zero_val;
            
            #if defined(AFLOAT_CHECK_ZERO)
            if ( std::abs( val ) >= fmin )
            #endif
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
        
        const uint   prec_ofs = fp64_mant_bits - prec_bits;
        const ulong  zero_val = fp64_zero_val & (( 1ul << nbits) - 1 );
        size_t       pos      = 10; // data starts after scaling factor, exponent bits and precision bits
        uint         bpos     = 0;  // start bit position in current byte

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  val  = data[i];
            ulong         zval = zero_val;
            
            //
            // Use absolute value and scale v_i and add 1 such that v_i >= 2.
            // With this, highest exponent bit is 1 and we only need to store
            // lowest <exp_bits> exponent bits
            //
        
            #if defined(AFLOAT_CHECK_ZERO)
            if ( std::abs( val ) >= vmin )
            #endif
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
    const uint  zero_val    = fp32_zero_val & (( 1 << nbits) - 1 );

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

            if ( crest <= zrest ) { bpos  = 0; ++pos; }
            else                  { bpos += zrest; }
        } while ( sbits < nbits );

        #if defined(AFLOAT_CHECK_ZERO)
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
                       const size_t    dim3 )
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
        // read and convert compressed data
        //
    
        const uint  prec_mask  = ( 1 << prec_bits ) - 1;
        const uint  prec_ofs   = fp32_mant_bits - prec_bits;
        const uint  exp_mask   = ( 1 << exp_bits ) - 1;
        const uint  sign_shift = exp_bits + prec_bits;
        const uint  zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );

        // number of values to read before decoding
        constexpr size_t  nchunk = 64;
    
        size_t  pos    = 6;
        uint    bpos   = 0;                          // bit position in current byte
        size_t  ncsize = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
        size_t  i      = 0;
        uint    zval_buf[ nchunk ];

        for ( ; i < ncsize; i += nchunk )
        {
            //
            // read next values into local buffer
            //
        
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                uint  zval  = 0;
                uint  sbits = 0;  // already read bits of zval
            
                do
                {
                    HLR_DBG_ASSERT( pos < zdata.size() );
        
                    const uint    crest = 8 - bpos;                               // remaining bits in current byte
                    const uint    zrest = nbits - sbits;                          // remaining bits to read for zval
                    const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                    const byte_t  data  = (zdata[pos] >> bpos) & zmask;           // part of zval in current byte
                
                    zval  |= (uint(data) << sbits); // lowest to highest bit in zdata
                    sbits += crest;

                    if ( crest <= zrest ) { bpos  = 0; ++pos; }
                    else                  { bpos += zrest; }
                } while ( sbits < nbits );

                zval_buf[lpos] = zval;
            }// for

            //
            // convert all values
            //
        
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const uint  zval = zval_buf[lpos];

                #if defined(AFLOAT_CHECK_ZERO)
                if ( zval == zero_val )
                    dest[i+lpos] = 0;
                else
                #endif
                {
                    const uint   mant  = zval & prec_mask;
                    const uint   exp   = (zval >> prec_bits) & exp_mask;
                    const bool   sign  = zval >> sign_shift;

                    const uint   rexp  = exp | fp32_exp_highbit; // re-add leading bit
                    const uint   rmant = mant << prec_ofs;
                    const uint   irval = (rexp << fp32_mant_bits) | rmant;
                    const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

                    dest[i+lpos] = double( rval );
                }// else
            }// for
        }// for

        for ( ; i < nsize; ++i )
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

                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );

            #if defined(AFLOAT_CHECK_ZERO)
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
        //
        // read scaling factor
        //
        
        double  scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        //
        // read and convert compressed data
        //
    
        const ulong  prec_mask  = ( 1ul << prec_bits ) - 1;
        const uint   prec_ofs   = fp64_mant_bits - prec_bits;
        const ulong  exp_mask   = ( 1ul << exp_bits ) - 1;
        const uint   sign_shift = exp_bits + prec_bits;
        const ulong  zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );

        // number of values to read before decoding
        constexpr size_t  nchunk = 32;
    
        size_t  pos    = 10;
        uint    bpos   = 0;                          // bit position in current byte
        size_t  ncsize = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
        size_t  i      = 0;

        for ( ; i < ncsize; i += nchunk )
        {
            //
            // read next values into local buffer
            //

            ulong  zval_buf[ nchunk ];
            
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                ulong  zval  = 0;
                uint   sbits = 0;  // already read bits of zval
            
                do
                {
                    HLR_DBG_ASSERT( pos < zdata.size() );
        
                    const uint    crest = 8 - bpos;                               // remaining bits in current byte
                    const uint    zrest = nbits - sbits;                          // remaining bits to read for zval
                    const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                    const byte_t  data  = (zdata[pos] >> bpos) & zmask;           // part of zval in current byte
                
                    zval  |= (ulong(data) << sbits); // lowest to highest bit in zdata
                    sbits += crest;

                    if ( crest <= zrest ) { bpos  = 0; ++pos; }
                    else                  { bpos += zrest; }
                } while ( sbits < nbits );

                zval_buf[lpos] = zval;
            }// for

            //
            // convert all values
            //
        
            for ( uint  lpos = 0; lpos < nchunk; ++lpos )
            {
                const ulong   zval  = zval_buf[lpos];

                #if defined(AFLOAT_CHECK_ZERO)
                if ( zval == zero_val )
                    dest[i+lpos] = 0;
                else
                #endif
                {
                    const ulong   mant  = zval & prec_mask;
                    const ulong   exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    
                    const ulong   rexp  = exp | fp64_exp_highbit; // re-add leading bit
                    const ulong   rmant = mant << prec_ofs;
                    const ulong   irval = (rexp << fp64_mant_bits) | rmant;
                    const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
                    
                    dest[i+lpos] = rval;
                }// else
            }// for
        }// for
        // size_t       pos        = 10;
        // uint         bpos       = 0;  // bit position in current byte

        for ( ; i < nsize; ++i )
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

                if ( crest <= zrest ) { bpos  = 0; ++pos; }
                else                  { bpos += zrest; }
            } while ( sbits < nbits );

            #if defined(AFLOAT_CHECK_ZERO)
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
compress_lr ( const blas::matrix< value_t > &                        U,
              const blas::vector< Hpro::real_type_t< value_t > > &   S );

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
        auto  vmin = std::abs( U(0,l) );
        auto  vmax = vmin;

        for ( size_t  i = 1; i < n; ++i )
        {
            const auto  u_il = std::abs( U(i,l) );
            
            vmin = std::min( vmin, u_il );
            vmax = std::max( vmax, u_il );
        }// for

        s[l] = real_t(1) / vmin;
        e[l] = uint( std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        m[l] = tol_to_rate( S(l) );

        // std::cout << m[l] << ", ";
        
        const size_t  nbits      = 1 + e[l] + m[l]; // number of bits per value
        const size_t  n_tot_bits = n * nbits;       // number of bits for all values in column
        
        if (( m[l] <= 23 ) && ( nbits <= 32 ))
            zsize += sizeof(float) + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
        else
            zsize += sizeof(double) + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
    }// for

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

        if (( prec_bits <= 23 ) && ( nbits <= 32 ))
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
            uint        bpos     = 0; // start bit position in current byte

            for ( size_t  i = 0; i < n; ++i )
            {
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
                const float  val  = U(i,l);
                uint         zval = zero_val;
            
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

            // pad last entry with "zeroes" for next column
            if ( bpos > 0 )
                ++pos;
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
            uint         bpos     = 0;  // start bit position in current byte

            for ( size_t  i = 0; i < n; ++i )
            {
                const double  val  = U(i,l);
                ulong         zval = zero_val;
            
                //
                // Use absolute value and scale v_i and add 1 such that v_i >= 2.
                // With this, highest exponent bit is 1 and we only need to store
                // lowest <exp_bits> exponent bits
                //
        
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

            // pad last entry with "zeroes" for next column
            if ( bpos > 0 )
                ++pos;
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
            uint        bpos       = 0; // bit position in current byte

            for ( size_t  i = 0; i < n; ++i )
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

                    if ( crest <= zrest ) { bpos  = 0; ++pos; }
                    else                  { bpos += zrest; }
                } while ( sbits < nbits );

                {
                    const uint   mant  = zval & prec_mask;
                    const uint   exp   = (zval >> prec_bits) & exp_mask;
                    const bool   sign  = zval >> sign_shift;
                    const uint   irval = ((exp | fp32_exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
                    const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;
                
                    U(i,l) = double( rval );
                }// else
            }// for

            // "read" padded zeros
            if ( bpos > 0 )
                ++pos;
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
            uint         bpos       = 0; // bit position in current byte

            for ( size_t  i = 0; i < n; ++i )
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

                    if ( crest <= zrest ) { bpos  = 0; ++pos; }
                    else                  { bpos += zrest; }
                } while ( sbits < nbits );

                {
                    const ulong   mant  = zval & prec_mask;
                    const ulong   exp   = (zval >> prec_bits) & exp_mask;
                    const bool    sign  = zval >> sign_shift;
                    const ulong   irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);
                    const double  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;

                    U(i,l) = rval;
                }// else
            }// for

            // "read" padded zeros
            if ( bpos > 0 )
                ++pos;
        }// else
    }// for
}


}}}// namespace hlr::compress::afloat

#endif // __HLR_UTILS_DETAIL_AFLOAT_HH
