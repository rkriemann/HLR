#ifndef __HLR_UTILS_DETAIL_AFLOAT_HH
#define __HLR_UTILS_DETAIL_AFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/afloat
// Description : functions for adaptive floating points
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

// DEBUG
// #include <bitset>

#include <chrono>

////////////////////////////////////////////////////////////
//
// compression using adaptive float representation
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace afloat {

// // timing
// using  my_clock = std::chrono::high_resolution_clock;

// extern double  t_load;
// extern double  t_decode;

using byte_t = unsigned char;

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
    else if ( eps >= 1e-8  ) return 23;
    else if ( eps >= 1e-9  ) return 36;
    else if ( eps >= 1e-10 ) return 40;
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
    zarray        zdata( nsize );

    HLR_ERROR( "TODO" );
    
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
    // (use "float" type to ensure "vmin" really is minimal value
    //  so we don't have values in [1,2) later)
    float  vmin = std::abs( data[0] );
    float  vmax = std::abs( data[0] );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        vmin = std::min< float >( vmin, std::abs( data[i] ) );
        vmax = std::max< float >( vmax, std::abs( data[i] ) );
    }// for

    // std::cout << vmin << " / " << vmax << " / " << std::ceil( std::log2( std::log2( vmax / vmin ) ) ) << std::endl;

    const byte_t  fp32_mant_bits = 23;
    const byte_t  fp32_exp_bits  = 8;
    const byte_t  fp32_sign_pos  = 31;

    // scale all values v_i such that we have |v_i| >= 1
    const float   scale     = 1.0 / vmin;
    // number of bits needed to represent exponent values
    const byte_t  exp_bits  = std::max< float >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );
    const uint    exp_mask  = ( 1 << exp_bits ) - 1;
    const byte_t  prec_bits = config.bitrate;
    const uint    prec_mask = ( 1 << prec_bits ) - 1;
    const byte_t  prec_ofs  = fp32_mant_bits - prec_bits;

    // std::cout << uint(exp_bits) << std::endl;
    // std::cout << std::bitset< 8 >( exp_mask ) << " / " << prec_ofs << " / " << std::bitset< 23 >( prec_mask ) << std::endl;

    const size_t  nbits      = 1 + exp_bits + prec_bits; // number of bits per value
    const size_t  n_tot_bits = nsize * nbits;            // number of bits for all values
    const size_t  zsize      = 4 + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
    auto          zdata      = std::vector< byte_t >( zsize );
    size_t        pos        = 6; // data starts after scaling factor, exponent bits and precision bits
    byte_t        bpos       = 0; // start bit position in current byte

    // std::cout << uint(nbits) << std::endl;
    
    // 32 bit integer as max for now
    HLR_ASSERT( nbits <= 32 );
    HLR_ASSERT( prec_bits <= 23 );
    
    // first, store scaling factor
    memcpy( zdata.data(), & scale, 4 );

    // then store number of exponents bits
    memcpy( zdata.data() + 4, & exp_bits, 1 );
            
    // and precision bits
    memcpy( zdata.data() + 5, & prec_bits, 1 );

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
        uint          zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

        // {
        //     const byte_t  fp32_sign_pos  = 31;
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
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = float( zdata[i] );
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
    constexpr byte_t  fp32_mant_bits = 23;
    constexpr byte_t  fp32_sign_pos  = 31;

    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    //
    
    float   scale;
    byte_t  exp_bits;
    byte_t  prec_bits;

    // extract scaling factor
    memcpy( & scale, zdata.data(), 4 );

    // and exponent bits
    memcpy( & exp_bits, zdata.data() + 4, 1 );

    // and precision bits
    memcpy( & prec_bits, zdata.data() + 5, 1 );

    //
    // read compressed data
    //
    
    const byte_t  nbits       = 1 + exp_bits + prec_bits;
    const uint    prec_mask   = ( 1 << prec_bits ) - 1;
    const uint    prec_ofs    = fp32_mant_bits - prec_bits;
    const uint    exp_mask    = ( 1 << exp_bits ) - 1;
    const uint    exp_highbit = 0b10000000;
    const byte_t  sign_shift  = exp_bits + prec_bits;

    // number of values to read before decoding
    constexpr size_t  nchunk = 32;
    
    size_t  pos        = 6;
    byte_t  bpos       = 0;                          // bit position in current byte
    size_t  ncsize     = (nsize / nchunk) * nchunk;  // largest multiple of <nchunk> below <nsize>
    size_t  i          = 0;
    uint    zval_buf[ nchunk ];

    for ( ; i < ncsize; i += nchunk )
    {
        //
        // read next 8 values into local buffer
        //

        // auto  tic = my_clock::now();
        
        for ( uint  lpos = 0; lpos < nchunk; ++lpos )
        {
            uint    zval  = 0;
            byte_t  sbits = 0;  // already read bits of zval
            
            do
            {
                HLR_DBG_ASSERT( pos < zdata.size() );
        
                const byte_t  crest = 8 - bpos;                               // remaining bits in current byte
                const byte_t  zrest = nbits - sbits;                          // remaining bits to read for zval
                const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff ); // mask for zval data
                const byte_t  data  = (zdata[pos] >> bpos) & zmask;           // part of zval in current byte
                
                zval  |= (data << sbits); // lowest to highest bit in zdata
                sbits += crest;

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

            zval_buf[lpos] = zval;
        }// for

        // auto  toc   = my_clock::now();
        // auto  since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

        // t_load += since;
        
        //
        // convert all 8 values
        //

        // tic = my_clock::now();
        
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

        // toc   = my_clock::now();
        // since = std::chrono::duration_cast< std::chrono::microseconds >( toc - tic ).count() / 1e6;

        // t_decode += since;
    }// for

    for ( ; i < nsize; ++i )
    {
        uint    zval  = 0;
        byte_t  sbits = 0;
            
        do
        {
            HLR_DBG_ASSERT( pos < zdata.size() );
        
            const byte_t  crest = 8 - bpos;
            const byte_t  zrest = nbits - sbits;
            const byte_t  zmask = (zrest < 8 ? (1 << zrest) - 1 : 0xff );
            const byte_t  data  = (zdata[pos] >> bpos) & zmask;
                
            zval  |= (data << sbits);
            sbits += crest;

            if ( crest <= zrest )
            {
                ++pos;
                bpos = 0;
            }// if
            else
                bpos += zrest;
        } while ( sbits < nbits );

        const uint   mant  = zval & prec_mask;
        const uint   exp   = (zval >> prec_bits) & exp_mask;
        const bool   sign  = zval >> sign_shift;
        const uint   irval = ((exp | exp_highbit) << fp32_mant_bits) | (mant << prec_ofs);
        const float  rval  = (sign ? -1 : 1 ) * ( * reinterpret_cast< const float * >( & irval ) - 1 ) / scale;

        dest[i] = double( rval );
    }// for
}

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
