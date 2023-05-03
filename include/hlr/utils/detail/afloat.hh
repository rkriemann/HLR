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
constexpr float  fp32_infinity    = std::numeric_limits< float >::infinity();

constexpr uint   fp64_mant_bits   = 52;
constexpr uint   fp64_exp_bits    = 11;
constexpr uint   fp64_sign_bit    = 63;
constexpr ulong  fp64_exp_highbit = 1 << (fp64_exp_bits-1);
constexpr ulong  fp64_zero_val    = 0xffffffffffffffff;
constexpr double fp64_infinity    = std::numeric_limits< double >::infinity();

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
    
        const float  fscale   = scale;
        const float  fmin     = vmin;
        const uint   prec_ofs = fp32_mant_bits - prec_bits;
        const uint   zero_val = fp32_zero_val & (( 1 << nbits) - 1 );
        
        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & fscale, 4 );

        //
        // compress data in "vectorized" form
        //
        
        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        bool              sign[ nbuf ]; // holds sign per entry
        float             fbuf[ nbuf ]; // holds rescaled value
        uint              ibuf[ nbuf ]; // holds value in compressed format
        size_t            pos  = 6;
        uint              bpos = 0; // start bit position in current byte
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
                fbuf[j] = std::max( fscale * aval + 1, 2.f ); // prevent rounding issues when converting from fp64

                HLR_DBG_ASSERT( fbuf[j] >= float(2) );
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

            // write into data buffer
            for ( size_t  j = 0; j < nbuf; ++j )
            {
                auto  zval  = ibuf[j];
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
        }// for
        
        // handle remaining values
        for ( ; i < nsize; ++i )
        {
            const float  val  = data[i];
            uint         zval = zero_val;
            
            if ( std::abs( val ) >= fmin )
            {
                const bool   zsign = ( val < 0 );
                
                const float  sval  = std::max( fscale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
                const uint   isval = (*reinterpret_cast< const uint * >( & sval ) );
                const uint   sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                const uint   smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                const uint   zexp  = sexp & exp_mask;
                const uint   zmant = smant >> prec_ofs;
                
                zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
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
        const size_t  zsize    = 8 + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
        auto          zdata    = std::vector< byte_t >( zsize );
        const uint    prec_ofs = fp64_mant_bits - prec_bits;
        const ulong   zero_val = fp64_zero_val & (( 1ul << nbits) - 1 );
        
        //
        // store header (exponents and precision bits and scaling factor)
        //
    
        zdata[0] = exp_bits;
        zdata[1] = prec_bits;
        memcpy( zdata.data() + 2, & scale, 8 );

        //
        // compress data in "vectorized" form
        //
        
        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        bool              sign[ nbuf ]; // holds sign per entry
        double            fbuf[ nbuf ]; // holds rescaled value
        ulong             ibuf[ nbuf ]; // holds value in compressed format
        size_t            pos  = 10;
        uint              bpos = 0; // start bit position in current byte
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
                const ulong  isval = (*reinterpret_cast< const ulong * >( & fbuf[j] ) );
                const ulong  sexp  = ( isval >> fp64_mant_bits ) & ((1ul << fp64_exp_bits) - 1);
                const ulong  smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
                const ulong  zexp  = sexp & exp_mask;
                const ulong  zmant = smant >> prec_ofs;
                
                ibuf[j] = (((sign[j] << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( ibuf[j] != zero_val );
            }// for

            // correct zeroes
            for ( size_t  j = 0; j < nbuf; ++j )
                if ( zero[j] )
                    ibuf[j] = zero_val;

            // write into data buffer
            for ( size_t  j = 0; j < nbuf; ++j )
            {
                auto  zval  = ibuf[j];
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
        }// for

        // handle remaining data
        for ( ; i < nsize; ++i )
        {
            const double  val  = data[i];
            ulong         zval = zero_val;
            
            if ( std::abs( val ) >= vmin )
            {
                const bool    zsign = ( val < 0 );
                const double  sval  = scale * std::abs(val) + 1;
                const ulong   isval = (*reinterpret_cast< const ulong * >( & sval ) );
                const ulong   sexp  = ( isval >> fp64_mant_bits ) & ((1ul << fp64_exp_bits) - 1);
                const ulong   smant = ( isval & ((1ul << fp64_mant_bits) - 1) );
                const ulong   zexp  = sexp & exp_mask;
                const ulong   zmant = smant >> prec_ofs;

                zval  = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                HLR_DBG_ASSERT( zval != zero_val );
            }// if
        
            uint  sbits = 0;
            
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

        if ( zval == zero_val )
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
        
        float       scale;
        const uint  prec_mask  = ( 1 << prec_bits ) - 1;
        const uint  prec_ofs   = fp32_mant_bits - prec_bits;
        const uint  exp_mask   = ( 1 << exp_bits ) - 1;
        const uint  sign_shift = exp_bits + prec_bits;
        const uint  zero_val   = fp32_zero_val & (( 1 << nbits) - 1 );

        memcpy( & scale, zdata.data() + 2, 4 );

        //
        // decompress in "vectorised" form
        //
        
        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        uint              ibuf[ nbuf ]; // holds value in compressed format
        float             fbuf[ nbuf ]; // holds uncompressed values
        size_t            pos  = 6;
        uint              bpos = 0;
        size_t            i    = 0;

        for ( ; i < nbsize; i += nbuf )
        {
            // read data
            for ( size_t  j = 0; j < nbuf; ++j )
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

                ibuf[j] = zval;
            }// for
            
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
        }// for

        // handle remaining values
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

            if ( zval == zero_val )
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
        const ulong  prec_mask  = ( 1ul << prec_bits ) - 1;
        const uint   prec_ofs   = fp64_mant_bits - prec_bits;
        const ulong  exp_mask   = ( 1ul << exp_bits ) - 1;
        const uint   sign_shift = exp_bits + prec_bits;
        const ulong  zero_val   = fp64_zero_val & (( 1ul << nbits) - 1 );

        //
        // read scaling factor
        //
        
        double  scale;

        memcpy( & scale, zdata.data() + 2, 8 );

        //
        // decompress in "vectorised" form
        //
        
        constexpr size_t  nbuf   = 64;
        const size_t      nbsize = nsize - nsize % nbuf;  // largest multiple of <nchunk> below <nsize>
        bool              zero[ nbuf ]; // mark zero entries
        ulong             ibuf[ nbuf ]; // holds value in compressed format
        double            fbuf[ nbuf ]; // holds uncompressed values
        size_t            pos  = 10;
        uint              bpos = 0;                          // bit position in current byte
        size_t            i    = 0;

        for ( ; i < nbsize; i += nbuf )
        {
            // read data
            for ( size_t  j = 0; j < nbuf; ++j )
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

                ibuf[j] = zval;
            }// for

            // convert from compressed format
            for ( size_t  j = 0; j < nbuf; ++j )
            {
                const auto   zval  = ibuf[j];
                const ulong  mant  = zval & prec_mask;
                const ulong  exp   = (zval >> prec_bits) & exp_mask;
                const bool   sign  = zval >> sign_shift;
                const ulong  irval = ((exp | fp64_exp_highbit) << fp64_mant_bits) | (mant << prec_ofs);

                zero[j] = ( zval == zero_val );
                fbuf[j] = (sign ? -1 : 1 ) * ( * reinterpret_cast< const double * >( & irval ) - 1 ) / scale;
            }// for

            // correct zeroes
            for ( size_t  j = 0; j < nbuf; ++j )
                if ( zero[j] )
                    fbuf[j] = double(0);

            // copy values
            for ( size_t  j = 0; j < nbuf; ++j )
                dest[i+j] = fbuf[j];
        }// for

        // handle remaining values
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

    const size_t  n          = U.nrows();
    const size_t  k          = U.ncols();
    auto          vmant_bits = std::vector< uint >( k );
    auto          vexp_bits  = std::vector< uint >( k );
    auto          vscale     = std::vector< real_t >( k );
    auto          vmin_val   = std::vector< real_t >( k );
    size_t        zsize      = 0;

    for ( uint  l = 0; l < k; ++l )
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

        vscale[l]     = real_t(1) / vmin;
        vexp_bits[l]  = uint( std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) ) );
        vmant_bits[l] = tol_to_rate( S(l) );
        vmin_val[l]   = vmin;

        const size_t  nbits      = 1 + vexp_bits[l] + vmant_bits[l]; // number of bits per value
        const size_t  n_tot_bits = n * nbits;                        // number of bits for all values in column
        
        if (( vmant_bits[l] <= 23 ) && ( nbits <= 32 ))
            zsize += sizeof(float) + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
        else
            zsize += sizeof(double) + 1 + 1 + n_tot_bits / 8 + ( n_tot_bits % 8 != 0 ? 1 : 0 );
    }// for

    // for ( uint  l = 0; l < k; ++l )
    //   std::cout << e[l] << '/' << m[l] << ", ";
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint  l = 0; l < k; ++l )
    {
        const double *  data      = U.data() + l * n;
        const uint      exp_bits  = vexp_bits[l];
        const uint      exp_mask  = ( 1 << exp_bits ) - 1;
        const uint      prec_bits = vmant_bits[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        if (( prec_bits <= 23 ) && ( nbits <= 32 ))
        {
            const uint   prec_ofs = fp32_mant_bits - prec_bits;
            const uint   zero_val = fp32_zero_val & (( 1 << nbits) - 1 );
            const float  fmin     = vmin_val[l];
            
            //
            // store header (exponents and precision bits and scaling factor)
            //
    
            const float  scale = vscale[l];
        
            zdata[pos]   = exp_bits;
            zdata[pos+1] = prec_bits;
            memcpy( zdata.data() + pos + 2, & scale, 4 );

            pos += 6;
            
            //
            // compress data in "vectorized" form
            //
        
            constexpr size_t  nbuf   = 64;
            const size_t      nbsize = n - n % nbuf;  // largest multiple of <nchunk> below <nsize>
            bool              zero[ nbuf ]; // mark zero entries
            bool              sign[ nbuf ]; // holds sign per entry
            float             fbuf[ nbuf ]; // holds rescaled value
            uint              ibuf[ nbuf ]; // holds value in compressed format
            size_t            pos  = 6;
            uint              bpos = 0; // start bit position in current byte
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

                // write into data buffer
                for ( size_t  j = 0; j < nbuf; ++j )
                {
                    auto  zval  = ibuf[j];
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
            }// for
            
            // handle remaining values
            for ( ; i < n; ++i )
            {
                const float  val  = U(i,l);
                uint         zval = zero_val;
            
                if ( std::abs( val ) >= fmin )
                {
                    const bool   zsign = ( val < 0 );
                
                    const float  sval  = std::max( scale * std::abs(val) + 1, 2.f ); // prevent rounding issues when converting from fp64
                    const uint   isval = (*reinterpret_cast< const uint * >( & sval ) );
                    const uint   sexp  = ( isval >> fp32_mant_bits ) & ((1u << fp32_exp_bits) - 1);
                    const uint   smant = ( isval & ((1u << fp32_mant_bits) - 1) );
                    const uint   zexp  = sexp & exp_mask;
                    const uint   zmant = smant >> prec_ofs;
                
                    zval = (((zsign << exp_bits) | zexp) << prec_bits) | zmant;

                    HLR_DBG_ASSERT( zval != zero_val );
                }// if
        
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
            const real_t  scale = vscale[l];
            
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
