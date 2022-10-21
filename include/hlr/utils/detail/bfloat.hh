#ifndef __HLR_UTILS_DETAIL_BFLOAT_HH
#define __HLR_UTILS_DETAIL_BFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/bfloat
// Description : bfloat related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using general bfloat format
// - use FP32 exponent size and precision dependend mantissa size (1+8+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bfloat {

using byte_t = unsigned char;

constexpr uint  fp32_exp_mask  = 0x7f800000;

constexpr ulong fp64_sign_mask = (1ul << 63);
constexpr uint  fp64_mant_bits = 52;
constexpr ulong fp64_mant_mask = 0x000fffffffffffff;
constexpr ulong fp64_exp_mask  = 0x7ff0000000000000;

constexpr uint  bf_header_ofs  = 1;

inline
byte_t
eps_to_rate ( const double eps )
{
    if      ( eps >= 1e-2  ) return 7;
    else if ( eps >= 1e-4  ) return 15;
    else if ( eps >= 1e-7  ) return 23;
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
    const size_t  nsize      = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint    nbits_min  = 1 + 8 + config.bitrate;                                 // minimal number of bits per value for precision
    const uint    nbits      = nbits_min + ( nbits_min % 8 == 0 ? 0 : 8 - ( nbits_min % 8 ) ); // round up to next multiple of 8
    const uint    nbyte      = nbits / 8;
    zarray        zdata;

    if ( nbyte == 2 )
    {
        //
        // BF16
        //
        
        zdata.resize( bf_header_ofs + nsize * 2 );
        zdata[0] = nbyte;
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            // reduce mantissa size by 8 bits
            const ushort  ival = (*reinterpret_cast< const uint * >( & data[i] ) ) >> 16;
            const size_t  zpos = 2*i + bf_header_ofs;
            
            zdata[zpos+1] = (ival & 0xff00) >> 8;
            zdata[zpos]   = (ival & 0x00ff);
        }// for

        return zdata;
    }// if
    else if ( nbyte == 3 )
    {
        //
        // BF24
        //
        
        zdata.resize( bf_header_ofs + nsize * 3 );
        zdata[0] = nbyte;
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            // reduce mantissa size by 8 bits
            const uint    ival = (*reinterpret_cast< const uint * >( & data[i] ) ) >> 8;
            const size_t  zpos = 3*i + bf_header_ofs;
            
            zdata[zpos+2] = (ival & 0xff0000) >> 16;
            zdata[zpos+1] = (ival & 0x00ff00) >> 8;
            zdata[zpos]   = (ival & 0x0000ff);
        }// for
    }// else
    else if ( nbyte == 4 )
    {
        //
        // BF32 == FP32
        //
        
        zarray  zdata( bf_header_ofs + nsize * 4 );

        zdata[0] = nbyte;
        std::copy( reinterpret_cast< const byte_t * >( data ),
                   reinterpret_cast< const byte_t * >( data + nsize ),
                   zdata.data() + bf_header_ofs );
    }// if
    else
        HLR_ERROR( "unsupported storage size" );

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
    const size_t  nsize      = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint    nbits_min  = 1 + 8 + config.bitrate;                                 // minimal number of bits per value for precision
    const uint    nbits      = nbits_min + ( nbits_min % 8 == 0 ? 0 : 8 - ( nbits_min % 8 ) ); // round up to next multiple of 8
    const uint    nbyte      = nbits / 8;
    zarray        zdata;
    
    if ( nbyte == 2 )
    {
        //
        // BF16
        //
        
        zdata.resize( bf_header_ofs + nsize * 2 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            // convert to float, reduce mantissa size by 8 bits
            const float   fval = data[i];
            const ushort  ival = (*reinterpret_cast< const uint * >( & fval ) ) >> 16;
            const size_t  zpos = 2*i + bf_header_ofs;

            zdata[zpos+1] = (ival & 0xff00) >> 8;
            zdata[zpos]   = (ival & 0x00ff);
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        //
        // BF24
        //
        
        zdata.resize( bf_header_ofs + nsize * 3 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            // convert to float, reduce mantissa size by 8 bits
            const float   fval = data[i];
            const uint    ival = (*reinterpret_cast< const uint * >( & fval ) ) >> 8;
            const size_t  zpos = 3*i + bf_header_ofs;

            zdata[zpos+2] = (ival & 0xff0000) >> 16;
            zdata[zpos+1] = (ival & 0x00ff00) >> 8;
            zdata[zpos]   = (ival & 0x0000ff);
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        //
        // BF32 == FP32
        //
        
        zdata.resize( bf_header_ofs + nsize * 4 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const float   fval = data[i];
            const uint    ival = (*reinterpret_cast< const uint * >( & fval ) );
            const size_t  zpos = 4*i + bf_header_ofs;

            zdata[zpos+3] = (ival & 0xff000000) >> 24;
            zdata[zpos+2] = (ival & 0x00ff0000) >> 16;
            zdata[zpos+1] = (ival & 0x0000ff00) >> 8;
            zdata[zpos]   = (ival & 0x000000ff);
        }// for
    }// if
    else if ( nbyte == 5 )
    {
        //
        // BF40 = 1 + 8 + 31 bits
        //

        constexpr uint  bf_mant_bits  = 31;
        constexpr uint  bf_sign_bit   = bf_mant_bits + 8;
        constexpr uint  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint  bf_sign_shift = 63 - bf_sign_bit;
        
        zdata.resize( bf_header_ofs + nsize * 5 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 5*i + bf_header_ofs;
            const double  val  = data[i];
            const ulong   ival = (*reinterpret_cast< const ulong * >( & val ) );
            const uint    exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const ulong   mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const ulong   sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const ulong   zval = sign | (ulong(exp - 0x381u) << bf_mant_bits) | mant;

            zdata[zpos+4] = (zval & 0x00ff00000000) >> 32;
            zdata[zpos+3] = (zval & 0x0000ff000000) >> 24;
            zdata[zpos+2] = (zval & 0x000000ff0000) >> 16;
            zdata[zpos+1] = (zval & 0x00000000ff00) >> 8;
            zdata[zpos]   = (zval & 0x0000000000ff);

            // // DEBUG
            // {
            //     constexpr ulong  bf_sign_mask  = (1ul    << bf_sign_bit);
            //     constexpr ulong  bf_exp_mask   = (0xfful << bf_mant_bits);
            //     constexpr ulong  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
                
            //     const ulong   dzval = ( (ulong(zdata[zpos+4]) << 32) |
            //                             (ulong(zdata[zpos+3]) << 24) |
            //                             (ulong(zdata[zpos+2]) << 16) |
            //                             (ulong(zdata[zpos+1]) <<  8) |
            //                             (ulong(zdata[zpos])        ) );
            //     const ulong   dsign = (dzval & sign_mask) << bf_sign_shift;
            //     const ulong   dexp  = (dzval & exp_mask ) >> bf_mant_bits;
            //     const ulong   dmant = (dzval & mant_mask) << bf_mant_shift;
            //     const ulong   dival = dsign | ((dexp + 0x381ul) << fp64_mant_bits) | dmant;
            //     const double  dfval = * reinterpret_cast< const double * >( & dival );

            //     std::cout << i << " : "
            //               << val << " / " << dfval << " / " << std::abs( val - dfval ) / std::abs( val ) << std::endl;
            // }
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        //
        // BF48 = 1 + 8 + 39 bits
        //

        constexpr uint  bf_mant_bits  = 39;
        constexpr uint  bf_sign_bit   = bf_mant_bits + 8;
        constexpr uint  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint  bf_sign_shift = 63 - bf_sign_bit;
        
        zdata.resize( bf_header_ofs + nsize * 6 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 6*i + bf_header_ofs;
            const double  val  = data[i];
            const ulong   ival = (*reinterpret_cast< const ulong * >( & val ) );
            const uint    exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const ulong   mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const ulong   sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const ulong   zval = sign | (ulong(exp - 0x381u) << bf_mant_bits) | mant;

            zdata[zpos+5] = (zval & 0xff0000000000) >> 40;
            zdata[zpos+4] = (zval & 0x00ff00000000) >> 32;
            zdata[zpos+3] = (zval & 0x0000ff000000) >> 24;
            zdata[zpos+2] = (zval & 0x000000ff0000) >> 16;
            zdata[zpos+1] = (zval & 0x00000000ff00) >> 8;
            zdata[zpos]   = (zval & 0x0000000000ff);

            // // DEBUG
            // {
            //     constexpr ulong  bf_sign_mask  = (1ul    << bf_sign_bit);
            //     constexpr ulong  bf_exp_mask   = (0xfful << bf_mant_bits);
            //     constexpr ulong  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
                
            //     const ulong   dzval = ( (ulong(zdata[zpos+5]) << 40) |
            //                             (ulong(zdata[zpos+4]) << 32) |
            //                             (ulong(zdata[zpos+3]) << 24) |
            //                             (ulong(zdata[zpos+2]) << 16) |
            //                             (ulong(zdata[zpos+1]) <<  8) |
            //                             (ulong(zdata[zpos])        ) );
            //     const ulong   dsign = (dzval & bf_sign_mask) << bf_sign_shift;
            //     const ulong   dexp  = (dzval & bf_exp_mask ) >> bf_mant_bits;
            //     const ulong   dmant = (dzval & bf_mant_mask) << bf_mant_shift;
            //     const ulong   dival = dsign | ((dexp + 0x381ul) << fp64_mant_bits) | dmant;
            //     const double  dfval = * reinterpret_cast< const double * >( & dival );

            //     std::cout << i << " : "
            //               << val << " / " << dfval << " / " << std::abs( val - dfval ) / std::abs( val ) << std::endl;
            // }
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        //
        // BF56
        //

        HLR_ERROR( "TODO" );
    }// if
    else if ( nbyte == 8 )
    {
        //
        // BF64 -> higher precision than FP64, so leave data untouched
        //
        
        zarray  zdata( bf_header_ofs + nsize * 8 );

        zdata[0] = nbyte;
        std::copy( reinterpret_cast< const byte_t * >( data ),
                   reinterpret_cast< const byte_t * >( data + nsize ),
                   zdata.data() + bf_header_ofs );
    }// if

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
    {
        const size_t  zpos = 3*i;
        const uint    ival = (zdata[zpos+2] << 24) | (zdata[zpos+1] << 16) | (zdata[zpos] << 8);
        
        dest[i] = * reinterpret_cast< const float * >( & ival );
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
    const uint    nbyte = zdata[0];

    if ( nbyte == 2 )
    {
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 2*i + bf_header_ofs;
            const uint    ival = (zdata[zpos+1] << 24) | (zdata[zpos] << 16);
            const float   fval = * reinterpret_cast< const float * >( & ival );
        
            dest[i] = double( fval );
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 3*i + bf_header_ofs;
            const uint    ival = (zdata[zpos+2] << 24) | (zdata[zpos+1] << 16) | (zdata[zpos] << 8);
            const float   fval = * reinterpret_cast< const float * >( & ival );
        
            dest[i] = double( fval );
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 4*i + bf_header_ofs;
            const uint    ival = (zdata[zpos+3] << 24) | (zdata[zpos+2] << 16) | (zdata[zpos+1] << 8) | zdata[zpos];
            const float   fval = * reinterpret_cast< const float * >( & ival );
        
            dest[i] = double( fval );
        }// for
    }// if
    else if ( nbyte == 5 )
    {
        constexpr uint   bf_mant_bits  = 31;
        constexpr uint   bf_sign_bit   = bf_mant_bits + 8;
        constexpr uint   bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint   bf_sign_shift = 63 - bf_sign_bit;
        constexpr ulong  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr ulong  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr ulong  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 5*i + bf_header_ofs;
            const ulong   zval = ( (ulong(zdata[zpos+4]) << 32) |
                                   (ulong(zdata[zpos+3]) << 24) |
                                   (ulong(zdata[zpos+2]) << 16) |
                                   (ulong(zdata[zpos+1]) <<  8) |
                                   (ulong(zdata[zpos])        ) );
            const ulong   sign = (zval & bf_sign_mask) << bf_sign_shift;
            const ulong   exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const ulong   mant = (zval & bf_mant_mask) << bf_mant_shift;
            const ulong   ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        constexpr uint   bf_mant_bits  = 39;
        constexpr uint   bf_sign_bit   = bf_mant_bits + 8;
        constexpr uint   bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint   bf_sign_shift = 63 - bf_sign_bit;
        constexpr ulong  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr ulong  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr ulong  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 6*i + bf_header_ofs;
            const ulong   zval = ( (ulong(zdata[zpos+5]) << 40) |
                                   (ulong(zdata[zpos+4]) << 32) |
                                   (ulong(zdata[zpos+3]) << 24) |
                                   (ulong(zdata[zpos+2]) << 16) |
                                   (ulong(zdata[zpos+1]) <<  8) |
                                   (ulong(zdata[zpos])        ) );
            const ulong   sign = (zval & bf_sign_mask) << bf_sign_shift;
            const ulong   exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const ulong   mant = (zval & bf_mant_mask) << bf_mant_shift;
            const ulong   ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        HLR_ERROR( "TODO" );
    }// if
    else if ( nbyte == 8 )
    {
        std::copy( zdata.data() + bf_header_ofs, zdata.data() + zdata.size(), reinterpret_cast< byte_t * >( dest ) );
    }// if
}

}}}// namespace hlr::compress::bfloat

#endif // __HLR_UTILS_DETAIL_BFLOAT_HH
