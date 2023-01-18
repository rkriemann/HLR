#ifndef __HLR_UTILS_DETAIL_DFLOAT_HH
#define __HLR_UTILS_DETAIL_DFLOAT_HH
//
// Project     : HLR
// Module      : utils/detail/dfloat
// Description : dfloat related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using general dfloat format
// - use FP64 exponent size and precision dependend mantissa size (1+11+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace dfloat {

using byte_t = unsigned char;

constexpr uint  bf_header_ofs  = 1;

inline
byte_t
eps_to_rate ( const double eps )
{
    if      ( eps >= 1e-2  ) return 7;
    else if ( eps >= 1e-3  ) return 14;
    else if ( eps >= 1e-4  ) return 23;
    else if ( eps >= 1e-6  ) return 23;
    else if ( eps >= 1e-7  ) return 31;
    else if ( eps >= 1e-8  ) return 31;
    else if ( eps >= 1e-9  ) return 39;
    else if ( eps >= 1e-10 ) return 39;
    else if ( eps >= 1e-12 ) return 44;
    else if ( eps >= 1e-14 ) return 54;
    else                     return 64;
}

// inline
// byte_t
// eps_to_rate ( const double eps )
// {
//     if      ( eps >= 1e-2  ) return 7;
//     else if ( eps >= 1e-4  ) return 15;
//     else if ( eps >= 1e-7  ) return 23;
//     else if ( eps >= 1e-8  ) return 24;
//     else if ( eps >= 1e-9  ) return 28;
//     else if ( eps >= 1e-10 ) return 32;
//     else if ( eps >= 1e-12 ) return 44;
//     else if ( eps >= 1e-14 ) return 54;
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
    const uint    nbits_min  = 1 + 11 + config.bitrate;                                 // minimal number of bits per value for precision
    const uint    nbits      = nbits_min + ( nbits_min % 8 == 0 ? 0 : 8 - ( nbits_min % 8 ) ); // round up to next multiple of 8
    const uint    nbyte      = nbits / 8;
    zarray        zdata;
    
    if ( nbyte == 2 )
    {
        //
        // 1-11-4
        //
        
        zdata.resize( bf_header_ofs + nsize * 2 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  fval = data[i];
            const ushort  ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-2) * 8);
            const size_t  zpos = 2*i + bf_header_ofs;

            zdata[zpos+1] = (ival & 0xff00) >> 8;
            zdata[zpos]   = (ival & 0x00ff);
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        //
        // 1-11-12
        //
        
        zdata.resize( bf_header_ofs + nsize * 3 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            // convert to float, reduce mantissa size by 8 bits
            const double  fval = data[i];
            const uint    ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-3)*8);
            const size_t  zpos = 3*i + bf_header_ofs;

            zdata[zpos+2] = (ival & 0xff0000) >> 16;
            zdata[zpos+1] = (ival & 0x00ff00) >> 8;
            zdata[zpos]   = (ival & 0x0000ff);
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        //
        // 1-11-20
        //
        
        zdata.resize( bf_header_ofs + nsize * 4 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  fval = data[i];
            const uint    ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-4)*8);
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
        // 1-11-28
        //

        zdata.resize( bf_header_ofs + nsize * 5 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  fval = data[i];
            const ulong   ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-5)*8);
            const size_t  zpos = 5*i + bf_header_ofs;

            zdata[zpos+4] = (ival & 0xff000000) >> 32;
            zdata[zpos+3] = (ival & 0xff000000) >> 24;
            zdata[zpos+2] = (ival & 0x00ff0000) >> 16;
            zdata[zpos+1] = (ival & 0x0000ff00) >> 8;
            zdata[zpos]   = (ival & 0x000000ff);
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        //
        // 1-11-36
        //

        zdata.resize( bf_header_ofs + nsize * 6 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  fval = data[i];
            const ulong   ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-6)*8);
            const size_t  zpos = 6*i + bf_header_ofs;

            zdata[zpos+5] = (ival & 0xff000000) >> 40;
            zdata[zpos+4] = (ival & 0xff000000) >> 32;
            zdata[zpos+3] = (ival & 0xff000000) >> 24;
            zdata[zpos+2] = (ival & 0x00ff0000) >> 16;
            zdata[zpos+1] = (ival & 0x0000ff00) >> 8;
            zdata[zpos]   = (ival & 0x000000ff);
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        //
        // 1-11-44
        //

        zdata.resize( bf_header_ofs + nsize * 7 );
        zdata[0] = nbyte;

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double  fval = data[i];
            const ulong   ival = (*reinterpret_cast< const ulong * >( & fval ) ) >> ((8-7)*8);
            const size_t  zpos = 7*i + bf_header_ofs;

            zdata[zpos+6] = (ival & 0xff000000) >> 48;
            zdata[zpos+5] = (ival & 0xff000000) >> 40;
            zdata[zpos+4] = (ival & 0xff000000) >> 32;
            zdata[zpos+3] = (ival & 0xff000000) >> 24;
            zdata[zpos+2] = (ival & 0x00ff0000) >> 16;
            zdata[zpos+1] = (ival & 0x0000ff00) >> 8;
            zdata[zpos]   = (ival & 0x000000ff);
        }// for
    }// if
    else if ( nbyte == 8 )
    {
        //
        // 1-11-52 = FP64
        //
        
        zarray  zdata( bf_header_ofs + nsize * 8 );

        zdata[0] = nbyte;
        std::copy( reinterpret_cast< const byte_t * >( data ),
                   reinterpret_cast< const byte_t * >( data + nsize ),
                   zdata.data() + bf_header_ofs );
    }// if

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
        //
        // 1-11-4
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 2*i + bf_header_ofs;
            const ulong   ival = ulong((zdata[zpos+1] << 8ul) |
                                       (zdata[zpos]         )) << ((8-2)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        //
        // 1-11-12
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 3*i + bf_header_ofs;
            const ulong   ival = ulong((zdata[zpos+2] << 16ul) |
                                       (zdata[zpos+1] <<  8ul) |
                                       (zdata[zpos]          )) << ((8-3)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        //
        // 1-11-20
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 4*i + bf_header_ofs;
            const ulong   ival = ulong((zdata[zpos+3] << 24ul) |
                                       (zdata[zpos+2] << 16ul) |
                                       (zdata[zpos+1] <<  8ul) |
                                       (zdata[zpos]          )) << ((8-4)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 5 )
    {
        //
        // 1-11-28
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 5*i + bf_header_ofs;
            const ulong   ival = ((ulong(zdata[zpos+4]) << 32ul) |
                                  (ulong(zdata[zpos+3]) << 24ul) |
                                  (ulong(zdata[zpos+2]) << 16ul) |
                                  (ulong(zdata[zpos+1]) <<  8ul) |
                                  (ulong(zdata[zpos])          )) << ((8-5)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        //
        // 1-11-36
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 6*i + bf_header_ofs;
            const ulong   ival = ((ulong(zdata[zpos+5]) << 40ul) |
                                  (ulong(zdata[zpos+4]) << 32ul) |
                                  (ulong(zdata[zpos+3]) << 24ul) |
                                  (ulong(zdata[zpos+2]) << 16ul) |
                                  (ulong(zdata[zpos+1]) <<  8ul) |
                                  (ulong(zdata[zpos])          )) << ((8-6)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        //
        // 1-11-48
        //
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const size_t  zpos = 7*i + bf_header_ofs;
            const ulong   ival = ((ulong(zdata[zpos+6]) << 48ul) |
                                  (ulong(zdata[zpos+5]) << 40ul) |
                                  (ulong(zdata[zpos+4]) << 32ul) |
                                  (ulong(zdata[zpos+3]) << 24ul) |
                                  (ulong(zdata[zpos+2]) << 16ul) |
                                  (ulong(zdata[zpos+1]) <<  8ul) |
                                  (ulong(zdata[zpos])          )) << ((8-7)*8ul);
            const double  fval = * reinterpret_cast< const double * >( & ival );
        
            dest[i] = fval;
        }// for
    }// if
    else if ( nbyte == 8 )
    {
        std::copy( zdata.data() + bf_header_ofs, zdata.data() + zdata.size(), reinterpret_cast< byte_t * >( dest ) );
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

}}}// namespace hlr::compress::dfloat

#endif // __HLR_UTILS_DETAIL_DFLOAT_HH
