#ifndef __HLR_UTILS_DETAIL_IEEE754_HH
#define __HLR_UTILS_DETAIL_IEEE754_HH
//
// Project     : HLR
// Module      : utils/detail/ieee754
// Description : IEEE754 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HAS_HALF)

#include <half.hpp>

#endif

////////////////////////////////////////////////////////////
//
// compression using IEEE754 types
// - chooses suitable type based on precision
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace ieee754 {

using byte_t   = unsigned char;
using ushort_t = unsigned short;
using half     = half_float::half;

//
// different types used
//
enum ieee754_t
{
    IEEE754_FP16,
    IEEE754_BF16,
    IEEE754_TF32,
    IEEE754_FP32,
    IEEE754_FP64
};

struct config
{
    ieee754_t  type;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }
inline config  get_config ( const double    eps )
{
    if      ( eps >= 3.9e-3  ) return config{ IEEE754_BF16 };
    else if ( eps >= 4.9e-4  ) return config{ IEEE754_FP16 };
    else if ( eps >= 6.0e-8  ) return config{ IEEE754_FP32 };
    else                       return config{ IEEE754_FP64 };
}

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
    HLR_ERROR( "TODO" );
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

    switch ( config.type )
    {
        case IEEE754_BF16 :
        {
            zarray      zdata( nsize * 2 + 1 );
            ushort_t *  zbuf = (ushort_t *) ( zdata.data() + 1 );

            zdata[0] = config.type;
            
            for ( size_t  i = 0; i < nsize; ++i )
            {
                const float  fval = float( data[i] );
                const uint   ival = (* reinterpret_cast< const uint * >( & fval ) ) >> 16;

                zbuf[i] = ival;
            }// for

            return zdata;
        }
        break;

        case IEEE754_FP16 :
        {
            zarray  zdata( nsize * 2 + 1 );
            half *  zbuf = (half *) ( zdata.data() + 1 );

            zdata[0] = config.type;
            
            for ( size_t  i = 0; i < nsize; ++i )
                zbuf[i] = half(data[i]);

            return zdata;
        }
        break;

        case IEEE754_TF32 :
        {
            using  hlr::compress::tf32::tensorfloat32;
            
            zarray           zdata( nsize * 3 + 1 );
            tensorfloat32 *  zbuf = (tensorfloat32 *) ( zdata.data() + 1 );

            zdata[0] = config.type;
            
            for ( size_t  i = 0; i < nsize; ++i )
                zbuf[i] = tensorfloat32(data[i]);

            return zdata;
        }
        break;

        case IEEE754_FP32 :
        {
            zarray   zdata( nsize * 4 + 1 );
            float *  zbuf = (float *) ( zdata.data() + 1 );

            zdata[0] = config.type;
            
            for ( size_t  i = 0; i < nsize; ++i )
                zbuf[i] = float(data[i]);

            return zdata;
        }
        break;

        case IEEE754_FP64 :
        {
            zarray    zdata( nsize * 8 + 1 );
            double *  zbuf = (double *) ( zdata.data() + 1 );

            zdata[0] = config.type;
            
            for ( size_t  i = 0; i < nsize; ++i )
                zbuf[i] = data[i];

            return zdata;
        }
        break;

        default:
            HLR_ERROR( "unknown IEEE754 type" );
            return zarray();
    }// switch

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
    HLR_ERROR( "TODO" );
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
    const size_t     nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const ieee754_t  type  = ieee754_t(zdata[0]);

    switch ( type )
    {
        case IEEE754_BF16 :
        {
            ushort_t *  zbuf = (ushort_t *) ( zdata.data() + 1 );
            
            for ( size_t  i = 0; i < nsize; ++i )
            {
                const uint   ival = zbuf[i] << 16;
                const float  fval = * reinterpret_cast< const float * >( & ival );
        
                dest[i] = double( fval );
            }// for
        }
        break;
        
        case IEEE754_FP16 :
        {
            half *  zbuf = (half *) ( zdata.data() + 1 );
            
            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = double( zbuf[i] );
        }
        break;
        
        case IEEE754_TF32 :
        {
            using  hlr::compress::tf32::tensorfloat32;
            
            tensorfloat32 *  zbuf = (tensorfloat32 *) ( zdata.data() + 1 );

            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = double( zbuf[i] );
        }
        break;
        
        case IEEE754_FP32 :
        {
            float *  zbuf = (float *) ( zdata.data() + 1 );
            
            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = double( zbuf[i] );
        }
        break;
        
        case IEEE754_FP64 :
        {
            double *  zbuf = (double *) ( zdata.data() + 1 );
            
            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = zbuf[i];
        }
        break;

        default:
            HLR_ERROR( "unknown IEEE754 type" );
    }// switch
}

}}}// namespace hlr::compress::ieee754

#endif // __HLR_UTILS_DETAIL_IEEE754_HH
