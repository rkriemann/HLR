#ifndef __HLR_UTILS_DETAIL_BF16_HH
#define __HLR_UTILS_DETAIL_BF16_HH
//
// Project     : HLR
// Module      : compress/bf16
// Description : BF16 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using BF16
// - only fixed compression size (1+8+7 bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bf16 {

using ushort_t = unsigned short;

struct config
{};

// holds compressed data
using  zarray = std::vector< ushort_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size() * sizeof(ushort_t); }
inline config  get_config ( const double    eps ) { return config{}; }

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

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const uint  ival = (* reinterpret_cast< const uint * >( & data[i] ) ) >> 16;

        zdata[i] = ival;
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
    zarray        zdata( nsize );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const float  fval = float( data[i] );
        const uint   ival = (* reinterpret_cast< const uint * >( & fval ) ) >> 16;

        zdata[i] = ival;
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
    {
        const uint  ival = zdata[i] << 16;
        
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
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const uint   ival = zdata[i] << 16;
        const float  fval = * reinterpret_cast< const float * >( & ival );
        
        dest[i] = double( fval );
    }// for
}

}}}// namespace hlr::compress::bf16

#endif // __HLR_UTILS_DETAIL_BF16_HH
