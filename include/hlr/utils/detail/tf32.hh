#ifndef __HLR_UTILS_DETAIL_TF32_HH
#define __HLR_UTILS_DETAIL_TF32_HH
//
// Project     : HLR
// Module      : utils/detail/tf32
// Description : TF32 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using TF32
// - only fixed compression size (1+8+10 bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace tf32 {

struct tensorfloat32
{
    unsigned char  data[3];
    
public:
    tensorfloat32 ()
            : data{ 0, 0, 0 }
    {}
    
    tensorfloat32 ( const float f )
    {
        *this = f;
    }
    
    // cast to float
    operator float () const
    {
        unsigned int  proc = (data[2] << 24) | (data[1] << 16) | (data[0] << 8);

        return * reinterpret_cast< float* >( & proc );
    }
    
    // cast to tensorfloat32
    tensorfloat32 &
    operator = ( float  float_val )
    {
        unsigned int  uf = (*reinterpret_cast< unsigned int * >( & float_val ) ) >> 8;

        data[2] = (uf & 0xff0000) >> 16;
        data[1] = (uf & 0xff00) >> 8;
        data[0] = (uf & 0xe0);
        
        return *this;
    }

    tensorfloat32 operator + ( tensorfloat32  f ) { return float(*this) + float(f); }
    tensorfloat32 operator - ( tensorfloat32  f ) { return float(*this) - float(f); }
    tensorfloat32 operator * ( tensorfloat32  f ) { return float(*this) * float(f); }
    tensorfloat32 operator / ( tensorfloat32  f ) { return float(*this) / float(f); }
};

struct config
{};

// holds compressed data
using  zarray = std::vector< tensorfloat32 >;

// assume 19 bits (1+8+10) storage
inline
size_t
byte_size  ( const zarray &  v   )
{
    const auto  bitsize = v.size() * 19;

    return bitsize / 8 + (bitsize % 8 == 0 ? 0 : 1);
}

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
        zdata[i] = tensorfloat32(data[i]);

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
        zdata[i] = tensorfloat32(data[i]);

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = double( zdata[i] );
}

}}}// namespace hlr::compress::tf32

#endif // __HLR_UTILS_DETAIL_TF32_HH
