#ifndef __HLR_UTILS_DETAIL_BF24_HH
#define __HLR_UTILS_DETAIL_BF24_HH
//
// Project     : HLR
// Module      : utils/detail/bf24
// Description : BF24 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using BF24 via half library
// - only fixed compression size (16 bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bf24 {

struct bfloat24
{
    unsigned char  data[3];
    
public:
    bfloat24 ()
            : data{ 0, 0, 0 }
    {}
    
    bfloat24 ( const float f )
    {
        *this = f;
    }
    
    // cast to float
    operator float () const
    {
        unsigned int  proc = (data[2] << 24) | (data[1] << 16) | (data[0] << 8);

        return * reinterpret_cast< float* >( & proc );
    }
    
    // cast to bfloat24
    bfloat24 &
    operator = ( float  float_val )
    {
        unsigned int  uf = (*reinterpret_cast< unsigned int * >( & float_val ) ) >> 8;

        data[2] = (uf & 0xff0000) >> 16;
        data[1] = (uf & 0xff00) >> 8;
        data[0] = (uf & 0xff);
        
        return *this;
    }

    bfloat24 operator + ( bfloat24  f ) { return float(*this) + float(f); }
    bfloat24 operator - ( bfloat24  f ) { return float(*this) - float(f); }
    bfloat24 operator * ( bfloat24  f ) { return float(*this) * float(f); }
    bfloat24 operator / ( bfloat24  f ) { return float(*this) / float(f); }
};

struct config
{};

// holds compressed data
using  zarray = std::vector< bfloat24 >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size() * sizeof(bfloat24); }
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
        zdata[i] = bfloat24(data[i]);

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

    // // look for max/min value
    // double  vmax = data[0];
    // double  vmin = data[0];

    // for ( size_t  i = 0; i < nsize; ++i )
    // {
    //     vmax = std::max( vmax, data[i] );
    //     vmin = std::min( vmin, data[i] );
    // }// for

    // std::cout << vmin << " / " << vmax << " / " << std::ceil( std::log2( std::ceil( std::log2( vmax / vmin ) ) ) ) << std::endl;
    
    for ( size_t  i = 0; i < nsize; ++i )
        zdata[i] = bfloat24(data[i]);

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
        return bf24::byte_size( v );
    }
    
private:

    mem_accessor ();
};
    
}}}// namespace hlr::compress::bf24

#endif // __HLR_UTILS_DETAIL_BF24_HH
