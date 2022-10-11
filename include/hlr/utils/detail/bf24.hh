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
// - only fixed compression size (1+8+15 bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bf24 {

using byte_t = unsigned char;

struct config
{};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }
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
    zarray        zdata( nsize * 3 );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        // reduce mantissa size by 8 bits
        const uint    ival = (*reinterpret_cast< const uint * >( & data[i] ) ) >> 8;
        const size_t  zpos = 3*i;

        zdata[zpos+2] = (ival & 0xff0000) >> 16;
        zdata[zpos+1] = (ival & 0x00ff00) >> 8;
        zdata[zpos]   = (ival & 0x0000ff);
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
    zarray        zdata( nsize * 3 );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        // convert to float, reduce mantissa size by 8 bits
        const float   fval = data[i];
        const uint    ival = (*reinterpret_cast< const uint * >( & fval ) ) >> 8;
        const size_t  zpos = 3*i;

        zdata[zpos+2] = (ival & 0xff0000) >> 16;
        zdata[zpos+1] = (ival & 0x00ff00) >> 8;
        zdata[zpos]   = (ival & 0x0000ff);
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
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const size_t  zpos = 3*i;
        const uint    ival = (zdata[zpos+2] << 24) | (zdata[zpos+1] << 16) | (zdata[zpos] << 8);
        const float   fval = * reinterpret_cast< const float * >( & ival );
        
        dest[i] = double( fval );
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
        return bf24::byte_size( v );
    }
    
private:

    mem_accessor ();
};
    
}}}// namespace hlr::compress::bf24

#endif // __HLR_UTILS_DETAIL_BF24_HH
