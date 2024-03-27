#ifndef __HLR_UTILS_DETAIL_MGARD_HH
#define __HLR_UTILS_DETAIL_MGARD_HH
//
// Project     : HLR
// Module      : compress/mgard
// Description : MGARD related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HLR_HAS_MGARD)

#include <mgard/compress_x.hpp>

namespace hlr { namespace compress { namespace mgard {

using byte_t = unsigned char;

//
// define compression mode
//
struct config
{
    double  tol;
};

//
// define various compression modes
//
inline config  get_config ( const double  acc  ) { return config{ acc }; }

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

//
// compression function
//
template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     adim0,
           const size_t     adim1 = 0,
           const size_t     adim2 = 0,
           const size_t     adim3 = 0 )
{
    size_t  dim0 = adim0;
    size_t  dim1 = adim1;
    size_t  dim2 = adim2;
    size_t  dim3 = adim3;

    if (( dim3 > 0 ) && ( dim3 < 3 )) { dim2 *= dim3; dim3  = 0; }
    if (( dim2 > 0 ) && ( dim2 < 3 )) { dim1 *= dim2; dim2  = 0; }
    if (( dim1 > 0 ) && ( dim1 < 3 )) { dim0 *= dim1; dim1  = 0; }

    HLR_ASSERT( dim0 >= 3 );

    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );

    mgard_x::Config  mconfig;
    double           s = 0;  // s = 0 for L2 norm
    void *           c_array;
    size_t           c_size;

    mconfig.lossless = mgard_x::lossless_type::Huffman_Zstd;
    mconfig.dev_type = mgard_x::device_type::AUTO;

    auto  shape = std::vector< mgard_x::SIZE >( ndims );

    switch ( ndims )
    {
        case 4  : shape[3] = dim3;
        case 3  : shape[2] = dim2;
        case 2  : shape[1] = dim1;
        case 1  : shape[0] = dim0; break;
        default : HLR_ERROR( "unsupported number of dimensions" );
    }// switch
    
    mgard_x::compress( ndims,
                       mgard_x::data_type::Double,
                       shape,
                       config.tol,
                       s, // s = 0 for L2 norm
                       mgard_x::error_bound_type::REL,
                       data,
                       c_array,
                       c_size,
                       mconfig,
                       false );

    auto  result = zarray( c_size );

    std::copy( reinterpret_cast< byte_t * >( c_array ),
               reinterpret_cast< byte_t * >( c_array ) + c_size,
               result.begin() );

    free( c_array );
    
    return result;
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

//
// decompression function
//
template < typename value_t >
void
decompress ( const byte_t *  zptr,
             const size_t    zsize,
             value_t *       dest,
             const size_t    adim0,
             const size_t    adim1 = 0,
             const size_t    adim2 = 0,
             const size_t    adim3 = 0 )
{
    size_t  dim0 = adim0;
    size_t  dim1 = adim1;
    size_t  dim2 = adim2;
    size_t  dim3 = adim3;

    if (( dim3 > 0 ) && ( dim3 < 3 )) { dim2 *= dim3; dim3  = 0; }
    if (( dim2 > 0 ) && ( dim2 < 3 )) { dim1 *= dim2; dim2  = 0; }
    if (( dim1 > 0 ) && ( dim1 < 3 )) { dim0 *= dim1; dim1  = 0; }

    HLR_ASSERT( dim0 >= 3 );
        
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    mgard_x::Config  config;

    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.dev_type = mgard_x::device_type::AUTO;

    void *  d_array = nullptr;
    
    mgard_x::decompress( zptr, zsize, d_array, config, false );
    std::copy( reinterpret_cast< value_t * >( d_array ),
               reinterpret_cast< value_t * >( d_array ) + nsize,
               dest );
    
}

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    decompress( zdata.data(), zdata.size(), dest, dim0, dim1, dim2, dim3 );
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
              const blas::vector< Hpro::real_type_t< value_t > > &  S )
{
    //
    // first, determine exponent bits and mantissa bits for all
    // columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    size_t        zsize = 0;
    auto          zlist = std::vector< zarray >( k );

    for ( uint  l = 0; l < k; ++l )
    {
        auto  zconf = get_config( S(l) );
        auto  z_i   = compress( zconf, U.data() + l * n, n );

        // {
        //     blas::vector< value_t >  t( n );
        //     auto                     u_l = U.column( l );
            
        //     decompress( z_i.data(), z_i.size(), t.data(), n );

        //     blas::add( -1.0, u_l, t );
        //     std::cout << blas::norm_2( t ) << std::endl;
        // }
        
        zsize   += z_i.size();
        zlist[l] = std::move( z_i );
    }// for

    zarray  zdata( zsize + sizeof(size_t) * k );
    size_t  pos = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        auto          z_i = std::move( zlist[l] );
        const size_t  s_i = z_i.size();
        
        memcpy( zdata.data() + pos, & s_i, sizeof(size_t) );
        pos += sizeof(size_t);
        
        memcpy( zdata.data() + pos, z_i.data(), s_i );
        pos += s_i;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    const size_t  n   = U.nrows();
    const uint    k   = U.ncols();
    size_t        pos = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        size_t  s_i = 0;

        memcpy( & s_i, zdata.data() + pos, sizeof(size_t) );
        pos += sizeof(size_t);
        
        decompress( zdata.data() + pos, s_i, U.data() + l*n, n );
        pos += s_i;
    }// for
}

}}}// namespace hlr::compress::mgard

#endif // HLR_HAS_MGARD

#endif // __HLR_UTILS_DETAIL_MGARD_HH
