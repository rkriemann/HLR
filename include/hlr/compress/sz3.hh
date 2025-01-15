#ifndef __HLR_UTILS_DETAIL_SZ3_HH
#define __HLR_UTILS_DETAIL_SZ3_HH
//
// Project     : HLR
// Module      : compress/sz3
// Description : SZ3 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// SZ3 related functions and types
//
////////////////////////////////////////////////////////////

#if defined(HLR_HAS_SZ3)

#include <string.h>

#if defined(HLR_HAS_SZ3)
#include <SZ3/api/sz.hpp>
#endif

namespace hlr { namespace compress { namespace sz3 {

using byte_t = char;

//
// holds compression parameters
//
struct config
{
    int     mode;
    double  eps;
};

inline config  get_config ( double  eps ) { return config{ SZ3::EB_REL, 0.2 * eps }; }
// inline config  get_config ( double  eps ) { return config{ SZ3::EB_L2NORM, eps }; }
// inline config  get_config ( double  eps ) { return config{ SZ3::EB_ABS,    eps }; }

using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v ) { return sizeof(zarray) + v.size(); }
inline size_t  compressed_size ( const zarray &  v ) { return v.size(); }

template < typename value_t >
zarray
compress ( const config &   config,
           const value_t *  data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0,
           const size_t     dim4 = 0 )
{
    const uint    ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    SZ3::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ3::Config( dim0 ); break;
        case  2 : conf = SZ3::Config( dim1, dim0 ); break;
        case  3 : conf = SZ3::Config( dim2, dim1, dim0 ); break;
        case  4 : conf = SZ3::Config( dim3, dim2, dim1, dim0 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch
    
    conf.errorBoundMode = config.mode;
    
    switch ( config.mode )
    {
        case SZ3::EB_REL :
            conf.errorBoundMode = SZ3::EB_REL;
            conf.relErrorBound  = config.eps;
            break;
            
        case SZ3::EB_ABS :
            conf.errorBoundMode = SZ3::EB_ABS;
            conf.absErrorBound  = config.eps;
            break;
            
        case SZ3::EB_L2NORM :
            conf.errorBoundMode   = SZ3::EB_L2NORM;
            conf.l2normErrorBound = config.eps;
            break;

        default:
            HLR_ERROR( "unsupported compression mode in SZ3" );
    }// switch
    
    auto    zdata  = std::vector< char >( 2 * nsize * sizeof(value_t) ); // as recommended by SZ3
    auto    zsize  = SZ_compress< value_t >( conf, data, zdata.data(), zdata.size() );
    auto    result = zarray( zsize );

    std::copy( zdata.begin(), zdata.begin() + zsize, result.begin() );
    
    return result;
}

template <>
inline
zarray
compress< std::complex< float > > ( const config &                 config,
                                    const std::complex< float > *  data,
                                    const size_t                   dim0,
                                    const size_t                   dim1,
                                    const size_t                   dim2,
                                    const size_t                   dim3,
                                    const size_t                   dim4 )
{
    HLR_ERROR( "TO DO" );
}

template <>
inline
zarray
compress< std::complex< double > > ( const config &                 config,
                                     const std::complex< double > * data,
                                     const size_t                   dim0,
                                     const size_t                   dim1,
                                     const size_t                   dim2,
                                     const size_t                   dim3,
                                     const size_t                   dim4 )
{
    HLR_ERROR( "TO DO" );
}

template < typename value_t >
void
decompress ( const byte_t *  zptr,
             const size_t    zsize,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{
    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    SZ3::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ3::Config( dim0 ); break;
        case  2 : conf = SZ3::Config( dim1, dim0 ); break;
        case  3 : conf = SZ3::Config( dim2, dim1, dim0 ); break;
        case  4 : conf = SZ3::Config( dim3, dim2, dim1, dim0 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch

    SZ_decompress< value_t >( conf, const_cast< byte_t * >( zptr ), zsize, dest );
}

template <>
inline
void
decompress< std::complex< float > > ( const byte_t *            zptr,
                                      const size_t              zsize,
                                      std::complex< float > *   dest,
                                      const size_t              dim0,
                                      const size_t              dim1,
                                      const size_t              dim2,
                                      const size_t              dim3,
                                      const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

template <>
inline
void
decompress< std::complex< double > > ( const byte_t *            zptr,
                                       const size_t              zsize,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

template < typename value_t >
void
decompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{
    decompress( v.data(), v.size(), dest, dim0, dim1, dim2, dim3, dim4 );
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

        zsize   += z_i.size();
        zlist[l] = std::move( z_i );
    }// for

    zarray  zdata( zsize + sizeof(size_t) * k );
    size_t  pos = 0;

    for ( auto &  z_i : zlist )
    {
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

}}}// namespace hlr::compress::sz

#endif // HLR_HAS_SZ3

#endif // __HLR_UTILS_DETAIL_SZ3_HH
