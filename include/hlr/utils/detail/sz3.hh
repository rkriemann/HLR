#ifndef __HLR_UTILS_DETAIL_SZ3_HH
#define __HLR_UTILS_DETAIL_SZ3_HH
//
// Project     : HLR
// Module      : utils/detail/sz3
// Description : SZ3 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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

//
// holds compression parameters
//
struct config
{
    int     mode;
    double  eps;
};

inline config  get_config ( double  eps ) { return config{ SZ::EB_REL, eps }; }
// inline config  get_config ( double  eps ) { return config{ SZ::EB_L2NORM, eps }; }
// inline config  get_config ( double  eps ) { return config{ SZ::EB_ABS,    eps }; }

//
// handles arrays allocated within SZ
//
struct zarray
{
    using value_t = char;
    using byte_t  = char;

private:
    byte_t *  _ptr;
    size_t    _size;

public:
    zarray ()
            : _ptr( nullptr )
            , _size( 0 )
    {}

    zarray ( const size_t  asize )
            : _ptr( nullptr )
            , _size( asize )
    {
        if ( asize > 0 )
            _ptr = new byte_t[ asize ];
    }

    zarray ( byte_t *      aptr,
             const size_t  asize )
            : _ptr( aptr )
            , _size( asize )
    {}

    zarray ( zarray &&  v )
            : _ptr( v._ptr )
            , _size( v._size )
    {
        v._ptr  = nullptr;
        v._size = 0;
    }

    ~zarray ()
    {
        free();
    }
    
    zarray &  operator = ( zarray &&  v )
    {
        free();
        
        _ptr  = v._ptr;
        _size = v._size;
        
        v._ptr  = nullptr;
        v._size = 0;

        return *this;
    }
    
    byte_t *  begin () const { return _ptr; }
    byte_t *  end   () const { return _ptr + _size; }
    
    byte_t *  data () const { return _ptr; }
    size_t    size () const { return _size; }

    void  free ()
    {
        delete[] _ptr;
        _ptr  = nullptr;
        _size = 0;
    }
};

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

template < typename value_t >
zarray
compress ( const config &   config,
           const value_t *  data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0,
           const size_t     dim4 = 0 );

template <>
inline
zarray
compress< float > ( const config &  config,
                    const float *   data,
                    const size_t    dim0,
                    const size_t    dim1,
                    const size_t    dim2,
                    const size_t    dim3,
                    const size_t    dim4 )
{
    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    SZ::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ::Config( dim0 ); break;
        case  2 : conf = SZ::Config( dim0, dim1 ); break;
        case  3 : conf = SZ::Config( dim0, dim1, dim2 ); break;
        case  4 : conf = SZ::Config( dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch
    
    conf.errorBoundMode = config.mode;
    
    switch ( config.mode )
    {
        case SZ::EB_REL :
            conf.errorBoundMode = SZ::EB_REL;
            conf.relErrorBound  = config.eps;
            break;
            
        case SZ::EB_ABS :
            conf.errorBoundMode = SZ::EB_ABS;
            conf.absErrorBound  = config.eps;
            break;
            
        case SZ::EB_L2NORM :
            conf.errorBoundMode   = SZ::EB_L2NORM;
            conf.l2normErrorBound = config.eps;
            break;

        default:
            HLR_ERROR( "unsupported compression mode in SZ3" );
    }// switch
    
    size_t  csize = 0;
    auto    ptr   = SZ_compress< float >( conf, data, csize );

    return zarray( ptr, csize );
}

template <>
inline
zarray
compress< double > ( const config &  config,
                     const double *  data,
                     const size_t    dim0,
                     const size_t    dim1,
                     const size_t    dim2,
                     const size_t    dim3,
                     const size_t    dim4 )
{
    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    SZ::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ::Config( dim0 ); break;
        case  2 : conf = SZ::Config( dim0, dim1 ); break;
        case  3 : conf = SZ::Config( dim0, dim1, dim2 ); break;
        case  4 : conf = SZ::Config( dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch
    
    conf.errorBoundMode = config.mode;
    
    switch ( config.mode )
    {
        case SZ::EB_REL :
            conf.errorBoundMode = SZ::EB_REL;
            conf.relErrorBound  = config.eps;
            break;
            
        case SZ::EB_ABS :
            conf.errorBoundMode = SZ::EB_ABS;
            conf.absErrorBound  = config.eps;
            break;
            
        case SZ::EB_L2NORM :
            conf.errorBoundMode   = SZ::EB_L2NORM;
            conf.l2normErrorBound = config.eps;
            break;

        default:
            HLR_ERROR( "unsupported compression mode in SZ3" );
    }// switch
    
    size_t  csize = 0;
    auto    ptr   = SZ_compress< double >( conf, data, csize );

    return zarray( ptr, csize );
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
decompress ( const zarray &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 );

template <>
inline
void
decompress< float > ( const zarray &  v,
                      float *        dest,
                      const size_t   dim0,
                      const size_t   dim1,
                      const size_t   dim2,
                      const size_t   dim3,
                      const size_t   dim4 )
{
    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    SZ::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ::Config( dim0 ); break;
        case  2 : conf = SZ::Config( dim0, dim1 ); break;
        case  3 : conf = SZ::Config( dim0, dim1, dim2 ); break;
        case  4 : conf = SZ::Config( dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch

    SZ_decompress< float >( conf, v.data(), v.size(), dest );
}

template <>
inline
void
decompress< double > ( const zarray &  v,
                       double *       dest,
                       const size_t   dim0,
                       const size_t   dim1,
                       const size_t   dim2,
                       const size_t   dim3,
                       const size_t   dim4 )
{
    const uint  ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    SZ::Config  conf;

    switch ( ndims )
    {
        case  1 : conf = SZ::Config( dim0 ); break;
        case  2 : conf = SZ::Config( dim0, dim1 ); break;
        case  3 : conf = SZ::Config( dim0, dim1, dim2 ); break;
        case  4 : conf = SZ::Config( dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of dimensions for SZ3" );
    }// switch

    SZ_decompress< double >( conf, v.data(), v.size(), dest );
}

template <>
inline
void
decompress< std::complex< float > > ( const zarray &            v,
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
decompress< std::complex< double > > ( const zarray &            v,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

}}}// namespace hlr::compress::sz

#endif // HLR_HAS_SZ3

#endif // __HLR_UTILS_DETAIL_SZ3_HH
