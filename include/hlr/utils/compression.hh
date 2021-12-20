#ifndef __HLR_UTILS_COMPRESSION_HH
#define __HLR_UTILS_COMPRESSION_HH
//
// Project     : HLR
// Module      : utils/compression
// Description : compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// ZFP related functions
//
////////////////////////////////////////////////////////////

#if defined(HAS_ZFP)

#include <zfpcarray2.h>

#endif

////////////////////////////////////////////////////////////
//
// SZ related functions and types
//
////////////////////////////////////////////////////////////

#if defined(HAS_SZ)

#include <string.h>

#if defined(HAS_SZ)
#include <sz.h>
#include <zlib.h>
#endif

#include <hlr/utils/log.hh>

namespace hlr { namespace sz {

//
// holds compression parameters
//
struct sz_config
{
    int     err_bound_mode;
    double  abs_err_bound;
    double  rel_bound_ratio;
    double  pwr_bound_ratio;
};

inline sz_config  sz_config_rel ( double rel_bound_ratio ) { return sz_config{ REL, 0.0, rel_bound_ratio, 0.0 }; }
inline sz_config  sz_config_abs ( double abs_err_bound   ) { return sz_config{ ABS, abs_err_bound, 0.0, 0.0 }; }

//
// handles arrays allocated within SZ
//
struct carray_view
{
    using value_t = unsigned char;
    using byte_t  = unsigned char;

private:
    byte_t *  _ptr;
    size_t    _size;

public:
    carray_view ()
            : _ptr( nullptr )
            , _size( 0 )
    {}

    carray_view ( byte_t *      aptr,
                  const size_t  asize )
            : _ptr( aptr )
            , _size( asize )
    {}

    carray_view ( carray_view &&  v )
            : _ptr( v._ptr )
            , _size( v._size )
    {
        v._ptr  = nullptr;
        v._size = 0;
    }

    ~carray_view ()
    {
        free();
    }
    
    carray_view &  operator = ( carray_view &&  v )
    {
        free();
        
        _ptr  = v._ptr;
        _size = v._size;
        
        v._ptr  = nullptr;
        v._size = 0;

        return *this;
    }
    
    byte_t *  data () const { return _ptr; }
    size_t    size () const { return _size; }

    void  free ()
    {
        ::free( _ptr );
        _ptr  = nullptr;
        _size = 0;
    }
};
    
template < typename value_t >
carray_view
compress ( const sz_config &  config,
           const value_t *    data,
           const size_t       dim0,
           const size_t       dim1 = 0,
           const size_t       dim2 = 0,
           const size_t       dim3 = 0,
           const size_t       dim4 = 0 );

template <>
inline
carray_view
compress< float > ( const sz_config &  config,
                    const float *      data,
                    const size_t       dim0,
                    const size_t       dim1,
                    const size_t       dim2,
                    const size_t       dim3,
                    const size_t       dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_FLOAT, const_cast< float * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return carray_view( ptr, csize );
}

template <>
inline
carray_view
compress< double > ( const sz_config &  config,
                     const double *     data,
                     const size_t       dim0,
                     const size_t       dim1,
                     const size_t       dim2,
                     const size_t       dim3,
                     const size_t       dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_DOUBLE, const_cast< double * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return carray_view( ptr, csize );
}

template <>
inline
carray_view
compress< std::complex< float > > ( const sz_config &              config,
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
carray_view
compress< std::complex< double > > ( const sz_config &              config,
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
uncompress ( const carray_view &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 );

template <>
inline
void
uncompress< float > ( const carray_view &  v,
                      float *        dest,
                      const size_t   dim0,
                      const size_t   dim1,
                      const size_t   dim2,
                      const size_t   dim3,
                      const size_t   dim4 )
{
    SZ_decompress_args( SZ_FLOAT, v.data(), v.size(),
                        dest,
                        dim4, dim3, dim2, dim1, dim0 );
}

template <>
inline
void
uncompress< double > ( const carray_view &  v,
                       double *       dest,
                       const size_t   dim0,
                       const size_t   dim1,
                       const size_t   dim2,
                       const size_t   dim3,
                       const size_t   dim4 )
{
    SZ_decompress_args( SZ_DOUBLE, v.data(), v.size(),
                        dest,
                        dim4, dim3, dim2, dim1, dim0 );
}

template <>
inline
void
uncompress< std::complex< float > > ( const carray_view &       v,
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
uncompress< std::complex< double > > ( const carray_view &       v,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

}}// namespace hlr::sz

#endif // HAS_SZ


////////////////////////////////////////////////////////////
//
// compression configuration type
//
////////////////////////////////////////////////////////////

namespace hlr
{

#if defined(HAS_SZ)

using  zconfig_t = hlr::sz::sz_config;

#elif defined(HAS_ZFP)

using  zconfig_t = zfp_config;

#else 

struct zconfig_t {};

#endif

}// namespace hlr

#endif // __HLR_UTILS_ZFP_HH
