#ifndef __HLR_UTILS_COMPRESSION_HH
#define __HLR_UTILS_COMPRESSION_HH
//
// Project     : HLR
// Module      : utils/compression
// Description : compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/utils/log.hh>

////////////////////////////////////////////////////////////
//
// ZFP related functions
//
////////////////////////////////////////////////////////////

#if defined(HAS_ZFP)

#include <zfp.h>

namespace hlr { namespace zfp {

//
// define compression mode
//
struct config
{
    zfp_mode  mode;
    double    accuracy;
    uint      precision;
    uint      rate;
};

inline config reversible     ()                     { return config{ zfp_mode_reversible, 0.0, 0, 0 }; }
inline config fixed_rate     ( const uint    rate ) { return config{ zfp_mode_fixed_rate, 0.0, 0, rate }; }
inline config fixed_accuracy ( const double  acc  ) { return config{ zfp_mode_fixed_accuracy, acc, 0, 0 }; }

// holds compressed data
using  zarray = std::vector< unsigned char >;

//
// compression functions
//
template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    const uint   ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    zfp_field *  field = nullptr;
    zfp_type     type;

    if      constexpr ( std::is_same_v< value_t, double > ) type = zfp_type_double;
    else if constexpr ( std::is_same_v< value_t, float >  ) type = zfp_type_float;
    else
        HLR_ERROR( "unsupported type" );
    
    switch ( ndims )
    {
        case  1 : field = zfp_field_1d( data, type, dim0 ); break;
        case  2 : field = zfp_field_2d( data, type, dim0, dim1 ); break;
        case  3 : field = zfp_field_3d( data, type, dim0, dim1, dim2 ); break;
        case  4 : field = zfp_field_4d( data, type, dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of ZFP dimensions" );
    }// switch

    auto  zfp = zfp_stream_open( nullptr );

    switch ( config.mode )
    {
        case zfp_mode_fixed_rate      : zfp_stream_set_rate( zfp, config.rate, type, ndims, zfp_false ); break;
        case zfp_mode_fixed_precision : zfp_stream_set_precision( zfp, config.precision ); break;
        case zfp_mode_fixed_accuracy  : zfp_stream_set_accuracy( zfp, config.accuracy ); break;
        case zfp_mode_reversible      : zfp_stream_set_reversible( zfp ); break;
            
        default :
            HLR_ASSERT( "unsupported ZFP mode" );
    }// switch

    // parallelism via hierarchy not within ZFP
    zfp_stream_set_execution( zfp, zfp_exec_serial );

    auto  bufsize = zfp_stream_maximum_size( zfp, field );
    auto  buffer  = zarray( bufsize );
    auto  stream  = stream_open( buffer.data(), bufsize );

    zfp_stream_set_bit_stream( zfp, stream );
    zfp_stream_rewind( zfp );

    if ( ! zfp_write_header( zfp, field, ZFP_HEADER_FULL ) )
        HLR_ERROR( "error in zfp_write_header" );

    auto  c_size = zfp_compress( zfp, field );

    if ( c_size == 0 )
        HLR_ERROR( "error in zfp_compress" );
    
    auto  result = zarray( c_size );

    std::copy( buffer.begin(), buffer.begin() + c_size, result.begin() );

    zfp_field_free( field );    
    zfp_stream_close( zfp );
    stream_close( stream );
    
    return result;
}

//
// decompression functions
//
template < typename value_t >
void
uncompress ( const zarray &  buffer,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    const uint   ndims = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? 1 : 2 ) : 3 ) : 4 );
    zfp_field *  field = nullptr;
    zfp_type     type;

    if      constexpr ( std::is_same_v< value_t, double > ) type = zfp_type_double;
    else if constexpr ( std::is_same_v< value_t, float >  ) type = zfp_type_float;
    else
        HLR_ERROR( "unsupported type" );
    
    switch ( ndims )
    {
        case  1 : field = zfp_field_1d( dest, type, dim0 ); break;
        case  2 : field = zfp_field_2d( dest, type, dim0, dim1 ); break;
        case  3 : field = zfp_field_3d( dest, type, dim0, dim1, dim2 ); break;
        case  4 : field = zfp_field_4d( dest, type, dim0, dim1, dim2, dim3 ); break;
        default :
            HLR_ASSERT( "unsupported number of ZFP dimensions" );
    }// switch

    auto  zfp = zfp_stream_open( nullptr );

    // zfp_field_set_type( field, type );
    // zfp_field_set_pointer( field, dest );

    switch ( ndims )
    {
        case  1 : zfp_field_set_size_1d( field, dim0 ); break;
        case  2 : zfp_field_set_size_2d( field, dim0, dim1 ); break;
        case  3 : zfp_field_set_size_3d( field, dim0, dim1, dim2 ); break;
        case  4 : zfp_field_set_size_4d( field, dim0, dim1, dim2, dim3 ); break;
        default:
            HLR_ASSERT( "unsupported number of ZFP dimensions" );
    }// switch

    // parallelism via hierarchy not within ZFP
    zfp_stream_set_execution( zfp, zfp_exec_serial );

    auto  stream  = stream_open( const_cast< unsigned char * >( buffer.data() ), buffer.size() );

    zfp_stream_set_bit_stream( zfp, stream );
    zfp_stream_rewind( zfp );

    if ( ! zfp_read_header( zfp, field, ZFP_HEADER_FULL ) )
        HLR_ERROR( "error in zfp_read_header" );
    
    if ( ! zfp_decompress( zfp, field ) )
        HLR_ERROR( "error in zfp_decompress" );

    zfp_field_free( field );    
    zfp_stream_close( zfp );
    stream_close( stream );
}

}}// namespace hlr::zfp

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

namespace hlr { namespace sz {

//
// holds compression parameters
//
struct config
{
    int     err_bound_mode;
    double  abs_err_bound;
    double  rel_bound_ratio;
    double  pwr_bound_ratio;
};

inline config  config_rel ( double rel_bound_ratio ) { return config{ REL, 0.0, rel_bound_ratio, 0.0 }; }
inline config  config_abs ( double abs_err_bound   ) { return config{ ABS, abs_err_bound, 0.0, 0.0 }; }

//
// handles arrays allocated within SZ
//
struct zarray_view
{
    using value_t = unsigned char;
    using byte_t  = unsigned char;

private:
    byte_t *  _ptr;
    size_t    _size;

public:
    zarray_view ()
            : _ptr( nullptr )
            , _size( 0 )
    {}

    zarray_view ( byte_t *      aptr,
                  const size_t  asize )
            : _ptr( aptr )
            , _size( asize )
    {}

    zarray_view ( zarray_view &&  v )
            : _ptr( v._ptr )
            , _size( v._size )
    {
        v._ptr  = nullptr;
        v._size = 0;
    }

    ~zarray_view ()
    {
        free();
    }
    
    zarray_view &  operator = ( zarray_view &&  v )
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
zarray_view
compress ( const config &   config,
           const value_t *  data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0,
           const size_t     dim4 = 0 );

template <>
inline
zarray_view
compress< float > ( const config &  config,
                    const float *   data,
                    const size_t    dim0,
                    const size_t    dim1,
                    const size_t    dim2,
                    const size_t    dim3,
                    const size_t    dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_FLOAT, const_cast< float * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return zarray_view( ptr, csize );
}

template <>
inline
zarray_view
compress< double > ( const config &  config,
                     const double *  data,
                     const size_t    dim0,
                     const size_t    dim1,
                     const size_t    dim2,
                     const size_t    dim3,
                     const size_t    dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_DOUBLE, const_cast< double * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return zarray_view( ptr, csize );
}

template <>
inline
zarray_view
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
zarray_view
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
uncompress ( const zarray_view &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 );

template <>
inline
void
uncompress< float > ( const zarray_view &  v,
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
uncompress< double > ( const zarray_view &  v,
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
uncompress< std::complex< float > > ( const zarray_view &       v,
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
uncompress< std::complex< double > > ( const zarray_view &       v,
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

using  zconfig_t = hlr::sz::config;

#elif defined(HAS_ZFP)

using  zconfig_t = hlr::zfp::config;

#else 

struct zconfig_t {};

#endif

}// namespace hlr

#endif // __HLR_UTILS_ZFP_HH
