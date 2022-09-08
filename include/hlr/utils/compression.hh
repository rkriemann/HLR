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

namespace hlr { namespace compress { namespace zfp {

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

inline
uint
eps_to_rate ( const double eps )
{
    // if      ( eps >= 1e-2 ) return 8;
    // else if ( eps >= 1e-3 ) return 10;
    // else if ( eps >= 1e-4 ) return 14;
    // else if ( eps >= 1e-5 ) return 16;
    // else if ( eps >= 1e-6 ) return 20;
    // else if ( eps >= 1e-7 ) return 24;
    // else if ( eps >= 1e-8 ) return 28;
    // else if ( eps >= 1e-9 ) return 30;
    // else if ( eps >= 1e-9 ) return 30;
    return uint( std::ceil( std::abs( std::log2( eps ) ) ) );
}

inline config  reversible        ()                     { return config{ zfp_mode_reversible, 0.0, 0, 0 }; }
inline config  fixed_rate        ( const uint    rate ) { return config{ zfp_mode_fixed_rate, 0.0, 0, rate }; }

inline config  absolute_accuracy ( const double  acc  ) { return config{ zfp_mode_fixed_rate, 0.0, 0, eps_to_rate( acc ) }; } // config{ zfp_mode_fixed_accuracy, acc, 0, 0 }; }
inline config  relative_accuracy ( const double  acc  ) { HLR_ERROR( "no supported by ZFP" ); return config{ zfp_mode_fixed_accuracy, acc, 0, 0 }; }

// holds compressed data
using  zarray = std::vector< unsigned char >;

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

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

}}}// namespace hlr::compress::zfp

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

namespace hlr { namespace compress { namespace sz {

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

inline config  relative_accuracy ( double  eps ) { return config{ REL, 0.0, eps, 0.0 }; }
inline config  absolute_accuracy ( double  eps ) { return config{ ABS, eps, 0.0, 0.0 }; }

//
// handles arrays allocated within SZ
//
struct zarray
{
    using value_t = unsigned char;
    using byte_t  = unsigned char;

private:
    byte_t *  _ptr;
    size_t    _size;

public:
    zarray ()
            : _ptr( nullptr )
            , _size( 0 )
    {}

    zarray ( const size_t  asize )
            : _ptr( reinterpret_cast< byte_t * >( ::malloc( asize ) ) )
            , _size( asize )
    {}

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
        ::free( _ptr );
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
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_FLOAT, const_cast< float * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

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
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_DOUBLE, const_cast< double * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

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
uncompress ( const zarray &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 );

template <>
inline
void
uncompress< float > ( const zarray &  v,
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
uncompress< double > ( const zarray &  v,
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
uncompress< std::complex< float > > ( const zarray &            v,
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
uncompress< std::complex< double > > ( const zarray &            v,
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

#endif // HAS_SZ

////////////////////////////////////////////////////////////
//
// SZ3 related functions and types
//
////////////////////////////////////////////////////////////

#if defined(HAS_SZ3)

#include <string.h>

#if defined(HAS_SZ3)
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

// inline config  relative_accuracy ( double  eps ) { return config{ SZ::EB_L2NORM, eps }; }
inline config  absolute_accuracy ( double  eps ) { return config{ SZ::EB_ABS,    eps }; }

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
            : _ptr( reinterpret_cast< byte_t * >( ::malloc( asize ) ) )
            , _size( asize )
    {}

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
        ::free( _ptr );
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
uncompress ( const zarray &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 );

template <>
inline
void
uncompress< float > ( const zarray &  v,
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
uncompress< double > ( const zarray &  v,
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
uncompress< std::complex< float > > ( const zarray &            v,
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
uncompress< std::complex< double > > ( const zarray &            v,
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

#endif // HAS_SZ3

////////////////////////////////////////////////////////////
//
// compression using Posits
//
////////////////////////////////////////////////////////////

#if defined(HAS_UNIVERSAL)

#include <universal/number/posit/posit.hpp>

namespace hlr { namespace compress { namespace posits {

struct config
{
    uint  bitsize;
};

// holds compressed data
using  zarray = std::vector< unsigned char >;

inline
size_t
byte_size ( const zarray &  v )
{
    if ( v.size() > 0 )
    {
        const auto  bitsize = v[0];
        const auto  nposits = (v.size() - 8) / bitsize;

        return std::ceil( nposits * bitsize / 8.0 );
    }// if

    return 0;
}

inline config  relative_accuracy ( const double  eps  ) { return config{ 24 }; } // TODO!!!
inline config  absolute_accuracy ( const double  eps  ) { return config{ 24 }; }

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
    zarray        zdata( 8 + nsize * 8 ); // SW-Posits have fixed size 8!

    zdata[0] = config.bitsize;
    
    switch ( config.bitsize )
    {
        case 24:
        {
            using  posit_t = sw::universal::posit< 24, 2 >;

            auto  ptr = reinterpret_cast< posit_t * >( zdata.data() + 8 );

            for ( size_t  i = 0; i < nsize; ++i )
                ptr[i] = posit_t( data[i] );
        }
        break;

        default:
            HLR_ERROR( "TODO" );
    }// switch

    return zdata;
}

template < typename value_t >
void
uncompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 );

template <>
inline
void
uncompress< float > ( const zarray &  v,
                      float *        dest,
                      const size_t   dim0,
                      const size_t   dim1,
                      const size_t   dim2,
                      const size_t   dim3,
                      const size_t   dim4 )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
uncompress< double > ( const zarray &  zdata,
                       double *        dest,
                       const size_t    dim0,
                       const size_t    dim1,
                       const size_t    dim2,
                       const size_t    dim3,
                       const size_t    dim4 )
{
    const size_t  nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    bitsize = zdata[0];
    
    switch ( bitsize )
    {
        case 24:
        {
            using  posit_t = sw::universal::posit< 24, 2 >;

            auto  ptr = reinterpret_cast< const posit_t * >( zdata.data() + 8 );

            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = double( ptr[i] );
        }
        break;

        default:
            HLR_ERROR( "TODO" );
    }// switch
}

}}}// namespace hlr::compress::posits

#endif

////////////////////////////////////////////////////////////
//
// compression configuration type
//
////////////////////////////////////////////////////////////

namespace hlr
{

namespace compress
{

#if defined(HAS_UNIVERSAL)

#  define HLR_HAS_COMPRESSION  1

using  zconfig_t = hlr::compress::posits::config;
using  zarray    = hlr::compress::posits::zarray;

using hlr::compress::posits::compress;
using hlr::compress::posits::uncompress;
using hlr::compress::posits::absolute_accuracy;
using hlr::compress::posits::byte_size;

#elif defined(HAS_SZ3)

#  define HLR_HAS_COMPRESSION  1

using  zconfig_t = hlr::compress::sz3::config;
using  zarray    = hlr::compress::sz3::zarray;

using hlr::compress::sz3::compress;
using hlr::compress::sz3::uncompress;
using hlr::compress::sz3::absolute_accuracy;
using hlr::compress::sz3::byte_size;

#elif defined(HAS_SZ)

#  define HLR_HAS_COMPRESSION  1

using  zconfig_t = hlr::compress::sz::config;
using  zarray    = hlr::compress::sz::zarray;

using hlr::compress::sz::compress;
using hlr::compress::sz::uncompress;
using hlr::compress::sz::absolute_accuracy;
using hlr::compress::sz::byte_size;

#elif defined(HAS_ZFP)

#  define HLR_HAS_COMPRESSION  1

using  zconfig_t = hlr::compress::zfp::config;
using  zarray    = hlr::compress::zfp::zarray;

using hlr::compress::zfp::compress;
using hlr::compress::zfp::uncompress;
using hlr::compress::zfp::absolute_accuracy;
using hlr::compress::zfp::byte_size;

#else 

#  define HLR_HAS_COMPRESSION  0

struct zconfig_t {};
using  zarray    = std::vector< char >;

#endif

}// namespace compress

}// namespace hlr

#endif // __HLR_UTILS_ZFP_HH
