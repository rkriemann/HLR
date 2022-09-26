#ifndef __HLR_UTILS_DETAIL_ZFP_HH
#define __HLR_UTILS_DETAIL_ZFP_HH
//
// Project     : HLR
// Module      : utils/detail/zfp
// Description : ZFP related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

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

//
// convert precision to bitrate
//
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

//
// define various compression modes
//
inline config  reversible        ()                     { return config{ zfp_mode_reversible, 0.0, 0, 0 }; }
inline config  fixed_rate        ( const uint    rate ) { return config{ zfp_mode_fixed_rate, 0.0, 0, rate }; }

inline config  absolute_accuracy ( const double  acc  ) { return config{ zfp_mode_fixed_rate, 0.0, 0, eps_to_rate( acc ) }; } // config{ zfp_mode_fixed_accuracy, acc, 0, 0 }; }
inline config  relative_accuracy ( const double  acc  ) { HLR_ERROR( "no supported by ZFP" ); return config{ zfp_mode_fixed_accuracy, acc, 0, 0 }; }

// holds compressed data
using  zarray = std::vector< unsigned char >;

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

//
// compression function
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
// decompression function
//
template < typename value_t >
void
decompress ( const zarray &  buffer,
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

//
// memory accessor
//
struct mem_accessor
{
    config  mode;

    mem_accessor ( const double  eps )
            : mode( absolute_accuracy( eps ) )
    {}
    
    template < typename value_t >
    zarray
    encode ( value_t *        data,
             const size_t     dim0,
             const size_t     dim1 = 0,
             const size_t     dim2 = 0,
             const size_t     dim3 = 0 )
    {
        return compress( mode, data, dim0, dim1, dim2, dim3 );
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
        return zfp::byte_size( v );
    }

private:

    mem_accessor ();
};

}}}// namespace hlr::compress::zfp

#endif

#endif // __HLR_UTILS_DETAIL_ZFP_HH
