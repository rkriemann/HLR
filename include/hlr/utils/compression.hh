#ifndef __HLR_UTILS_COMPRESSION_HH
#define __HLR_UTILS_COMPRESSION_HH
//
// Project     : HLR
// Module      : utils/compression
// Description : compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

// define for arithmetic specific compression bit rates 
#define HLR_COMPRESS_RATE_ARITH

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>

////////////////////////////////////////////////////////////
//
// compression configuration type
//
////////////////////////////////////////////////////////////

#if defined(COMPRESSOR)

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if COMPRESSOR == 1

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/fp32.hh>

namespace hlr { namespace compress {

static const char provider[] = "FP32";

using  zconfig_t = hlr::compress::fp32::config;
using  zarray    = hlr::compress::fp32::zarray;

using hlr::compress::fp32::compress;
using hlr::compress::fp32::decompress;
using hlr::compress::fp32::get_config;
using hlr::compress::fp32::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 2

#  if !defined(HAS_HALF)
#    error "half library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/fp16.hh>

namespace hlr { namespace compress {

static const char provider[] = "FP16";

using  zconfig_t = hlr::compress::fp16::config;
using  zarray    = hlr::compress::fp16::zarray;

using hlr::compress::fp16::compress;
using hlr::compress::fp16::decompress;
using hlr::compress::fp16::get_config;
using hlr::compress::fp16::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 3

#  if !defined(HAS_ZFP)
#    error "ZFP library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/zfp.hh>

namespace hlr { namespace compress {

static const char provider[] = "ZFP";

using  zconfig_t = hlr::compress::zfp::config;
using  zarray    = hlr::compress::zfp::zarray;

using hlr::compress::zfp::compress;
using hlr::compress::zfp::decompress;
using hlr::compress::zfp::get_config;
using hlr::compress::zfp::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 4

#  if !defined(HAS_UNIVERSAL)
#    error "Universal library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/posits.hh>

namespace hlr { namespace compress {

static const char provider[] = "Posits";

using  zconfig_t = hlr::compress::posits::config;
using  zarray    = hlr::compress::posits::zarray;

using hlr::compress::posits::compress;
using hlr::compress::posits::decompress;
using hlr::compress::posits::get_config;
using hlr::compress::posits::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 5

#  if !defined(HAS_SZ)
#    error "SZ library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/sz.hh>

namespace hlr { namespace compress {

static const char provider[] = "SZ";

using  zconfig_t = hlr::compress::sz::config;
using  zarray    = hlr::compress::sz::zarray;

using hlr::compress::sz::compress;
using hlr::compress::sz::decompress;
using hlr::compress::sz::get_config;
using hlr::compress::sz::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 6

#  if !defined(HAS_SZ3)
#    error "SZ3 library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/sz3.hh>

namespace hlr { namespace compress {

static const char provider[] = "SZ3";

using  zconfig_t = hlr::compress::sz3::config;
using  zarray    = hlr::compress::sz3::zarray;

using hlr::compress::sz3::compress;
using hlr::compress::sz3::decompress;
using hlr::compress::sz3::get_config;
using hlr::compress::sz3::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 7

#  if !defined(HAS_LZ4)
#    error "LZ4 library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/lz4.hh>

namespace hlr { namespace compress {

static const char provider[] = "LZ4";

using  zconfig_t = hlr::compress::lz4::config;
using  zarray    = hlr::compress::lz4::zarray;

using hlr::compress::lz4::compress;
using hlr::compress::lz4::decompress;
using hlr::compress::lz4::get_config;
using hlr::compress::lz4::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 8

#  if !defined(HAS_ZLIB)
#    error "ZLIB library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/zlib.hh>

namespace hlr { namespace compress {

static const char provider[] = "zlib";

using  zconfig_t = hlr::compress::zlib::config;
using  zarray    = hlr::compress::zlib::zarray;

using hlr::compress::zlib::compress;
using hlr::compress::zlib::decompress;
using hlr::compress::zlib::get_config;
using hlr::compress::zlib::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 9

#  if !defined(HAS_ZSTD)
#    error "ZSTD library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/zstd.hh>

namespace hlr { namespace compress {

static const char provider[] = "Zstd";

using  zconfig_t = hlr::compress::zstd::config;
using  zarray    = hlr::compress::zstd::zarray;

using hlr::compress::zstd::compress;
using hlr::compress::zstd::decompress;
using hlr::compress::zstd::get_config;
using hlr::compress::zstd::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 10

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/bf16.hh>

namespace hlr { namespace compress {

static const char provider[] = "BF16";

using  zconfig_t = hlr::compress::bf16::config;
using  zarray    = hlr::compress::bf16::zarray;

using hlr::compress::bf16::compress;
using hlr::compress::bf16::decompress;
using hlr::compress::bf16::get_config;
using hlr::compress::bf16::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 11

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/tf32.hh>

namespace hlr { namespace compress {

static const char provider[] = "TF32";

using  zconfig_t = hlr::compress::tf32::config;
using  zarray    = hlr::compress::tf32::zarray;

using hlr::compress::tf32::compress;
using hlr::compress::tf32::decompress;
using hlr::compress::tf32::get_config;
using hlr::compress::tf32::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 12

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/bf24.hh>

namespace hlr { namespace compress {

static const char provider[] = "BF24";

using  zconfig_t = hlr::compress::bf24::config;
using  zarray    = hlr::compress::bf24::zarray;

using hlr::compress::bf24::compress;
using hlr::compress::bf24::decompress;
using hlr::compress::bf24::get_config;
using hlr::compress::bf24::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 13

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/afloat.hh>

namespace hlr { namespace compress {

static const char provider[] = "afloat";

using  zconfig_t = hlr::compress::afloat::config;
using  zarray    = hlr::compress::afloat::zarray;

using hlr::compress::afloat::compress;
using hlr::compress::afloat::decompress;
using hlr::compress::afloat::get_config;
using hlr::compress::afloat::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 14

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/apfloat.hh>

namespace hlr { namespace compress {

static const char provider[] = "apfloat";

using  zconfig_t = hlr::compress::apfloat::config;
using  zarray    = hlr::compress::apfloat::zarray;

using hlr::compress::apfloat::compress;
using hlr::compress::apfloat::decompress;
using hlr::compress::apfloat::get_config;
using hlr::compress::apfloat::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 15

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/bfloat.hh>

namespace hlr { namespace compress {

static const char provider[] = "bfloat";

using  zconfig_t = hlr::compress::bfloat::config;
using  zarray    = hlr::compress::bfloat::zarray;

using hlr::compress::bfloat::compress;
using hlr::compress::bfloat::decompress;
using hlr::compress::bfloat::get_config;
using hlr::compress::bfloat::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 16

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/dfloat.hh>

namespace hlr { namespace compress {

static const char provider[] = "dfloat";

using  zconfig_t = hlr::compress::dfloat::config;
using  zarray    = hlr::compress::dfloat::zarray;

using hlr::compress::dfloat::compress;
using hlr::compress::dfloat::decompress;
using hlr::compress::dfloat::get_config;
using hlr::compress::dfloat::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 17

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/mgard.hh>

namespace hlr { namespace compress {

static const char provider[] = "mgard";

using  zconfig_t = hlr::compress::mgard::config;
using  zarray    = hlr::compress::mgard::zarray;

using hlr::compress::mgard::compress;
using hlr::compress::mgard::decompress;
using hlr::compress::mgard::get_config;
using hlr::compress::mgard::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif COMPRESSOR == 18

#  define HLR_HAS_COMPRESSION  1

#include <hlr/utils/detail/dummy.hh>

namespace hlr { namespace compress {

static const char provider[] = "dummy";

using  zconfig_t = hlr::compress::dummy::config;
using  zarray    = hlr::compress::dummy::zarray;

using hlr::compress::dummy::compress;
using hlr::compress::dummy::decompress;
using hlr::compress::dummy::get_config;
using hlr::compress::dummy::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else // no library available

#  define HLR_HAS_COMPRESSION  0

namespace hlr { namespace compress {

static const char provider[] = "none";

struct zconfig_t {};
struct zarray    {};

}} // namespace hlr::compress

#endif

#else // COMPRESSOR == none

#  define HLR_HAS_COMPRESSION  0

namespace hlr { namespace compress {

static const char provider[] = "none";

struct zconfig_t {};
struct zarray    {};

inline zconfig_t  get_config ( double /* eps */ ) { return zconfig_t{}; }
inline size_t     byte_size  ( const zarray &   ) { return SIZE_MAX; } // ensures maximal memory size

}} // namespace hlr::compress
    
#endif

namespace hlr { namespace compress {

//
// wrappers for blas::matrix and blas::vector
//
template < typename value_t >
zarray
compress ( const zconfig_t &                config,
           const blas::matrix< value_t > &  M )
{
    return compress< value_t >( config, M.data(), M.nrows(), M.ncols() );
}

template < typename value_t >
zarray
compress ( const zconfig_t &                config,
           const blas::vector< value_t > &  v )
{
    return compress< value_t >( config, v.data(), v.length() );
}

template < typename value_t >
void
decompress ( const zarray &             zdata,
             blas::matrix< value_t > &  M )
{
    return decompress< value_t >( zdata, M.data(), M.nrows(), M.ncols() );
}

template < typename value_t >
void
decompress ( const zarray &             zdata,
             blas::vector< value_t > &  v )
{
    return decompress< value_t >( zdata, v.data(), v.length() );
}

//
// memory accessor
//
struct mem_accessor
{
    zconfig_t  mode;

    mem_accessor ( const double  eps )
            : mode( compress::get_config( eps ) )
    {}
    
    template < typename value_t >
    zarray
    encode ( value_t *        data,
             const size_t     dim0,
             const size_t     dim1 = 0,
             const size_t     dim2 = 0,
             const size_t     dim3 = 0 )
    {
        return compress::compress( mode, data, dim0, dim1, dim2, dim3 );
    }
    
    template < typename value_t >
    zarray
    encode ( const blas::matrix< value_t > &  M )
    {
        return compress::compress( mode, M );
    }
    
    template < typename value_t >
    zarray
    encode ( const blas::vector< value_t > &  v )
    {
        return compress::compress( mode, v );
    }
    
    template < typename value_t >
    void
    decode ( const zarray &  zbuf,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
    {
        compress::decompress( zbuf, dest, dim0, dim1, dim2, dim3 );
    }

    template < typename value_t >
    void
    decode ( const zarray &             zbuf,
             blas::matrix< value_t > &  M )
    {
        compress::decompress( zbuf, M );
    }

    template < typename value_t >
    void
    decode ( const zarray &             zbuf,
             blas::vector< value_t > &  v )
    {
        compress::decompress( zbuf, v );
    }

    size_t
    byte_size ( const zarray &  zbuf )
    {
        return compress::byte_size( zbuf );
    }

private:

    mem_accessor ();
};

}} // namespace hlr::compress

#endif // __HLR_UTILS_COMPRESSION_HH
