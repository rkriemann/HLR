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
// #define HLR_COMPRESS_RATE_ARITH

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>

// different compressor types
#define HLR_COMPRESSOR_AFLOAT   1
#define HLR_COMPRESSOR_APFLOAT  2
#define HLR_COMPRESSOR_BFLOAT   3
#define HLR_COMPRESSOR_DFLOAT   4
#define HLR_COMPRESSOR_ZFP      5
#define HLR_COMPRESSOR_SZ       6
#define HLR_COMPRESSOR_SZ3      6
#define HLR_COMPRESSOR_MGARD    7
#define HLR_COMPRESSOR_LZ4      8
#define HLR_COMPRESSOR_ZLIB     9
#define HLR_COMPRESSOR_ZSTD     10
#define HLR_COMPRESSOR_POSITS   11
#define HLR_COMPRESSOR_FP32     12
#define HLR_COMPRESSOR_FP16     13
#define HLR_COMPRESSOR_BF16     14
#define HLR_COMPRESSOR_TF32     15
#define HLR_COMPRESSOR_BF24     16
#define HLR_COMPRESSOR_NONE     17

#if defined(HLR_COMPRESSOR)

#  define HLR_HAS_COMPRESSION  1

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if HLR_COMPRESSOR == HLR_COMPRESSOR_FP32

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_FP16

#  if !defined(HLR_HAS_HALF)
#    error "half library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_ZFP

#  if !defined(HLR_HAS_ZFP)
#    error "ZFP library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_POSITS

#  if !defined(HLR_HAS_UNIVERSAL)
#    error "Universal library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_SZ

#  if !defined(HLR_HAS_SZ)
#    error "SZ library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_SZ3

#  if !defined(HLR_HAS_SZ3)
#    error "SZ3 library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_LZ4

#  if !defined(HLR_HAS_LZ4)
#    error "LZ4 library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_ZLIB

#  if !defined(HLR_HAS_ZLIB)
#    error "ZLIB library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_ZSTD

#  if !defined(HLR_HAS_ZSTD)
#    error "ZSTD library not available"
#  endif

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_BF16

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_TF32

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_BF24

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_AFLOAT

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_APFLOAT

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_BFLOAT

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_DFLOAT

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

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_MGARD

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

// #elif HLR_COMPRESSOR == HLR_COMPRESSOR_NONE

// #include <hlr/utils/detail/dummy.hh>

// namespace hlr { namespace compress {

// static const char provider[] = "dummy";

// using  zconfig_t = hlr::compress::dummy::config;
// using  zarray    = hlr::compress::dummy::zarray;

// using hlr::compress::dummy::compress;
// using hlr::compress::dummy::decompress;
// using hlr::compress::dummy::get_config;
// using hlr::compress::dummy::byte_size;

// }} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else

#  error "unsupported HLR_COMPRESSOR value"

#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else // HLR_COMPRESSOR not defined

#  define HLR_HAS_COMPRESSION  0

namespace hlr { namespace compress {

static const char provider[] = "none";

struct zconfig_t {};
struct zarray    {};

inline zconfig_t  get_config ( double /* eps */ ) { return zconfig_t{}; }
inline size_t     byte_size  ( const zarray &   ) { return SIZE_MAX; } // ensures maximal memory size

}} // namespace hlr::compress
    
#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

namespace hlr { namespace compress {

//
// wrappers for blas::matrix and blas::vector
//
template < typename value_t >
zarray
compress ( const zconfig_t &                 config,
           const blas::tensor3< value_t > &  T )
{
    return compress< value_t >( config, T.data(), T.size(0), T.size(1), T.size(2) );
}

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
decompress ( const zarray &              zdata,
             blas::tensor3< value_t > &  T )
{
    return decompress< value_t >( zdata, T.data(), T.size(0), T.size(1), T.size(2) );
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
// general interface for compressible objects
//
struct compressible
{
public:
    //
    // compression interface
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig ) = 0;

    // compress data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc ) = 0;

    // decompress internal data
    virtual void   decompress    () = 0;

    // return true if data is compressed
    virtual bool   is_compressed () const = 0;
};

//
// test functions
//
template < typename T >
bool
is_compressible ( const T &  ref )
{
    const auto  C = dynamic_cast< const compressible * >( &ref );
    
    return ! is_null( C );
}

template < typename T >
bool
is_compressible ( const T *  ptr )
{
    const auto  C = dynamic_cast< const compressible * >( ptr );

    return ! is_null( C );
}

template < typename T >
bool
is_compressed ( const T &  ref )
{
    const auto  C = dynamic_cast< const compressible * >( &ref );
    
    return ! is_null( C ) && C->is_compressed();
}

template < typename T >
bool
is_compressed ( const T *  ptr )
{
    const auto  C = dynamic_cast< const compressible * >( ptr );
    
    return ! is_null( C ) && C->is_compressed();
}

//
// define implementation for adaptive precision compression
// for lowrank matrices
//
#if HLR_COMPRESSOR == HLR_COMPRESSOR_APFLOAT

namespace ap = hlr::compress::apfloat;

#else

namespace ap = hlr::compress::afloat;

#endif

}} // namespace hlr::compress

#endif // __HLR_UTILS_COMPRESSION_HH
