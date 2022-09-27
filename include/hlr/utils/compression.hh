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

#include <hlr/utils/detail/zfp.hh>
#include <hlr/utils/detail/sz.hh>
#include <hlr/utils/detail/sz3.hh>
#include <hlr/utils/detail/posits.hh>
#include <hlr/utils/detail/fp16.hh>
#include <hlr/utils/detail/fp32.hh>

#include <hlr/arith/blas.hh>

////////////////////////////////////////////////////////////
//
// compression configuration type
//
////////////////////////////////////////////////////////////

namespace hlr
{

namespace compress
{

#if defined(COMPRESSOR)

#if COMPRESSOR == 1

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "FP32";

using  zconfig_t = hlr::compress::fp32::config;
using  zarray    = hlr::compress::fp32::zarray;

using hlr::compress::fp32::compress;
using hlr::compress::fp32::decompress;
using hlr::compress::fp32::get_config;
using hlr::compress::fp32::byte_size;

#elif COMPRESSOR == 2

#  if !defined(HAS_HALF)
#    error "half library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "FP16";

using  zconfig_t = hlr::compress::fp16::config;
using  zarray    = hlr::compress::fp16::zarray;

using hlr::compress::fp16::compress;
using hlr::compress::fp16::decompress;
using hlr::compress::fp16::get_config;
using hlr::compress::fp16::byte_size;

#elif COMPRESSOR == 3 && defined(HAS_ZFP)

#  if !defined(HAS_ZFP)
#    error "ZFP library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "ZFP";

using  zconfig_t = hlr::compress::zfp::config;
using  zarray    = hlr::compress::zfp::zarray;

using hlr::compress::zfp::compress;
using hlr::compress::zfp::decompress;
using hlr::compress::zfp::get_config;
using hlr::compress::zfp::byte_size;

#elif COMPRESSOR == 4

#  if !defined(HAS_UNIVERSAL)
#    error "Universal library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "Posits";

using  zconfig_t = hlr::compress::posits::config;
using  zarray    = hlr::compress::posits::zarray;

using hlr::compress::posits::compress;
using hlr::compress::posits::decompress;
using hlr::compress::posits::get_config;
using hlr::compress::posits::byte_size;

#elif COMPRESSOR == 5

#  if !defined(HAS_SZ)
#    error "SZ library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "SZ";

using  zconfig_t = hlr::compress::sz::config;
using  zarray    = hlr::compress::sz::zarray;

using hlr::compress::sz::compress;
using hlr::compress::sz::decompress;
using hlr::compress::sz::get_config;
using hlr::compress::sz::byte_size;

#elif COMPRESSOR == 6

#  if !defined(HAS_SZ3)
#    error "SZ3 library not available"
#  endif

#  define HLR_HAS_COMPRESSION  1

static const char provider[] = "SZ3";

using  zconfig_t = hlr::compress::sz3::config;
using  zarray    = hlr::compress::sz3::zarray;

using hlr::compress::sz3::compress;
using hlr::compress::sz3::decompress;
using hlr::compress::sz3::get_config;
using hlr::compress::sz3::byte_size;

#else // no library available

#  define HLR_HAS_COMPRESSION  0

static const char provider[] = "none";

struct zconfig_t {};
struct zarray    {};

#endif

#else // COMPRESSOR == none

#  define HLR_HAS_COMPRESSION  0

static const char provider[] = "none";

struct zconfig_t {};
struct zarray    {};

inline zconfig_t  get_config ( double /* eps */ ) { return zconfig_t{}; }
inline size_t     byte_size  ( const zarray &   ) { return SIZE_MAX; } // ensures maximal memory size
    
#endif

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

}// namespace compress

}// namespace hlr

#endif // __HLR_UTILS_COMPRESSION_HH
