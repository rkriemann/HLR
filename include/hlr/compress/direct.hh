#ifndef __HLR_COMPRESS_DIRECT_HH
#define __HLR_COMPRESS_DIRECT_HH
//
// Project     : HLR
// Module      : compress/direct
// Description : direct compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdlib>

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/compress/ztypes.hh>

#if defined(HLR_COMPRESSOR)

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if HLR_COMPRESSOR == HLR_COMPRESSOR_FP32

#include <hlr/compress/fp32.hh>

namespace hlr { namespace compress {

static const char provider[] = "fp32";

using  zconfig_t = hlr::compress::fp32::config;
using  zarray    = hlr::compress::fp32::zarray;

using hlr::compress::fp32::compress;
using hlr::compress::fp32::decompress;
using hlr::compress::fp32::get_config;
using hlr::compress::fp32::byte_size;
using hlr::compress::fp32::compressed_size;

namespace zblas { using hlr::compress::fp32::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_FP16

#  if !defined(HLR_HAS_HALF)
#    error "half library not available"
#  endif

#include <hlr/compress/fp16.hh>

namespace hlr { namespace compress {

static const char provider[] = "fp16";

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

#include <hlr/compress/zfp.hh>

namespace hlr { namespace compress {

#if HLR_ZFP_MODE == 0
static const char provider[] = "zfp fr";
#elif HLR_ZFP_MODE == 1
static const char provider[] = "zfp fp";
#elif HLR_ZFP_MODE == 2
static const char provider[] = "zfp fa";
#endif

using  zconfig_t = hlr::compress::zfp::config;
using  zarray    = hlr::compress::zfp::zarray;

using hlr::compress::zfp::compress;
using hlr::compress::zfp::decompress;
using hlr::compress::zfp::get_config;
using hlr::compress::zfp::byte_size;
using hlr::compress::zfp::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_POSITS

#  if !defined(HLR_HAS_UNIVERSAL)
#    error "Universal library not available"
#  endif

#include <hlr/compress/posits.hh>

namespace hlr { namespace compress {

static const char provider[] = "posits";

using  zconfig_t = hlr::compress::posits::config;
using  zarray    = hlr::compress::posits::zarray;

using hlr::compress::posits::compress;
using hlr::compress::posits::decompress;
using hlr::compress::posits::get_config;
using hlr::compress::posits::byte_size;
using hlr::compress::posits::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_CFLOAT

#  if !defined(HLR_HAS_UNIVERSAL)
#    error "Universal library not available"
#  endif

#include <hlr/compress/cfloat.hh>

namespace hlr { namespace compress {

static const char provider[] = "cfloat";

using  zconfig_t = hlr::compress::cfloat::config;
using  zarray    = hlr::compress::cfloat::zarray;

using hlr::compress::cfloat::compress;
using hlr::compress::cfloat::decompress;
using hlr::compress::cfloat::get_config;
using hlr::compress::cfloat::byte_size;
using hlr::compress::cfloat::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_BLOSC

#include <hlr/compress/blosc.hh>

namespace hlr { namespace compress {

static const char provider[] = "blosc";

using  zconfig_t = hlr::compress::blosc::config;
using  zarray    = hlr::compress::blosc::zarray;

using hlr::compress::blosc::compress;
using hlr::compress::blosc::decompress;
using hlr::compress::blosc::get_config;
using hlr::compress::blosc::byte_size;
using hlr::compress::blosc::compressed_size;

namespace zblas { using hlr::compress::blosc::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_SZ

#  if !defined(HLR_HAS_SZ)
#    error "SZ library not available"
#  endif

#include <hlr/compress/sz.hh>

namespace hlr { namespace compress {

static const char provider[] = "sz";

using  zconfig_t = hlr::compress::sz::config;
using  zarray    = hlr::compress::sz::zarray;

using hlr::compress::sz::compress;
using hlr::compress::sz::decompress;
using hlr::compress::sz::get_config;
using hlr::compress::sz::byte_size;
using hlr::compress::sz::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_SZ3

#  if !defined(HLR_HAS_SZ3)
#    error "SZ3 library not available"
#  endif

#include <hlr/compress/sz3.hh>

namespace hlr { namespace compress {

static const char provider[] = "sz3";

using  zconfig_t = hlr::compress::sz3::config;
using  zarray    = hlr::compress::sz3::zarray;

using hlr::compress::sz3::compress;
using hlr::compress::sz3::decompress;
using hlr::compress::sz3::get_config;
using hlr::compress::sz3::byte_size;
using hlr::compress::sz3::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_LZ4

#  if !defined(HLR_HAS_LZ4)
#    error "LZ4 library not available"
#  endif

#include <hlr/compress/lz4.hh>

namespace hlr { namespace compress {

static const char provider[] = "lz4";

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

#include <hlr/compress/zlib.hh>

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

#include <hlr/compress/zstd.hh>

namespace hlr { namespace compress {

static const char provider[] = "zstd";

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

#include <hlr/compress/bf16.hh>

namespace hlr { namespace compress {

static const char provider[] = "bf16";

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

#include <hlr/compress/tf32.hh>

namespace hlr { namespace compress {

static const char provider[] = "tf32";

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

#include <hlr/compress/bf24.hh>

namespace hlr { namespace compress {

static const char provider[] = "bf24";

using  zconfig_t = hlr::compress::bf24::config;
using  zarray    = hlr::compress::bf24::zarray;

using hlr::compress::bf24::compress;
using hlr::compress::bf24::decompress;
using hlr::compress::bf24::get_config;
using hlr::compress::bf24::byte_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_AFL

#include <hlr/compress/afl.hh>

namespace hlr { namespace compress {

static const char provider[] = "afl";

using  zconfig_t = hlr::compress::afl::config;
using  zarray    = hlr::compress::afl::zarray;

using hlr::compress::afl::compress;
using hlr::compress::afl::decompress;
using hlr::compress::afl::get_config;
using hlr::compress::afl::byte_size;
using hlr::compress::afl::compressed_size;

namespace zblas { using hlr::compress::afl::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_AFLP

#include <hlr/compress/aflp.hh>

namespace hlr { namespace compress {

static const char provider[] = "aflp";

using  zconfig_t = hlr::compress::aflp::config;
using  zarray    = hlr::compress::aflp::zarray;

using hlr::compress::aflp::compress;
using hlr::compress::aflp::decompress;
using hlr::compress::aflp::get_config;
using hlr::compress::aflp::byte_size;
using hlr::compress::aflp::compressed_size;

namespace zblas { using hlr::compress::aflp::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_SFL

#include <hlr/compress/sfl.hh>

namespace hlr { namespace compress {

static const char provider[] = "sfl";

using  zconfig_t = hlr::compress::sfl::config;
using  zarray    = hlr::compress::sfl::zarray;

using hlr::compress::sfl::compress;
using hlr::compress::sfl::decompress;
using hlr::compress::sfl::get_config;
using hlr::compress::sfl::byte_size;
using hlr::compress::sfl::compressed_size;

namespace zblas { using hlr::compress::sfl::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_DFL

#include <hlr/compress/dfl.hh>

namespace hlr { namespace compress {

static const char provider[] = "dfl";

using  zconfig_t = hlr::compress::dfl::config;
using  zarray    = hlr::compress::dfl::zarray;

using hlr::compress::dfl::compress;
using hlr::compress::dfl::decompress;
using hlr::compress::dfl::get_config;
using hlr::compress::dfl::byte_size;
using hlr::compress::dfl::compressed_size;

namespace zblas { using hlr::compress::dfl::mulvec; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_DFL2

#include <hlr/compress/dfl2.hh>

namespace hlr { namespace compress {

static const char provider[] = "dfl2";

using  zconfig_t = hlr::compress::dfl2::config;
using  zarray    = hlr::compress::dfl2::zarray;

using hlr::compress::dfl2::compress;
using hlr::compress::dfl2::decompress;
using hlr::compress::dfl2::get_config;
using hlr::compress::dfl2::byte_size;
using hlr::compress::dfl2::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_MGARD

#include <hlr/compress/mgard.hh>

namespace hlr { namespace compress {

static const char provider[] = "mgard";

using  zconfig_t = hlr::compress::mgard::config;
using  zarray    = hlr::compress::mgard::zarray;

using hlr::compress::mgard::compress;
using hlr::compress::mgard::decompress;
using hlr::compress::mgard::get_config;
using hlr::compress::mgard::byte_size;
using hlr::compress::mgard::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_COMPRESSOR == HLR_COMPRESSOR_MP

#include <hlr/compress/mixedprec.hh>

namespace hlr { namespace compress {

static const char provider[] = "mixedprec";

using  zconfig_t = hlr::compress::mixedprec::config;
using  zarray    = hlr::compress::mixedprec::zarray;

using hlr::compress::mixedprec::compress;
using hlr::compress::mixedprec::decompress;
using hlr::compress::mixedprec::get_config;
using hlr::compress::mixedprec::byte_size;
using hlr::compress::mixedprec::compressed_size;

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif  ( HLR_COMPRESSOR == HLR_COMPRESSOR_NONE ) || ( HLR_COMPRESSOR == HLR_COMPRESSOR_MP )

//
// dummy compressor
//

#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress {

static const char provider[] = "none";

struct zconfig_t {};

struct zarray
{
    zarray ()               {}
    zarray ( const size_t ) {}
    
    byte_t *  data  ()       { return nullptr; }
    byte_t *  data  () const { return nullptr; }

    size_t    size  () const { return 0; }

    byte_t *  begin () const { return nullptr; }
    byte_t *  end   () const { return nullptr; }

    bool      empty () const { return true; }
};

template < typename value_t >
zarray
compress ( const zconfig_t &   config,
           value_t *           data,
           const size_t        dim0,
           const size_t        dim1 = 0,
           const size_t        dim2 = 0,
           const size_t        dim3 = 0 )
{
    return zarray();
}

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{}

inline zconfig_t  get_config      ( double /* eps */ ) { return zconfig_t{}; }
inline size_t     byte_size       ( const zarray &   ) { return 0; } // signals failed compression
inline size_t     compressed_size ( const zarray &   ) { return 0; }

}} // namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else

#  error "unsupported HLR_COMPRESSOR value"

#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else // HLR_COMPRESSOR not defined

#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress {

static const char provider[] = "none";

struct zconfig_t {};

struct zarray
{
    zarray ()               {}
    zarray ( const size_t ) {}
    
    byte_t *  data  ()       { return nullptr; }
    byte_t *  data  () const { return nullptr; }

    size_t    size  () const { return 0; }

    byte_t *  begin () const { return nullptr; }
    byte_t *  end   () const { return nullptr; }
};

template < typename value_t >
zarray
compress ( const zconfig_t &   config,
           value_t *           data,
           const size_t        dim0,
           const size_t        dim1 = 0,
           const size_t        dim2 = 0,
           const size_t        dim3 = 0 )
{
    return zarray();
}

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{}

inline zconfig_t  get_config      ( double /* eps */ ) { return zconfig_t{}; }
inline size_t     byte_size       ( const zarray &   ) { return 0; } // signals failed compression
inline size_t     compressed_size ( const zarray &   ) { return 0; }

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
compress ( const zconfig_t &                      config,
           const hlr::blas::tensor3< value_t > &  T )
{
    return compress< value_t >( config, T.data(), T.size(0), T.size(1), T.size(2) );
}

template < typename value_t >
zarray
compress ( const zconfig_t &                      config,
           const hlr::blas::tensor4< value_t > &  T )
{
    return compress< value_t >( config, T.data(), T.size(0), T.size(1), T.size(2), T.size(3) );
}

template < typename value_t >
zarray
compress ( const zconfig_t &                     config,
           const hlr::blas::matrix< value_t > &  M )
{
    return compress< value_t >( config, M.data(), M.nrows(), M.ncols() );
}

template < typename value_t >
zarray
compress ( const zconfig_t &                     config,
           const hlr::blas::vector< value_t > &  v )
{
    return compress< value_t >( config, v.data(), v.length() );
}

template < typename value_t >
void
decompress ( const zarray &                   zdata,
             hlr::blas::tensor3< value_t > &  T )
{
    return decompress< value_t >( zdata, T.data(), T.size(0), T.size(1), T.size(2) );
}

template < typename value_t >
void
decompress ( const zarray &                   zdata,
             hlr::blas::tensor4< value_t > &  T )
{
    return decompress< value_t >( zdata, T.data(), T.size(0), T.size(1), T.size(2), T.size(3) );
}

template < typename value_t >
void
decompress ( const zarray &                  zdata,
             hlr::blas::matrix< value_t > &  M )
{
    return decompress< value_t >( zdata, M.data(), M.nrows(), M.ncols() );
}

template < typename value_t >
void
decompress ( const zarray &                  zdata,
             hlr::blas::vector< value_t > &  v )
{
    return decompress< value_t >( zdata, v.data(), v.length() );
}

//
// return compression configuration based on accuracy and matrix
//
template < typename value_t >
zconfig_t
get_config ( const accuracy &                 acc,
             const blas::matrix< value_t > &  M )
{
    // in case ZFP is not used
    #if not defined( HLR_ZFP_MODE )
    #  define HLR_ZFP_MODE 0
    #endif
    
    //
    // Assumption: compression with relative error per coefficient, i.e. |a_ij - ã_ij| ≤ ε|a_ij|
    // Then also |M-~M|_F ≤ ε|M|_F holds.
    //
    // TODO: adjust for |·|_2
    //
    
    auto  tol = acc.abs_eps();

    if constexpr (( HLR_COMPRESSOR == HLR_COMPRESSOR_ZFP ) && ( HLR_ZFP_MODE == 2 ))
    {
        //
        // ZFP with fixed accuracy uses absolute error
        //
    
        if ( acc.abs_eps() != 0 )
        {
            switch ( acc.norm_mode() )
            {
                case  Hpro::spectral_norm  : tol = acc.abs_eps(); break;
                case  Hpro::frobenius_norm : tol = acc.abs_eps(); break;
                default :
                    HLR_ERROR( "unsupported norm mode" );
            }// switch
        }// if
        else if ( acc.rel_eps() != 0 )
        {
            switch ( acc.norm_mode() )
            {
                case  Hpro::spectral_norm  : tol = acc.rel_eps() * blas::norm_2( M ); break;
                case  Hpro::frobenius_norm : tol = acc.rel_eps() * blas::norm_F( M ); break;
                default :
                    HLR_ERROR( "unsupported norm mode" );
            }// switch
        }// if
        else
            HLR_ERROR( "zero error" );

        // compensate for dimension growth (why ???)
        tol /= std::max( M.nrows(), M.ncols() );
    
    }// if
    else
    {
        //
        // assuming relative error within compressor
        //
    
        if ( acc.abs_eps() != 0 )
        {
            switch ( acc.norm_mode() )
            {
                case  Hpro::spectral_norm  : tol = acc.abs_eps() / blas::norm_2( M ); break;
                case  Hpro::frobenius_norm : tol = acc.abs_eps() / blas::norm_F( M ); break;
                default :
                    HLR_ERROR( "unsupported norm mode" );
            }// switch
        }// if
        else if ( acc.rel_eps() != 0 )
        {
            switch ( acc.norm_mode() )
            {
                case  Hpro::spectral_norm  : tol = acc.rel_eps(); break;
                case  Hpro::frobenius_norm : tol = acc.rel_eps(); break;
                default :
                    HLR_ERROR( "unsupported norm mode" );
            }// switch
        }// if
        else
            HLR_ERROR( "zero error" );
    }// else
    
    return get_config( tol );
}

//
// test compression/decompression
//
inline
void
test ()
{
    srand48( 1 );
    
    auto    rand  = [] () { return 2.0 * drand48() - 1.0; };
    size_t  n     = 12;
    auto    M     = blas::matrix< double >( n, n );

    blas::fill_fn( M, rand );
    
    auto    zconf = get_config( relative_prec( Hpro::frobenius_norm, 1e-4 ), M );
    auto    zM    = compress( zconf, M );
    auto    D     = blas::matrix< double >( n, n );

    decompress( zM, D );

    blas::add( -1.0, M, D );

    std::cout << blas::norm_F( D ) << std::endl
              << blas::norm_F( D ) / blas::norm_F( M ) << std::endl;
}

}}// namespace hlr::compress

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

//
// deactivate if requested by user
//

#if HLR_USE_ZBLAS == 0 && defined(HLR_HAS_ZBLAS_DIRECT)
#  undef HLR_HAS_ZBLAS_DIRECT
#endif

#endif // __HLR_UTILS_DIRECT_HH
