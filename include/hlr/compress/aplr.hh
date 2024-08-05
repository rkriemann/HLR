#ifndef __HLR_COMPRESS_APLR_HH
#define __HLR_COMPRESS_APLR_HH
//
// Project     : HLR
// Module      : compress/aplr
// Description : aplr compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/compress/ztypes.hh>

//
// deactivate by default
//
#if defined(HLR_APLR_COMPRESSOR)

#  define HLR_HAS_APLR_COMPRESSION  1

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_MP

#include <hlr/compress/mixedprec.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::mixedprec::zarray;
using hlr::compress::mixedprec::compress_lr;
using hlr::compress::mixedprec::decompress_lr;
using hlr::compress::mixedprec::byte_size;
using hlr::compress::mixedprec::compressed_size;

static const char provider[] = "mixedprec";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::mixedprec::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_MP2

#include <hlr/compress/mixedprec2.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::mixedprec2::zarray;
using hlr::compress::mixedprec2::compress_lr;
using hlr::compress::mixedprec2::decompress_lr;
using hlr::compress::mixedprec2::byte_size;
using hlr::compress::mixedprec2::compressed_size;

static const char provider[] = "mixedprec2";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::mixedprec2::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_ZFP

#include <hlr/compress/zfp.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::zfp::zarray;
using hlr::compress::zfp::compress_lr;
using hlr::compress::zfp::decompress_lr;
using hlr::compress::zfp::byte_size;
using hlr::compress::zfp::compressed_size;

static const char provider[] = "zfp";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_SZ3

#include <hlr/compress/sz3.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::sz3::zarray;
using hlr::compress::sz3::compress_lr;
using hlr::compress::sz3::decompress_lr;
using hlr::compress::sz3::byte_size;

static const char provider[] = "sz3";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_MGARD

#include <hlr/compress/mgard.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::mgard::zarray;
using hlr::compress::mgard::compress_lr;
using hlr::compress::mgard::decompress_lr;
using hlr::compress::mgard::byte_size;

static const char provider[] = "mgard";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_BFL

#include <hlr/compress/bfl.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::bfl::zarray;
using hlr::compress::bfl::compress_lr;
using hlr::compress::bfl::decompress_lr;
using hlr::compress::bfl::byte_size;
using hlr::compress::bfl::compressed_size;

static const char provider[] = "bfl";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::bfl::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_DFL

#include <hlr/compress/dfl.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::dfl::zarray;
using hlr::compress::dfl::compress_lr;
using hlr::compress::dfl::decompress_lr;
using hlr::compress::dfl::byte_size;
using hlr::compress::dfl::compressed_size;

static const char provider[] = "dfl";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::dfl::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_DFL2

#include <hlr/compress/dfl2.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::dfl2::zarray;
using hlr::compress::dfl2::compress_lr;
using hlr::compress::dfl2::decompress_lr;
using hlr::compress::dfl2::byte_size;
using hlr::compress::dfl2::compressed_size;

static const char provider[] = "dfl2";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_AFLP

#include <hlr/compress/aflp.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::aflp::zarray;
using hlr::compress::aflp::compress_lr;
using hlr::compress::aflp::decompress_lr;
using hlr::compress::aflp::byte_size;
using hlr::compress::aflp::compressed_size;

static const char provider[] = "aflp";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::aflp::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_AFL

#include <hlr/compress/afl.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::afl::zarray;
using hlr::compress::afl::compress_lr;
using hlr::compress::afl::decompress_lr;
using hlr::compress::afl::byte_size;
using hlr::compress::afl::compressed_size;

static const char provider[] = "afl";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_POSITS

#include <hlr/compress/posits.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::posits::zarray;
using hlr::compress::posits::compress_lr;
using hlr::compress::posits::decompress_lr;
using hlr::compress::posits::byte_size;
using hlr::compress::posits::compressed_size;

static const char provider[] = "posits";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_BLOSC

#include <hlr/compress/blosc.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::blosc::zarray;
using hlr::compress::blosc::compress_lr;
using hlr::compress::blosc::decompress_lr;
using hlr::compress::blosc::byte_size;
using hlr::compress::blosc::compressed_size;

static const char provider[] = "blosc";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::blosc::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_APLR_COMPRESSOR == HLR_COMPRESSOR_NONE

#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress { namespace aplr {

struct zconfig_t {};

struct zarray
{
    zarray ()               {}
    zarray ( const size_t ) {}
    
    byte_t *  data   ()       { return nullptr; }
    byte_t *  data   () const { return nullptr; }

    size_t    size   () const           { return 0; }
    void      resize ( const size_t n ) {}

    byte_t *  begin  () const { return nullptr; }
    byte_t *  end    () const { return nullptr; }
};

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > & , const blas::vector< Hpro::real_type_t< value_t > > & )
{
    return zarray();
}

template < typename value_t >
void
decompress_lr ( const zarray & , blas::matrix< value_t > & )
{}

inline size_t  byte_size       ( const zarray & ) { return 0; } // signals failure
inline size_t  compressed_size ( const zarray & ) { return 0; }

static const char provider[] = "none";

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else

#  define HLR_HAS_APLR_COMPRESSION  1

#include <hlr/compress/mixedprec.hh>

namespace hlr { namespace compress { namespace aplr {

using zarray = hlr::compress::mixedprec::zarray;
using hlr::compress::mixedprec::compress_lr;
using hlr::compress::mixedprec::decompress_lr;
using hlr::compress::mixedprec::byte_size;
using hlr::compress::mixedprec::compressed_size;

static const char provider[] = "mixedprec";

namespace zblas
{

template < typename value_t >
void
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    hlr::compress::mixedprec::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::aplr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#endif // compare(HLR_APLR_COMPRESSOR)

//
// deactivate if requested by user
//

#if HLR_USE_ZBLAS == 0 && defined(HLR_HAS_ZBLAS_APLR)
#  undef HLR_HAS_ZBLAS_APLR
#endif

#endif // defined(HLR_APLR_COMPRESSOR)

#endif // __HLR_COMPRESS_APLR_HH
