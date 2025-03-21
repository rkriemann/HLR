#ifndef __HLR_COMPRESS_VALR_HH
#define __HLR_COMPRESS_VALR_HH
//
// Project     : HLR
// Module      : compress/valr
// Description : valr compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/compress/ztypes.hh>

//
// deactivate by default
//
#if defined(HLR_VALR_COMPRESSOR)

#  define HLR_HAS_VALR_COMPRESSION  1

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_MP

#include <hlr/compress/mixedprec.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_MP2

#include <hlr/compress/mixedprec2.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_ZFP

#include <hlr/compress/zfp.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::zfp::zarray;
using hlr::compress::zfp::compress_lr;
using hlr::compress::zfp::decompress_lr;
using hlr::compress::zfp::byte_size;
using hlr::compress::zfp::compressed_size;

#if HLR_ZFP_MODE == 0
static const char provider[] = "zfp fr";
#elif HLR_ZFP_MODE == 1
static const char provider[] = "zfp fp";
#elif HLR_ZFP_MODE == 2
static const char provider[] = "zfp fa";
#endif

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_SZ3

#include <hlr/compress/sz3.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::sz3::zarray;
using hlr::compress::sz3::compress_lr;
using hlr::compress::sz3::decompress_lr;
using hlr::compress::sz3::byte_size;
using hlr::compress::sz3::compressed_size;

static const char provider[] = "sz3";

}}}// namespace hlr::compress::valr

// //////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////

// #elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_MGARD

// #include <hlr/compress/mgard.hh>

// namespace hlr { namespace compress { namespace valr {

// using zarray = hlr::compress::mgard::zarray;
// using hlr::compress::mgard::compress_lr;
// using hlr::compress::mgard::decompress_lr;
// using hlr::compress::mgard::byte_size;

// static const char provider[] = "mgard";

// }}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_SFL

#include <hlr/compress/sfl.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::sfl::zarray;
using hlr::compress::sfl::compress_lr;
using hlr::compress::sfl::decompress_lr;
using hlr::compress::sfl::byte_size;
using hlr::compress::sfl::compressed_size;

static const char provider[] = "sfl";

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
    hlr::compress::sfl::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_DFL

#include <hlr/compress/dfl.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_DFL2

#include <hlr/compress/dfl2.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::dfl2::zarray;
using hlr::compress::dfl2::compress_lr;
using hlr::compress::dfl2::decompress_lr;
using hlr::compress::dfl2::byte_size;
using hlr::compress::dfl2::compressed_size;

static const char provider[] = "dfl2";

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_AFLP

#include <hlr/compress/aflp.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_AFL

#include <hlr/compress/afl.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::afl::zarray;
using hlr::compress::afl::compress_lr;
using hlr::compress::afl::decompress_lr;
using hlr::compress::afl::byte_size;
using hlr::compress::afl::compressed_size;

static const char provider[] = "afl";

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
    hlr::compress::afl::mulvec_lr( nrows, ncols, op_A, alpha, zA, x, y );
}

}// namespace zblas

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_POSITS

#include <hlr/compress/posits.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::posits::zarray;
using hlr::compress::posits::compress_lr;
using hlr::compress::posits::decompress_lr;
using hlr::compress::posits::byte_size;
using hlr::compress::posits::compressed_size;

static const char provider[] = "posits";

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_CFLOAT

#include <hlr/compress/cfloat.hh>

namespace hlr { namespace compress { namespace valr {

using zarray = hlr::compress::cfloat::zarray;
using hlr::compress::cfloat::compress_lr;
using hlr::compress::cfloat::decompress_lr;
using hlr::compress::cfloat::byte_size;
using hlr::compress::cfloat::compressed_size;

static const char provider[] = "cfloat";

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_BLOSC

#include <hlr/compress/blosc.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#elif HLR_VALR_COMPRESSOR == HLR_COMPRESSOR_NONE

#include <hlr/compress/byte_n.hh>

namespace hlr { namespace compress { namespace valr {

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

    bool      empty () const { return true; }
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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#else

#  define HLR_HAS_VALR_COMPRESSION  1

#include <hlr/compress/mixedprec.hh>

namespace hlr { namespace compress { namespace valr {

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

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#endif // compare(HLR_VALR_COMPRESSOR)

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace valr {

template < typename value_t >
blas::vector< value_t >
get_tolerances ( const accuracy &                 acc,
                 const blas::vector< value_t > &  sv )
{
    // in case ZFP is not used
    #if not defined( HLR_ZFP_MODE )
    #  define HLR_ZFP_MODE 0
    #endif
    
    auto  tol = acc.abs_eps();

    if ( acc.abs_eps() != 0 )
    {
        tol = acc.abs_eps();
    }// if
    else if ( acc.rel_eps() != 0 )
    {
        // use relative error: δ = ε |M|
        auto  norm = value_t(0);

        switch ( acc.norm_mode() )
        {
            case  Hpro::spectral_norm  : norm = sv(0); break;
                
            case  Hpro::frobenius_norm :
            {
                for ( uint  i = 0; i < sv.length(); ++i )
                    norm += math::square( sv(i) );

                norm = math::sqrt( norm );
            }// case
            break;

            default : HLR_ERROR( "unsupported norm mode" );
        }// switch
        
        tol = acc.rel_eps() * norm;
    }// if

    //
    // correct for ZFP fixed accuracy mode
    //

    if constexpr (( HLR_COMPRESSOR == HLR_COMPRESSOR_ZFP ) && ( HLR_ZFP_MODE == 2 ))
    {
        tol /= sv.length();
    }// if
    
    //
    // adjust tolerance for additional error due to VALR,
    // e.g., look for ε, such that Σ 1/σ_i ε² + (1+2k)ε = δ
    //
    
    {
        auto  k = sv.length();
        auto  δ = tol;
        auto  Σ = value_t(0);

        for ( uint  i = 0; i < sv.length(); ++i )
            Σ += 1.0 / sv(i);

        // std::cout << "δ: " << δ << " / sum: " << Σ << " / err: " << δ*(1 + 2*k) + δ*δ * Σ;

        // //
        // // using both terms in the error formula
        // //
        
        // // solve via p/q since more robust than direct approach
        // auto  p = ( 1 + 2*k ) / Σ;
        // auto  q = δ / Σ;
        // auto  t = p*p / 4 - q;

        // if ( t < 0 ) t = 0;
            
        // auto  d1 = std::abs( p/2 + std::sqrt( t ) );
        // auto  d2 = std::abs( p/2 - std::sqrt( t ) );
        
        // tol = std::min( std::abs( d1 ), std::abs( d2 ) ); // std::max( δ, δ * δ * Σ ) / ( 3 * k ); // "/ 2"

        //
        // using just sum of sing. values.
        //
        
        auto  p = 1 / Σ;
        auto  q = δ / Σ;
        auto  t = p*p / 4 - q;

        if ( t < 0 ) t = 0;
            
        auto  d1 = std::abs( p/2 + std::sqrt( t ) );
        auto  d2 = std::abs( p/2 - std::sqrt( t ) );
        
        tol = std::min( std::abs( d1 ), std::abs( d2 ) );

        // std::cout << " / tol: " << tol << ", " << tol * ( 1 + 2*k + tol * Σ );
        // std::cout << std::endl;
    }
    
    //
    // we aim for σ_i ≈ δ u_i and hence choose u_i = δ / σ_i
    //
    
    auto        sv_tol = blas::copy( sv );
    const auto  k      = sv.length();

    for ( uint  l = 0; l < k; ++l )
        sv_tol(l) = tol / sv(l);

    return  sv_tol;
}

}}}// namespace hlr::compress::valr

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

//
// deactivate if requested by user
//

#if HLR_USE_ZBLAS == 0 && defined(HLR_HAS_ZBLAS_VALR)
#  undef HLR_HAS_ZBLAS_VALR
#endif

#endif // defined(HLR_VALR_COMPRESSOR)

#endif // __HLR_COMPRESS_VALR_HH
