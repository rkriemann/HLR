#ifndef __HLR_UTILS_DETAIL_MIXEDPREC2_HH
#define __HLR_UTILS_DETAIL_MIXEDPREC2_HH
//
// Project     : HLR
// Module      : compress/mixedprec2
// Description : functions for mixed precision representation of LR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstdint>

#include <hlr/arith/blas.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_APLR

////////////////////////////////////////////////////////////
//
// compression using mixed precision representation
// of lowrank matrices using double and float.
//
// Arithmetic is also implemented with native formats.
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace mixedprec2 {

using byte_t = uint8_t;

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v ) { return v.size(); }

//////////////////////////////////////////////////////////////////////////////////////
//
// different floating point types for mixed precision
//
//////////////////////////////////////////////////////////////////////////////////////

//
// floating point types
// - assuming double as base type!!!
//

// finest precision
using             fp64_t    = double;

// middle precision
using             fp32_t    = float;
constexpr double  fp32_prec = 6.0e-8;

//////////////////////////////////////////////////////////////////////////////////////
//
// compression
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const hlr::blas::matrix< value_t > &                       U,
              const hlr::blas::vector< Hpro::real_type_t< value_t > > &  S );

template <>
inline
zarray
compress_lr< float > ( const hlr::blas::matrix< float > &  U,
                       const hlr::blas::vector< float > &  S )
{
    HLR_ERROR( "not supported" );
}

template <>
inline
zarray
compress_lr< double > ( const hlr::blas::matrix< double > &  U,
                        const hlr::blas::vector< double > &  S )
{
    //
    // determine corresponding parts for FP64/FP32
    //

    const size_t    n    = U.nrows();
    const uint32_t  rank = U.ncols();
    int             i    = rank-1;

    auto  test_prec = [&i,&S] ( double  u )
    {
        uint32_t  nprec = 0;
            
        while ( i >= 0 )
        {
            // test u ≤ tol / σ_i = S_i
            if ( u <= S(i) ) nprec++; 
            else             break;
            --i;
        }// while

        return nprec;
    };

    const uint32_t  n_fp32 = test_prec( fp32_prec );
    const uint32_t  n_fp64 = i+1; // remaining singular values
    size_t          s      = 0;

    HLR_ASSERT( n_fp64 >= 0 );
    HLR_ASSERT( n_fp32 + n_fp64 == rank );

    //
    // copy into storage
    //

    const size_t  zsize = ( 2 * sizeof(uint32_t) +
                            sizeof(fp64_t) * n * n_fp64 +
                            sizeof(fp32_t) * n * n_fp32 );
    zarray        zdata( zsize );

    reinterpret_cast< uint32_t * >( zdata.data() )[0] = n_fp64;
    reinterpret_cast< uint32_t * >( zdata.data() )[1] = n_fp32;

    uint32_t  k   = 0;
    size_t    pos = 2 * sizeof(uint32_t);
        
    {
        auto    zptr = reinterpret_cast< fp64_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp64; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = fp64_t( U(i,k) );
        }// for
        pos += n_fp64 * n * sizeof(fp64_t);
    }

    {
        auto    zptr = reinterpret_cast< fp32_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp32; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = fp32_t( U(i,k) );
        }// for
        pos += n_fp32 * n * sizeof(fp32_t);
    }

    HLR_ASSERT( k == rank );

    return zdata;
}

template <>
inline
zarray
compress_lr< std::complex< float > > ( const hlr::blas::matrix< std::complex< float > > &  U,
                                       const hlr::blas::vector< float > &                  S )
{
    HLR_ERROR( "not supported" );
}

template <>
inline
zarray
compress_lr< std::complex< double > > ( const hlr::blas::matrix< std::complex< double > > &  U,
                                        const hlr::blas::vector< double > &                  S )
{
    HLR_ERROR( "todo" );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// compression
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
decompress_lr ( const zarray &                  zdata,
                hlr::blas::matrix< value_t > &  U );

template <>
inline
void
decompress_lr< float > ( const zarray &                zdata,
                         hlr::blas::matrix< float > &  U )
{
    HLR_ERROR( "not supported" );
}

template <>
inline
void
decompress_lr< double > ( const zarray &                 zdata,
                          hlr::blas::matrix< double > &  U )
{
    const size_t    nrows  = U.nrows();
    const uint32_t  rank   = U.ncols();
    const uint32_t  n_fp64 = reinterpret_cast< const uint32_t * >( zdata.data() )[0];
    const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zdata.data() )[1];
    size_t          pos    = 2 * sizeof(uint32_t);
    uint32_t        k      = 0;
    
    if ( n_fp64 > 0 )
    {
        auto    zptr = reinterpret_cast< const fp64_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp64; ++l, ++k )
        {
            for ( size_t  i = 0; i < nrows; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_fp64 * nrows * sizeof(fp64_t);
    }// if

    if ( n_fp32 > 0 )
    {
        auto    zptr = reinterpret_cast< const fp32_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp32; ++l, ++k )
        {
            for ( size_t  i = 0; i < nrows; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_fp32 * nrows * sizeof(fp32_t);
    }// if

    HLR_ASSERT( k == rank );
}

template <>
inline
void
decompress_lr< std::complex< float > > ( const zarray &                                zdata,
                                         hlr::blas::matrix< std::complex< float > > &  U )
{
    HLR_ERROR( "not supported" );
}

template <>
inline
void
decompress_lr< std::complex< double > > ( const zarray &                                zdata,
                                         hlr::blas::matrix< std::complex< double > > &  U )
{
    HLR_ERROR( "todo" );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// BLAS
//
//////////////////////////////////////////////////////////////////////////////////////

namespace
{

template < typename value_t,
           typename storage_t >
void
mulvec ( const size_t       nrows,
         const size_t       ncols,
         const matop_t      op_A,
         const value_t      alpha,
         const storage_t *  zA,
         const value_t *    x,
         value_t *          y )
{
    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = x[j];
                
                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                    y[i] += value_t(zA[pos]) * x_j;
            }// for
        }// case
        break;
        
        case  apply_adjoint :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                value_t  y_j = value_t(0);
                
                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                    y_j += value_t(zA[pos]) * x[i];

                y[j] += y_j;
            }// for
        }// case
        break;

        default:
            HLR_ERROR( "TODO" );
    }// switch
}

}// namespace anonymous

template < typename value_t >
void
mulvec_lr ( const size_t     nrows,
            const size_t     ncols,
            const matop_t    op_A,
            const value_t    alpha,
            const zarray &   zA,
            const value_t *  x,
            value_t *        y )
{
    const uint32_t  n_fp64 = reinterpret_cast< const uint32_t * >( zA.data() )[0];
    const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zA.data() )[1];
    size_t          zpos   = 2 * sizeof(uint32_t);
    size_t          pos    = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            if ( n_fp64 > 0 )
            {
                auto  zptr = reinterpret_cast< const fp64_t * >( zA.data() + zpos );
                
                if constexpr ( std::is_same_v< fp64_t, value_t > )
                {
                    blas::gemv( 'N', nrows, n_fp64, alpha, zptr, nrows, x + pos, 1, fp64_t(1), y, 1 );
                }// if
                else
                {
                    mulvec< value_t, fp64_t >( nrows, n_fp64, op_A, alpha, zptr, x + pos, y );
                }// else

                zpos += n_fp64 * nrows * sizeof(fp64_t);
                pos  += n_fp64;
            }// if

            if ( n_fp32 > 0 )
            {
                auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                if constexpr ( std::is_same_v< fp64_t, value_t > )
                {
                    auto  tx = blas::vector< fp32_t >( n_fp32 );
                    auto  ty = blas::vector< fp32_t >( nrows );

                    for ( size_t  i = 0; i < n_fp32; ++i )
                        tx(i) = *(x + pos + i);
                    
                    blas::gemv( 'N', nrows, n_fp32, alpha, zptr, nrows, tx.data(), 1, fp32_t(0), ty.data(), 1 );

                    for ( size_t  i = 0; i < nrows; ++i )
                        y[i] += ty(i);
                }// if
                else if constexpr ( std::is_same_v< fp32_t, value_t > )
                {
                    blas::gemv( 'N', nrows, n_fp32, alpha, zptr, nrows, x + pos, 1, fp32_t(1), y, 1 );
                }// if
                else
                {
                    mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x + pos, y );
                }// else
                
                zpos += n_fp32 * nrows * sizeof(fp32_t);
                pos  += n_fp32;
            }// if
        }// case
        break;
        
        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            if ( n_fp64 > 0 )
            {
                auto  zptr = reinterpret_cast< const fp64_t * >( zA.data() + zpos );
                
                mulvec< value_t, fp64_t >( nrows, n_fp64, op_A, alpha, zptr, x, y + pos );
                zpos += n_fp64 * nrows * sizeof(fp64_t);
                pos  += n_fp64;
            }// if

            if ( n_fp32 > 0 )
            {
                auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x, y + pos );
                zpos += n_fp32 * nrows * sizeof(fp32_t);
                pos  += n_fp32;
            }// if
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::mixedprec2

#endif // __HLR_UTILS_DETAIL_MIXEDPREC2_HH
