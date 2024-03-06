#ifndef __HLR_UTILS_DETAIL_MIXEDPREC_HH
#define __HLR_UTILS_DETAIL_MIXEDPREC_HH
//
// Project     : HLR
// Module      : utils/detail/mixedprec
// Description : functions for mixed precision representation of LR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstdint>

#include <hlr/arith/blas.hh>

////////////////////////////////////////////////////////////
//
// compression using mixed precision representation
// of lowrank matrices using three different float
// precisions:
//
//    double + single + half
//
// with half either float16 or bfloat16 (depending on compiler
// support)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace mixedprec {

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
// GCC >= 12 and Clang >= 15 support _Float16
// check some of the default defines for this
//
#if defined(__FLT16_EPSILON__)
#  define HLR_HAS_FLOAT16  1
#else
#  define HLR_HAS_FLOAT16  0
#endif

//
// software floating point implementation for BF16: 1+8+7
//
struct bf16
{
    unsigned short  data;
    
public:
    bf16 () : data{ 0 }      {}
    bf16 ( const float   f ) { *this = f; }
    bf16 ( const double  f ) { *this = f; }
    
    // cast to float/double
    operator float () const
    {
        const uint32_t  ival = data << 16;

        return * reinterpret_cast< const float * >( & ival );
    }
    
    operator double () const { return float(*this); }
    
    // cast to bf16
    bf16 &
    operator = ( const float  val )
    {
        data = (* reinterpret_cast< const uint32_t * >( & val ) ) >> 16;
        
        return *this;
    }

    bf16 &
    operator = ( const double  val )
    {
        return *this = float(val);
    }

    bf16 operator + ( bf16  f ) { return float(*this) + float(f); }
    bf16 operator - ( bf16  f ) { return float(*this) - float(f); }
    bf16 operator * ( bf16  f ) { return float(*this) * float(f); }
    bf16 operator / ( bf16  f ) { return float(*this) / float(f); }
};

//
// floating point types
// - assuming double as base type!!!
//

// finest precision
using             mptype1_t = double;

// middle precision
using             mptype2_t = float;
constexpr double  mpprec2   = 6.0e-8;

// coarsest precision
#if HLR_HAS_FLOAT16
using             mptype3_t = _Float16;
constexpr double  mpprec3   = 4.9e-4;
#else
using             mptype3_t = bf16;
constexpr double  mpprec3   = 3.9e-3;
#endif

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
    // determine corresponding parts for MP1, MP2, MP3
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

    const uint32_t  n_mp3 = test_prec( mpprec3 );
    const uint32_t  n_mp2 = test_prec( mpprec2 );
    const uint32_t  n_mp1 = i+1;                   // remaining singular values
    size_t          s     = 0;

    // std::cout << n_mp3 << " / " << n_mp2 << " / " << n_mp1 << std::endl;
    
    HLR_ASSERT( n_mp1 >= 0 );
    HLR_ASSERT( n_mp3 + n_mp2 + n_mp1 == rank );

    //
    // copy into storage
    //

    const size_t  zsize = ( 3 * sizeof(uint32_t) +
                            sizeof(mptype1_t) * n * n_mp1 +
                            sizeof(mptype2_t) * n * n_mp2 +
                            sizeof(mptype3_t) * n * n_mp3 );
    zarray        zdata( zsize );

    reinterpret_cast< uint32_t * >( zdata.data() )[0] = n_mp1;
    reinterpret_cast< uint32_t * >( zdata.data() )[1] = n_mp2;
    reinterpret_cast< uint32_t * >( zdata.data() )[2] = n_mp3;

    uint32_t  k   = 0;
    size_t    pos = 3 * sizeof(uint32_t);
        
    {
        auto    zptr = reinterpret_cast< mptype1_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp1; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = mptype1_t( U(i,k) );
        }// for
        pos += n_mp1 * n * sizeof(mptype1_t);
    }

    {
        auto    zptr = reinterpret_cast< mptype2_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp2; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = mptype2_t( U(i,k) );
        }// for
        pos += n_mp2 * n * sizeof(mptype2_t);
    }

    {
        auto    zptr = reinterpret_cast< mptype3_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp3; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = mptype3_t( U(i,k) );
        }// for
        pos += n_mp3 * n * sizeof(mptype3_t);
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

    // if constexpr ( Hpro::is_complex_type_v< value_t > )
    // {
    //     uint32_t    k     = 0;
    //     size_t  pos_U = 0;
    //     size_t  pos_V = 0;

    //     _mpdata.U1 = std::vector< mptype1_t >( 2 * n_mp1 * oU.nrows() );
    //     _mpdata.V1 = std::vector< mptype1_t >( 2 * n_mp1 * oV.nrows() );

    //     for ( size_t  i = 0; i < _mpdata.U1.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U1[i]   = mptype1_t( std::real( u_i ) );
    //         _mpdata.U1[i+1] = mptype1_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V1.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V1[i]   = mptype1_t( std::real( v_i ) );
    //         _mpdata.V1[i+1] = mptype1_t( std::imag( v_i ) );
    //     }// for

    //     _mpdata.U2 = std::vector< mptype2_t >( 2 * n_mp2 * oU.nrows() );
    //     _mpdata.V2 = std::vector< mptype2_t >( 2 * n_mp2 * oV.nrows() );

    //     for ( size_t  i = 0; i < _mpdata.U2.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U2[i]   = mptype2_t( std::real( u_i ) );
    //         _mpdata.U2[i+1] = mptype2_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V2.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V2[i]   = mptype2_t( std::real( v_i ) );
    //         _mpdata.V2[i+1] = mptype2_t( std::imag( v_i ) );
    //     }// for


    //     _mpdata.U3 = std::vector< mptype3_t >( 2 * n_mp3 * oU.nrows() );
    //     _mpdata.V3 = std::vector< mptype3_t >( 2 * n_mp3 * oV.nrows() );
            
    //     for ( size_t  i = 0; i < _mpdata.U3.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U3[i]   = mptype3_t( std::real( u_i ) );
    //         _mpdata.U3[i+1] = mptype3_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V3.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V3[i]   = mptype3_t( std::real( v_i ) );
    //         _mpdata.V3[i+1] = mptype3_t( std::imag( v_i ) );
    //     }// for
    // }// if
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
    const size_t    n     = U.nrows();
    const uint32_t  rank  = U.ncols();
    const uint32_t  n_mp1 = reinterpret_cast< const uint32_t * >( zdata.data() )[0];
    const uint32_t  n_mp2 = reinterpret_cast< const uint32_t * >( zdata.data() )[1];
    const uint32_t  n_mp3 = reinterpret_cast< const uint32_t * >( zdata.data() )[2];
    size_t          pos   = 3 * sizeof(uint32_t);
    uint32_t        k     = 0;
    
    if ( n_mp1 > 0 )
    {
        auto    zptr = reinterpret_cast< const mptype1_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp1; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_mp1 * n * sizeof(mptype1_t);
    }// if

    if ( n_mp2 > 0 )
    {
        auto    zptr = reinterpret_cast< const mptype2_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp2; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_mp2 * n * sizeof(mptype2_t);
    }// if

    if ( n_mp3 > 0 )
    {
        auto    zptr = reinterpret_cast< const mptype3_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_mp3; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_mp3 * n * sizeof(mptype3_t);
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
    
    // size_t          pos   = 0;
    // const uint32_t  n_mp1 = _mpdata.U1.size() / (2 * dU.nrows());
    // const uint32_t  n_mp2 = _mpdata.U2.size() / (2 * dU.nrows());
    // const uint32_t  n_mp3 = _mpdata.U3.size() / (2 * dU.nrows());
                
    // for ( uint32_t  k1 = 0; k1 < n_mp1; ++k1, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U1[ pos ], _mpdata.U1[ pos+1 ] );
    // }// for

    // pos = 0;
    // for ( uint32_t  k2 = 0; k2 < n_mp2; ++k2, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U2[ pos ], _mpdata.U2[ pos+1 ] );
    // }// for
                
    // pos = 0;
    // for ( uint32_t  k3 = 0; k3 < n_mp3; ++k3, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U3[ pos ], _mpdata.U3[ pos+1 ] );
    // }// for
}

}}}// namespace hlr::compress::mixedprec

#endif // __HLR_UTILS_DETAIL_MIXEDPREC_HH
