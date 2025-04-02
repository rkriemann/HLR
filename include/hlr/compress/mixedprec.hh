#ifndef __HLR_UTILS_DETAIL_MIXEDPREC_HH
#define __HLR_UTILS_DETAIL_MIXEDPREC_HH
//
// Project     : HLR
// Module      : compress/mixedprec
// Description : functions for mixed precision representation of LR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdint>

#include <hlr/arith/blas.hh>
#include <hlr/compress/byte_n.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_APLR

////////////////////////////////////////////////////////////
//
// compression using mixed precision representation
// of lowrank matrices using three different float
// precisions:
//
//    double + single + half
//
// with half either float16 (depending on compiler support)
// or bfloat16 (preferred)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace mixedprec {

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 0, std::ceil( -std::log2( eps ) ) ); }
// inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ) + 1; }

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v   ) { return v.size(); }
inline config  get_config      ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

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
#  define HLR_HAS_FLOAT16  0 // prefer BF16 anyway because much faster
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
using             fp64_t         = double;
constexpr byte_t  fp64_prec_bits = 53;

// middle precision
using             fp32_t         = float;
constexpr double  fp32_prec      = 6.0e-8;
constexpr byte_t  fp32_prec_bits = 24;

// coarsest precision
#if HLR_HAS_FLOAT16
using             fp16_t         = _Float16;
constexpr double  fp16_prec      = 4.9e-4;
constexpr byte_t  fp16_prec_bits = 11;
#else
using             fp16_t         = bf16;
constexpr double  fp16_prec      = 3.9e-3;
constexpr byte_t  fp16_prec_bits = 8;
#endif

//
// floating point data
//
template < typename real_t > struct prec_bits_s {};

template <> struct prec_bits_s< float >  { constexpr static byte_t  value = 24; };
template <> struct prec_bits_s< double > { constexpr static byte_t  value = 53; };

template < typename real_t > inline constexpr byte_t prec_bits_v = prec_bits_s< real_t >::value;

////////////////////////////////////////////////////////////////////////////////
//
// compression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    //
    // precision defines format to use, e.g., FP64/32/16
    // exponent range is assumed to be sufficient (NOT CHECKED!!!)
    //
    
    const size_t  nsize     = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    prec_bits = std::min< byte_t >( prec_bits_v< real_t >, config.bitrate );
    
    if ( prec_bits <= fp16_prec_bits )
    {
        const auto  nbytes = 2 + nsize * 2;
        auto        zdata  = std::vector< byte_t >( nbytes );

        zdata[0] = 1;

        auto  ptr = reinterpret_cast< fp16_t * >( zdata.data() + 2 );

        for ( size_t  i = 0; i < nsize; ++i )
            *(ptr++) = fp16_t( data[i] );

        return zdata;
    }// if
    else if ( prec_bits <= fp32_prec_bits )
    {
        const auto  nbytes = 4 + nsize * 4;
        auto        zdata  = std::vector< byte_t >( nbytes );

        zdata[0] = 2;

        auto  ptr = reinterpret_cast< fp32_t * >( zdata.data() + 4 );

        for ( size_t  i = 0; i < nsize; ++i )
            *(ptr++) = fp32_t( data[i] );

        return zdata;
    }// if
    else
    {
        const auto  nbytes = 8 + nsize * 8;
        auto        zdata  = std::vector< byte_t >( nbytes );

        zdata[0] = 3;

        auto  ptr = reinterpret_cast< fp64_t * >( zdata.data() + 8 );

        for ( size_t  i = 0; i < nsize; ++i )
            *(ptr++) = fp64_t( data[i] );

        return zdata;
    }// else
}

template <>
inline
zarray
compress< std::complex< float > > ( const config &           config,
                                    std::complex< float > *  data,
                                    const size_t             dim0,
                                    const size_t             dim1,
                                    const size_t             dim2,
                                    const size_t             dim3 )
{
    if      ( dim1 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, 2 );
    else                  return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, dim3 * 2 );
}

template <>
inline
zarray
compress< std::complex< double > > ( const config &            config,
                                     std::complex< double > *  data,
                                     const size_t              dim0,
                                     const size_t              dim1,
                                     const size_t              dim2,
                                     const size_t              dim3 )
{
    if      ( dim1 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, 2 );
    else                  return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, dim3 * 2 );
}

////////////////////////////////////////////////////////////////////////////////
//
// decompression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // convert back based on given FP type
    //

    switch ( zdata[0] )
    {
        case 1 :
        {
            auto  ptr = reinterpret_cast< const fp16_t * >( zdata.data() + 2 );

            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = value_t( *(ptr++) );
        }
        break;
        
        case 2 :
        {
            auto  ptr = reinterpret_cast< const fp32_t * >( zdata.data() + 4 );

            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = value_t( *(ptr++) );
        }
        break;
        
        case 3 :
        {
            auto  ptr = reinterpret_cast< const fp64_t * >( zdata.data() + 8 );

            for ( size_t  i = 0; i < nsize; ++i )
                dest[i] = value_t( *(ptr++) );
        }
        break;

        default :
            HLR_ERROR( "invalid FP type in compressed data" );
    }// switch
}

template <>
inline
void
decompress< std::complex< float > > ( const zarray &           zdata,
                                      std::complex< float > *  dest,
                                      const size_t             dim0,
                                      const size_t             dim1,
                                      const size_t             dim2,
                                      const size_t             dim3 )
{
    if      ( dim1 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, 2 );
    else                  decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, dim3 * 2 );
}
    
template <>
inline
void
decompress< std::complex< double > > ( const zarray &            zdata,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3 )
{
    if      ( dim1 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 ) decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, 2 );
    else                  decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, dim3 * 2 );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// special versions for lowrank data
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
    //
    // determine corresponding parts for FP32, FP16
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

    const uint32_t  n_fp16 = test_prec( fp16_prec );
    const uint32_t  n_fp32 = i+1;                   // remaining singular values
    size_t          s     = 0;

    // std::cout << n_fp16 << " / " << n_fp32 << " / " << n_fp64 << std::endl;
    
    HLR_ASSERT( n_fp32 >= 0 );
    HLR_ASSERT( n_fp16 + n_fp32 == rank );

    //
    // copy into storage
    //

    const size_t  zsize = ( 2 * sizeof(uint32_t) +
                            sizeof(fp32_t) * n * n_fp32 +
                            sizeof(fp16_t) * n * n_fp16 );
    zarray        zdata( zsize );

    reinterpret_cast< uint32_t * >( zdata.data() )[0] = n_fp32;
    reinterpret_cast< uint32_t * >( zdata.data() )[1] = n_fp16;

    uint32_t  k   = 0;
    size_t    pos = 2 * sizeof(uint32_t);
        
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

    {
        auto    zptr = reinterpret_cast< fp16_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp16; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = fp16_t( U(i,k) );
        }// for
        pos += n_fp16 * n * sizeof(fp16_t);
    }

    HLR_ASSERT( k == rank );

    return zdata;
}

template <>
inline
zarray
compress_lr< double > ( const hlr::blas::matrix< double > &  U,
                        const hlr::blas::vector< double > &  S )
{
    //
    // determine corresponding parts for FP64, FP32, FP16
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

    const uint32_t  n_fp16 = test_prec( fp16_prec );
    const uint32_t  n_fp32 = test_prec( fp32_prec );
    const uint32_t  n_fp64 = i+1;                   // remaining singular values
    size_t          s     = 0;

    // std::cout << n_fp16 << " / " << n_fp32 << " / " << n_fp64 << std::endl;
    
    HLR_ASSERT( n_fp64 >= 0 );
    HLR_ASSERT( n_fp16 + n_fp32 + n_fp64 == rank );

    //
    // copy into storage
    //

    const size_t  zsize = ( 3 * sizeof(uint32_t) +
                            sizeof(fp64_t) * n * n_fp64 +
                            sizeof(fp32_t) * n * n_fp32 +
                            sizeof(fp16_t) * n * n_fp16 );
    zarray        zdata( zsize );

    reinterpret_cast< uint32_t * >( zdata.data() )[0] = n_fp64;
    reinterpret_cast< uint32_t * >( zdata.data() )[1] = n_fp32;
    reinterpret_cast< uint32_t * >( zdata.data() )[2] = n_fp16;

    uint32_t  k   = 0;
    size_t    pos = 3 * sizeof(uint32_t);
        
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

    {
        auto    zptr = reinterpret_cast< fp16_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp16; ++l, ++k )
        {
            for ( size_t  i = 0; i < n; ++i, ++zpos )
                zptr[zpos] = fp16_t( U(i,k) );
        }// for
        pos += n_fp16 * n * sizeof(fp16_t);
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

    //     _mpdata.U1 = std::vector< fp64_t >( 2 * n_fp64 * oU.nrows() );
    //     _mpdata.V1 = std::vector< fp64_t >( 2 * n_fp64 * oV.nrows() );

    //     for ( size_t  i = 0; i < _mpdata.U1.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U1[i]   = fp64_t( std::real( u_i ) );
    //         _mpdata.U1[i+1] = fp64_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V1.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V1[i]   = fp64_t( std::real( v_i ) );
    //         _mpdata.V1[i+1] = fp64_t( std::imag( v_i ) );
    //     }// for

    //     _mpdata.U2 = std::vector< fp32_t >( 2 * n_fp32 * oU.nrows() );
    //     _mpdata.V2 = std::vector< fp32_t >( 2 * n_fp32 * oV.nrows() );

    //     for ( size_t  i = 0; i < _mpdata.U2.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U2[i]   = fp32_t( std::real( u_i ) );
    //         _mpdata.U2[i+1] = fp32_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V2.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V2[i]   = fp32_t( std::real( v_i ) );
    //         _mpdata.V2[i+1] = fp32_t( std::imag( v_i ) );
    //     }// for


    //     _mpdata.U3 = std::vector< fp16_t >( 2 * n_fp16 * oU.nrows() );
    //     _mpdata.V3 = std::vector< fp16_t >( 2 * n_fp16 * oV.nrows() );
            
    //     for ( size_t  i = 0; i < _mpdata.U3.size(); i += 2 )
    //     {
    //         const auto  u_i = oU.data()[ pos_U++ ];

    //         _mpdata.U3[i]   = fp16_t( std::real( u_i ) );
    //         _mpdata.U3[i+1] = fp16_t( std::imag( u_i ) );
    //     }// for

    //     for ( size_t  i = 0; i < _mpdata.V3.size(); i += 2 )
    //     {
    //         const auto  v_i = oV.data()[ pos_V++ ];

    //         _mpdata.V3[i]   = fp16_t( std::real( v_i ) );
    //         _mpdata.V3[i+1] = fp16_t( std::imag( v_i ) );
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
    const size_t    nrows = U.nrows();
    const uint32_t  rank  = U.ncols();
    const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zdata.data() )[0];
    const uint32_t  n_fp16 = reinterpret_cast< const uint32_t * >( zdata.data() )[1];
    size_t          pos   = 2 * sizeof(uint32_t);
    uint32_t        k     = 0;
    
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

    if ( n_fp16 > 0 )
    {
        auto    zptr = reinterpret_cast< const fp16_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp16; ++l, ++k )
        {
            for ( size_t  i = 0; i < nrows; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_fp16 * nrows * sizeof(fp16_t);
    }// if

    HLR_ASSERT( k == rank );
}

template <>
inline
void
decompress_lr< double > ( const zarray &                 zdata,
                          hlr::blas::matrix< double > &  U )
{
    const size_t    nrows = U.nrows();
    const uint32_t  rank  = U.ncols();
    const uint32_t  n_fp64 = reinterpret_cast< const uint32_t * >( zdata.data() )[0];
    const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zdata.data() )[1];
    const uint32_t  n_fp16 = reinterpret_cast< const uint32_t * >( zdata.data() )[2];
    size_t          pos   = 3 * sizeof(uint32_t);
    uint32_t        k     = 0;
    
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

    if ( n_fp16 > 0 )
    {
        auto    zptr = reinterpret_cast< const fp16_t * >( zdata.data() + pos );
        size_t  zpos = 0;

        for ( uint32_t  l = 0; l < n_fp16; ++l, ++k )
        {
            for ( size_t  i = 0; i < nrows; ++i, ++zpos )
                U(i,k) = zptr[zpos];
        }// for
        pos += n_fp16 * nrows * sizeof(fp16_t);
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
    // const uint32_t  n_fp64 = _mpdata.U1.size() / (2 * dU.nrows());
    // const uint32_t  n_fp32 = _mpdata.U2.size() / (2 * dU.nrows());
    // const uint32_t  n_fp16 = _mpdata.U3.size() / (2 * dU.nrows());
                
    // for ( uint32_t  k1 = 0; k1 < n_fp64; ++k1, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U1[ pos ], _mpdata.U1[ pos+1 ] );
    // }// for

    // pos = 0;
    // for ( uint32_t  k2 = 0; k2 < n_fp32; ++k2, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U2[ pos ], _mpdata.U2[ pos+1 ] );
    // }// for
                
    // pos = 0;
    // for ( uint32_t  k3 = 0; k3 < n_fp16; ++k3, ++k )
    // {
    //     const auto  s_k = _S(k);
                    
    //     for ( uint32_t  i = 0; i < dU.nrows(); ++i, pos += 2 )
    //         dU(i,k) = s_k * value_t( _mpdata.U3[ pos ], _mpdata.U3[ pos+1 ] );
    // }// for
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
            value_t *        y );

template <>
inline
void
mulvec_lr< float > ( const size_t     nrows,
                     const size_t     ncols,
                     const matop_t    op_A,
                     const float      alpha,
                     const zarray &   zA,
                     const float *    x,
                     float *          y )
{
    using  value_t = float;
    
    const auto  zsize = zA.size();
    size_t      zpos  = 0;
    size_t      dpos  = 0;

    //
    // in case of joined blocks, iterate over all
    //
    
    while ( zpos < zsize )
    {
        const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zA.data() + zpos )[0];
        const uint32_t  n_fp16 = reinterpret_cast< const uint32_t * >( zA.data() + zpos )[1];

        zpos += 2 * sizeof(uint32_t);
        
        switch ( op_A )
        {
            case  apply_normal :
            {
                if ( n_fp32 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                    {
                        blas::gemv( 'N', nrows, n_fp32, alpha, zptr, nrows, x + dpos, 1, fp32_t(1), y, 1 );
                    }// if
                    // {
                    //     mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x + dpos, y );
                    // }// else
                
                    zpos += n_fp32 * nrows * sizeof(fp32_t);
                    dpos += n_fp32;
                }// if

                if ( n_fp16 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp16_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp16_t >( nrows, n_fp16, op_A, alpha, zptr, x + dpos, y );
                    zpos += n_fp16 * nrows * sizeof(fp16_t);
                    dpos += n_fp16;
                }// if
            }// case
            break;
        
            case  apply_conjugate  : HLR_ERROR( "TODO" );
            
            case  apply_transposed : HLR_ERROR( "TODO" );

            case  apply_adjoint :
            {
                if ( n_fp32 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x, y + dpos );
                    zpos += n_fp32 * nrows * sizeof(fp32_t);
                    dpos += n_fp32;
                }// if

                if ( n_fp16 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp16_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp16_t >( nrows, n_fp16, op_A, alpha, zptr, x, y + dpos );
                    zpos += n_fp16 * nrows * sizeof(fp16_t);
                    dpos += n_fp16;
                }// if
            }// case
            break;
        }// switch
    }// while
}

template <>
inline
void
mulvec_lr< double > ( const size_t     nrows,
                      const size_t     ncols,
                      const matop_t    op_A,
                      const double     alpha,
                      const zarray &   zA,
                      const double *   x,
                      double *         y )
{
    using  value_t = double;
    
    const auto  zsize = zA.size();
    size_t      zpos  = 0;
    size_t      dpos  = 0;

    //
    // in case of joined blocks, iterate over all
    //
    
    while ( zpos < zsize )
    {
        const uint32_t  n_fp64 = reinterpret_cast< const uint32_t * >( zA.data() + zpos )[0];
        const uint32_t  n_fp32 = reinterpret_cast< const uint32_t * >( zA.data() + zpos )[1];
        const uint32_t  n_fp16 = reinterpret_cast< const uint32_t * >( zA.data() + zpos )[2];

        zpos += 3 * sizeof(uint32_t);
        
        switch ( op_A )
        {
            case  apply_normal :
            {
                if ( n_fp64 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp64_t * >( zA.data() + zpos );
                
                    blas::gemv( 'N', nrows, n_fp64, alpha, zptr, nrows, x + dpos, 1, fp64_t(1), y, 1 );
                
                    zpos += n_fp64 * nrows * sizeof(fp64_t);
                    dpos += n_fp64;
                }// if

                if ( n_fp32 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                    {
                        auto  tx = blas::vector< fp32_t >( n_fp32 );
                        auto  ty = blas::vector< fp32_t >( nrows );

                        for ( size_t  i = 0; i < n_fp32; ++i )
                            tx(i) = *(x + dpos + i);
                    
                        blas::gemv( 'N', nrows, n_fp32, alpha, zptr, nrows, tx.data(), 1, fp32_t(0), ty.data(), 1 );

                        for ( size_t  i = 0; i < nrows; ++i )
                            y[i] += ty(i);
                    }// if
                    // {
                    //     mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x + dpos, y );
                    // }// else
                
                    zpos += n_fp32 * nrows * sizeof(fp32_t);
                    dpos += n_fp32;
                }// if

                if ( n_fp16 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp16_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp16_t >( nrows, n_fp16, op_A, alpha, zptr, x + dpos, y );
                    zpos += n_fp16 * nrows * sizeof(fp16_t);
                    dpos += n_fp16;
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
                
                    mulvec< value_t, fp64_t >( nrows, n_fp64, op_A, alpha, zptr, x, y + dpos );
                    zpos += n_fp64 * nrows * sizeof(fp64_t);
                    dpos += n_fp64;
                }// if

                if ( n_fp32 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp32_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp32_t >( nrows, n_fp32, op_A, alpha, zptr, x, y + dpos );
                    zpos += n_fp32 * nrows * sizeof(fp32_t);
                    dpos += n_fp32;
                }// if

                if ( n_fp16 > 0 )
                {
                    auto  zptr = reinterpret_cast< const fp16_t * >( zA.data() + zpos );
                
                    mulvec< value_t, fp16_t >( nrows, n_fp16, op_A, alpha, zptr, x, y + dpos );
                    zpos += n_fp16 * nrows * sizeof(fp16_t);
                    dpos += n_fp16;
                }// if
            }// case
            break;
        }// switch
    }// while
}

}}}// namespace hlr::compress::mixedprec

#endif // __HLR_UTILS_DETAIL_MIXEDPREC_HH
