#ifndef __HLR_UTILS_DETAIL_POSITS_HH
#define __HLR_UTILS_DETAIL_POSITS_HH
//
// Project     : HLR
// Module      : utils/detail/posits
// Description : posits related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HLR_HAS_UNIVERSAL)

#include <cstdint>

#include <universal/number/posit/posit.hpp>

#include <hlr/utils/detail/byte_n.hh>

namespace hlr { namespace compress { namespace posits {

using byte_t = uint8_t;

// return byte padded value of <n>
inline size_t byte_pad ( size_t  n )
{
    return ( n % 8 != 0 ) ? n + (8 - n%8) : n;
}

inline
uint
eps_to_rate ( const double eps )
{
    // |d_i - ~d_i| ≤ 2^(-m) ≤ ε with m = remaining mantissa length
    return std::max< double >( 1, std::ceil( -std::log2( eps ) ) );
}

struct config
{
    uint  bitsize;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v )  { return v.size(); }

inline config  get_config ( const double  eps  ) { return config{ eps_to_rate( eps ) }; }

//
// convert given array <data> into posits and store results in <cptr>
//
template < typename value_t, int bitsize, typename storage_t >
struct convert
{
    static constexpr uint64_t  mask = ( 1ul << bitsize ) - 1ul;
    
    static void
    to_posit ( byte_t *         cptr,
               const value_t *  data,
               const size_t     nsize,
               const value_t    scale )
    {
        using  posit_t = sw::universal::posit< bitsize, 2 >;
        
        auto  ptr = reinterpret_cast< storage_t * >( cptr );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            auto  p    = posit_t( data[i] * scale );
            auto  bits = p.get();

            ptr[i] = bits.to_ullong() & mask;
        }// if
    }// for

    static void
    from_posit ( const byte_t *  cptr,
                 value_t *       data,
                 const size_t    nsize,
                 const value_t   scale )
    {
        using  posit_t    = sw::universal::posit< bitsize, 2 >;
        using  bitblock_t = sw::universal::bitblock< bitsize >;

        auto  ptr = reinterpret_cast< const storage_t * >( cptr );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            auto        raw  = ptr[i];
            bitblock_t  bits;
            posit_t     p;
            
            bits = uint64_t(raw);
            p.set( bits );

            data[i] = value_t( p ) / scale;
        }// for
    }
};

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    constexpr float   fp32_infinity = std::numeric_limits< float >::infinity();
    constexpr double  fp64_infinity = std::numeric_limits< double >::infinity();
    
    //
    // look for min/max value (> 0!)
    //
    
    value_t  vmin = fp64_infinity;
    value_t  vmax = 0;

    for ( size_t  i = 0; i < nsize; ++i )
    {
        const auto  d_i = std::abs( data[i] );
        const auto  val = ( d_i == value_t(0) ? value_t(fp64_infinity) : d_i );

        vmin = std::min( vmin, val );
        vmax = std::max( vmax, d_i );
    }// for

    HLR_DBG_ASSERT( vmin > value_t(0) );
    
    const value_t  scale = 1.0 / vmax;
    const auto     nbits = config.bitsize + 2; // 2 exponent bits
    const auto     pbits = byte_pad( nbits );
    const auto     ofs   = 1 + sizeof(value_t);
    zarray         zdata( ofs + nsize * pbits / 8 );

    zdata[0] = nbits;
    memcpy( zdata.data() + 1, & scale, sizeof(value_t) );
    
    switch ( nbits )
    {
        case  8: { convert< value_t,  8, byte_t   >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case  9: { convert< value_t,  9, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 10: { convert< value_t, 10, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 11: { convert< value_t, 11, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 12: { convert< value_t, 12, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 13: { convert< value_t, 13, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 14: { convert< value_t, 14, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 15: { convert< value_t, 15, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 16: { convert< value_t, 16, uint16_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 17: { convert< value_t, 17, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 18: { convert< value_t, 18, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 19: { convert< value_t, 19, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 20: { convert< value_t, 20, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 21: { convert< value_t, 21, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 22: { convert< value_t, 22, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 23: { convert< value_t, 23, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 24: { convert< value_t, 24, byte3_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 25: { convert< value_t, 25, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 26: { convert< value_t, 26, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 27: { convert< value_t, 27, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 28: { convert< value_t, 28, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 29: { convert< value_t, 29, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 30: { convert< value_t, 30, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 31: { convert< value_t, 31, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 32: { convert< value_t, 32, uint32_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 40: { convert< value_t, 40, byte5_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 48: { convert< value_t, 48, byte6_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        case 56: { convert< value_t, 56, byte7_t  >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;
        // case 64: { convert< value_t, 64, uint64_t >::to_posit( zdata.data() + ofs, data, nsize, scale ); } break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( config.bitsize ) );
    }// switch

    return zdata;
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

//
// decompression function
//
template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    const size_t  nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    bitsize = zdata[0];
    auto          scale   = value_t(0);
    const auto    ofs     = 1 + sizeof(value_t);

    memcpy( & scale, zdata.data() + 1, sizeof(value_t) );
    
    switch ( bitsize )
    {
        case  8: { convert< value_t,  8, byte_t   >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case  9: { convert< value_t,  9, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 10: { convert< value_t, 10, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 11: { convert< value_t, 11, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 12: { convert< value_t, 12, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 13: { convert< value_t, 13, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 14: { convert< value_t, 14, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 15: { convert< value_t, 15, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 16: { convert< value_t, 16, uint16_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 17: { convert< value_t, 17, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 18: { convert< value_t, 18, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 19: { convert< value_t, 19, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 20: { convert< value_t, 20, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 21: { convert< value_t, 21, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 22: { convert< value_t, 22, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 23: { convert< value_t, 23, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 24: { convert< value_t, 24, byte3_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 25: { convert< value_t, 25, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 26: { convert< value_t, 26, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 27: { convert< value_t, 27, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 28: { convert< value_t, 28, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 29: { convert< value_t, 29, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 30: { convert< value_t, 30, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 31: { convert< value_t, 31, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 32: { convert< value_t, 32, uint32_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 40: { convert< value_t, 40, byte5_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 48: { convert< value_t, 48, byte6_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        case 56: { convert< value_t, 56, byte7_t  >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;
        // case 64: { convert< value_t, 64, uint64_t >::from_posit( zdata.data() + ofs, dest, nsize, scale ); } break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
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

// namespace detail
// {
//
// //
// // some basic blas functions
// //
// template < size_t nbits,
//            size_t es >
// inline
// void
// mulvec ( const size_t           nrows,
//          const size_t           ncols,
//          const Hpro::matop_t    op_A,
//          const double           dalpha,
//          const byte_t *         A_ptr,
//          const double *         x_ptr,
//          const double           beta,
//          double *               y_ptr )
// {
//     using  posit_t = sw::universal::posit< nbits, es >;

//     auto           A     = reinterpret_cast< const posit_t * >( A_ptr );
//     const posit_t  alpha = dalpha;

//     if ( op_A == Hpro::apply_normal )
//     {
//         auto  y = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             y[i] = beta * y_ptr[i];

//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             const posit_t  x_j = x_ptr[j];
            
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y[i] += alpha * A[j*nrows+i] * x_j;
//         }// for

//         for ( size_t  i = 0; i < nrows; ++i )
//             y_ptr[i] = double( y[i] );
//     }// if
//     else if ( op_A == Hpro::apply_transposed )
//     {
//         auto  x = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             x[i] = x_ptr[i];
        
//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             posit_t  y_j = beta * y_ptr[j];
        
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y_j += alpha * A[j*nrows+i] * x[i];

//             y_ptr[j] = double( y_j );
//         }// for
//     }// if
//     else if ( op_A == Hpro::apply_adjoint )
//     {
//         auto  x = std::vector< posit_t >( nrows );
        
//         for ( size_t  i = 0; i < nrows; ++i )
//             x[i] = x_ptr[i];
        
//         for ( size_t  j = 0; j < ncols; ++j )
//         {
//             posit_t  y_j = beta * y_ptr[j];
        
//             for ( size_t  i = 0; i < nrows; ++i )
//                 y_j += alpha * A[j*nrows+i] * x[i];

//             y_ptr[j] = double( y_j );
//         }// for
//     }// if
//     else
//         HLR_ERROR( "TODO" );
// }

// }// namespace detail

// template < typename value_t >
// void
// mulvec ( const size_t         nrows,
//          const size_t         ncols,
//          const Hpro::matop_t  op_A,
//          const value_t        alpha,
//          const zarray &       A,
//          const value_t *      x,
//          const value_t        beta,
//          value_t *            y );

// template <>
// inline
// void
// mulvec< double > ( const size_t         nrows,
//                    const size_t         ncols,
//                    const Hpro::matop_t  op_A,
//                    const double         alpha,
//                    const zarray &       A,
//                    const double *       x,
//                    const double         beta,
//                    double *             y )
// {
//     const auto  bitsize = A[0];

//     switch ( bitsize )
//     {
//         case  8: detail::mulvec<  8, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 10: detail::mulvec< 10, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 12: detail::mulvec< 12, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 14: detail::mulvec< 14, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 16: detail::mulvec< 16, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 18: detail::mulvec< 18, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 20: detail::mulvec< 20, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 22: detail::mulvec< 22, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 24: detail::mulvec< 24, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 26: detail::mulvec< 26, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 28: detail::mulvec< 28, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 30: detail::mulvec< 30, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 32: detail::mulvec< 32, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 34: detail::mulvec< 34, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 36: detail::mulvec< 36, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 40: detail::mulvec< 40, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 44: detail::mulvec< 44, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 54: detail::mulvec< 54, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
//         case 64: detail::mulvec< 64, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;

//         default:
//             HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
//     }// switch
// }
    
}}}// namespace hlr::compress::posits

#endif // HLR_HAS_UNIVERSAL

#endif // __HLR_UTILS_DETAIL_POSITS_HH
