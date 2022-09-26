#ifndef __HLR_UTILS_DETAIL_POSITS_HH
#define __HLR_UTILS_DETAIL_POSITS_HH
//
// Project     : HLR
// Module      : utils/detail/posits
// Description : posits related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using Posits
//
////////////////////////////////////////////////////////////

#if defined(HAS_UNIVERSAL)

#include <universal/number/posit/posit.hpp>

namespace hlr { namespace compress { namespace posits {

inline
uint
eps_to_rate ( const double eps )
{
    if      ( eps >= 1e-2  ) return 12;
    else if ( eps >= 1e-3  ) return 14;
    else if ( eps >= 1e-4  ) return 18;
    else if ( eps >= 1e-5  ) return 22;
    else if ( eps >= 1e-6  ) return 26;
    else if ( eps >= 1e-7  ) return 30;
    else if ( eps >= 1e-8  ) return 34;
    else if ( eps >= 1e-9  ) return 36;
    else if ( eps >= 1e-10 ) return 40;
    else if ( eps >= 1e-12 ) return 44;
    else if ( eps >= 1e-14 ) return 54;
    else                     return 64;
}

struct config
{
    uint  bitsize;
};

// holds compressed data
using  zarray = std::vector< unsigned char >;

inline
size_t
byte_size ( const zarray &  v )
{
    if ( v.size() > 0 )
    {
        const auto  bitsize = v[0];
        const auto  nposits = (v.size() - 8) / 8;

        return std::ceil( nposits * bitsize / 8.0 );
    }// if

    return 0;
}

inline config  relative_accuracy ( const double  eps  ) { return config{ eps_to_rate( eps ) }; }
inline config  absolute_accuracy ( const double  eps  ) { return config{ eps_to_rate( eps ) }; }

template < typename value_t, int bitsize, int expsize >
void
to_posit ( unsigned char *  cptr,
           const value_t *  data,
           const size_t     nsize )
{
    using  posit_t = sw::universal::posit< bitsize, expsize >;

    auto  ptr = reinterpret_cast< posit_t * >( cptr );
    
    for ( size_t  i = 0; i < nsize; ++i )
        ptr[i] = posit_t( data[i] );
}

template < typename value_t, int bitsize, int expsize >
void
from_posit ( const unsigned char *  cptr,
             value_t *              data,
             const size_t           nsize )
{
    using  posit_t = sw::universal::posit< bitsize, expsize >;

    auto  ptr = reinterpret_cast< const posit_t * >( cptr );
    
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = value_t( ptr[i] );
}

template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 );

template <>
inline
zarray
compress< float > ( const config &   config,
                    float *          data,
                    const size_t     dim0,
                    const size_t     dim1,
                    const size_t     dim2,
                    const size_t     dim3 )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
zarray
compress< double > ( const config &   config,
                     double *         data,
                     const size_t     dim0,
                     const size_t     dim1,
                     const size_t     dim2,
                     const size_t     dim3 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( 8 + nsize * 8 ); // SW-Posits have fixed size 8!

    zdata[0] = config.bitsize;
    
    switch ( config.bitsize )
    {
        case  8: to_posit< double,  8, 1 >( zdata.data() + 8, data, nsize ); break;
        case 10: to_posit< double, 10, 1 >( zdata.data() + 8, data, nsize ); break;
        case 12: to_posit< double, 12, 2 >( zdata.data() + 8, data, nsize ); break;
        case 14: to_posit< double, 14, 2 >( zdata.data() + 8, data, nsize ); break;
        case 16: to_posit< double, 16, 2 >( zdata.data() + 8, data, nsize ); break;
        case 18: to_posit< double, 18, 2 >( zdata.data() + 8, data, nsize ); break;
        case 20: to_posit< double, 20, 2 >( zdata.data() + 8, data, nsize ); break;
        case 22: to_posit< double, 22, 2 >( zdata.data() + 8, data, nsize ); break;
        case 24: to_posit< double, 24, 2 >( zdata.data() + 8, data, nsize ); break;
        case 26: to_posit< double, 26, 2 >( zdata.data() + 8, data, nsize ); break;
        case 28: to_posit< double, 28, 2 >( zdata.data() + 8, data, nsize ); break;
        case 30: to_posit< double, 30, 3 >( zdata.data() + 8, data, nsize ); break;
        case 32: to_posit< double, 32, 3 >( zdata.data() + 8, data, nsize ); break;
        case 34: to_posit< double, 34, 3 >( zdata.data() + 8, data, nsize ); break;
        case 36: to_posit< double, 36, 3 >( zdata.data() + 8, data, nsize ); break;
        case 40: to_posit< double, 40, 3 >( zdata.data() + 8, data, nsize ); break;
        case 44: to_posit< double, 44, 3 >( zdata.data() + 8, data, nsize ); break;
        case 54: to_posit< double, 54, 3 >( zdata.data() + 8, data, nsize ); break;
        case 64: to_posit< double, 64, 3 >( zdata.data() + 8, data, nsize ); break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( config.bitsize ) );
    }// switch

    return zdata;
}

template < typename value_t >
void
decompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 );

template <>
inline
void
decompress< float > ( const zarray &  v,
                      float *        dest,
                      const size_t   dim0,
                      const size_t   dim1,
                      const size_t   dim2,
                      const size_t   dim3,
                      const size_t   dim4 )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
decompress< double > ( const zarray &  zdata,
                       double *        dest,
                       const size_t    dim0,
                       const size_t    dim1,
                       const size_t    dim2,
                       const size_t    dim3,
                       const size_t    dim4 )
{
    const size_t  nsize   = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    bitsize = zdata[0];
    
    switch ( bitsize )
    {
        case  8: from_posit< double,  8, 1 >( zdata.data() + 8, dest, nsize ); break;
        case 10: from_posit< double, 10, 1 >( zdata.data() + 8, dest, nsize ); break;
        case 12: from_posit< double, 12, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 14: from_posit< double, 14, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 16: from_posit< double, 16, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 18: from_posit< double, 18, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 20: from_posit< double, 20, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 22: from_posit< double, 22, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 24: from_posit< double, 24, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 26: from_posit< double, 26, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 28: from_posit< double, 28, 2 >( zdata.data() + 8, dest, nsize ); break;
        case 30: from_posit< double, 30, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 32: from_posit< double, 32, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 34: from_posit< double, 34, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 36: from_posit< double, 36, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 40: from_posit< double, 40, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 44: from_posit< double, 44, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 54: from_posit< double, 54, 3 >( zdata.data() + 8, dest, nsize ); break;
        case 64: from_posit< double, 64, 3 >( zdata.data() + 8, dest, nsize ); break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
    }// switch
}

namespace detail
{

//
// some basic blas functions
//
template < size_t nbits,
           size_t es >
inline
void
mulvec ( const size_t           nrows,
         const size_t           ncols,
         const Hpro::matop_t    op_A,
         const double           dalpha,
         const unsigned char *  A_ptr,
         const double *         x_ptr,
         const double           beta,
         double *               y_ptr )
{
    using  posit_t = sw::universal::posit< nbits, es >;

    auto           A     = reinterpret_cast< const posit_t * >( A_ptr );
    const posit_t  alpha = dalpha;

    if ( op_A == Hpro::apply_normal )
    {
        auto  y = std::vector< posit_t >( nrows );
        
        for ( size_t  i = 0; i < nrows; ++i )
            y[i] = beta * y_ptr[i];

        for ( size_t  j = 0; j < ncols; ++j )
        {
            const posit_t  x_j = x_ptr[j];
            
            for ( size_t  i = 0; i < nrows; ++i )
                y[i] += alpha * A[j*nrows+i] * x_j;
        }// for

        for ( size_t  i = 0; i < nrows; ++i )
            y_ptr[i] = double( y[i] );
    }// if
    else if ( op_A == Hpro::apply_transposed )
    {
        auto  x = std::vector< posit_t >( nrows );
        
        for ( size_t  i = 0; i < nrows; ++i )
            x[i] = x_ptr[i];
        
        for ( size_t  j = 0; j < ncols; ++j )
        {
            posit_t  y_j = beta * y_ptr[j];
        
            for ( size_t  i = 0; i < nrows; ++i )
                y_j += alpha * A[j*nrows+i] * x[i];

            y_ptr[j] = double( y_j );
        }// for
    }// if
    else if ( op_A == Hpro::apply_adjoint )
    {
        auto  x = std::vector< posit_t >( nrows );
        
        for ( size_t  i = 0; i < nrows; ++i )
            x[i] = x_ptr[i];
        
        for ( size_t  j = 0; j < ncols; ++j )
        {
            posit_t  y_j = beta * y_ptr[j];
        
            for ( size_t  i = 0; i < nrows; ++i )
                y_j += alpha * A[j*nrows+i] * x[i];

            y_ptr[j] = double( y_j );
        }// for
    }// if
    else
        HLR_ERROR( "TODO" );
}

}// namespace detail

template < typename value_t >
void
mulvec ( const size_t         nrows,
         const size_t         ncols,
         const Hpro::matop_t  op_A,
         const value_t        alpha,
         const zarray &       A,
         const value_t *      x,
         const value_t        beta,
         value_t *            y );

template <>
inline
void
mulvec< double > ( const size_t         nrows,
                   const size_t         ncols,
                   const Hpro::matop_t  op_A,
                   const double         alpha,
                   const zarray &       A,
                   const double *       x,
                   const double         beta,
                   double *             y )
{
    const auto  bitsize = A[0];

    switch ( bitsize )
    {
        case  8: detail::mulvec<  8, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 10: detail::mulvec< 10, 1 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 12: detail::mulvec< 12, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 14: detail::mulvec< 14, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 16: detail::mulvec< 16, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 18: detail::mulvec< 18, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 20: detail::mulvec< 20, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 22: detail::mulvec< 22, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 24: detail::mulvec< 24, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 26: detail::mulvec< 26, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 28: detail::mulvec< 28, 2 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 30: detail::mulvec< 30, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 32: detail::mulvec< 32, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 34: detail::mulvec< 34, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 36: detail::mulvec< 36, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 40: detail::mulvec< 40, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 44: detail::mulvec< 44, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 54: detail::mulvec< 54, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;
        case 64: detail::mulvec< 64, 3 >( nrows, ncols, op_A, alpha, A.data() + 8, x, beta, y ); break;

        default:
            HLR_ERROR( "unsupported bitsize " + Hpro::to_string( bitsize ) );
    }// switch
}
    
//
// memory accessor
//
struct mem_accessor
{
    config  mode;

    mem_accessor ( const double  eps )
            : mode( absolute_accuracy( eps ) )
    {}
    
    template < typename value_t >
    zarray
    encode ( value_t *        data,
             const size_t     dim0,
             const size_t     dim1 = 0,
             const size_t     dim2 = 0,
             const size_t     dim3 = 0 )
    {
        return compress( mode, data, dim0, dim1, dim2, dim3 );
    }
    
    template < typename value_t >
    void
    decode ( const zarray &  buffer,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
    {
        decompress( buffer, dest, dim0, dim1, dim2, dim3 );
    }

    size_t
    byte_size ( const zarray &  v )
    {
        return posits::byte_size( v );
    }

private:

    mem_accessor ();
};

}}}// namespace hlr::compress::posits

#endif // HAS_UNIVERSAL

#endif // __HLR_UTILS_DETAIL_POSITS_HH
