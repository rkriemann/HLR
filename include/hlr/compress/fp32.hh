#ifndef __HLR_UTILS_DETAIL_FP32_HH
#define __HLR_UTILS_DETAIL_FP32_HH
//
// Project     : HLR
// Module      : compress/fp32
// Description : FP32 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// compression using FP32
// - only fixed compression size (1+8+23 bits)
//
////////////////////////////////////////////////////////////

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT

namespace hlr { namespace compress { namespace fp32 {

using fp32_t = float;

struct config
{};

// holds compressed data
using  zarray = std::vector< fp32_t >;

inline size_t  compressed_size ( const zarray &  v   ) { return v.size() * sizeof(fp32_t); }
inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + compressed_size( v ); }
inline config  get_config      ( const double    eps ) { return config{}; }

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( nsize );

    for ( size_t  i = 0; i < nsize; ++i )
        zdata[i] = fp32_t(data[i]);

    return zdata;
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
    zarray        zdata( nsize );

    for ( size_t  i = 0; i < nsize; ++i )
        zdata[i] = fp32_t(data[i]);

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( nsize*2 );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        zdata[2*i]   = fp32_t( std::real( data[i] ) );
        zdata[2*i+1] = fp32_t( std::imag( data[i] ) );
    }// for

    return zdata;
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( nsize*2 );

    for ( size_t  i = 0; i < nsize; ++i )
    {
        zdata[2*i]   = fp32_t( std::real( data[i] ) );
        zdata[2*i+1] = fp32_t( std::imag( data[i] ) );
    }// for

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
decompress< float > ( const zarray &  zdata,
                      float *         dest,
                      const size_t    dim0,
                      const size_t    dim1,
                      const size_t    dim2,
                      const size_t    dim3,
                      const size_t    dim4 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = float( zdata[i] );
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = double( zdata[i] );
}

template <>
inline
void
decompress< std::complex< float > > ( const zarray &           zdata,
                                      std::complex< float > *  dest,
                                      const size_t             dim0,
                                      const size_t             dim1,
                                      const size_t             dim2,
                                      const size_t             dim3,
                                      const size_t             dim4 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = std::complex< float >( zdata[2*i], zdata[2*i+1] );
}

template <>
inline
void
decompress< std::complex< double > > ( const zarray &            zdata,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    for ( size_t  i = 0; i < nsize; ++i )
        dest[i] = std::complex< double >( zdata[2*i], zdata[2*i+1] );
}

//
// compressed blas
//

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
    using  real_t = Hpro::real_type_t< value_t >;

    // if constexpr ( std::same_as< value_t, float > )
    // {
    //     blas::gemv( char( op_A ),
    //                 blas::blas_int_t(nrows),
    //                 blas::blas_int_t(ncols),
    //                 alpha,
    //                 zA.data(),
    //                 blas::blas_int_t(nrows),
    //                 x,
    //                 blas::blas_int_t(1),
    //                 float(1),
    //                 y,
    //                 blas::blas_int_t(1) );
    // }// if
    // else if constexpr ( std::same_as< value_t, std::complex< float > > )
    // {
    //     blas::gemv( char( op_A ),
    //                 blas::blas_int_t(nrows),
    //                 blas::blas_int_t(ncols),
    //                 alpha,
    //                 zA.data(),
    //                 blas::blas_int_t(nrows),
    //                 x,
    //                 blas::blas_int_t(1),
    //                 std::complex< float >( 1 ),
    //                 y,
    //                 blas::blas_int_t(1) );
    // }// if
    // else
    {
        auto  ptr = zA.data();
        
        switch ( op_A )
        {
            case apply_normal :
            {
                for ( blas::idx_t j = 0; j < ncols; ++j )
                {
                    const auto  x_j = alpha * x[j];
            
                    for ( blas::idx_t i = 0; i < nrows; ++i )
                        y[i] += *ptr++ * x_j;
                }// for
            }
            break;
            
            case apply_conjugate :
            {
                for ( blas::idx_t j = 0; j < ncols; ++j )
                {
                    const auto  x_j = alpha * x[j];
            
                    for ( blas::idx_t i = 0; i < nrows; ++i )
                        y[i] += math::conj( *ptr++ ) * x_j;
                }// for
            }
            break;
            
            case apply_transposed :
            {
                for ( blas::idx_t j = 0; j < ncols; ++j )
                {
                    value_t  f = value_t(0);
            
                    for ( blas::idx_t i = 0; i < nrows; ++i )
                        f += *ptr++ * x[i];
            
                    y[j] += alpha * f;
                }// for
            }
            break;
            
            case apply_adjoint :
            {
                for ( blas::idx_t j = 0; j < ncols; ++j )
                {
                    value_t  f = value_t(0);
            
                    for ( blas::idx_t i = 0; i < nrows; ++i )
                        f += math::conj(*ptr++) * x[i];
            
                    y[j] += alpha * f;
                }// for
            }
            break;

            default:
                HLR_ERROR( "unknown matrix mode" );
        }// switch
    }// else
}

}}}// namespace hlr::compress::fp32

#endif // __HLR_UTILS_DETAIL_FP32_HH
