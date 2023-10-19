#ifndef __HLR_UTILS_DETAIL_DFL_HH
#define __HLR_UTILS_DETAIL_DFL_HH
//
// Project     : HLR
// Module      : utils/detail/dfl
// Description : dfl related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/detail/bfl.hh>

////////////////////////////////////////////////////////////
//
// compression using general dfl format
// - use FP64 exponent size and precision dependend mantissa size (1+11+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace dfl {

using byte_t = uint8_t;

constexpr uint32_t  fp64_mant_bits = 52;
constexpr uint32_t  dfl_header_ofs = 1;

inline
byte_t
eps_to_rate ( const double eps )
{
    // |d_i - ~d_i| ≤ 2^(-m) ≤ ε with m = remaining mantissa length
    return std::max< double >( 1, std::ceil( -std::log2( eps ) ) );
}

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }
inline config  get_config ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

//
// shift bits to the left thereby reducing mantissa size
//
template < typename  storage_t >
struct dfl
{
    static constexpr uint64_t  dfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 11;  // 1 sign bit, 11 exponent bits
    static constexpr uint64_t  dfl_mant_shift = fp64_mant_bits - dfl_mant_bits;
    
    static
    void
    compress ( const double *  data,
               const size_t    nsize,
               byte_t *        zdata )
    {
        auto  zptr = reinterpret_cast< storage_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint64_t  zval  = ival >> dfl_mant_shift;
            
            zptr[i] = zval;
        }// for
    }

    static
    void
    decompress ( double *        data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
        auto  zptr = reinterpret_cast< const storage_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  ztmp = zval << dfl_mant_shift;
            const double    fval = * reinterpret_cast< const double * >( & ztmp );
            
            data[i] = fval;
        }// for
    }
};

template <>
struct dfl< byte8_t >
{
    static
    void
    compress ( const double *  data,
               const size_t    nsize,
               byte_t *        zdata )
    {
        std::copy( data, data + nsize, reinterpret_cast< double * >( zdata ) );
    }

    static
    void
    decompress ( double *        data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
        std::copy( reinterpret_cast< const double * >( zdata ),
                   reinterpret_cast< const double * >( zdata ) + nsize,
                   data );
    }
};

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
    return bfl::compress( bfl::config( config.bitrate ), data, dim0, dim1, dim2, dim3 );
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
    const size_t    nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint32_t  nbits = byte_pad( 1 + 11 + config.bitrate ); // total no. of bits per value
    const uint32_t  nbyte = nbits / 8;
    zarray          zdata( dfl_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    switch ( nbyte )
    {
        case  2 : dfl< byte2_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  3 : dfl< byte3_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  4 : dfl< byte4_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  5 : dfl< byte5_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  6 : dfl< byte6_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  7 : dfl< byte7_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
        case  8 : dfl< byte8_t >::compress( data, nsize, zdata.data() + dfl_header_ofs ); break;
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

////////////////////////////////////////////////////////////////////////////////
//
// decompression functions
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
decompress ( const zarray &  v,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 );

template <>
inline
void
decompress< float > ( const zarray &  zdata,
                      float *         dest,
                      const size_t    dim0,
                      const size_t    dim1,
                      const size_t    dim2,
                      const size_t    dim3 )
{
    bfl::decompress( zdata, dest, dim0, dim1, dim2, dim3 );
}

template <>
inline
void
decompress< double > ( const zarray &  zdata,
                       double *        dest,
                       const size_t    dim0,
                       const size_t    dim1,
                       const size_t    dim2,
                       const size_t    dim3 )
{
    const size_t    nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint32_t  nbyte = zdata[0];

    switch ( nbyte )
    {
        case  2 : dfl< byte2_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  3 : dfl< byte3_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  4 : dfl< byte4_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  5 : dfl< byte5_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  6 : dfl< byte6_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  7 : dfl< byte7_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
        case  8 : dfl< byte8_t >::decompress( dest, nsize, zdata.data() + dfl_header_ofs ); break;
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
// special version for lowrank matrices
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > &                       U,
              const blas::vector< Hpro::real_type_t< value_t > > &  S )
{
    HLR_ERROR( "TODO" );
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
zarray
compress_lr< float > ( const blas::matrix< float > &  U,
                       const blas::vector< float > &  S )
{
    return bfl::compress_lr( U, S );
}

template <>
inline
zarray
compress_lr< double > ( const blas::matrix< double > &  U,
                        const blas::vector< double > &  S )
{
    //
    // first, determine mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          b     = std::vector< uint32_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto      nprecbits = eps_to_rate( S(l) );
        const uint32_t  nbits     = byte_pad( 1 + 11 + nprecbits );
        const uint32_t  nbyte     = nbits / 8;

        b[l]   = nbyte;
        zsize += 1 + nbyte * n;
    }// for

    //
    // convert each column to compressed form
    //

    auto    zdata = std::vector< byte_t >( zsize );
    size_t  pos   = 0;
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  nbyte = b[l];

        zdata[pos] = nbyte;
        pos += 1;

        switch ( nbyte )
        {
            case  2 : dfl< byte2_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  3 : dfl< byte3_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  4 : dfl< byte4_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  5 : dfl< byte5_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  6 : dfl< byte6_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  7 : dfl< byte7_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  8 : dfl< byte8_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
        }// switch
        
        pos += n*nbyte;
    }// for

    return zdata;
}

template <>
inline
void
decompress_lr< float > ( const zarray &           zdata,
                         blas::matrix< float > &  U )
{
    bfl::decompress_lr( zdata, U );
}

template <>
inline
void
decompress_lr< double > ( const zarray &            zdata,
                          blas::matrix< double > &  U )
{
    const size_t    n   = U.nrows();
    const uint32_t  k   = U.ncols();
    size_t          pos = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  nbyte = zdata[ pos ];

        pos += 1;

        switch ( nbyte )
        {
            case  2 : dfl< byte2_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  3 : dfl< byte3_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  4 : dfl< byte4_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  5 : dfl< byte5_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  6 : dfl< byte6_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  7 : dfl< byte7_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  8 : dfl< byte8_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
        }// switch
        
        pos += nbyte * n;
    }// for
}

}}}// namespace hlr::compress::dfl

#endif // __HLR_UTILS_DETAIL_DFL_HH
