#ifndef __HLR_UTILS_DETAIL_BFL_HH
#define __HLR_UTILS_DETAIL_BFL_HH
//
// Project     : HLR
// Module      : utils/detail/bfl
// Description : bfl related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstdint>

#include <hlr/utils/detail/byte_n.hh>

////////////////////////////////////////////////////////////
//
// compression using general bfl format
// - use FP32 exponent size and precision dependend mantissa size (1+8+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bfl {

using byte_t = uint8_t;

constexpr uint32_t  fp32_mant_bits = 23;

constexpr uint64_t  fp64_sign_mask = (1ul << 63);
constexpr uint32_t  fp64_mant_bits = 52;
constexpr uint64_t  fp64_mant_mask = 0x000fffffffffffff;
constexpr uint64_t  fp64_exp_mask  = 0x7ff0000000000000;

constexpr uint32_t  bfl_header_ofs = 1;

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
// convert to float and simply shift bits to the left, 
// thereby reducing mantissa size
//
template < typename  value_t,
           typename  storage_t >
struct bfl32
{
    static constexpr uint32_t  bfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 8;  // 1 sign bit, 8 exponent bits
    static constexpr uint32_t  bfl_mant_shift = fp32_mant_bits - bfl_mant_bits;
    
    static
    void
    compress ( const value_t *  data,
               const size_t     nsize,
               byte_t *         zdata )
    {
        auto  zptr = reinterpret_cast< storage_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const float     fval  = float(data[i]);
            const uint32_t  ival = (*reinterpret_cast< const uint32_t * >( & fval ) );
            const uint32_t  zval  = ival >> bfl_mant_shift;
            
            zptr[i] = zval;
        }// for
    }

    static
    void
    decompress ( value_t *       data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
        auto  zptr = reinterpret_cast< const storage_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint32_t  zval = zptr[i];
            const uint32_t  ztmp = zval << bfl_mant_shift;
            const float     fval = * reinterpret_cast< const float * >( & ztmp );
            
            data[i] = value_t(fval);
        }// for
    }
};

//
// specialization for 4 bytes: just convert to float
//
template < typename  value_t >
struct bfl32< value_t, byte4_t >
{
    static
    void
    compress ( const value_t *  data,
               const size_t     nsize,
               byte_t *         zdata )
    {
        auto  zptr = reinterpret_cast< float * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
            zptr[i] = float(data[i]);
    }

    static
    void
    decompress ( value_t *       data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
        auto  zptr = reinterpret_cast< const float * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
            data[i] = value_t(zptr[i]);
    }
};

//
// extract sign, exponent and mantissa, then reduce exponent to 8 bits
// and shorten mantissa
//
template < typename  value_t,
           typename  storage_t >
struct bfl64
{
    static constexpr uint32_t  bfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 8;  // 1 sign bit, 8 exponent bits
    static constexpr uint32_t  bfl_sign_bit   = bfl_mant_bits + 8;
    static constexpr uint32_t  bfl_mant_shift = fp64_mant_bits - bfl_mant_bits;
    static constexpr uint32_t  bfl_sign_shift = 63 - bfl_sign_bit;
    static constexpr uint64_t  bfl_sign_mask  = (1ul    << bfl_sign_bit);
    static constexpr uint64_t  bfl_exp_mask   = (0xfful << bfl_mant_bits);
    static constexpr uint64_t  bfl_mant_mask  = (1ul    << bfl_mant_bits) - 1;
        
    static
    void
    compress ( const value_t *  data,
               const size_t     nsize,
               byte_t *         zdata )
    {
        auto  zptr = reinterpret_cast< storage_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bfl_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bfl_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bfl_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }

    static
    void
    decompress ( value_t *       data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
    
        auto  zptr = reinterpret_cast< const storage_t * >( zdata );
    
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bfl_sign_mask) << bfl_sign_shift;
            const uint64_t  exp  = (zval & bfl_exp_mask ) >> bfl_mant_bits;
            const uint64_t  mant = (zval & bfl_mant_mask) << bfl_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }
};

//
// specialization when using 8 bytes: just stick to double
//
template <>
struct bfl64< float, byte8_t >
{
    static void compress   ( const float *, const size_t, byte_t * ) { HLR_ERROR( "FP32 with 8 bytes???" ); }
    static void decompress ( float *, const size_t, const byte_t * ) { HLR_ERROR( "FP32 with 8 bytes???" ); }
};

template <>
struct bfl64< double, byte8_t >
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
           const size_t     dim3 = 0 )
{
    const size_t    nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint32_t  nbits = byte_pad( 1 + 8 + config.bitrate ); // total no. of bits per value
    const uint32_t  nbyte = nbits / 8;
    zarray          zdata( bfl_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    switch ( nbyte )
    {
        case 2  : bfl32< value_t, byte2_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 3  : bfl32< value_t, byte3_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 4  : bfl32< value_t, byte4_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 5  : bfl64< value_t, byte5_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 6  : bfl64< value_t, byte6_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 7  : bfl64< value_t, byte7_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        default : bfl64< value_t, byte8_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
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
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    nbyte = zdata[0];

    switch ( nbyte )
    {
        case 2  : bfl32< value_t, byte2_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 3  : bfl32< value_t, byte3_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 4  : bfl32< value_t, byte4_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 5  : bfl64< value_t, byte5_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 6  : bfl64< value_t, byte6_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 7  : bfl64< value_t, byte7_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        default : bfl64< value_t, byte8_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
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
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          b     = std::vector< uint32_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto      nprecbits = eps_to_rate( S(l) );
        const uint32_t  nbits     = byte_pad( 1 + 8 + nprecbits );
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
        const auto  nbyte = b[l];

        zdata[pos] = nbyte;
        pos += 1;

        switch ( nbyte )
        {
            case  2 : bfl32< value_t, byte2_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  3 : bfl32< value_t, byte3_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  4 : bfl32< value_t, byte4_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  5 : bfl64< value_t, byte5_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  6 : bfl64< value_t, byte6_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  7 : bfl64< value_t, byte7_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            default : bfl64< value_t, byte8_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
        }// switch
        
        pos += n*nbyte;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    const size_t    n   = U.nrows();
    const uint32_t  k   = U.ncols();
    size_t          pos = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto  nbyte = zdata[ pos ];

        pos += 1;

        switch ( nbyte )
        {
            case  2 : bfl32< value_t, byte2_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  3 : bfl32< value_t, byte3_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  4 : bfl32< value_t, byte4_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  5 : bfl64< value_t, byte5_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  6 : bfl64< value_t, byte6_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            case  7 : bfl64< value_t, byte7_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            default : bfl64< value_t, byte8_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
        }// switch
        
        pos += nbyte * n;
    }// for
}

template <>
inline
zarray
compress_lr< std::complex< float > > ( const blas::matrix< std::complex< float > > &  U,
                                       const blas::vector< float > &                  S )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
zarray
compress_lr< std::complex< double > > ( const blas::matrix< std::complex< double > > &  U,
                                        const blas::vector< double > &                  S )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
decompress_lr< std::complex< float > > ( const zarray &                           zdata,
                                         blas::matrix< std::complex< float > > &  U )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
decompress_lr< std::complex< double > > ( const zarray &                            zdata,
                                          blas::matrix< std::complex< double > > &  U )
{
    HLR_ERROR( "TODO" );
}

}}}// namespace hlr::compress::bfl

#endif // __HLR_UTILS_DETAIL_BFL_HH
