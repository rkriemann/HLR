#ifndef __HLR_UTILS_DETAIL_BFL_HH
#define __HLR_UTILS_DETAIL_BFL_HH
//
// Project     : HLR
// Module      : compress/bfl
// Description : bfl related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstdint>

#include <hlr/compress/byte_n.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_APLR

////////////////////////////////////////////////////////////
//
// compression using general bfl format
// - use 8 bit exponent and precision dependend mantissa (1+8+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace bfl {

using byte_t = uint8_t;

constexpr uint8_t   fp32_mant_bits = 23;

constexpr uint8_t   fp64_sign_bit  = 63;
constexpr uint64_t  fp64_sign_mask = (1ul << fp64_sign_bit);
constexpr uint8_t   fp64_mant_bits = 52;
constexpr uint64_t  fp64_mant_mask = 0x000fffffffffffff;
constexpr uint64_t  fp64_exp_mask  = 0x7ff0000000000000;

constexpr uint8_t   bfl_header_ofs = 4;

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_aplr ( const double  eps ) { return eps_to_rate( eps ); }

struct config
{
    byte_t  bitrate;
};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size       ( const zarray &  v   ) { return sizeof(v) + v.size(); }
inline size_t  compressed_size ( const zarray &  v   ) { return v.size(); }
inline config  get_config      ( const double    eps ) { return config{ eps_to_rate( eps ) }; }

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

//
// simply shift mantissa bits to the left
//
template < typename  storage_t >
struct bfl32
{
    static constexpr uint8_t  bfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 8;  // 1 sign bit, 8 exponent bits
    static constexpr uint8_t  bfl_mant_shift = fp32_mant_bits - bfl_mant_bits;
    
    
    static
    void
    compress ( const float *  data,
               const size_t   nsize,
               byte_t *       zdata )
    {
        auto  zptr = reinterpret_cast< storage_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint32_t  ival = (*reinterpret_cast< const uint32_t * >( data + i ) );
            const uint32_t  zval  = ival >> bfl_mant_shift;
            
            zptr[i] = zval;
        }// for
    }

    static
    void
    decompress ( float *         data,
                 const size_t    nsize,
                 const byte_t *  zdata )
    {
        auto  zptr = reinterpret_cast< const storage_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint32_t  zval = zptr[i];
            const uint32_t  ztmp = zval << uint32_t(bfl_mant_shift);
            
            data[i] = * reinterpret_cast< const float * >( & ztmp );
        }// for
    }
};

//
// specialization for 4 bytes: just copy memory
//
template <>
struct bfl32< byte4_t >
{
    static void  compress   ( const float *  data,
                              const size_t   nsize,
                              byte_t *       zdata )
    {
        std::copy( data, data + nsize, reinterpret_cast< float * >( zdata ) );
    }

    static void  decompress ( float *         data,
                              const size_t    nsize,
                              const byte_t *  zdata )
    {
        std::copy( reinterpret_cast< const float * >( zdata ),
                   reinterpret_cast< const float * >( zdata ) + nsize,
                   data );
    }
};

//
// extract sign, exponent and mantissa, then reduce exponent to 8 bits
// and shorten mantissa
//
template < typename  storage_t >
struct bfl64
{
    static constexpr uint8_t   bfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 8;  // 1 sign bit, 8 exponent bits
    static constexpr uint8_t   bfl_sign_bit   = bfl_mant_bits + 8;
    static constexpr uint8_t   bfl_mant_shift = fp64_mant_bits - bfl_mant_bits;
    static constexpr uint8_t   bfl_sign_shift = fp64_sign_bit  - bfl_sign_bit;
    static constexpr uint64_t  bfl_sign_mask  = (1ul    << bfl_sign_bit);
    static constexpr uint64_t  bfl_exp_mask   = (0xfful << bfl_mant_bits);
    static constexpr uint64_t  bfl_mant_mask  = (1ul    << bfl_mant_bits) - 1;
        
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
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bfl_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bfl_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381ul) << bfl_mant_bits) | mant;

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
// specialization when using 8 bytes: just copy memory
//
template <>
struct bfl64< byte8_t >
{
    static void compress   ( const double *  data,
                             const size_t    nsize,
                             byte_t *        zdata )
    {
        std::copy( data, data + nsize, reinterpret_cast< double * >( zdata ) );
    }

    static void decompress ( double *        data,
                             const size_t    nsize,
                             const byte_t *  zdata )
    {
        std::copy( reinterpret_cast< const double * >( zdata ),
                   reinterpret_cast< const double * >( zdata ) + nsize,
                   data );
    }
};

template < typename  value_t,
           typename  storage_t >
struct bfl
{
    static void  compress   ( const value_t *  data,
                              const size_t     nsize,
                              byte_t *         zdata );

    static void  decompress ( value_t *        data,
                              const size_t     nsize,
                              const byte_t *   zdata );
};

template < typename  storage_t >
struct bfl< float, storage_t >
{
    static inline void  compress   ( const float *  data,
                                     const size_t   nsize,
                                     byte_t *       zdata )
    {
        bfl32< storage_t >::compress( data, nsize, zdata );
    }

    static inline void  decompress ( float *         data,
                                     const size_t    nsize,
                                     const byte_t *  zdata )
    {
        bfl32< storage_t >::decompress( data, nsize, zdata );
    }
};

template < typename  storage_t >
struct bfl< double, storage_t >
{
    static inline void  compress   ( const double *  data,
                                     const size_t    nsize,
                                     byte_t *        zdata )
    {
        bfl64< storage_t >::compress( data, nsize, zdata );
    }

    static inline void  decompress ( double *        data,
                                     const size_t    nsize,
                                     const byte_t *  zdata )
    {
        bfl64< storage_t >::decompress( data, nsize, zdata );
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
    const size_t   nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint8_t  nbits = byte_pad( 1 + 8 + config.bitrate ); // total no. of bits per value
    const uint8_t  nbyte = nbits / 8;
    zarray         zdata( bfl_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    switch ( nbyte )
    {
        case 2  : bfl< float, byte2_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 3  : bfl< float, byte3_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 4  :
        default : bfl< float, byte4_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
    }// switch
        
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
    const size_t   nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint8_t  nbits = byte_pad( 1 + 8 + config.bitrate ); // total no. of bits per value
    const uint8_t  nbyte = nbits / 8;
    zarray         zdata( bfl_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    switch ( nbyte )
    {
        case 2  : bfl< double, byte2_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 3  : bfl< double, byte3_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 4  : bfl< double, byte4_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 5  : bfl< double, byte5_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 6  : bfl< double, byte6_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 7  : bfl< double, byte7_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
        case 8  :
        default : bfl< double, byte8_t >::compress( data, nsize, zdata.data() + bfl_header_ofs ); break;
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    nbyte = zdata[0];

    switch ( nbyte )
    {
        case 2  : bfl< float, byte2_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 3  : bfl< float, byte3_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case 4  :
        default : bfl< float, byte4_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
    }// switch
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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    nbyte = zdata[0];

    switch ( nbyte )
    {
        case  2 : bfl< double, byte2_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  3 : bfl< double, byte3_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  4 : bfl< double, byte4_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  5 : bfl< double, byte5_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  6 : bfl< double, byte6_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  7 : bfl< double, byte7_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
        case  8 :
        default : bfl< double, byte8_t >::decompress( dest, nsize, zdata.data() + bfl_header_ofs ); break;
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
    using  real_t = Hpro::real_type_t< value_t >;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    auto          b     = std::vector< uint32_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto      nprecbits = eps_to_rate_aplr( S(l) );
        const uint32_t  nbits     = byte_pad( 1 + 8 + nprecbits );
        const uint32_t  nbyte     = nbits / 8;

        b[l]   = nbyte;
        zsize += bfl_header_ofs + nbyte * n;
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
        pos       += bfl_header_ofs;

        if constexpr ( sizeof(real_t) == 4 )
        {
            switch ( nbyte )
            {
                case  2 : bfl< value_t, byte2_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  3 : bfl< value_t, byte3_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  4 :
                default : bfl< value_t, byte4_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            }// switch
        }// if
        else
        {
            switch ( nbyte )
            {
                case  2 : bfl< value_t, byte2_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  3 : bfl< value_t, byte3_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  4 : bfl< value_t, byte4_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  5 : bfl< value_t, byte5_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  6 : bfl< value_t, byte6_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  7 : bfl< value_t, byte7_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  8 :
                default : bfl< value_t, byte8_t >::compress( U.data() + l*n, n, zdata.data() + pos ); break;
            }// switch
        }// else
        
        pos += n*nbyte;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    using  real_t = Hpro::real_type_t< value_t >;

    const size_t    n   = U.nrows();
    const uint32_t  k   = U.ncols();
    size_t          pos = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const auto  nbyte = zdata[ pos ];

        pos += bfl_header_ofs;

        if constexpr ( sizeof(real_t) == 4 )
        {
            switch ( nbyte )
            {
                case  2 : bfl< value_t, byte2_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  3 : bfl< value_t, byte3_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  4 : 
                default : bfl< value_t, byte4_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            }// switch
        }// if
        else
        {
            switch ( nbyte )
            {
                case  2 : bfl< value_t, byte2_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  3 : bfl< value_t, byte3_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  4 : bfl< value_t, byte4_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  5 : bfl< value_t, byte5_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  6 : bfl< value_t, byte6_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  7 : bfl< value_t, byte7_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
                case  8 :
                default : bfl< value_t, byte8_t >::decompress( U.data() + l*n, n, zdata.data() + pos ); break;
            }// switch
        }// else
        
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

//
// compressed blas
//

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
    static constexpr uint8_t   bfl_mant_bits  = 8 * sizeof(storage_t) - 1 - 8;  // 1 sign bit, 8 exponent bits
    static constexpr uint8_t   bfl_sign_bit   = bfl_mant_bits + 8;
    static constexpr uint8_t   bfl_mant_shift = fp64_mant_bits - bfl_mant_bits;
    static constexpr uint8_t   bfl_sign_shift = fp64_sign_bit  - bfl_sign_bit;
    static constexpr uint64_t  bfl_sign_mask  = (1ul    << bfl_sign_bit);
    static constexpr uint64_t  bfl_exp_mask   = (0xfful << bfl_mant_bits);
    static constexpr uint64_t  bfl_mant_mask  = (1ul    << bfl_mant_bits) - 1;

    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = alpha * x[j];
                
                for ( size_t  i = 0; i < nrows; ++i, pos++ )
                {
                    const uint64_t  zval = zA[pos];
                    const uint64_t  sign = (zval & bfl_sign_mask) << bfl_sign_shift;
                    const uint64_t  exp  = (zval & bfl_exp_mask ) >> bfl_mant_bits;
                    const uint64_t  mant = (zval & bfl_mant_mask) << bfl_mant_shift;
                    const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
                    const double    fval = * reinterpret_cast< const double * >( & ival );
                    
                    y[i] += fval * x_j;
                }// for
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
                {
                    const uint64_t  zval = zA[pos];
                    const uint64_t  sign = (zval & bfl_sign_mask) << bfl_sign_shift;
                    const uint64_t  exp  = (zval & bfl_exp_mask ) >> bfl_mant_bits;
                    const uint64_t  mant = (zval & bfl_mant_mask) << bfl_mant_shift;
                    const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
                    const double    fval = * reinterpret_cast< const double * >( & ival );

                    y_j += fval * x[i];
                }// for

                y[j] += alpha * y_j;
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
mulvec ( const size_t     nrows,
         const size_t     ncols,
         const matop_t    op_A,
         const value_t    alpha,
         const zarray &   zA,
         const value_t *  x,
         value_t *        y )
{
    using  real_t = Hpro::real_type_t< value_t >;

    const uint8_t  nbyte = zA[0];
    
    switch ( nbyte )
    {
        case  2 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte2_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  3 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte3_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  4 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte4_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  5 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte5_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  6 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte6_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  7 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte7_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        case  8 : mulvec( nrows, ncols, op_A, alpha, reinterpret_cast< const byte8_t * >( zA.data() + bfl_header_ofs ), x, y ); break;
        default :
            HLR_ERROR( "unsupported byte size" );
    }// switch
}

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
    using  real_t = Hpro::real_type_t< value_t >;

    size_t  pos = 0;

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = zA[pos];
        
                switch ( nbyte )
                {
                    case  2 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte2_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte3_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte4_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte5_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte6_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte7_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte8_t * >( zA.data() + pos + bfl_header_ofs ), x+l, y ); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                pos += bfl_header_ofs + nbyte * nrows;
            }// for
        }// case
        break;

        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = zA[pos];
        
                switch ( nbyte )
                {
                    case  2 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte2_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  3 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte3_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  4 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte4_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  5 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte5_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  6 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte6_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  7 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte7_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    case  8 : mulvec( nrows, 1, op_A, alpha, reinterpret_cast< const byte8_t * >( zA.data() + pos + bfl_header_ofs ), x, y+l ); break;
                    default :
                        HLR_ERROR( "unsupported byte size" );
                }// switch

                pos += bfl_header_ofs + nbyte * nrows;
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::bfl

#endif // __HLR_UTILS_DETAIL_BFL_HH
