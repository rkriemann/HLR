#ifndef __HLR_UTILS_DETAIL_DFL_HH
#define __HLR_UTILS_DETAIL_DFL_HH
//
// Project     : HLR
// Module      : utils/detail/dfl
// Description : dfl related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/detail/bfloat.hh>

////////////////////////////////////////////////////////////
//
// compression using general dfl format
// - use FP64 exponent size and precision dependend mantissa size (1+11+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace dfl {

using byte_t = uint8_t;

constexpr uint64_t  fp64_sign_mask = (1ul << 63);
constexpr uint32_t  fp64_mant_bits = 52;
constexpr uint64_t  fp64_mant_mask = 0x000fffffffffffff;
constexpr uint64_t  fp64_exp_mask  = 0x7ff0000000000000;

constexpr uint32_t  bf_header_ofs  = 1;

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

inline
void
compress_fp64 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata,
                const uint32_t  nbyte )
{
    if ( nbyte == 2 )
    {
        //
        // 1 + 11 + 4 bits
        //

        constexpr uint32_t  bf_mant_bits  = 4;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte2_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        //
        // 1 + 11 + 12 bits
        //

        constexpr uint32_t  bf_mant_bits  = 12;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte3_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        //
        // 1 + 11 + 20 bits
        //

        constexpr uint32_t  bf_mant_bits  = 20;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte4_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 5 )
    {
        //
        // 1 + 11 + 28 bits
        //

        constexpr uint32_t  bf_mant_bits  = 28;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte5_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        //
        // 1 + 11 + 36 bits
        //

        constexpr uint32_t  bf_mant_bits  = 36;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte6_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        //
        // 1 + 11 + 44 bits
        //

        constexpr uint32_t  bf_mant_bits  = 44;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        
        auto  zptr = reinterpret_cast< byte7_t * >( zdata );

        for ( size_t  i = 0; i < nsize; ++i )
        {
            const double    fval  = data[i];
            const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & fval ) );
            const uint32_t  exp  = (ival & fp64_exp_mask ) >> fp64_mant_bits;
            const uint64_t  mant = (ival & fp64_mant_mask) >> bf_mant_shift;
            const uint64_t  sign = (ival & fp64_sign_mask) >> bf_sign_shift;
            const uint64_t  zval = sign | (uint64_t(exp - 0x381u) << bf_mant_bits) | mant;

            zptr[i] = zval;
        }// for
    }// if
    else if ( nbyte == 8 )
    {
        //
        // BF64 -> higher precision than FP64, so leave data untouched
        //
        
        std::copy( reinterpret_cast< const double * >( data ),
                   reinterpret_cast< const double * >( data + nsize ),
                   zdata );
    }// if
    else
        HLR_ERROR( "unsupported storage size" );
}

template < typename value_t >
void
decompress_fp64 ( value_t *       data,
                  const size_t    nsize,
                  const byte_t *  zdata,
                  const size_t    nbyte )
{
    if ( nbyte == 2 )
    {
        constexpr uint32_t  bf_mant_bits  = 4;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        auto  zptr = reinterpret_cast< const byte2_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 3 )
    {
        constexpr uint32_t  bf_mant_bits  = 12;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        auto  zptr = reinterpret_cast< const byte3_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 4 )
    {
        constexpr uint32_t  bf_mant_bits  = 20;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        auto  zptr = reinterpret_cast< const byte4_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 5 )
    {
        constexpr uint32_t  bf_mant_bits  = 28;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;
        
        auto  zptr = reinterpret_cast< const byte5_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 6 )
    {
        constexpr uint32_t  bf_mant_bits  = 36;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;

        auto  zptr = reinterpret_cast< const byte6_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 7 )
    {
        constexpr uint32_t  bf_mant_bits  = 44;
        constexpr uint32_t  bf_sign_bit   = bf_mant_bits + 11;
        constexpr uint32_t  bf_mant_shift = fp64_mant_bits - bf_mant_bits;
        constexpr uint32_t  bf_sign_shift = 63 - bf_sign_bit;
        constexpr uint64_t  bf_sign_mask  = (1ul    << bf_sign_bit);
        constexpr uint64_t  bf_exp_mask   = (0xfful << bf_mant_bits);
        constexpr uint64_t  bf_mant_mask  = (1ul    << bf_mant_bits) - 1;

        auto  zptr = reinterpret_cast< const byte7_t * >( zdata );
        
        for ( size_t  i = 0; i < nsize; ++i )
        {
            const uint64_t  zval = zptr[i];
            const uint64_t  sign = (zval & bf_sign_mask) << bf_sign_shift;
            const uint64_t  exp  = (zval & bf_exp_mask ) >> bf_mant_bits;
            const uint64_t  mant = (zval & bf_mant_mask) << bf_mant_shift;
            const uint64_t  ival = sign | ((exp + 0x381ul) << fp64_mant_bits) | mant;
            const double    fval = * reinterpret_cast< const double * >( & ival );
        
            data[i] = fval;
        }// for
    }// if
    else if ( nbyte == 8 )
    {
        std::copy( reinterpret_cast< const double * >( zdata ),
                   reinterpret_cast< const double * >( zdata + nsize ),
                   data );
    }// if
}

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
    return bfloat::compress( bfloat::config( config.bitrate ), data, dim0, dim1, dim2, dim3 );
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
    zarray          zdata( bf_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    compress_fp64( data, nsize, zdata.data() + bf_header_ofs, nbyte );
        
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
    if ( dim1 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, 2 );
    else
        return compress< float >( config, reinterpret_cast< float * >( data ), dim0, dim1, dim2, dim3 * 2 );
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
    if ( dim1 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, 2 );
    else
        return compress< double >( config, reinterpret_cast< double * >( data ), dim0, dim1, dim2, dim3 * 2 );
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
    bfloat::decompress( zdata, dest, dim0, dim1, dim2, dim3 );
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
    const uint32_t    nbyte = zdata[0];

    decompress_fp64( dest, nsize, zdata.data() + bf_header_ofs, nbyte );
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
    if ( dim1 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, 2 );
    else
        decompress< float >( zdata, reinterpret_cast< float * >( dest ), dim0, dim1, dim2, dim3 * 2 );
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
    if ( dim1 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, 2, 0, 0 );
    else if ( dim2 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, 2, 0 );
    else if ( dim3 == 0 )
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, 2 );
    else
        decompress< double >( zdata, reinterpret_cast< double * >( dest ), dim0, dim1, dim2, dim3 * 2 );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// special version for lowrank matrices
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > &                       U,
              const blas::vector< Hpro::real_type_t< value_t > > &  S );

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U );

template <>
inline
zarray
compress_lr< float > ( const blas::matrix< float > &  U,
                       const blas::vector< float > &  S )
{
    return bfloat::compress_lr( U, S );
}

template <>
inline
zarray
compress_lr< double > ( const blas::matrix< double > &  U,
                        const blas::vector< double > &  S )
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

        compress_fp64( U.data() + l*n, n, zdata.data() + pos, nbyte );
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
    bfloat::decompress_lr( zdata, U );
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

        decompress_fp64( U.data() + l * n, n, zdata.data() + pos, nbyte );
        pos += nbyte * n;
    }// for
}

template <>
inline
void
decompress_lr< std::complex< double > > ( const zarray &                            zdata,
                                          blas::matrix< std::complex< double > > &  U )
{
    HLR_ERROR( "TODO" );
}

}}}// namespace hlr::compress::dfl

#endif // __HLR_UTILS_DETAIL_DFL_HH
