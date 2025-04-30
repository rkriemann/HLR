#ifndef __HLR_UTILS_DETAIL_SFL2_HH
#define __HLR_UTILS_DETAIL_SFL2_HH
//
// Project     : HLR
// Module      : compress/sfl2
// Description : compression with fixed exponent and variable mantissa
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstring>
#include <cstdint>
#include <limits>

#include <hlr/arith/blas.hh>
#include <hlr/compress/byte_n.hh>
#include <hlr/compress/ztypes.hh>

// activate/deactivate bitstreams
#define HLR_USE_BITSTREAM
#include <hlr/compress/bitstream.hh>

////////////////////////////////////////////////////////////
//
// compression using adaptive float representation
//
// - fixed exponent size based on datatype (FP32/FP64)
// - mantissa size depends on precision
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace sfl2 {

template < typename real_t >
struct fp_info
{};

template <>
struct fp_info< float >
{
    constexpr static uint32_t  exp_bits  = 8;
    constexpr static uint32_t  mant_bits = 23;
};
    
template <>
struct fp_info< double >
{
    constexpr static uint32_t  exp_bits  = 11;
    constexpr static uint32_t  mant_bits = 52;
    constexpr static uint8_t   sign_bit  = 63;
    constexpr static uint64_t  sign_mask = (1ul << sign_bit);
    constexpr static uint64_t  mant_mask = 0x000fffffffffffff;
    constexpr static uint64_t  exp_mask  = 0x7ff0000000000000;
};

using FP32 = fp_info< float >;
using FP64 = fp_info< double >;

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 1, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ); }

struct config
{
    byte_t  bitrate;
};

inline
std::ostream &
operator << ( std::ostream &  os, const config &  conf )
{
    return os << "rate " << conf.bitrate;
}

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
// compress data needing more than 32 bits
//
inline
void
compress ( const float *   data,
           const size_t    nsize,
           byte_t *        zdata,
           const uint32_t  prec_bits )
{
    const uint32_t  nbits = 1 + FP32::exp_bits + prec_bits;
    const uint32_t  shift = FP32::mant_bits - prec_bits;

    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    zdata[0] = prec_bits;

    //
    // compress data in "vectorized" form
    //
        
    size_t    pos  = 1; // position in compressed storage
    uint32_t  bpos = 0; // start bit position in current byte

    const size_t  bssize = pad_bs< uint32_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< uint32_t >( zdata + pos, bssize );
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const float     val   = data[i];
        const uint32_t  isval = (*reinterpret_cast< const uint32_t * >( & val ) );
        const uint32_t  zval  = ( isval >> shift );

        bs.write_bits( zval, nbits );
    }// for
}

inline
void
decompress ( float *         data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint32_t  prec_bits )
{
    const uint32_t  nbits = 1 + FP32::exp_bits + prec_bits;
    const uint32_t  shift = FP32::mant_bits - prec_bits;

    //
    // decompress in "vectorised" form
    //
        
    size_t    pos  = 1;
    uint32_t  bpos = 0;                          // bit position in current byte

    const size_t  bssize = pad_bs< uint32_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< uint32_t >( const_cast< byte_t * >( zdata ) + pos, bssize );
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const uint32_t  zval  = bs.read_bits( nbits );
        const uint32_t  irval = ( zval << shift );
        const float     rval  = * reinterpret_cast< const float * >( & irval );

        data[i] = rval;
    }// for
}

//
// compress data needing more than 32 bits
//
inline
void
compress ( const double *  data,
           const size_t    nsize,
           byte_t *        zdata,
           const uint32_t  prec_bits )
{
    const uint32_t  nbits          = 1 + FP32::exp_bits + prec_bits;
    const uint8_t   sfl_mant_bits  = prec_bits;
    const uint8_t   sfl_sign_bit   = sfl_mant_bits + 8;
    const uint8_t   sfl_mant_shift = FP64::mant_bits - sfl_mant_bits;
    const uint8_t   sfl_sign_shift = FP64::sign_bit  - sfl_sign_bit;
    
    //
    // store header (exponents and precision bits and scaling factor)
    //
    
    zdata[0] = prec_bits;

    //
    // compress data in "vectorized" form
    //
        
    size_t        pos    = 1; // position in compressed storage
    const size_t  bssize = pad_bs< uint64_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< uint64_t >( zdata + pos, bssize );
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const double    val  = data[i];
        const uint64_t  ival = (*reinterpret_cast< const uint64_t * >( & val ) );
        const uint32_t  exp  = (ival & FP64::exp_mask ) >> FP64::mant_bits;
        const uint64_t  mant = (ival & FP64::mant_mask) >> sfl_mant_shift;
        const uint64_t  sign = (ival & FP64::sign_mask) >> sfl_sign_shift;
        uint64_t        zval = 0;

        HLR_DBG_ASSERT( exp - 0x381ul >= 0 );
        
        if ( exp == 0 ) zval = sign | mant;
        else            zval = sign | (uint64_t(exp - 0x381ul) << sfl_mant_bits) | mant;

        bs.write_bits( zval, nbits );
    }// for
}

inline
void
decompress ( double *        data,
             const size_t    nsize,
             const byte_t *  zdata,
             const uint32_t  prec_bits )
{
    const uint32_t  nbits          = 1 + FP32::exp_bits + prec_bits;
    const uint8_t   sfl_mant_bits  = prec_bits;
    const uint8_t   sfl_sign_bit   = sfl_mant_bits + 8;
    const uint8_t   sfl_mant_shift = FP64::mant_bits - sfl_mant_bits;
    const uint8_t   sfl_sign_shift = FP64::sign_bit  - sfl_sign_bit;
    const uint64_t  sfl_sign_mask  = (1ul    << sfl_sign_bit);
    const uint64_t  sfl_exp_mask   = (0xfful << sfl_mant_bits);
    const uint64_t  sfl_mant_mask  = (1ul    << sfl_mant_bits) - 1;

    //
    // decompress in "vectorised" form
    //
        
    size_t        pos    = 1;
    const size_t  bssize = pad_bs< uint64_t >( byte_pad( nsize * nbits ) / 8 );
    auto          bs     = bitstream< uint64_t >( const_cast< byte_t * >( zdata ) + pos, bssize );
    
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const uint64_t  zval = bs.read_bits( nbits );
        const uint64_t  sign = (zval & sfl_sign_mask) << sfl_sign_shift;
        const uint64_t  exp  = (zval & sfl_exp_mask ) >> sfl_mant_bits;
        const uint64_t  mant = (zval & sfl_mant_mask) << sfl_mant_shift;
        fp64int_t       val{ 0 };
        
        if ( exp == 0 ) val.u = sign | mant;
        else            val.u = sign | ((exp + 0x381ul) << FP64::mant_bits) | mant;
        
        data[i] = val.f;
    }// for
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
           const size_t     dim3 = 0 )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    //
    // look for min/max value (> 0!)
    //
    
    const size_t    nsize     = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint32_t  prec_bits = std::min< uint32_t >( fp_info< real_t >::mant_bits, config.bitrate );        // total no. of bits per value
    const size_t    nbits     = 1 + FP32::exp_bits + prec_bits;                                                    // number of bits per value
    auto            zdata     = std::vector< byte_t >( 1 + 1 + pad_bs< uint32_t >( byte_pad( nsize * nbits ) / 8 ) );

    compress( data, nsize, zdata.data(), prec_bits );

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
    using  real_t = Hpro::real_type_t< value_t >;
    
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );

    //
    // read compression header (scaling, exponent and precision bits)
    //
    
    const uint32_t  prec_bits = zdata[0];
    const uint32_t  nbits     = 1 + FP32::exp_bits + prec_bits;
    
    HLR_ASSERT( nbits     <= sizeof(value_t) * 8 );
    HLR_ASSERT( prec_bits <= fp_info< real_t >::mant_bits );

    if ( prec_bits == 0 )
    {
        // zero data
        for ( size_t  i = 0; i < nsize; ++i )
            dest[i] = value_t(0);
    }// if
    else
        decompress( dest, nsize, zdata.data(), prec_bits );
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
    
    constexpr auto  exp_bits   = fp_info< real_t >::exp_bits;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n = U.nrows();
    const size_t  k = U.ncols();
    auto          m = std::vector< uint32_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        m[l] = eps_to_rate_valr( S(l) );

        const size_t  nbits = 1 + exp_bits + m[l]; // number of bits per value
        
        zsize += 1 + 1 + pad_bs< uint32_t >( byte_pad( n * nbits ) / 8 );
    }// for

    // for ( uint32_t  l = 0; l < k; ++l )
    //     std::cout << e[l] << '/' << m[l] << std::endl;
    // std::cout << std::endl;

    //
    // convert each column to compressed form
    //

    auto              zdata       = std::vector< byte_t >( zsize );
    size_t            pos         = 0;
    constexpr size_t  header_size = 2;
    const real_t *    U_ptr       = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  prec_bits = m[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U.data() + l*n, n, zdata.data() + pos, exp_bits, prec_bits );
        pos += header_size + pad_bs< uint32_t >( byte_pad( n * nbits ) / 8 );
    }// for

    return zdata;
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
    using  real_t = double;
    
    constexpr auto  exp_bits   = fp_info< real_t >::exp_bits;
    
    //
    // first, determine exponent bits and mantissa bits for all columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    const size_t  n2    = 2 * n;
    auto          m     = std::vector< uint32_t >( k );
    size_t        zsize = 0;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        m[l] = eps_to_rate_valr( S(l) );

        const size_t  nbits = 1 + exp_bits + m[l]; // number of bits per value
        
        zsize += 1 + pad_bs< uint32_t >( byte_pad( n2 * nbits ) / 8 ); // twice because real+imag
    }// for

    //
    // convert each column to compressed form
    //

    auto              zdata       = std::vector< byte_t >( zsize );
    size_t            pos         = 0;
    constexpr size_t  header_size = 1;
    const real_t *    U_ptr       = reinterpret_cast< const real_t * >( U.data() );
        
    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint32_t  prec_bits = m[l];
        const size_t    nbits     = 1 + exp_bits + prec_bits; // number of bits per value

        compress( U_ptr + l * n2, n2, zdata.data() + pos, prec_bits );
        pos += header_size + pad_bs< uint32_t >( byte_pad( n2 * nbits ) / 8 );
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const size_t      n           = U.nrows();
    const uint32_t    k           = U.ncols();
    size_t            pos         = 0;
    constexpr size_t  header_size = 1;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  prec_bits = zdata[ pos ];
        const uint32_t  nbits     = 1 + FP32::exp_bits + prec_bits;

        decompress( U.data() + l * n, n, zdata.data() + pos, prec_bits );
        pos += header_size + pad_bs< uint32_t >( byte_pad( nbits * n ) / 8 );
    }// for
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
    using  real_t = double;
    
    const size_t      n           = U.nrows();
    const uint32_t    k           = U.ncols();
    size_t            pos         = 0;
    constexpr size_t  header_size = 2;
    real_t *          U_ptr       = reinterpret_cast< real_t * >( U.data() );
    const size_t      n2          = 2 * n;

    for ( uint32_t  l = 0; l < k; ++l )
    {
        //
        // read compression header (scaling, exponent and precision bits)
        // and decompress data
        //
    
        const uint32_t  exp_bits  = zdata[ pos ];
        const uint32_t  prec_bits = zdata[ pos+1 ];
        const uint32_t  nbits     = 1 + exp_bits + prec_bits;

        decompress( U_ptr + l * n2, n2, zdata.data() + pos, prec_bits );
        pos += header_size + pad_bs< uint32_t >( byte_pad( nbits * n2 ) / 8 );
    }// for
}

}}}// namespace hlr::compress::sfl2

#endif // __HLR_UTILS_DETAIL_SFL2_HH
