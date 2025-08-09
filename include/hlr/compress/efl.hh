#ifndef __HLR_UTILS_DETAIL_EFL_HH
#define __HLR_UTILS_DETAIL_EFL_HH
//
// Project     : HLR
// Module      : compress/efl
// Description : extended floating point compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2025. All Rights Reserved.
//

#include <cstring>

#include <hlr/compress/byte_n.hh>

//
// signal availability of compressed BLAS
//
#define HLR_HAS_ZBLAS_DIRECT
#define HLR_HAS_ZBLAS_VALR

////////////////////////////////////////////////////////////
//
// compression using general efl format
// - use FP64 exponent size and precision dependend mantissa size (1+11+X bits)
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace efl {

constexpr uint8_t  efl_header_ofs = 8;

//
// return bitrate for given accuracy
//
//   |d_i - ~d_i| ≤ 2^(-m) ≤ ε with mantissa length m = ⌈-log₂ ε⌉
//
inline byte_t eps_to_rate      ( const double  eps ) { return std::max< double >( 0, std::ceil( -std::log2( eps ) ) ); }
inline byte_t eps_to_rate_valr ( const double  eps ) { return eps_to_rate( eps ); }

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
// conversion masks/indices for AVX512
//
////////////////////////////////////////////////////////////////////////////////

#if defined (__AVX512VBMI__) && defined (__EVEX512__)

//
// convert 8 FP64 <-> FP32 <-> FP16 (1-8-7)
//

static const __mmask32  to_fp16_mask_8 = 0b00000000000000001111111111111111;  // upper 16 bytes will be zero
static const int8_t     to_fp16_8[32]  = { // use upper 2 bytes for FP16
    2,  3,    6,  7,    10, 11,   14, 15,
    18, 19,   22, 23,   26, 27,   30, 31,

    -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1
};
static const auto       to_fp16_idxs_8 = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( to_fp16_8 ) );

static const __mmask32  from_fp16_mask_8 = 0b11001100110011001100110011001100;  // first two bytes are zet to zero
static const int8_t     from_fp16_8[32]  = {  
    -1, -1, 0, 1,   -1, -1,  2,  3,   -1, -1,  4,  5,   -1, -1,  6,  7,
    -1, -1, 8, 9,   -1, -1, 10, 11,   -1, -1, 12, 13,   -1, -1, 14, 15
};
static const auto       from_fp16_idxs_8 = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( from_fp16_8 ) );


//
// convert 8 FP64 <-> FP32 <-> FP24 (1-8-15)
//

static const __mmask32  to_fp24_mask_8 = 0b00000000111111111111111111111111;  // upper 8 bytes will be zero
static const int8_t     to_fp24_8[32]  = { // use upper 3 bytes for FP24
    1,   2,  3,    5,  6,  7,    9, 10, 11,   13, 14, 15,
    17, 18, 19,   21, 22, 23,   25, 26, 27,   29, 30, 31,

    -1, -1, -1, -1,   -1, -1, -1, -1
};
static const auto       to_fp24_idxs_8 = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( to_fp24_8 ) );

static const __mmask32  from_fp24_mask_8 = 0b11101110111011101110111011101110;  // first byte is zet to zero
static const int8_t     from_fp24_8[32]  = {  
    -1,  0,  1,  2,   -1,  3,  4,  5,   -1,  6,  7,  8,   -1,  9, 10, 11,
    -1, 12, 13, 14,   -1, 15, 16, 17,   -1, 18, 19, 20,   -1, 21, 22, 23
};
static const auto       from_fp24_idxs_8 = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( from_fp24_8 ) );


//
// convert 16 FP32 <-> FP24 (1-8-15)
//
static const __mmask64  to_fp24_mask_16 = 0b0000000000000000111111111111111111111111111111111111111111111111;  // upper 16 bytes will be zero
static const int8_t     to_fp24_16[64]  = {   // use upper 3 bytes for FP24
    1,   2,  3,    5,  6,  7,    9, 10, 11,   13, 14, 15,
    17, 18, 19,   21, 22, 23,   25, 26, 27,   29, 30, 31,
    33, 34, 35,   37, 38, 39,   41, 42, 43,   45, 46, 47,
    49, 50, 51,   53, 54, 55,   57, 58, 59,   61, 62, 63,
            
    -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1
};
static const auto       to_fp24_idxs_16 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( to_fp24_16 ) );

static const __mmask64  from_fp24_mask_16 = 0b1110111011101110111011101110111011101110111011101110111011101110;  // first byte is zet to zero
static const int8_t     from_fp24_16[64]  = {  
    -1,  0,  1,  2,   -1,  3,  4,  5,   -1,  6,  7,  8,   -1,  9, 10, 11,
    -1, 12, 13, 14,   -1, 15, 16, 17,   -1, 18, 19, 20,   -1, 21, 22, 23,
    -1, 24, 25, 26,   -1, 27, 28, 29,   -1, 30, 31, 32,   -1, 33, 34, 35,
    -1, 36, 37, 38,   -1, 39, 40, 41,   -1, 42, 43, 45,   -1, 45, 46, 47
};
static const auto       from_fp24_idxs_16 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( from_fp24_16 ) );

//
// convert 8 FP64 <-> FP40 (1-11-28) :   seeeeeee eeemmmmm mmmmmmmm mmmmmmmm mmmmmmmm
//                              byte :       4        3        2        1        0
//

static const __mmask64  to_fp40_mask_8 = 0b0000000000000000000000001111111111111111111111111111111111111111;  // upper 24 bytes will be zero
static const int8_t     to_fp40_8[64]  = { // use upper 5 bytes for FP40
    3,   4,  5,  6,  7,   11, 12, 13, 14, 15,
    19, 20, 21, 22, 23,   27, 28, 29, 30, 31,
    35, 36, 37, 38, 39,   43, 44, 45, 46, 47,
    51, 52, 53, 54, 55,   59, 60, 61, 62, 63,

    -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1,
    -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1
};
static const auto       to_fp40_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( to_fp40_8 ) );

static const __mmask64  from_fp40_mask_8 = 0b1111100011111000111110001111100011111000111110001111100011111000;  // first three bytes are zet to zero
static const int8_t     from_fp40_8[64]  = {  
    -1, -1, -1,  0,  1,  2,  3,  4,    -1, -1, -1,  5,  6,  7,  8,  9,
    -1, -1, -1, 10, 11, 12, 13, 14,    -1, -1, -1, 15, 16, 17, 18, 19,
    -1, -1, -1, 20, 21, 22, 23, 24,    -1, -1, -1, 25, 26, 27, 28, 29,
    -1, -1, -1, 30, 31, 32, 33, 34,    -1, -1, -1, 35, 36, 37, 38, 39
};
static const auto       from_fp40_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( from_fp40_8 ) );

//
// convert 8 FP64 <-> FP48 (1-11-36) :   seeeeeee eeemmmmm mmmmmmmm mmmmmmmm mmmmmmmm mmmmmmmm
//                              byte :       5        4        3        2        1        0
//

static const __mmask64  to_fp48_mask_8 = 0b0000000000000000111111111111111111111111111111111111111111111111;  // upper 16 bytes will be zero
static const int8_t     to_fp48_8[64]  = { // use upper 6 bytes for FP48
    2,   3,  4,  5,  6,  7,    10, 11, 12, 13, 14, 15,
    18, 19, 20, 21, 22, 23,    26, 27, 28, 29, 30, 31,
    34, 35, 36, 37, 38, 39,    42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55,    58, 59, 60, 61, 62, 63,

    -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1,   -1, -1, -1, -1,
};
static const auto       to_fp48_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( to_fp48_8 ) );

static const __mmask64  from_fp48_mask_8 = 0b1111110011111100111111001111110011111100111111001111110011111100;  // first two bytes are zet to zero
static const int8_t     from_fp48_8[64]  = {  
    -1, -1,  0,  1,  2,  3,  4,  5,    -1, -1,  6,  7,  8,  9, 10, 11
    -1, -1, 12, 13, 14, 15, 16, 17     -1, -1, 18, 19, 20, 21, 22, 23,
    -1, -1, 24, 25, 26, 27, 28, 29     -1, -1, 30, 31, 32, 33, 34, 35,
    -1, -1, 36, 37, 38, 39, 40, 41,    -1, -1, 42, 43, 44, 45, 46, 47
};
static const auto       from_fp48_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( from_fp48_8 ) );

//
// convert 8 FP64 <-> FP56 (1-11-44) :   seeeeeee eeemmmmm mmmmmmmm mmmmmmmm mmmmmmmm mmmmmmmm mmmmmmmm
//                              byte :       6        5        4        3        2        1        0
//

static const __mmask64  to_fp56_mask_8 = 0b0000000011111111111111111111111111111111111111111111111111111111;  // upper 8 bytes will be zero
static const int8_t     to_fp56_8[64]  = { // use upper 7 bytes for FP56
    1,  2,   3,  4,  5,  6,  7,     9, 10, 11, 12, 13, 14, 15,
    17, 18, 19, 20, 21, 22, 23,    25, 26, 27, 28, 29, 30, 31,
    33, 34, 35, 36, 37, 38, 39,    41, 42, 43, 44, 45, 46, 47,
    49, 50, 51, 52, 53, 54, 55,    57, 58, 59, 60, 61, 62, 63,

    -1, -1, -1, -1,   -1, -1, -1, -1,
};
static const auto       to_fp56_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( to_fp56_8 ) );

static const __mmask64  from_fp56_mask_8 = 0b1111111011111110111111101111111011111110111111101111111011111110;  // first two bytes are zet to zero
static const int8_t     from_fp56_8[64]  = {  
    -1,  0,  1,  2,  3,  4,  5,  6,    -1,  7,  8,  9, 10, 11, 12, 13
    -1, 14, 15, 16, 17, 18, 19, 20     -1, 21, 22, 23, 24, 25, 26, 27,
    -1, 28, 29, 30, 31, 32, 33, 34     -1, 35, 36, 37, 38, 39, 40, 41,
    -1, 42, 43, 44, 45, 46, 47, 48,    -1, 49, 50, 51, 52, 53, 54, 55,
};
static const auto       from_fp56_idxs_8 = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( from_fp56_8 ) );

#endif

////////////////////////////////////////////////////////////////////////////////
//
// low-level compression functions
//
////////////////////////////////////////////////////////////////////////////////

//
// return per entry byte size based on precision bits
//
inline
uint8_t
precision_byte_size ( const uint8_t  bitrate )
{
    if      ( bitrate <=  7 ) return 2;
    else if ( bitrate <= 15 ) return 3;
    else if ( bitrate <= 23 ) return 4;
    else if ( bitrate <= 28 ) return 5;
    else if ( bitrate <= 36 ) return 6;
    else if ( bitrate <= 44 ) return 7;
    else                      return 8;
}

//
// FP16 : 1-8-7  (aka BF16)
//
static
void
compress_fp16 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    #if defined (__AVX512F__)

    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 16 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vf = _mm512_cvtpd_ps( vd );
        const auto  vb = _mm256_maskz_permutexvar_epi8( to_fp16_mask_8, to_fp16_idxs_8, reinterpret_cast< __m256i >( vf ) );

        _mm256_storeu_si256( reinterpret_cast< __m256i * >( zptr ), vb );
    }// for

    #else

    auto  zptr = reinterpret_cast< byte2_t * >( zdata );

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp32int_t  v{ .f = float(data[i]) };
        
        zptr[i] = v.u >> 16;
    }// for
    
    #endif
}

static
void
decompress_fp16 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    #if defined (__AVX512F__)

    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 16 )
    {
        const auto  vb = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( zptr ) );
        const auto  vf = _mm256_maskz_permutexvar_epi8( from_fp16_mask_8, from_fp16_idxs_8, vb );
        const auto  vd = _mm512_cvtps_pd( reinterpret_cast< __m256 >( vf ) );

        _mm512_storeu_pd( data + i, vd );
    }// for
    
    #else

    auto  zptr = reinterpret_cast< const byte2_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp32int_t  v{ .u = uint32_t(zptr[i]) << 16 };
        
        data[i] = double(v.f);

        HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    }// for

    #endif
}

//
// FP24 : 1-8-15
//
static
void
compress_fp24 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 24 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vf = _mm512_cvtpd_ps( vd );
        const auto  vb = _mm256_maskz_permutexvar_epi8( to_fp24_mask_8, to_fp24_idxs_8, reinterpret_cast< __m256i >( vf ) );

        _mm256_storeu_si256( reinterpret_cast< __m256i * >( zptr ), vb );
    }// for

    #else

    auto  zptr = reinterpret_cast< byte3_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp32int_t  v{ .f = float(data[i]) };
        
        zptr[i] = v.u >> 8;
    }// for

    #endif
}

static
void
decompress_fp24 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 24 )
    {
        const auto  vb = _mm256_loadu_si256( reinterpret_cast< const __m256i * >( zptr ) );
        const auto  vf = _mm256_maskz_permutexvar_epi8( from_fp24_mask_8, from_fp24_idxs_8, vb );
        const auto  vd = _mm512_cvtps_pd( reinterpret_cast< __m256 >( vf ) );

        _mm512_storeu_pd( data + i, vd );
    }// for

    #else

    auto  zptr = reinterpret_cast< const byte3_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp32int_t  v{ .u = uint32_t( zptr[i] ) << 8 };
        
        data[i] = double(v.f);

        HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    }// for

    #endif
}

//
// FP32 : 1-8-23
//
static
void
compress_fp32 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    auto  zptr = reinterpret_cast< float * >( zdata );

    #if defined (__AVX512F__)
    
    for ( size_t  i = 0; i < nsize; i += 8 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vf = _mm512_cvtpd_ps( vd );

        _mm256_storeu_ps( zptr + i, vf );
    }// for
    
    #elif defined(__AVX__)

    for ( size_t  i = 0; i < nsize; i += 4 )
    {
        const auto  vd = _mm256_loadu_pd( data + i );
        const auto  vf = _mm256_cvtpd_ps( vd );

        _mm_storeu_ps( zptr + i, vf );
    }// for

    #else

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        zptr[i] = float( data[i] ) );
    
    #endif
}

static
void
decompress_fp32 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    auto  zptr = reinterpret_cast< const float * >( zdata );

    #if defined (__AVX512F__)
    
    for ( size_t  i = 0; i < nsize; i += 8 )
    {
        const auto  vf = _mm256_loadu_ps( zptr + i );
        const auto  vd = _mm512_cvtps_pd( vf );

        _mm512_storeu_pd( data + i, vd );
    }// for
    
    #elif defined(__AVX__)

    for ( size_t  i = 0; i < nsize; i += 4 )
    {
        const auto  vf = _mm_loadu_ps( zptr + i );
        const auto  vd = _mm256_cvtps_pd( vf );

        _mm256_storeu_pd( data + i, vd );
    }// for

    #else

    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
        data[i] = double( zptr[i] ) );
    
    #endif
}

//
// FP40 : 1-11-28
//
static
void
compress_fp40 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 40 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vb = _mm512_maskz_permutexvar_epi8( to_fp40_mask_8, to_fp40_idxs_8, reinterpret_cast< __m512i >( vd ) );

        _mm512_storeu_si512( reinterpret_cast< __m512i * >( zptr ), vb );
    }// for

    #else

    auto  zptr = reinterpret_cast< byte5_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .f = data[i] };
        
        zptr[i] = v.u >> 24;
    }// for

    #endif
}

static
void
decompress_fp40 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 40 )
    {
        const auto  vb = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( zptr ) );
        const auto  vd = reinterpret_cast< __m512d >( _mm512_maskz_permutexvar_epi8( from_fp40_mask_8, from_fp40_idxs_8, vb ) );

        _mm512_storeu_pd( data + i, vd );
    }// for

    #else

    auto  zptr = reinterpret_cast< const byte5_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .u = uint64_t( zptr[i] ) << 24 };
        
        data[i] = v.f;

        HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    }// for

    #endif
}

//
// FP48 : 1-11-36
//
static
void
compress_fp48 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 48 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vb = _mm512_maskz_permutexvar_epi8( to_fp48_mask_8, to_fp48_idxs_8, reinterpret_cast< __m512i >( vd ) );

        _mm512_storeu_si512( reinterpret_cast< __m512i * >( zptr ), vb );
    }// for

    #else

    auto  zptr = reinterpret_cast< byte6_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .f = data[i] };
        
        zptr[i] = v.u >> 16;
    }// for

    #endif
}

static
void
decompress_fp48 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 48 )
    {
        const auto  vb = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( zptr ) );
        const auto  vd = reinterpret_cast< __m512d >( _mm512_maskz_permutexvar_epi8( from_fp48_mask_8, from_fp48_idxs_8, vb ) );

        _mm512_storeu_pd( data + i, vd );
    }// for

    #else

    auto  zptr = reinterpret_cast< const byte6_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .u = uint64_t( zptr[i] ) << 16 };
        
        data[i] = v.f;

        HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    }// for

    #endif
}

//
// FP56 : 1-11-44
//
static
void
compress_fp56 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 56 )
    {
        const auto  vd = _mm512_loadu_pd( data + i );
        const auto  vb = _mm512_maskz_permutexvar_epi8( to_fp56_mask_8, to_fp56_idxs_8, reinterpret_cast< __m512i >( vd ) );

        _mm512_storeu_si512( reinterpret_cast< __m512i * >( zptr ), vb );
    }// for

    #else

    auto  zptr = reinterpret_cast< byte7_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .f = data[i] };
        
        zptr[i] = v.u >> 8;
    }// for

    #endif
}

static
void
decompress_fp56 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    #if defined (__AVX512F__)
    
    auto  zptr = zdata;
    
    for ( size_t  i = 0; i < nsize; i += 8, zptr += 56 )
    {
        const auto  vb = _mm512_loadu_si512( reinterpret_cast< const __m512i * >( zptr ) );
        const auto  vd = reinterpret_cast< __m512d >( _mm512_maskz_permutexvar_epi8( from_fp56_mask_8, from_fp56_idxs_8, vb ) );

        _mm512_storeu_pd( data + i, vd );
    }// for

    #else

    auto  zptr = reinterpret_cast< const byte7_t * >( zdata );
    
    #pragma GCC ivdep
    for ( size_t  i = 0; i < nsize; ++i )
    {
        const fp64int_t  v{ .u = uint64_t( zptr[i] ) << 8 };
        
        data[i] = v.f;

        HLR_DBG_ASSERT( std::isfinite( data[i] ) );
    }// for

    #endif
}

//
// FP64 : 1-11-52
//
static
void
compress_fp64 ( const double *  data,
                const size_t    nsize,
                byte_t *        zdata )
{
    std::memcpy( zdata, data, sizeof(double) * nsize );
}

static
void
decompress_fp64 ( double *        data,
                  const size_t    nsize,
                  const byte_t *  zdata )
{
    std::memcpy( data, zdata, sizeof(double) * nsize );
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
    HLR_ERROR( "TODO" );
    // return sfl::compress( sfl::config{ config.bitrate }, data, dim0, dim1, dim2, dim3 );
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
    const auto    nbyte = precision_byte_size( config.bitrate );
    zarray        zdata( efl_header_ofs + nbyte * nsize );

    zdata[0] = nbyte;

    switch ( nbyte )
    {
        case  2 : compress_fp16( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  3 : compress_fp24( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  4 : compress_fp32( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  5 : compress_fp40( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  6 : compress_fp48( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  7 : compress_fp56( data, nsize, zdata.data() + efl_header_ofs ); break;
        case  8 : compress_fp64( data, nsize, zdata.data() + efl_header_ofs ); break;
        default : HLR_ERROR( "invalid byte size" );
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
                       const size_t    dim3 )
{
    const size_t   nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const uint8_t  nbyte = zdata[0];

    switch ( nbyte )
    {
        case  2 : decompress_fp16( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  3 : decompress_fp24( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  4 : decompress_fp32( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  5 : decompress_fp40( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  6 : decompress_fp48( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  7 : decompress_fp56( dest, nsize, zdata.data() + efl_header_ofs ); break;
        case  8 : decompress_fp64( dest, nsize, zdata.data() + efl_header_ofs ); break;
        default : HLR_ERROR( "invalid byte size" );
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
    HLR_ERROR( "TODO" );
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
    auto          b     = std::vector< uint8_t >( k );
    size_t        zsize = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        const uint8_t  nbyte = precision_byte_size( eps_to_rate_valr( S(l) ) );

        b[l]   = nbyte;
        zsize += efl_header_ofs + n * nbyte;
    }// for

    //
    // convert each column to compressed form
    //

    auto  zdata = std::vector< byte_t >( zsize );
    auto  zptr  = zdata.data();
    auto  vptr  = U.data();
        
    for ( uint  l = 0; l < k; ++l )
    {
        const auto  nbyte = b[l];

        *zptr  = nbyte;
        zptr  += efl_header_ofs;

        switch ( nbyte )
        {
            case  2 : compress_fp16( vptr, n, zptr ); break;
            case  3 : compress_fp24( vptr, n, zptr ); break;
            case  4 : compress_fp32( vptr, n, zptr ); break;
            case  5 : compress_fp40( vptr, n, zptr ); break;
            case  6 : compress_fp48( vptr, n, zptr ); break;
            case  7 : compress_fp56( vptr, n, zptr ); break;
            case  8 : compress_fp64( vptr, n, zptr ); break;
            default : HLR_ERROR( "invalid byte size" );
        }// switch
        
        zptr += n*nbyte;
        vptr += n;
    }// for

    return zdata;
}

template <>
inline
void
decompress_lr< float > ( const zarray &           zdata,
                         blas::matrix< float > &  U )
{
    HLR_ERROR( "TODO" );
}

template <>
inline
void
decompress_lr< double > ( const zarray &            zdata,
                          blas::matrix< double > &  U )
{
    const size_t    n    = U.nrows();
    const uint32_t  k    = U.ncols();
    size_t          pos  = 0;
    auto            vptr = U.data();
    auto            zptr = zdata.data();

    for ( uint32_t  l = 0; l < k; ++l )
    {
        const uint8_t  nbyte = *zptr;

        zptr += efl_header_ofs;

        switch ( nbyte )
        {
            case  2 : decompress_fp16( vptr, n, zptr ); break;
            case  3 : decompress_fp24( vptr, n, zptr ); break;
            case  4 : decompress_fp32( vptr, n, zptr ); break;
            case  5 : decompress_fp40( vptr, n, zptr ); break;
            case  6 : decompress_fp48( vptr, n, zptr ); break;
            case  7 : decompress_fp56( vptr, n, zptr ); break;
            case  8 : decompress_fp64( vptr, n, zptr ); break;
            default : HLR_ERROR( "invalid byte size" );
        }// switch
        
        zptr += nbyte * n;
        vptr += n;
    }// for
}

//
// compressed blas
//

namespace
{

inline
void
mulvec ( const uint8_t   nbyte,
         const size_t    nrows,
         const size_t    ncols,
         const matop_t   op_A,
         const float     alpha,
         const byte_t *  zA,
         const float *   x,
         float *         y )
{
    HLR_ERROR( "TODO" );
}

inline
void
mulvec ( const uint8_t   nbyte,
         const size_t    nrows,
         const size_t    ncols,
         const matop_t   op_A,
         const double    alpha,
         const byte_t *  zA,
         const double *  x,
         double *        y )
{
    using  value_t = double;
    
    constexpr size_t  NB  = 64;
    value_t           row[ NB ];

    HLR_DBG_ASSERT( nrows % NB == 0 );
    
    switch ( op_A )
    {
        case  apply_normal :
        {
            size_t  pos = 0;
            
            for ( size_t  j = 0; j < ncols; ++j )
            {
                const auto  x_j = alpha * x[j];

                for ( size_t  i = 0; i < nrows; i += NB )
                {
                    switch ( nbyte )
                    {
                        case  2 : decompress_fp16( row, NB, zA ); break;
                        case  3 : decompress_fp24( row, NB, zA ); break;
                        case  4 : decompress_fp32( row, NB, zA ); break;
                        case  5 : decompress_fp40( row, NB, zA ); break;
                        case  6 : decompress_fp48( row, NB, zA ); break;
                        case  7 : decompress_fp56( row, NB, zA ); break;
                        case  8 : decompress_fp64( row, NB, zA ); break;
                        default : HLR_ERROR( "invalid byte size" );
                    }// switch

                    zA += NB * nbyte;
                    
                    for ( size_t  k = 0; k < NB; ++k )
                        y[i+k] += row[k] * x_j;
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
                
                for ( size_t  i = 0; i < nrows; i += NB )
                {
                    switch ( nbyte )
                    {
                        case  2 : decompress_fp16( row, NB, zA ); break;
                        case  3 : decompress_fp24( row, NB, zA ); break;
                        case  4 : decompress_fp32( row, NB, zA ); break;
                        case  5 : decompress_fp40( row, NB, zA ); break;
                        case  6 : decompress_fp48( row, NB, zA ); break;
                        case  7 : decompress_fp56( row, NB, zA ); break;
                        case  8 : decompress_fp64( row, NB, zA ); break;
                        default : HLR_ERROR( "invalid byte size" );
                    }// switch

                    zA += NB * nbyte;
                    
                    for ( size_t  k = 0; k < NB; ++k )
                        y_j += row[k] * x[i+k];
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
    
    mulvec( nbyte, nrows, ncols, op_A, alpha, zA.data() + efl_header_ofs, x, y );
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

    auto  zptr = zA.data();

    switch ( op_A )
    {
        case  apply_normal :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = *zptr;

                zptr += efl_header_ofs;
                mulvec( nbyte, nrows, 1, op_A, alpha, zptr, x+l, y );
                zptr += nbyte * nrows;
            }// for
        }// case
        break;
        
        case  apply_conjugate  : HLR_ERROR( "TODO" );
            
        case  apply_transposed : HLR_ERROR( "TODO" );

        case  apply_adjoint :
        {
            for ( uint  l = 0; l < ncols; ++l )
            {
                const uint8_t  nbyte = *zptr;
        
                zptr += efl_header_ofs;
                mulvec( nbyte, nrows, 1, op_A, alpha, zptr, x, y+l );
                zptr += nbyte * nrows;
            }// for
        }// case
        break;
    }// switch
}

}}}// namespace hlr::compress::efl

#endif // __HLR_UTILS_DETAIL_EFL_HH
