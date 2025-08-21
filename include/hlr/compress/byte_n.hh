#ifndef __HLR_UTILS_DETAIL_BYTE_N_HH
#define __HLR_UTILS_DETAIL_BYTE_N_HH
//
// Project     : HLR
// Module      : compress/byte_n
// Description : optimized types for multiple bytes
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdint>

namespace hlr { namespace compress { 

// general byte type
using byte_t = uint8_t;

// return byte padded value of <n>
constexpr size_t byte_pad ( const size_t  n )
{
    return ( n % 8 != 0 ) ? n + (8 - n % 8) : n;
}

////////////////////////////////////////////////////////////
//
// shared float/int union
//
////////////////////////////////////////////////////////////

union fp32int_t
{
    float     f;
    uint32_t  u;
};
    
union fp64int_t
{
    double    f;
    uint64_t  u;
};
    
////////////////////////////////////////////////////////////
//
// datatypes to store multiple bytes
//
////////////////////////////////////////////////////////////

using byte1_t = uint8_t;
using byte2_t = uint16_t;
using byte4_t = uint32_t;
using byte8_t = uint64_t;

struct byte3_t
{
    byte_t  data[3];

    byte3_t ( const uint32_t  n )
    {
        *this = n;
    }
    
    void operator = ( const uint32_t  n )
    {
        data[0] = (n & 0x0000ff);
        data[1] = (n & 0x00ff00) >> 8;
        data[2] = (n & 0xff0000) >> 16;
    }

    void operator = ( const uint64_t  n )
    {
        data[0] = (n & 0x0000ff);
        data[1] = (n & 0x00ff00) >> 8;
        data[2] = (n & 0xff0000) >> 16;
    }

    operator uint32_t () const { return ( data[2] << 16 ) | ( data[1] << 8 ) | data[0]; }
};

struct __attribute__((packed)) byte5_t
{
    uint32_t  data0;
    uint8_t   data1;
    
    byte5_t ( const uint64_t  n )
    {
        *this = n;
    }
    
    void operator = ( const uint64_t  n )
    {
        data0 = uint32_t( n & 0xffffffff );
        data1 = uint8_t( (n >> 32) & 0xff );
    }

    operator uint64_t () const { return uint64_t(data1) << 32 | uint64_t(data0); }
};

struct __attribute__((packed)) byte6_t
{
    uint32_t  data0;
    uint16_t  data1;
    
    byte6_t ( const uint64_t  n )
    {
        *this = n;
    }
    
    void operator = ( const uint64_t  n )
    {
        data0 = uint32_t( n & 0xffffffff );
        data1 = uint16_t( (n >> 32) & 0xffff );
    }

    operator uint64_t () const { return uint64_t(data1) << 32 | uint64_t(data0); }
};

struct __attribute__((packed)) byte7_t
{
    uint32_t  data0;
    uint16_t  data1;
    uint8_t   data2;
    
    byte7_t ( const uint64_t  n )
    {
        *this = n;
    }
    
    void operator = ( const uint64_t  n )
    {
        data0 = uint32_t( n & 0xffffffff );
        data1 = uint16_t( (n >> 32) & 0xffff );
        data2 = uint8_t( (n >> 48) & 0xff );
    }

    operator uint64_t () const { return uint64_t(data2) << 48 | uint64_t(data1) << 32 | uint64_t(data0); }
};

}}// hlr::compress

#endif // __HLR_UTILS_DETAIL_BYTE_N_HH

