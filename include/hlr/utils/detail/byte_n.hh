#ifndef __HLR_UTILS_DETAIL_BYTE_N_HH
#define __HLR_UTILS_DETAIL_BYTE_N_HH
//
// Project     : HLR
// Module      : utils/detail/byte_n
// Description : optimized types for multiple bytes
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cstdint>

namespace hlr { namespace compress { 

// general byte type
using byte_t = unsigned char;

// return byte padded value of <n>
inline size_t byte_pad ( size_t  n )
{
    return ( n % 8 != 0 ) ? n + (8 - n%8) : n;
}

////////////////////////////////////////////////////////////
//
// datatypes to store multiple bytes
//
////////////////////////////////////////////////////////////

using byte2_t = uint16_t;

struct byte3_t
{
    byte_t  data[3];
    
    void operator = ( const uint32_t  n )
    {
        data[0] = (n & 0x0000ff);
        data[1] = (n & 0x00ff00) >> 8;
        data[2] = (n & 0xff0000) >> 16;
    }

    operator uint32_t () const { return ( data[2] << 16 ) | ( data[1] << 8 ) | data[0]; }
};

using byte4_t = uint32_t;

struct byte5_t
{
    byte_t  data[5];
    
    void operator = ( const uint64_t  n )
    {
        *reinterpret_cast< uint32_t * >( data ) = uint32_t(n & 0xffffffff);
        data[4] = byte_t( (n >> 32) & 0xff );
    }

    operator uint64_t () const { return uint64_t(data[4]) << 32 | uint64_t(*reinterpret_cast< const uint32_t * >( data )); }
};

struct byte6_t
{
    byte_t  data[6];
    
    void operator = ( const uint64_t  n )
    {
        *reinterpret_cast< uint32_t * >( data   ) = uint32_t( n & 0xffffffff );
        *reinterpret_cast< uint16_t * >( data+4 ) = uint16_t( (n >> 32) & 0xffff );
    }

    operator uint64_t () const { return ( uint64_t(*reinterpret_cast< const uint16_t * >( data+4 )) << 32 |
                                          uint64_t(*reinterpret_cast< const uint32_t * >( data   )) ); }
};

struct byte7_t
{
    byte_t  data[7];
    
    void operator = ( const uint64_t  n )
    {
        *reinterpret_cast< uint32_t * >( data ) = uint32_t( n & 0xffffffff );

        const uint32_t  n1 = n >> 32;
        
        data[4] = (n1 & 0x0000ff);
        data[5] = (n1 & 0x00ff00) >> 8;
        data[6] = (n1 & 0xff0000) >> 16;
    }

    operator uint64_t () const { return uint64_t(data[6]) << 48 | uint64_t(data[5]) << 40 | uint64_t(data[4]) << 32 | uint64_t(*reinterpret_cast< const uint32_t * >( data )); }
};

using byte8_t = unsigned long;

}}// hlr::compress

#endif // __HLR_UTILS_DETAIL_BYTE_N_HH

