#ifndef __HLR_UTILS_DETAIL_BYTE_N_HH
#define __HLR_UTILS_DETAIL_BYTE_N_HH
//
// Project     : HLR
// Module      : utils/detail/byte_n
// Description : optimized types for multiple bytes
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

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

using byte2_t = unsigned short;

struct byte3_t
{
    byte_t  data[3];
    
    void operator = ( const uint  n )
    {
        data[0] = (n & 0x0000ff);
        data[1] = (n & 0x00ff00) >> 8;
        data[2] = (n & 0xff0000) >> 16;
    }

    operator uint () const { return ( data[2] << 16 ) | ( data[1] << 8 ) | data[0]; }
};

using byte4_t = unsigned int;

struct byte5_t
{
    byte_t  data[5];
    
    void operator = ( const ulong  n )
    {
        *reinterpret_cast< uint * >( data ) = uint(n & 0xffffffff);
        data[4] = byte_t( (n >> 32) & 0xff );
    }

    operator ulong () const { return ulong(data[4]) << 32 | ulong(*reinterpret_cast< const uint * >( data )); }
};

struct byte6_t
{
    byte_t  data[6];
    
    void operator = ( const ulong  n )
    {
        *reinterpret_cast< uint *   >( data   ) = uint( n & 0xffffffff );
        *reinterpret_cast< ushort * >( data+4 ) = ushort( (n >> 32) & 0xffff );
    }

    operator ulong () const { return ( ulong(*reinterpret_cast< const ushort * >( data+4 )) << 32 |
                                       ulong(*reinterpret_cast< const uint *   >( data   )) ); }
};

struct byte7_t
{
    byte_t  data[7];
    
    void operator = ( const ulong  n )
    {
        *reinterpret_cast< uint *   >( data ) = uint( n & 0xffffffff );

        const uint  n1 = n >> 32;
        
        data[4] = (n1 & 0x0000ff);
        data[5] = (n1 & 0x00ff00) >> 8;
        data[6] = (n1 & 0xff0000) >> 16;
    }

    operator ulong () const { return ulong(data[6]) << 48 | ulong(data[5]) << 40 | ulong(data[4]) << 32 | ulong(*reinterpret_cast< const uint *   >( data   )); }
};

using byte8_t = unsigned long;

}}// hlr::compress

#endif // __HLR_UTILS_DETAIL_BYTE_N_HH

