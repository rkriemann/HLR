#ifndef __HLR_COMPRESS_BITSTREAM_HH
#define __HLR_COMPRESS_BITSTREAM_HH
//
// Project     : HLR
// Module      : compress/bitstream
// Description : bitstream based on (copied from) ZFP bitstream
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdint>

namespace hlr { namespace compress {

//
// based on ZFP bitstream but simplified internal usage
//
// TODO: if size of given buffer is not multiple of 8 byte, data
//       is written beyond buffer end
//
struct bitstream
{
public:
    constexpr static auto  WSIZE = sizeof(uint64_t) * CHAR_BIT;

    size_t      bits;   // number of buffered bits (0 <= bits < WSIZE)
    uint64_t    buffer; // incoming/outgoing bits (buffer < 2^bits)
    uint64_t *  ptr;    // pointer to next word to be read/written
    uint64_t *  begin;  // beginning of stream
    uint64_t *  end;    // end of stream (not enforced)

public:
    //
    // ctor for given buffer
    //
    bitstream ( void *  data,
                size_t  bytes )
    {
        begin  = reinterpret_cast< uint64_t * >( data );
        end    = begin + bytes / sizeof(uint64_t);

        ptr    = begin;
        buffer = 0;
        bits   = 0;
    }

    //
    // dtor
    //
    ~bitstream ()
    {
        flush();
    }
    
    //
    // write given bits to stream
    //
    uint64_t
    write_bits ( uint64_t  value,
                 size_t    n )
    {
        constexpr auto  WSIZE = sizeof(uint64_t) * CHAR_BIT;
        
        // append bit string to buffer
        buffer += value << bits;
        bits   += n;
        
        // is buffer full?
        if ( bits >= WSIZE )
        {
            // 1 <= n <= 64; decrement n to ensure valid right shifts below
            value >>= 1;
            n--;
            
            // assert: 0 <= n < 64; WSIZE <= s->bits <= WSIZE + n
            do
            {
                
                bits  -= WSIZE;                // output WSIZE bits while buffer is full
                *ptr++ = buffer;               // assert: 0 <= s->bits <= n
                buffer = value >> (n - bits);  // assert: 0 <= n - s->bits < 64
            } while ( bits >= WSIZE );
        }// if
        
        // assert: 0 <= s->bits < WSIZE
        buffer &= (1ul << bits) - 1;
        
        // assert: 0 <= n < 64
        return value >> n;
    }

    //
    // return <n> bits from stream
    //
    uint64_t
    read_bits ( const size_t  n )
    {
        constexpr auto  WSIZE = sizeof(uint64_t) * CHAR_BIT;

        uint64_t  value = buffer;
        
        if ( bits < n )
        {
            // keep fetching WSIZE bits until enough bits are buffered
            do {
                buffer = *ptr++; // assert: 0 <= bits < n <= 64
                value += buffer << bits;
                bits  += WSIZE;
            } while ( bits < n );

            // assert: 1 <= n <= bits < n + WSIZE
            bits -= n;
            
            if ( ! bits )
            {
                // value holds exactly n bits; no need for masking
                buffer = 0;
            }// if
            else
            {
                buffer >>= WSIZE - bits;         // assert: 1 <= bits < WSIZE
                value   &= (2ul << (n - 1)) - 1; // assert: 1 <= n <= 64
            }// else
        }// if
        else
        {
            // assert: 0 <= n <= bits < WSIZE <= 64
            bits    -= n;
            buffer >>= n;
            value   &= (1ul << n) - 1;
        }// else
        
        return value;
    }

    //
    // write remaining bits in buffer to stream
    //
    void
    flush ()
    {
        auto  rest = (WSIZE - bits) % WSIZE;
        
        if ( rest )
        {
            auto  b = bits;
            
            for ( b += rest; b >= WSIZE; b -= WSIZE )
            {
                *ptr++ = buffer;
                buffer = 0;
            }// for
            
            bits = b;
        }// if
    }
};

}}// namespace hlr::compress

#endif // __HLR_COMPRESS_BITSTREAM_HH
