#ifndef __HLR_UTILS_DETAIL_ZLIB_HH
#define __HLR_UTILS_DETAIL_ZLIB_HH
//
// Project     : HLR
// Module      : utils/detail/zlib
// Description : ZLIB related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HLR_HAS_ZLIB)

#include <zlib.h>

namespace hlr { namespace compress { namespace zlib {

//
// fixed compression mode
//
struct config {};

inline config  get_config  ( const double  /* acc */  ) { return config{}; }

// holds compressed data
using  zarray = std::vector< Bytef >;

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

//
// compression function
//
template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    auto          csize = compressBound( nsize * sizeof(value_t) );
    zarray        zdata( csize );

    const auto  retval = compress2( zdata.data(), & csize,
                                    reinterpret_cast< const Bytef * >( data ), nsize * sizeof(value_t),
                                    9 ); // Z_DEFAULT_COMPRESSION );

    HLR_ASSERT( retval == Z_OK );

    zdata.resize( csize );

    return zdata;
}

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{
    const size_t  nsize  = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    uLongf        dsize  = nsize * sizeof(value_t);
    const auto    retval = uncompress( reinterpret_cast< Bytef * >( dest ), & dsize,
                                       zdata.data(), zdata.size() );

    HLR_ASSERT( retval == Z_OK );
    HLR_ASSERT( dsize  == nsize * sizeof(value_t) );
}

}}}// namespace hlr::compress::zlib

#endif // HLR_HAS_ZLIB

#endif // __HLR_UTILS_DETAIL_ZLIB_HH
