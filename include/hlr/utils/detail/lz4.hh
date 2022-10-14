#ifndef __HLR_UTILS_DETAIL_LZ4_HH
#define __HLR_UTILS_DETAIL_LZ4_HH
//
// Project     : HLR
// Module      : utils/detail/lz4
// Description : LZ4 related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#if defined(HAS_LZ4)

#include <lz4.h>

namespace hlr { namespace compress { namespace lz4 {

//
// fixed compression mode
//
struct config {};

inline config  get_config  ( const double  /* acc */  ) { return config{}; }

// holds compressed data
using  zarray = std::vector< char >;

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
    const size_t  nsize     = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const size_t  max_csize = LZ4_compressBound( nsize * sizeof(value_t) );
    zarray        zdata( max_csize );

    const auto  csize = LZ4_compress_default( reinterpret_cast< const char * >( data ), zdata.data(), nsize * sizeof(value_t), max_csize );

    HLR_ASSERT( csize > 0 );

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
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    const auto    dsize = LZ4_decompress_safe( zdata.data(), reinterpret_cast< char * >( dest ), zdata.size(), nsize * sizeof(value_t) );

    HLR_ASSERT( dsize >= 0 );
}

}}}// namespace hlr::compress::lz4

#endif // HAS_LZ4

#endif // __HLR_UTILS_DETAIL_LZ4_HH
