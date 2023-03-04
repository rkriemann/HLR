#ifndef __HLR_UTILS_DETAIL_ZSTD_HH
#define __HLR_UTILS_DETAIL_ZSTD_HH
//
// Project     : HLR
// Module      : utils/detail/zstd
// Description : ZSTD related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HAS_ZSTD)

#include <zstd.h>

namespace hlr { namespace compress { namespace zstd {

//
// fixed compression mode
//
struct config {};

inline config  get_config  ( const double  /* acc */  ) { return config{}; }

// holds compressed data
using  zarray = std::vector< unsigned char >;

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
    auto          max_csize = ZSTD_compressBound( nsize * sizeof(value_t) );
    zarray        zdata( max_csize );

    const auto  csize = ZSTD_compress( reinterpret_cast< void * >( zdata.data() ), max_csize,
                                       reinterpret_cast< const void * >( data ), nsize * sizeof(value_t),
                                       ZSTD_maxCLevel() ); // ZSTD_CLEVEL_DEFAULT );

    HLR_ASSERT( ! ZSTD_isError( csize ) );

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
    const auto    dsize = ZSTD_decompress( reinterpret_cast< void * >( dest ), nsize * sizeof(value_t),
                                           reinterpret_cast< const void * >( zdata.data() ), zdata.size() );

    HLR_ASSERT( ! ZSTD_isError( dsize ) );
    HLR_ASSERT( dsize  == nsize * sizeof(value_t) );
}

}}}// namespace hlr::compress::zstd

#endif // HAS_ZSTD

#endif // __HLR_UTILS_DETAIL_ZSTD_HH
