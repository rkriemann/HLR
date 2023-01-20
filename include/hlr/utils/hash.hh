#ifndef __HLR_UTILS_HASH_HH
#define __HLR_UTILS_HASH_HH
//
// Project     : HLib
// Module      : hash.hh
// Description : hash functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TIndexSet.hh>

namespace hlr { 

namespace hpro = HLIB;

using indexset       = hpro::TIndexSet;
using block_indexset = hpro::TBlockIndexSet;

// hash function for index sets
struct indexset_hash
{
    size_t operator () ( const indexset &  is ) const
    {
        return ( std::hash< hpro::idx_t >()( is.first() ) ^ std::hash< hpro::idx_t >()( is.last()  ) );
    }
};

// hash function for index sets
struct block_indexset_hash
{
    size_t operator () ( const block_indexset &  is ) const
    {
        return ( indexset_hash()( is.row_is() ) ^ indexset_hash()( is.col_is() ) );
    }
};

}// namespace hlr

#endif // __HLR_UTILS_HASH_HH
