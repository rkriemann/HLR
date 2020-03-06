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

// hash function for index sets
struct indexset_hash
{
    size_t operator () ( const HLIB::TIndexSet &  is ) const
    {
        return ( std::hash< HLIB::idx_t >()( is.first() ) ^
                 std::hash< HLIB::idx_t >()( is.last()  ) );
    }
};

}// namespace hlr

#endif // __HLR_UTILS_HASH_HH
