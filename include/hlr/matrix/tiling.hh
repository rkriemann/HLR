#ifndef __HLR_MATRIX_TILING_HH
#define __HLR_MATRIX_TILING_HH
//
// Project     : HLR
// Module      : matrix/tiling
// Description : tiling related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <unordered_map>

#include <hpro/cluster/TCluster.hh>

#include <hlr/utils/checks.hh>
#include <hlr/utils/hash.hh>
#include <hlr/utils/log.hh>

namespace hlr { namespace matrix {

// map HLIB types to HLR 
using  indexset = Hpro::TIndexSet;

// mapping of clusters (indexsets) to tile indexsets
using  tile_is_map_t = std::unordered_map< indexset, std::deque< indexset >, indexset_hash >;

//
// Construct map of cluster to tile-indexsets for all clusters
// in given custer tree. The leaf clusters have a single tile
// while all inner nodes have the union of all tiles of all sons.
//
namespace detail
{

inline
void
setup_tiling ( const Hpro::TCluster &  cl,
               tile_is_map_t &         tile_map )
{
    if ( cl.nsons() > 0 )
    {
        tile_map[ cl ] = {};
            
        for ( uint  i = 0; i < cl.nsons(); ++i )
        {
            auto  son = cl.son( i );
            
            if ( ! is_null( son ) )
            {
                setup_tiling( * son, tile_map );

                if ( tile_map.find( *son ) == tile_map.end() )
                    HLR_ERROR( "setup_tiling : son tiles missing" );
                
                if ( tile_map.at( *son ).size() == 0 )
                    HLR_ERROR( "setup_tiling : no son tiles" );
                
                for ( auto  is : tile_map.at( *son ) )
                    tile_map[ cl ].push_back( is );
            }// if
        }// for
    }// if
    else
        tile_map[ cl ] = { cl };
}

}// namespace detail

inline
tile_is_map_t
setup_tiling ( const Hpro::TCluster &  cl )
{
    tile_is_map_t  tile_map;

    detail::setup_tiling( cl, tile_map );

    return tile_map;
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_TILING_HH
