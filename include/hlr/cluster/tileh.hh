#ifndef __HLR_CLUSTER_TILEH_HH
#define __HLR_CLUSTER_TILEH_HH
//
// Project     : HLR
// File        : tileh.hh
// Description : TileH related clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cluster/TCoordinate.hh>
#include <cluster/TClusterTree.hh>
#include <cluster/TBlockClusterTree.hh>

namespace hlr { namespace cluster { namespace tileh {

//
// cluster set of coordinates with minimal block size <ntile>
// and top <nlvl> levels being combined (flattened)
//
std::unique_ptr< HLIB::TClusterTree >
cluster      ( HLIB::TCoordinate *   coords,
               const size_t          ntile,
               const size_t          nlvl );

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< HLIB::TBlockClusterTree >
blockcluster ( HLIB::TClusterTree *  rowct,
               HLIB::TClusterTree *  colct );

}}}// namespace hlr::cluster::tileh

#endif // __HLR_CLUSTER_TILEH_HH
