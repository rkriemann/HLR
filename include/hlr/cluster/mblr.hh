#ifndef __HLR_CLUSTER_MBLR_HH
#define __HLR_CLUSTER_MBLR_HH
//
// Project     : HLR
// File        : mblr.hh
// Description : MBLR related clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/cluster/TClusterTree.hh>
#include <hpro/cluster/TBlockClusterTree.hh>

namespace hlr { namespace cluster { namespace mblr {

//
// cluster set of coordinates with minimal block size <ntile>
// and hierarchy of depth <nlvl> 
//
std::unique_ptr< HLIB::TClusterTree >
cluster      ( HLIB::TCoordinate &   coords,
               const size_t          ntile,
               const size_t          nlvl );

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< HLIB::TBlockClusterTree >
blockcluster ( HLIB::TClusterTree &  rowct,
               HLIB::TClusterTree &  colct );

}}}// namespace hlr::cluster::mblr

#endif // __HLR_CLUSTER_MBLR_HH
