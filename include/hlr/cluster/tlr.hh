#ifndef __HLR_CLUSTER_TLR_HH
#define __HLR_CLUSTER_TLR_HH
//
// Project     : HLR
// Module      : tlr.hh
// Description : TLR related clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TClusterTree.hh>
#include <hpro/cluster/TBlockClusterTree.hh>

namespace hlr { namespace cluster { namespace tlr {

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< Hpro::TClusterTree >
cluster      ( Hpro::TCoordinate &          coords,
               const Hpro::TBSPPartStrat &  part,
               const size_t                 ntile );

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< Hpro::TBlockClusterTree >
blockcluster ( Hpro::TClusterTree &  rowct,
               Hpro::TClusterTree &  colct );

}}}// namespace hlr::cluster::tlr

#endif // __HLR_CLUSTER_TLR_HH
