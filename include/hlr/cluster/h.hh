#ifndef __HLR_CLUSTER_H_HH
#define __HLR_CLUSTER_H_HH
//
// Project     : HLib
// File        : H.hh
// Description : H related clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/cluster/TClusterTree.hh>
#include <hpro/cluster/TBlockClusterTree.hh>
#include <hpro/cluster/TAdmCondition.hh>

namespace hlr { namespace cluster { namespace h {

namespace  hpro = HLIB;

using  coordinates       = hpro::TCoordinate;
using  cluster_tree      = hpro::TClusterTree;
using  blockcluster_tree = hpro::TBlockClusterTree;
using  admissibility     = hpro::TAdmCondition;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< cluster_tree >
cluster      ( coordinates &   coords,
               const size_t    ntile );

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< blockcluster_tree >
blockcluster ( cluster_tree &  rowct,
               cluster_tree &  colct );

//
// build block cluster tree based on given row/column cluster trees
// with given admissibility condition
//
std::unique_ptr< blockcluster_tree >
blockcluster ( cluster_tree &         rowct,
               cluster_tree &         colct,
               const admissibility &  adm );

}}}// namespace hlr::cluster::h

#endif // __HLR_CLUSTER_H_HH
