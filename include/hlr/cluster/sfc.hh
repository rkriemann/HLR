#ifndef __HLR_CLUSTER_SFC_HH
#define __HLR_CLUSTER_SFC_HH
//
// Project     : HLR
// Module      : sfc.hh
// Description : Hilbert curve based clustering
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/cluster/TClusterTree.hh>
#include <hpro/cluster/TBlockClusterTree.hh>
#include <hpro/cluster/TAdmCondition.hh>
#include <hpro/cluster/TBSPCTBuilder.hh>

namespace hlr { namespace cluster { namespace sfc {

using  Hpro::TSFCCTBuilder::binary;
using  Hpro::TSFCCTBuilder::blr;

using  cluster_type_t    = Hpro::TSFCCTBuilder::cluster_type_t;
using  coordinates       = Hpro::TCoordinate;
using  cluster_tree      = Hpro::TClusterTree;
using  blockcluster_tree = Hpro::TBlockClusterTree;
using  admissibility     = Hpro::TAdmCondition;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< cluster_tree >
cluster      ( const cluster_type_t  cl_type,
               coordinates &         coords,
               const size_t          ntile );

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

}}}// namespace hlr::cluster::sfc

#endif // __HLR_CLUSTER_SFC_HH
