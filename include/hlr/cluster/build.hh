#ifndef __HLR_CLUSTER_BUILD_HH
#define __HLR_CLUSTER_BUILD_HH
//
// Project     : HLR
// Module      : cluster/build
// Description : cluster tree and block tree constructions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBCBuilder.hh>

namespace hlr { namespace cluster {

using coordinates           = Hpro::TCoordinate;
using permutation           = Hpro::TPermutation;
using cluster               = Hpro::TCluster;
using block                 = Hpro::TBlockCluster;
using partitioning_strategy = Hpro::TBSPPartStrat;
using admissibility         = Hpro::TAdmCondition;

//
// build cluster tree
//
std::tuple< std::unique_ptr< cluster >, std::unique_ptr< permutation > >
build_cluster_tree ( const coordinates &            coord,
                     const partitioning_strategy &  part,
                     const uint                     ntile )
{
    auto  builder = Hpro::TBSPCTBuilder( &part, ntile );
    auto  ct      = builder.build( & coord );
    auto  root    = ct->root();
    auto  pe2i    = ct->perm_e2i();
    auto  pi2e    = ct->perm_i2e();

    ct->release_data();

    return { std::unique_ptr< cluster >( const_cast< cluster * >( root ) ),
             std::unique_ptr< permutation >( const_cast< permutation * >( pe2i ) ) };
}
    
//
// build block tree
//
std::unique_ptr< block >
build_block_tree ( const cluster &        rowcl,
                   const cluster &        colcl,
                   const admissibility &  adm )
{
    auto  builder = Hpro::TBCBuilder();

    return builder.build( & rowcl, & colcl, & adm );
}
    
}}// hlr::cluster

#endif  // __HLR_CLUSTER_BUILD_HH
