//
// Project     : HLR
// Module      : hodlr.cc
// Description : HODLR specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/hodlr.hh"

namespace hlr { namespace cluster { namespace hodlr {

using namespace Hpro;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< Hpro::TClusterTree >
cluster ( Hpro::TCoordinate &          coords,
          const Hpro::TBSPPartStrat &  part,
          const size_t                 ntile )
{
    TBSPCTBuilder      ct_builder( & part, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< Hpro::TBlockClusterTree >
blockcluster ( Hpro::TClusterTree &  rowct,
               Hpro::TClusterTree &  colct )
{
    TOffDiagAdmCond  adm_cond;
    TBCBuilder       bct_builder;

    return bct_builder.build( & rowct, & colct, & adm_cond );
}

}}}// namespace hlr::cluster::hodlr
