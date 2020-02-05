//
// Project     : HLib
// File        : hodlr.cc
// Description : HODLR specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/hodlr.hh"

namespace hlr { namespace cluster { namespace hodlr {

using namespace HLIB;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< HLIB::TClusterTree >
cluster ( HLIB::TCoordinate &  coords,
          const size_t         ntile )
{
    TCardBSPPartStrat  part_strat;
    TBSPCTBuilder      ct_builder( & part_strat, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< HLIB::TBlockClusterTree >
blockcluster ( HLIB::TClusterTree &  rowct,
               HLIB::TClusterTree &  colct )
{
    TOffDiagAdmCond  adm_cond;
    TBCBuilder       bct_builder;

    return bct_builder.build( & rowct, & colct, & adm_cond );
}

}}}// namespace hlr::cluster::hodlr
