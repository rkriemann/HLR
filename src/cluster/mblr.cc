//
// Project     : HLR
// Module      : mblr.cc
// Description : MBLR specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/mblr.hh"

namespace hlr { namespace cluster { namespace mblr {

using namespace HLIB;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< HLIB::TClusterTree >
cluster ( HLIB::TCoordinate &  coords,
          const size_t         ntile,
          const size_t         nlvl )
{
    TCardBSPPartStrat  part_strat;
    TMBLRCTBuilder     ct_builder( nlvl, & part_strat, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< HLIB::TBlockClusterTree >
blockcluster ( HLIB::TClusterTree &  rowct,
               HLIB::TClusterTree &  colct )
{
    TWeakStdGeomAdmCond  adm_cond;
    TBCBuilder           bct_builder;

    return bct_builder.build( & rowct, & colct, & adm_cond );
}

}}}// namespace hlr::cluster::mblr
