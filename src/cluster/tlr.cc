//
// Project     : HLR
// Module      : tlr.cc
// Description : TLR specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/tlr.hh"

namespace hlr { namespace cluster { namespace tlr {

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< Hpro::TClusterTree >
cluster ( Hpro::TCoordinate &          coords,
          const Hpro::TBSPPartStrat &  part,
          const size_t                 ntile )
{
    Hpro::TMBLRCTBuilder  ct_builder( 1, & part, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< Hpro::TBlockClusterTree >
blockcluster ( Hpro::TClusterTree &  rowct,
               Hpro::TClusterTree &  colct )
{
    Hpro::TWeakStdGeomAdmCond  adm_cond;
    Hpro::TBCBuilder           bct_builder;

    return bct_builder.build( & rowct, & colct, & adm_cond );
}

}}}// namespace hlr::cluster::tlr
