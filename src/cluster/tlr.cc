//
// Project     : HLib
// File        : tlr.cc
// Description : TLR specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cluster/TBSPCTBuilder.hh>
#include <cluster/TBSPPartStrat.hh>
#include <cluster/TBCBuilder.hh>
#include <cluster/TGeomAdmCond.hh>

#include "hlr/cluster/tlr.hh"

namespace hlr { namespace cluster { namespace tlr {

using namespace HLIB;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< HLIB::TClusterTree >
cluster ( HLIB::TCoordinate *   coords,
          const size_t          ntile )
{
    TCardBSPPartStrat    part_strat;
    TMBLRCTBuilder       ct_builder( 1, & part_strat, ntile );

    return ct_builder.build( coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< HLIB::TBlockClusterTree >
blockcluster ( HLIB::TClusterTree *  rowct,
               HLIB::TClusterTree *  colct )
{
    TWeakStdGeomAdmCond  adm_cond;
    TBCBuilder           bct_builder;

    return bct_builder.build( rowct, colct, & adm_cond );
}

}}}// namespace hlr::cluster::tlr
