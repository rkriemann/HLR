//
// Project     : HLib
// File        : stdh.cc
// Description : common functions for Standard H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <cluster/TBSPCTBuilder.hh>
#include <cluster/TBSPPartStrat.hh>
#include <cluster/TBCBuilder.hh>
#include <cluster/TGeomAdmCond.hh>

#include "stdh.hh"

namespace StdH
{

using namespace HLIB;

//
// set up cluster and block cluster tree
//
std::pair< std::unique_ptr< TClusterTree >,
           std::unique_ptr< TBlockClusterTree > >
cluster ( TCoordinate *  coords,
          const size_t   ntile )
{
    TCardBSPPartStrat    part_strat;
    TBSPCTBuilder        ct_builder( & part_strat, ntile );

    auto  ct = ct_builder.build( coords );

    TWeakStdGeomAdmCond  adm_cond;
    TBCBuilder           bct_builder;

    auto  bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

    return { std::move( ct ), std::move( bct ) };
}

}// namespace StdH
