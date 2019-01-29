//
// Project     : HLib
// File        : tlr.cc
// Description : TLR specific functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "tlr.hh"

namespace TLR
{

using namespace HLIB;

//
// clustering
//

std::pair< std::unique_ptr< TClusterTree >,
           std::unique_ptr< TBlockClusterTree > >
cluster ( TCoordinate *  coords,
          const size_t   ntile )
{
    TCardBSPPartStrat    part_strat;
    TMBLRCTBuilder       ct_builder( 1, & part_strat, ntile );

    auto  ct = ct_builder.build( coords );

    TWeakStdGeomAdmCond  adm_cond( 4.0 );
    TBCBuilder           bct_builder;

    auto  bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

    return { std::move( ct ), std::move( bct ) };
}

}// namespace TLR
