//
// Project     : HLib
// File        : hodlr.cc
// Description : HODLR specific functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hodlr.hh"

namespace HODLR
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
    TCardBSPPartStrat  part_strat;
    TBSPCTBuilder      ct_builder( & part_strat, ntile );
    
    auto  ct = ct_builder.build( coords );
    
    TOffDiagAdmCond    adm_cond;
    TBCBuilder         bct_builder;
    
    auto  bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
    
    return { std::move( ct ), std::move( bct ) };
}

}// namespace HODLR
