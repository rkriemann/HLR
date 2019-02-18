#ifndef __HLR_STDH_HH
#define __HLR_STDH_HH
//
// Project     : HLib
// File        : stdh.hh
// Description : common functions for Standard H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <cluster/TCoordinate.hh>
#include <cluster/TClusterTree.hh>
#include <cluster/TBlockClusterTree.hh>

namespace StdH
{

//
// set up cluster and block cluster tree
//
std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile );

}// namespace StdH

#endif // __HLR_STDH_HH
