#ifndef __HLR_TILEH_HH
#define __HLR_TILEH_HH
//
// Project     : HLib
// File        : TileH.hh
// Description : common Tile-H functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <cluster/TCoordinate.hh>
#include <cluster/TClusterTree.hh>
#include <cluster/TBlockClusterTree.hh>

namespace TileH
{

//
// flatten top <n> levels of cluster tree
//
void
flatten ( HLIB::TCluster *  cl,
          const HLIB::uint  nlevel );

//
// set up cluster and block cluster tree
//
std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile,
          const int            nprocs );

}// namespace TileH

#endif // __HLR_TILEH_HH
