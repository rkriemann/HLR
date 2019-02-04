#ifndef __HLR_TLR_HH
#define __HLR_TLR_HH
//
// Project     : HLib
// File        : tlr.hh
// Description : TLR arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <hlib.hh>

namespace TLR
{

//
// clustering
//

std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile );

}// namespace TLR

#endif // __HLR_TLR_HH
