#ifndef __HLR_HODLR_HH
#define __HLR_HODLR_HH
//
// Project     : HLib
// File        : hodlr.hh
// Description : HODLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <hlib.hh>

namespace HODLR
{

//
// clustering
//

std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile );

//
// LU factorization
//

namespace SEQ
{

template < typename value_t >
void lu ( HLIB::TMatrix *          A,
          const HLIB::TTruncAcc &  acc );

}// namespace SEQ

namespace TBB
{

template < typename value_t >
void lu ( HLIB::TMatrix *          A,
          const HLIB::TTruncAcc &  acc );

}// namespace TBB

}// namespace HODLR

#endif // __HLR_HODLR_HH
