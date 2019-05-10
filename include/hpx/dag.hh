#ifndef __HLR_HPX_DAG_HH
#define __HLR_HPX_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : execute DAG using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

namespace HPX
{

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc );

}// namespace HPX

}// namespace DAG

}// namespace HLR

#endif // __HLR_HPX_DAG_HH
