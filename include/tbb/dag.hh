#ifndef __HLR_TBB_DAG_HH
#define __HLR_TBB_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

namespace TBB
{

//
// construct DAG using refinement of given node
//
DAG::Graph
refine ( DAG::Node *  root );

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc );

}// namespace TBB

}// namespace DAG

}// namespace HLR

#endif // __HLR_TBB_DAG_HH
