#ifndef __HLR_TF_DAG_HH
#define __HLR_TF_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : execute DAG using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

namespace TF
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

}// namespace TF

}// namespace DAG

}// namespace HLR

#endif // __HLR_TF_DAG_HH
