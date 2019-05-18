#ifndef __HLR_OMP_DAG_HH
#define __HLR_OMP_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : execute DAG using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

namespace OMP
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

}// namespace OMP

}// namespace DAG

}// namespace HLR

#endif // __HLR_OMP_DAG_HH
