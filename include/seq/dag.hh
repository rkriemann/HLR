#ifndef __HLR_SEQ_DAG_HH
#define __HLR_SEQ_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : DAG execution function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <base/TTruncAcc.hh>

#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

namespace Seq
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
run ( Graph &                  dag,
      const HLIB::TTruncAcc &  acc );

}// namespace Seq

}// namespace DAG

}// namespace HLR

#endif // __HLR_SEQ_DAG_HH
