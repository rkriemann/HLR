#ifndef __HLR_TBB_DAG_HH
#define __HLR_TBB_DAG_HH
//
// Project     : HLR
// Module      : dag.hh
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace tbb
{

namespace dag
{

//
// construct DAG using refinement of given node
//
hlr::dag::graph
refine ( hlr::dag::node *                  root,
         const size_t                      min_size,
         const hlr::dag::end_nodes_mode_t  end_mode = hlr::dag::use_single_end_node );

//
// execute DAG <dag>
//
void
run ( hlr::dag::graph &        dag,
      const HLIB::TTruncAcc &  acc );

}// namespace TBB

}// namespace DAG

}// namespace HLR

#endif // __HLR_TBB_DAG_HH
