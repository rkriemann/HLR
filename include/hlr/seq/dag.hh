#ifndef __HLR_SEQ_DAG_HH
#define __HLR_SEQ_DAG_HH
//
// Project     : HLR
// Module      : dag.hh
// Description : DAG execution function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/base/TTruncAcc.hh>

#include "hlr/dag/graph.hh"

namespace hlr { namespace seq { namespace dag {

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
      const Hpro::TTruncAcc &  acc );

}}}// namespace hlr::seq::dag

#endif // __HLR_SEQ_DAG_HH
