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

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace seq
{

namespace dag
{

//
// construct DAG using refinement of given node
//
hlr::dag::graph
refine ( hlr::dag::node *  root,
         const size_t      min_size );

//
// execute DAG <dag>
//
void
run ( hlr::dag::graph &        dag,
      const HLIB::TTruncAcc &  acc );

}// namespace dag

}// namespace seq

}// namespace hlr

#endif // __HLR_SEQ_DAG_HH
