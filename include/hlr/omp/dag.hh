#ifndef __HLR_OMP_DAG_HH
#define __HLR_OMP_DAG_HH
//
// Project     : HLib
// File        : dag.hh
// Description : execute DAG using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace omp
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

}// namespace OMP

}// namespace DAG

}// namespace HLR

#endif // __HLR_OMP_DAG_HH
