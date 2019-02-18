#ifndef __HLR_TBB_RUN_HH
#define __HLR_TBB_RUN_HH
//
// Project     : HLib
// File        : tbb_run.hh
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "Graph.hh"

namespace DAG
{

namespace TBB
{

//
// execute DAG <dag>
//
void
run ( Graph &                  dag,
      const HLIB::TTruncAcc &  acc );

}// namespace TBB

}// namespace DAG

#endif // __HLR_TBB_RUN_HH
