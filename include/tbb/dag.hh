#ifndef __HLR_TBB_DAG_HH
#define __HLR_TBB_RUN_HH
//
// Project     : HLib
// File        : tbb_run.hh
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "dag/Graph.hh"

namespace HLR
{

namespace TBB
{

struct DAGExecution
{
    //
    // execute DAG <dag>
    //
    void
    run ( DAG::Graph &             dag,
          const HLIB::TTruncAcc &  acc ) const;
};

}// namespace TBB

}// namespace HLR

#endif // __HLR_TBB_DAG_HH
