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

namespace SEQ
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

}// namespace SEQ

}// namespace HLR

#endif // __HLR_SEQ_DAG_HH
