#ifndef __HLR_SEQ_RUN_HH
#define __HLR_SEQ_RUN_HH
//
// Project     : HLib
// File        : seq_run.hh
// Description : execute DAG sequentially
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "Graph.hh"

namespace DAG
{

namespace SEQ
{

//
// execute DAG <dag>
//
void
run ( Graph &                  dag,
      const HLIB::TTruncAcc &  acc );

}// namespace SEQ

}// namespace DAG

#endif // __HLR_SEQ_RUN_HH
