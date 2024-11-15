//
// Project     : HLR
// Module      : dag-tbb.cc
// Description : DAG based H-LU using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/dag.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "dag-lu.hh"
#include "tbb.hh"
