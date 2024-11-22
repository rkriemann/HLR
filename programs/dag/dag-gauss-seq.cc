//
// Project     : HLR
// Module      : gauss-seq.cc
// Description : DAG based Gaussian Elimination using sequential execution
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/dag.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "dag-gauss.hh"
#include "seq.hh"
