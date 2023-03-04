//
// Project     : HLR
// Module      : dag-tf.cc
// Description : DAG based H-LU using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/dag.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "dag-lu.hh"
#include "seq.hh"
