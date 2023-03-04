//
// Project     : HLR
// Module      : gauss-tf.cc
// Description : DAG based Gaussian Elimination using C++-TaskFlow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"
#include "hlr/tf/dag.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "dag-gauss.hh"
#include "seq.hh"
