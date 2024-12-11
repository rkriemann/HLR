//
// Project     : HLR
// Module      : gauss-omp.cc
// Description : DAG based Gaussian Elimination using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/dag.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "dag-gauss.hh"
#include "seq.hh"
