//
// Project     : HLR
// Module      : dag-omp.cc
// Description : DAG based H-LU using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/dag.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "dag-lu.hh"
#include "seq.hh"
