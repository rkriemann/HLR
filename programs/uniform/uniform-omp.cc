//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/cluster_basis.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "uniform.hh"
#include "seq.hh"
