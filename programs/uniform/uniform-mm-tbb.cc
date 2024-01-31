//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_accu.hh"
#include "hlr/tbb/arith_uniform.hh"
#include "hlr/tbb/cluster_basis.hh"
#include "hlr/tbb/norm.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "uniform-mm.hh"
#include "tbb.hh"
