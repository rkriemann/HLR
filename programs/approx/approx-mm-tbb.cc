//
// Project     : HLR
// Program     : accu-tbb.cc
// Description : testing accumulator arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_accu.hh"
#include "hlr/tbb/arith_lazy.hh"
#include "hlr/tbb/dag.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "approx-mm.hh"
#include "tbb.hh"
