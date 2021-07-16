//
// Project     : HLR
// Program     : approx-lu-tbb.cc
// Description : comparing H-LU for different arith./approx.
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_accu.hh"
#include "hlr/tbb/arith_lazy.hh"
#include "hlr/tbb/dag.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "approx-lu.hh"
#include "tbb.hh"
