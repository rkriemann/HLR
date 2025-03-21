//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_uniform.hh"
#include "hlr/tbb/convert.hh"
#include "hlr/tbb/compress.hh"
#include "hlr/tbb/norm.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "mvm-uniform.hh"
#include "tbb.hh"
