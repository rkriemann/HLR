//
// Project     : HLR
// File        : rec-lu.hh
// Description : recursive LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "rec-lu.hh"
#include "tbb.hh"
