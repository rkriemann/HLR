//
// Project     : HLR
// Module      : hodlr-tbb.cc
// Description : HODLR LU using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "hodlr.hh"
#include "tbb.hh"
