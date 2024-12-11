//
// Project     : HLR
// Module      : hodlr-tbb.cc
// Description : tile-based HODLR-LU using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_tiled.hh"
#include "hlr/tbb/arith_tiled_v2.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "tiled-hodlr.hh"
#include "tbb.hh"
