//
// Project     : HLR
// File        : tiled-hca-tbb.cc
// Description : tiled HCA based construction using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_tiled_v2.hh"
#include "hlr/tbb/hca.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "tiled-hca.hh"
#include "tbb.hh"
