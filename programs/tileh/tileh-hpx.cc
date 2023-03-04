//
// Project     : HLR
// Module      : tileh-hpx.cc
// Description : Tile-H arithmetic using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>

#include "hlr/hpx/matrix.hh"
#include "hlr/hpx/arith.hh"
#include "hlr/hpx/arith_accu.hh"
#include "hlr/hpx/dag.hh"

namespace          impl      = hlr::hpx;
const std::string  impl_name = "hpx";

#include "tileh.hh"
#include "hpx.hh"
