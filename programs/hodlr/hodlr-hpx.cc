//
// Project     : HLR
// Module      : tlr-hpx.cc
// Description : HODLR LU with HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>

#include "hlr/hpx/matrix.hh"
#include "hlr/hpx/arith.hh"

namespace          impl      = hlr::hpx;
const std::string  impl_name = "hpx";

#include "hodlr.hh"
#include "hpx.hh"
