//
// Project     : HLR
// Module      : dag-hpx.cc
// Description : DAG based H-LU using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>

#include "hlr/hpx/matrix.hh"
#include "hlr/hpx/dag.hh"

namespace          impl      = hlr::hpx;
const std::string  impl_name = "hpx";

#include "dag-lu.hh"
#include "hpx.hh"
