//
// Project     : HLR
// Module      : hodlr-seq.cc
// Description : sequential tile-based HODLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_tiled.hh"
#include "hlr/seq/arith_tiled_v2.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "tiled-hodlr.hh"
#include "seq.hh"
