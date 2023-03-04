//
// Project     : HLR
// Module      : hodlr-tf.cc
// Description : HODLR LU using cpp-taskflow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "hodlr.hh"
#include "seq.hh"
