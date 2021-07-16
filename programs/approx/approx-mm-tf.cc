//
// Project     : HLR
// Program     : accu-tf.cc
// Description : testing accumulator arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"
#include "hlr/tf/arith_accu.hh"
#include "hlr/tf/arith_lazy.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "approx-mm.hh"
#include "seq.hh"
