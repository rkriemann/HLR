//
// Project     : HLR
// Program     : accu-seq.cc
// Description : testing accumulator arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_accu.hh"
#include "hlr/seq/arith_lazy.hh"
#include "hlr/seq/dag.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "approx-mm.hh"
#include "seq.hh"
