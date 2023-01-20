//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlr/seq/matrix.hh>
#include <hlr/seq/convert.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_accu.hh>
#include <hlr/seq/dag.hh>

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "single-lu.hh"
#include "seq.hh"
