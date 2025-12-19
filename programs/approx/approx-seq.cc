//
// Project     : HLR
// Program     : approx-lu-seq.cc
// Description : comparing H-LU for different arith./approx.
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/seq/matrix.hh>
#include <hlr/seq/norm.hh>

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "approx.hh"
#include "seq.hh"
