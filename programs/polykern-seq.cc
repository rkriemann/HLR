//
// Project     : HLR
// Program     : polykern-seq.cc
// Description : computation of kernel of matrix defined by polynomials
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/arith/blas.hh"
#include "hlr/seq/arith.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "polykern.hh"
#include "seq.hh"
