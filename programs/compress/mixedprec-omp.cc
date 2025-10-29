//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/convert.hh"
#include "hlr/omp/compress.hh"
#include "hlr/omp/norm.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "mixedprec.hh"
#include "seq.hh"
