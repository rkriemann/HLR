//
// Project     : HLR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/compress.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/norm.hh"
// #include "hlr/omp/convert.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "compress.hh"
#include "seq.hh"
