//
// Project     : HLR
// File        : tiled-hca-seq.cc
// Description : sequential HCA based construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_tiled_v2.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "tiled-hca.hh"

void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main();
}

HLR_DEFAULT_MAIN
