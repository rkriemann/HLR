//
// Project     : HLR
// File        : tiled-h-tbb.cc
// Description : tile-based H-arithmetic using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/arith_tiled_v2.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "tiled-h.hh"

void
framework_main ()
{
    auto                   param = ::tbb::global_control::max_allowed_parallelism;
    ::tbb::global_control  tbb_control( param, ( nthreads > 0 ? nthreads : ::tbb::global_control::active_value( param ) ) );
    
    program_main();
}

HLR_DEFAULT_MAIN
