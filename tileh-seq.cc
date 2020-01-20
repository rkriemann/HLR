//
// Project     : HLR
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/dag.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "tileh.hh"

void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    hpro::CFG::set_nthreads( 1 );

    program_main();
}

HLR_DEFAULT_MAIN
