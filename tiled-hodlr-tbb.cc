//
// Project     : HLR
// File        : hodlr-tbb.cc
// Description : tile-based HODLR-LU using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "tiled-hodlr.hh"

template < typename problem_t >
void
framework_main ()
{
    auto                   param = ::tbb::global_control::max_allowed_parallelism;
    ::tbb::global_control  tbb_control( param, ( nthreads > 0 ? nthreads : ::tbb::global_control::active_value( param ) ) );

    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
