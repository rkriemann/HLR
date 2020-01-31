//
// Project     : HLR
// File        : tileh-omp.cc
// Description : Tile-H arithmetic using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/dag.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "tileh.hh"

template < typename problem_t >
void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
