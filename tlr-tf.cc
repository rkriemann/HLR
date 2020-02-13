//
// Project     : HLR
// File        : tlr-tf.cc
// Description : TLR LU using cpp-taskflow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"
#include "hlr/tf/dag.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "tlr.hh"

template < typename problem_t >
void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
