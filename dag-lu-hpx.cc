//
// Project     : HLR
// File        : dag-hpx.cc
// Description : DAG based H-LU using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>

#include "hlr/hpx/matrix.hh"
#include "hlr/hpx/dag.hh"

namespace          impl      = hlr::hpx;
const std::string  impl_name = "hpx";

#include "dag-lu.hh"

template < typename problem_t >
void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main< problem_t >();
}

//
// HPX specific main functions
//
int
hpx_main ( int argc, char ** argv )
{
    hlr_main( argc, argv );
    
    return ::hpx::finalize();
}

int
main ( int argc, char ** argv )
{
    return ::hpx::init( argc, argv );
}
