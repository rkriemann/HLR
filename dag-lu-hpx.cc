//
// Project     : HLib
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

int
hpx_main ( int argc, char ** argv )
{
    hlrmain( argc, argv );
    
    return ::hpx::finalize();
}

int
main ( int argc, char ** argv )
{
    return ::hpx::init( argc, argv );
}
