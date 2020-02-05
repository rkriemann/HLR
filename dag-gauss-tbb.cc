//
// Project     : HLR
// File        : gauss-tbb.cc
// Description : DAG based Gaussian Elimination using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/tbb/dag.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "dag-gauss.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
