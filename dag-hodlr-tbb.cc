//
// Project     : HLib
// File        : dag-hodlr-tbb.cc
// Description : tiled HODLR-LU using DAG with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"
#include "hlr/seq/arith_tiled.hh"
#include "hlr/tbb/arith_tiled.hh"
#include "hlr/tbb/dag.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "dag-hodlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
