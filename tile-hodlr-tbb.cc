//
// Project     : HLib
// File        : hodlr-seq.cc
// Description : sequential tile-based HODLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/arith.hh"

namespace          impl      = hlr::tbb;
const std::string  impl_name = "tbb";

#include "tile-hodlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}

