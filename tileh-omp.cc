//
// Project     : HLR
// File        : tileh-omp.cc
// Description : ompuential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"
#include "hlr/omp/dag.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "tileh.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
