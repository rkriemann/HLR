//
// Project     : HLib
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

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
