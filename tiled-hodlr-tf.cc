//
// Project     : HLib
// File        : hodlr-tf.cc
// Description : tile-based HODLR-LU using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "tiled-hodlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}

