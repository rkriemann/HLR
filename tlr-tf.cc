//
// Project     : HLib
// File        : tlr-tf.cc
// Description : TLR LU using cpp-taskflow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "tlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}

