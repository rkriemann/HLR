//
// Project     : HLib
// File        : gauss-tf.cc
// Description : DAG based Gaussian Elimination using C++-TaskFlow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/arith.hh"
#include "hlr/tf/dag.hh"

namespace          impl      = hlr::tf;
const std::string  impl_name = "tf";

#include "gauss.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
