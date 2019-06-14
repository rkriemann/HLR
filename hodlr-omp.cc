//
// Project     : HLib
// File        : hodlr-omp.cc
// Description : HODLR LU using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/arith.hh"

namespace          impl      = hlr::omp;
const std::string  impl_name = "omp";

#include "hodlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}

