//
// Project     : HLib
// File        : dag-seq.cc
// Description : sequential H-LU using DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/dag.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "dag.hh"

int
main ( int argc, char ** argv )
{
    HLIB::CFG::set_nthreads( 1 );

    return hlrmain( argc, argv );
}
