//
// Project     : HLib
// File        : dag-tbb.cc
// Description : DAG based H-LU using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tbb/matrix.hh"
#include "hlr/tbb/dag.hh"

namespace impl = hlr::tbb;

#include "dag.hh"

template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    dag_main< problem_t >( argc, argv, "tbb" );
}

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
