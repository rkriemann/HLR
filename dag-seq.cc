//
// Project     : HLib
// File        : dag-seq.cc
// Description : sequential H-LU using DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/dag.hh"

namespace impl = hlr::seq;

#include "dag.hh"

template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    dag_main< problem_t >( argc, argv, "seq" );
}

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
