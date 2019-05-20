//
// Project     : HLib
// File        : dag-tf.cc
// Description : DAG based H-LU using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/tf/matrix.hh"
#include "hlr/tf/dag.hh"

namespace impl = hlr::tf;

#include "dag.hh"

template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    dag_main< problem_t >( argc, argv, "tf" );
}

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
