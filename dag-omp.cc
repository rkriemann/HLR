//
// Project     : HLib
// File        : dag-omp.cc
// Description : DAG based H-LU using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/omp/matrix.hh"
#include "hlr/omp/dag.hh"

namespace impl = hlr::omp;

#include "dag.hh"

template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    dag_main< problem_t >( argc, argv, "omp" );
}

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
