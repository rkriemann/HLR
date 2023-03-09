//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/utils/io.hh>

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = double;

    matrix::sparse_matrix< value_t >  S;
}
