//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hpro/io/TMatrixIO.hh>

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

    auto  M1 = Hpro::read_matrix< value_t >( "S.mat" );
    auto  S1 = ptrcast( M1.get(), Hpro::TSparseMatrix< value_t > );
    
    matrix::sparse_matrix< value_t >  S2( *S1 );
}
