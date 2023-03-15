//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hlr/tensor/dense_tensor.hh>
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
    using midx    = tensor::dense_tensor< value_t, 3 >::multiindex;
        
    tensor::dense_tensor< value_t, 3 >  t{ 3, 3, 3 };

    t( midx{ 0, 0, 0 } ) = 1;
    t( midx{ 1, 0, 0 } ) = 2;
    t( midx{ 0, 1, 0 } ) = 3;
    t( midx{ 0, 0, 1 } ) = 4;
    t( midx{ 1, 2, 2 } ) = 5;

    for ( uint  k = 0; k < t.dim(2); ++k )
    {
        for ( uint  i = 0; i < t.dim(0); ++i )
        {
            for ( uint  j = 0; j < t.dim(1); ++j )
                std::cout << t( midx{ i, j, k } ) << ", ";

            std::cout << std::endl;
        }// for

        std::cout << std::endl;
    }// for

    io::hdf5::write( t, "t" );
}
