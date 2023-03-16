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

template < typename value_t >
void
print ( const tensor::dense_tensor< value_t, 3 > &  t )
{
    using midx = tensor::dense_tensor< value_t, 3 >::multiindex;
    
    for ( uint  k = 0; k < t.dim(2); ++k )
    {
        for ( uint  j = 0; j < t.dim(1); ++j )
        {
            for ( uint  i = 0; i < t.dim(0); ++i )
                std::cout << t( midx{ i, j, k } ) << ", ";

            std::cout << std::endl;
        }// for

        std::cout << std::endl;
    }// for
}

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
    double                              v = 1.0;

    for ( uint  k = 0; k < t.dim(2); ++k )
        for ( uint  j = 0; j < t.dim(1); ++j )
            for ( uint  i = 0; i < t.dim(0); ++i )
                t( midx{ i, j, k } ) = v++;

    print( t );

    io::hdf5::write( t, "t" );

    auto  t2 = io::hdf5::read_tensor< value_t, 3 >( "u.h5" );

    print( t2 );
}
