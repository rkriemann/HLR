//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cmath>
#include <numbers>

#include <common.hh>
#include <common-main.hh>

#include <hlr/tensor/dense_tensor.hh>
#include <hlr/utils/io.hh>

using namespace hlr;

template < typename value_t >
void
print ( const tensor::dense_tensor3< value_t > &  t )
{
    for ( uint  l = 0; l < t.dim(2); ++l )
    {
        for ( uint  j = 0; j < t.dim(1); ++j )
        {
            for ( uint  i = 0; i < t.dim(0); ++i )
                std::cout << t( i, j, l ) << ", ";

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

    const size_t                      n = 10;
    const double                      h = std::numbers::pi / double(n-1);
    tensor::dense_tensor3< value_t >  t( is(0,n-1), is(0,n-1), is(0,n-1) );
    double                            v = 1.0;

    for ( uint  l = 0; l < n; ++l )
        for ( uint  j = 0; j < n; ++j )
            for ( uint  i = 0; i < n; ++i )
                t( i, j, l ) = std::sin( i * h ) + std::cos( j * h ) + std::sin( l * h );

    // print( t );
    io::vtk::print( t, "t.vtk" );

    io::hdf5::write( t, "t" );

    auto  t2 = io::hdf5::read_tensor< value_t >( "u.h5" );

    io::vtk::print( t2, "u.vtk" );
    
    // print( t2 );
}
