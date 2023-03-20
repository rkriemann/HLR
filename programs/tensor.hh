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
print ( const blas::tensor3< value_t > &  t )
{
    for ( uint  l = 0; l < t.size(2); ++l )
    {
        for ( uint  i = 0; i < t.size(0); ++i )
        {
            for ( uint  j = 0; j < t.size(1); ++j )
                std::cout << t( i, j, l ) << ", ";

            std::cout << std::endl;
        }// for

        std::cout << std::endl;
    }// for
}

template < typename value_t >
void
print ( const tensor::dense_tensor3< value_t > &  t )
{
    print( t.tensor() );
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = double;

    {
        auto  X = blas::tensor3< value_t >( 3, 4, 2 );
        uint  v = 1;

        for ( uint  l = 0; l < X.size(2); ++l )
            for ( uint  j = 0; j < X.size(1); ++j )
                for ( uint  i = 0; i < X.size(0); ++i )
                    X(i,j,l) = v++;

        print( X );

        auto  X0 = X.unfold( 0 );
        auto  X1 = X.unfold( 1 );
        auto  X2 = X.unfold( 2 );

        std::cout << X0 << std::endl;
        std::cout << X1 << std::endl;
        std::cout << X2 << std::endl;

        auto  M = blas::matrix< value_t >( 2, 3 );

        v = 1;
        for ( uint  j = 0; j < M.ncols(); ++j )
            for ( uint  i = 0; i < M.nrows(); ++i )
                M(i,j) = v++;

        std::cout << M << std::endl;

        auto  Y = blas::tensor_product( X, M, 0 );

        print( Y );

        auto  acc = Hpro::fixed_prec( 1e-4 );
        auto  [ G, Y0, Y1, Y2 ] = hosvd( X, acc );
    }

    if ( false )
    {
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
    }

    if ( false )
    {
        auto  t2 = io::hdf5::read_tensor< value_t >( "u.h5" );

        io::vtk::print( t2, "u.vtk" );
    
        // print( t2 );
    }
}
