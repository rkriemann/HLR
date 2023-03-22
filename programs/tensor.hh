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
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/construct.hh>
#include <hlr/tensor/convert.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/utils/io.hh>

using namespace hlr;

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

    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );

    if ( false )
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

        auto  acc               = Hpro::fixed_prec( 1e-4 );
        auto  [ G, Y0, Y1, Y2 ] = blas::hosvd( X, acc );

        print( G );
        
        std::cout << Y0 << std::endl;
        std::cout << Y1 << std::endl;
        std::cout << Y2 << std::endl;

        auto  W0 = blas::tensor_product( G,  Y0, 0 );
        auto  W1 = blas::tensor_product( W0, Y1, 1 );
        auto  W  = blas::tensor_product( W1, Y2, 2 );

        print( W );
    }

    if ( true )
    {
        blas::tensor3< value_t >  X;
        
        std::cout << term::bullet << term::bold << "dense tensor" << term::reset << std::endl;

        if ( cmdline::datafile != "" )
        {
            std::cout << "  " << term::bullet << term::bold << "reading from " << cmdline::datafile << term::reset << std::endl;

            X = io::hdf5::read< blas::tensor3< value_t > >( cmdline::datafile );
        }// if
        else
        {
            std::cout << "  " << term::bullet << term::bold << "building tensor" << term::reset << std::endl;
            
            const size_t  n = cmdline::n;
            const double  h = std::numbers::pi / double(n-1);
            // double        v = 1.0;
            
            X = std::move( blas::tensor3< value_t >( n, n, n ) );
            
            tic = timer::now();
        
            for ( uint  l = 0; l < n; ++l )
                for ( uint  j = 0; j < n; ++j )
                    for ( uint  i = 0; i < n; ++i )
                        // X( i, j, l ) = v++;
                        X( i, j, l ) = std::sin( 4.0 * i * h ) + std::cos( 2.0 * j * h ) + std::sin( l * h );
            
            toc = timer::since( tic );
            std::cout << "    done in  " << format_time( toc ) << std::endl;
        }// else
        
        std::cout << "    dims   = " << term::bold << X.size(0) << " × " << X.size(1) << " × " << X.size(2) << term::reset << std::endl;
        std::cout << "    mem    = " << format_mem( X.byte_size() ) << std::endl;
        
        // std::cout << X << std::endl;
        if ( verbose(3) ) io::vtk::print( X, "X.vtk" );
        if ( verbose(2) ) io::hdf5::write( X, "X" );

        {
            std::cout << term::bullet << term::bold << "HOSVD" << term::reset << std::endl;

            tic = timer::now();
        
            auto  acc               = Hpro::fixed_prec( cmdline::eps );
            auto  apx               = approx::SVD< value_t >();
            auto  [ G, X0, X1, X2 ] = blas::hosvd( X, acc, apx );
            
            toc = timer::since( tic );
            
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            std::cout << "    mem    = " << format_mem( G.byte_size() + X0.byte_size() + X1.byte_size() + X2.byte_size() ) << std::endl;
            
            auto  Z  = tensor::tucker_tensor3< value_t >( is( 0,n-1 ), is( 0,n-1 ), is( 0,n-1 ),
                                                          std::move( G ),
                                                          std::move( X0 ),
                                                          std::move( X1 ),
                                                          std::move( X2 ) );
            
            std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
            
            auto  T0 = blas::tensor_product( Z.G(), Z.X(0), 0 );
            auto  T1 = blas::tensor_product( T0,    Z.X(1), 1 );
            auto  Y  = blas::tensor_product( T1,    Z.X(2), 2 );
            
            if ( verbose(3) ) io::vtk::print( Y, "Y1" );
            if ( verbose(2) ) io::hdf5::write( Y, "Y1" );
            
            blas::add( -1, X, 1, Y );
            std::cout << "    error  = " << format_error( blas::norm_F( Y ), blas::norm_F( Y ) / blas::norm_F( X ) ) << std::endl;
            
            if ( verbose(3) ) io::vtk::print( Y, "error1" );
        }

        {
            std::cout << term::bullet << term::bold << "Hierarchical HOSVD" << term::reset << std::endl;

            tic = timer::now();
        
            auto  acc = Hpro::fixed_prec( cmdline::eps );
            auto  apx = approx::SVD< value_t >();
            auto  H   = impl::tensor::build_hierarchical_tucker( X, acc, apx, cmdline::ntile );
            
            toc = timer::since( tic );
            
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;

            auto  Y = impl::tensor::to_dense( *H );
            
            if ( verbose(3) ) io::vtk::print( *Y, "Y2" );
            if ( verbose(2) ) io::hdf5::write( Y->tensor(), "Y2" );
            
            blas::add( -1, X, 1, Y->tensor() );
            std::cout << "    error  = " << format_error( blas::norm_F( Y->tensor() ), blas::norm_F( Y->tensor() ) / blas::norm_F( X ) ) << std::endl;
            
            if ( verbose(3) ) io::vtk::print( *Y, "error2" );
        }
    }

    if ( false )
    {
        auto  t2 = io::hdf5::read< blas::tensor3< value_t > >( "u.h5" );

        io::vtk::print( t2, "u.vtk" );
    
        // print( t2 );
    }
}
