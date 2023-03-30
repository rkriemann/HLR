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

    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  apx = approx::RRQR< value_t >();
        
    std::cout << term::bullet << term::bold << "dense tensor" << term::reset << std::endl;

    auto  X = blas::tensor3< value_t >();
    
    if ( cmdline::datafile != "" )
    {
        std::cout << "  " << term::bullet << term::bold << "reading from " << cmdline::datafile << term::reset << std::endl;

        tic = timer::now();

        X = io::hdf5::read< blas::tensor3< value_t > >( cmdline::datafile );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
    }// if
    else
    {
        std::cout << "  " << term::bullet << term::bold << "building tensor" << term::reset << std::endl;
            
        const size_t  n = cmdline::n;
        const auto    π = std::numbers::pi;
        const double  h = π / double(n-1);
        // double        v = 1.0;
            
        X = std::move( blas::tensor3< value_t >( n, n, n ) );
            
        tic = timer::now();

        for ( uint  l = 0; l < n; ++l )
            for ( uint  j = 0; j < n; ++j )
                for ( uint  i = 0; i < n; ++i )
                {
                    // X( i, j, l ) = v++;
                    X( i, j, l ) = std::sin( 32.0 * i * h ) + std::cos( 16.0 * j * h ) + std::sin( 8.0 * l * h );
                }// for
            
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
    }// else
        
    std::cout << "    dims   = " << term::bold << X.size(0) << " × " << X.size(1) << " × " << X.size(2) << term::reset << std::endl;
    std::cout << "    mem    = " << format_mem( X.byte_size() ) << std::endl;
        
    // std::cout << X << std::endl;
    if ( verbose(3) ) io::vtk::print( X, "X.vtk" );
    if ( verbose(2) ) io::hdf5::write( X, "X" );

    //
    // HOSVD
    //
    
    if ( std::max({ X.size(0), X.size(1), X.size(2) }) <= 256 )
    {
        std::cout << term::bullet << term::bold << "HOSVD" << term::reset << std::endl;

        tic = timer::now();
        
        auto  acc               = Hpro::fixed_prec( cmdline::eps );
        auto  [ G, X0, X1, X2 ] = blas::hosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            
        auto  Z  = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                      std::move( G ),
                                                      std::move( X0 ),
                                                      std::move( X1 ),
                                                      std::move( X2 ) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;

        auto  T0 = blas::tensor_product( Z.G(), Z.X(0), 0 );
        auto  T1 = blas::tensor_product( T0,    Z.X(1), 1 );
        auto  Y  = blas::tensor_product( T1,    Z.X(2), 2 );
        
        if ( verbose(3) ) io::vtk::print( Y, "Y1" );
        if ( verbose(2) ) io::hdf5::write( Y, "Y1" );
            
        impl::blas::add( -1, X, Y );
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y ), impl::blas::norm_F( Y ) / impl::blas::norm_F( X ) ) << std::endl;
            
        if ( verbose(3) ) io::vtk::print( Y, "error1" );

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( acc );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;

        T0 = std::move( blas::tensor_product( Z.G_decompressed(), Z.X_decompressed(0), 0 ) );
        T1 = std::move( blas::tensor_product( T0, Z.X_decompressed(1), 1 ) );
        Y  = std::move( blas::tensor_product( T1, Z.X_decompressed(2), 2 ) );

        impl::blas::add( -1, X, Y );
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y ), impl::blas::norm_F( Y ) / impl::blas::norm_F( X ) ) << std::endl;
    }

    //
    // ST-HOSVD
    //

    if ( std::max({ X.size(0), X.size(1), X.size(2) }) <= 256 )
    {
        std::cout << term::bullet << term::bold << "ST-HOSVD" << term::reset << std::endl;

        tic = timer::now();
        
        auto  acc               = Hpro::fixed_prec( cmdline::eps );
        auto  [ G, X0, X1, X2 ] = blas::sthosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            
        auto  Z  = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                      std::move( G ),
                                                      std::move( X0 ),
                                                      std::move( X1 ),
                                                      std::move( X2 ) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
            
        auto  T0 = blas::tensor_product( Z.G(), Z.X(0), 0 );
        auto  T1 = blas::tensor_product( T0,    Z.X(1), 1 );
        auto  Y  = blas::tensor_product( T1,    Z.X(2), 2 );
            
        if ( verbose(3) ) io::vtk::print( Y, "Y1" );
        if ( verbose(2) ) io::hdf5::write( Y, "Y1" );
            
        impl::blas::add( -1, X, Y );
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y ), impl::blas::norm_F( Y ) / impl::blas::norm_F( X ) ) << std::endl;
            
        if ( verbose(3) ) io::vtk::print( Y, "error1" );

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( acc );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;

        T0 = std::move( blas::tensor_product( Z.G_decompressed(), Z.X_decompressed(0), 0 ) );
        T1 = std::move( blas::tensor_product( T0, Z.X_decompressed(1), 1 ) );
        Y  = std::move( blas::tensor_product( T1, Z.X_decompressed(2), 2 ) );

        impl::blas::add( -1, X, Y );
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y ), impl::blas::norm_F( Y ) / impl::blas::norm_F( X ) ) << std::endl;
    }

    //
    // Hierarchical HOSVD
    //
    
    {
        std::cout << term::bullet << term::bold << "Hierarchical HOSVD (" << "ntile = " << cmdline::ntile << ")" << term::reset << std::endl;

        tic = timer::now();
        
        auto  acc = Hpro::fixed_prec( cmdline::eps );
        auto  H   = impl::tensor::build_hierarchical_tucker( X, acc, apx, cmdline::ntile );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(H->byte_size()) ) << std::endl;

        if ( verbose(1) ) io::vtk::print( *H, "H" );

        auto  Y = impl::tensor::to_dense( *H );
            
        if ( verbose(3) ) io::vtk::print( *Y, "Y2" );
        if ( verbose(2) ) io::hdf5::write( Y->tensor(), "Y2" );
            
        impl::blas::add( -1, X, Y->tensor() );
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y->tensor() ), impl::blas::norm_F( Y->tensor() ) / impl::blas::norm_F( X ) ) << std::endl;
            
        if ( verbose(3) ) io::vtk::print( *Y, "error2" );

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        tic = timer::now();

        impl::tensor::compress( *H, acc );

        toc = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;
        std::cout << "      rate = " << boost::format( "%.02fx" ) % ( double(X.byte_size()) / double(H->byte_size()) ) << std::endl;

        tic = timer::now();
        
        Y = impl::tensor::to_dense( *H );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        
        tic = timer::now();

        impl::blas::add( -1, X, Y->tensor() );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    error  = " << format_error( impl::blas::norm_F( Y->tensor() ), impl::blas::norm_F( Y->tensor() ) / impl::blas::norm_F( X ) ) << std::endl;
    }
}
