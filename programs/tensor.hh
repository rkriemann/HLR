//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
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

struct local_accuracy : public hlr::tensor_accuracy
{
    local_accuracy ( const double  abs_eps )
            : tensor_accuracy( hlr::frobenius_norm, 0.0, abs_eps )
    {}
    
    virtual
    const hlr::tensor_accuracy
    acc ( const indexset &  is0,
          const indexset &  is1,
          const indexset &  is2 ) const
    {
        return absolute_prec( hlr::frobenius_norm, abs_eps() * std::sqrt( double(is0.size()) * double(is1.size()) * double(is2.size()) ) );
    }
    using accuracy::acc;
};

//
// construct X as
//                                        1
//    X_ijl = V(x^ijl) =    Σ    ───────────────────
//                       0≤i<j<3 |x^ijl_i - x^ijl_j|
//
template < typename value_t >
blas::tensor3< value_t >
coulomb_cost ( const size_t  n )
{
    // Ω = [0,1]³
    const double  h = 1.0 / (n-1);
    auto          X = blas::tensor3< value_t >( n, n, n );

    for ( size_t  l = 0; l < n; ++l )
    {
        const double  x_2 = l * h;
        
        for ( size_t  j = 0; j < n; ++j )
        {
            const double  x_1 = j * h;
        
            for ( size_t  i = 0; i < n; ++i )
            {
                const double  x_0 = i * h;

                if (( i != j ) && ( j != l ) && ( i != l ))
                    X(i,j,l) = ( 1.0 / std::abs( x_0 - x_1 ) +
                                 1.0 / std::abs( x_0 - x_2 ) +
                                 1.0 / std::abs( x_1 - x_2 ) );
            }// for
        }// for
    }// for
    
    return X;
}

//
// generate lowrank tensor from canonical form
//
template < typename value_t >
blas::tensor3< value_t >
rand_tensor ( const size_t  n,
              const size_t  rank )
{
    auto  X  = blas::tensor3< value_t >( n, n, n );
    auto  a0 = 1.0;
    auto  a1 = 1.0;
    auto  a2 = 1.0;
    
    for ( size_t  k = 0; k < rank; ++k )
    {
        auto  v0 = blas::random< value_t >( n );
        auto  v1 = blas::random< value_t >( n );
        auto  v2 = blas::random< value_t >( n );

        for ( size_t  l = 0; l < n; ++l )
        {
            const auto  xl = a2 * v2(l);
            
            for ( size_t  j = 0; j < n; ++j )
            {
                const auto  xj = xl * a1 * v1(j);
                
                for ( size_t  i = 0; i < n; ++i )
                    X(i,j,l) += xj * a0 * v0(i);
            }// for
        }// for

        a2 *= 0.9;
        a1 *= 0.8;
        a0 *= 0.7;
    }// for

    return  X;
}

//
// generate tensor based on sin/cos
//
template < typename value_t >
blas::tensor3< value_t >
sin_tensor ( const size_t  n )
{
    const auto    π = std::numbers::pi;
    const double  h = π / double(n-1);
    auto          X = blas::tensor3< value_t >( n, n, n );
            
    for ( uint  l = 0; l < n; ++l )
        for ( uint  j = 0; j < n; ++j )
            for ( uint  i = 0; i < n; ++i )
                X( i, j, l ) = std::sin( 32.0 * i * h ) + std::cos( 16.0 * j * h ) + std::sin( 8.0 * l * h );

    return X;
}

//
// generate tensor based on sin/cos
//
template < typename value_t >
blas::tensor3< value_t >
test_tensor ( const size_t  n )
{
    auto     X = blas::tensor3< value_t >( n, n, n );
    value_t  v = 1.0;
            
    for ( uint  l = 0; l < n; ++l )
        for ( uint  j = 0; j < n; ++j )
            for ( uint  i = 0; i < n; ++i )
                X( i, j, l ) = v++;

    return X;
}

//
// play around with tensors
//
template < typename value_t >
void
test_tensors ();

template < typename value_t >
void
test_compression ( const blas::tensor3< value_t > &  X,
                   const blas::tensor3< value_t > &  G,
                   const blas::matrix< value_t > &   X0,
                   const blas::matrix< value_t > &   X1,
                   const blas::matrix< value_t > &   X2 )
{
    auto          norm_X = blas::norm_F( X );
    auto          norm_G = blas::norm_F( G );
    const size_t  mem = sizeof(value_t) * ( G.size(0) * G.size(1) * G.size(2) +
                                            X0.nrows() + X0.ncols() +
                                            X1.nrows() + X1.ncols() +
                                            X2.nrows() + X2.ncols() );
    
    auto  T0     = tensor_product( G,  X0, 0 );
    auto  T1     = tensor_product( T0, X1, 1 );
    auto  Y      = tensor_product( T1, X2, 2 );
    auto  norm_Y = blas::norm_F( Y );
            
    for ( double  tol = 1e-2; tol >= eps; tol /= 10 )
    {
        size_t  zmem = 0;
        
        std::cout << term::bullet << term::bold << "ε = " << tol << term::reset << std::endl;

        #if 1
        auto  zX0 = compress::aplr::zarray();
        auto  zX1 = compress::aplr::zarray();
        auto  zX2 = compress::aplr::zarray();

        {
            auto  Gi    = G.unfold( 0 );
            auto  S_tol = blas::sv( Gi );

            for ( uint  l = 0; l < S_tol.length(); ++l )
                S_tol(l) = ( tol ) / S_tol(l);

            zX0   = std::move( compress::aplr::compress_lr( X0, S_tol ) );
            zmem += compress::aplr::byte_size( zX0 );
        }
        {
            auto  Gi    = G.unfold( 1 );
            auto  S_tol = blas::sv( Gi );

            for ( uint  l = 0; l < S_tol.length(); ++l )
                S_tol(l) = ( tol ) / S_tol(l);

            zX1 = std::move( compress::aplr::compress_lr( X1, S_tol ) );
            zmem += compress::aplr::byte_size( zX1 );
        }
        {
            auto  Gi    = G.unfold( 2 );
            auto  S_tol = blas::sv( Gi );

            for ( uint  l = 0; l < S_tol.length(); ++l )
                S_tol(l) = ( tol ) / S_tol(l);

            zX2 = std::move( compress::aplr::compress_lr( X2, S_tol ) );
            zmem += compress::aplr::byte_size( zX2 );
        }
        #else
        auto  zX0 = compress::compress< value_t >( compress::get_config( tol ), X0 );
        auto  zX1 = compress::compress< value_t >( compress::get_config( tol ), X1 );
        auto  zX2 = compress::compress< value_t >( compress::get_config( tol ), X2 );

        zmem += compress::byte_size( zX0 );
        zmem += compress::byte_size( zX1 );
        zmem += compress::byte_size( zX2 );
        #endif
        
        auto  zG = compress::compress< value_t >( compress::get_config( tol ), G );

        zmem += compress::byte_size( zG );

        {
            auto  dG  = blas::tensor3< value_t >( G.size(0), G.size(1), G.size(2) );
            auto  dX0 = blas::matrix< value_t >( X0.nrows(), X0.ncols() );
            auto  dX1 = blas::matrix< value_t >( X1.nrows(), X1.ncols() );
            auto  dX2 = blas::matrix< value_t >( X2.nrows(), X2.ncols() );

            compress::decompress( zG, dG );
            #if 1
            compress::aplr::decompress_lr( zX0, dX0 );
            compress::aplr::decompress_lr( zX1, dX1 );
            compress::aplr::decompress_lr( zX2, dX2 );
            #else
            compress::decompress( zX0, dX0 );
            compress::decompress( zX1, dX1 );
            compress::decompress( zX2, dX2 );
            #endif

            auto  errorY = blas::tucker_error( Y, dG, dX0, dX1, dX2 );
            auto  errorX = blas::tucker_error( X, dG, dX0, dX1, dX2 );

            blas::add( -1, G, dG );
            blas::add( -1, X0, dX0 );
            blas::add( -1, X1, dX1 );
            blas::add( -1, X2, dX2 );

            std::cout << "    rate    " << double(mem) / double(zmem) << std::endl;
            std::cout << "    error G / X0 / X1 / X2"
                      << " : " 
                      << boost::format( "%.4e" ) % ( blas::norm_F( dG ) / blas::norm_F( G ) )
                      << " / "
                      << boost::format( "%.4e" ) % ( blas::norm_F( dX0 ) / blas::norm_F( X0 ) )
                      << " / "
                      << boost::format( "%.4e" ) % ( blas::norm_F( dX1 ) / blas::norm_F( X1 ) )
                      << " / "
                      << boost::format( "%.4e" ) % ( blas::norm_F( dX2 ) / blas::norm_F( X2 ) )
                      << std::endl;

            std::cout << "    error X "
                      << " : " 
                      << boost::format( "%.4e" ) % ( errorX )
                      << " / "
                      << boost::format( "%.4e" ) % ( errorX / norm_X )
                      << std::endl;
            std::cout << "    error Y "
                      << " : " 
                      << boost::format( "%.4e" ) % ( errorY )
                      << " / "
                      << boost::format( "%.4e" ) % ( errorY / norm_Y )
                      << std::endl;
        }
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

    if ( true )
    {
        test_tensors< value_t >();
        return;
    }// if

    {
        auto  M = blas::random< value_t >( 4, 4 );
        auto  R = blas::matrix< value_t >();

        io::matlab::write( M, "M" );

        blas::lq( M, R );
        
        io::matlab::write( M, "Q" );
        io::matlab::write( R, "R" );
    }
    
    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  apx = approx::SVD< value_t >();
        
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
        tic = timer::now();

        switch ( 0 )
        {
            case 0:
                std::cout << "  " << term::bullet << term::bold << "building Coulomb cost tensor" << term::reset << std::endl;
                X = std::move( coulomb_cost< value_t >( n ) );
                break;

            case 1:
                std::cout << "  " << term::bullet << term::bold << "building random tensor" << term::reset << std::endl;
                X = std::move( rand_tensor< value_t >( n, 50 ) );
                break;

            case 2:
                std::cout << "  " << term::bullet << term::bold << "building sin/cos tensor" << term::reset << std::endl;
                X = std::move( sin_tensor< value_t >( n ) );
                break;
                
            case 3:
                std::cout << "  " << term::bullet << term::bold << "building test tensor" << term::reset << std::endl;
                X = std::move( test_tensor< value_t >( n ) );
                break;

            default:
                HLR_ERROR( "no tensor" );
        }// switch
            
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
    }// else
        
    std::cout << "    dims   = " << term::bold << X.size(0) << " × " << X.size(1) << " × " << X.size(2) << term::reset << std::endl;
    std::cout << "    mem    = " << format_mem( X.byte_size() ) << std::endl;

    const auto  norm_X = impl::blas::norm_F( X );
    
    std::cout << "    |X|_F  = " << format_norm( norm_X ) << std::endl;
        
    // {
    //     //
    //     // copy some part
    //     //

    //     auto  Y = blas::tensor3< value_t >( 256, 256, 256 );

    //     for ( uint l = 0; l < 256; l++ )
    //         for ( uint j = 0; j < 256; j++ )
    //             for ( uint i = 0; i < 256; i++ )
    //                 Y(i,j,l) = X(i + 1024,j,l);

    //     io::vtk::print( Y, "Y.vtk" );
    //     io::hdf5::write( Y, "Y" );
    //     return;
    // }

    if ( verbose(3) ) io::vtk::print( X, "X.vtk" );
    if ( verbose(2) ) io::hdf5::write( X, "X" );

    auto  tapprox = split( cmdline::tapprox, "," );
    
    //
    // HOSVD
    //
    
    if ( contains( tapprox, "hosvd" ) )
    {
        std::cout << term::bullet << term::bold << "HOSVD" << " ( ε = " << cmdline::eps << " )" << term::reset << std::endl;

        auto  dim_fac = 1.0 / std::sqrt( 3.0 );
        auto  tol     = cmdline::eps * norm_X / dim_fac;
        auto  acc     = absolute_prec( Hpro::frobenius_norm, tol );
        
        tic = timer::now();

        auto  [ G, X0, X1, X2 ] = impl::blas::hosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            
        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
            
        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }

    //
    // ST-HOSVD
    //

    if ( contains( tapprox, "sthosvd" ) )
    {
        std::cout << term::bullet << term::bold << "ST-HOSVD" << " ( ε = " << cmdline::eps << " )" << term::reset << std::endl;

        auto  dim_fac = 1.0 / std::sqrt( 3.0 );
        auto  tol     = cmdline::eps * norm_X / dim_fac;
        auto  acc     = absolute_prec( Hpro::frobenius_norm, tol );

        tic = timer::now();
        
        auto  [ G, X0, X1, X2 ] = blas::sthosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            
        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
            
        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }

    //
    // Greedy-HOSVD
    //

    if ( contains( tapprox, "ghosvd" ) )
    {
        std::cout << term::bullet << term::bold << "Greedy-HOSVD" << " ( ε = " << cmdline::eps << " )" << term::reset << std::endl;

        auto  tol  = cmdline::eps * norm_X;
        auto  acc  = absolute_prec( Hpro::frobenius_norm, tol );

        std::cout << "    tol    = " << boost::format( "%.4e" ) % tol << std::endl;

        tic = timer::now();
        
        auto  [ G, X0, X1, X2 ] = impl::blas::greedy_hosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;

        // {
        //     std::cout << "  " << term::bullet << term::bold << "recompression" << term::reset << std::endl;

        //     std::cout << "      G  = " << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << std::endl;
        //     std::cout << "      X0 = " << X0.nrows() << " × " << X0.ncols() << std::endl;
        //     std::cout << "      X1 = " << X1.nrows() << " × " << X1.ncols() << std::endl;
        //     std::cout << "      X2 = " << X2.nrows() << " × " << X2.ncols() << std::endl;

        //     auto  acc2               = absolute_prec( Hpro::frobenius_norm, 10.0 * tol );
        //     auto  [ G2, W0, W1, W2 ] = impl::blas::greedy_hosvd( G, acc2, apx );

        //     std::cout << "      G2 = " << G2.size(0) << " × " << G2.size(1) << " × " << G2.size(2) << std::endl;
        //     std::cout << "      W0 = " << W0.nrows() << " × " << W0.ncols() << std::endl;
        //     std::cout << "      W1 = " << W1.nrows() << " × " << W1.ncols() << std::endl;
        //     std::cout << "      W2 = " << W2.nrows() << " × " << W2.ncols() << std::endl;

        //     auto  V0 = blas::prod( X0, W0 );
        //     auto  V1 = blas::prod( X1, W1 );
        //     auto  V2 = blas::prod( X2, W2 );

        //     auto  error = blas::tucker_error( X, G2, V0, V1, V2 );
            
        //     std::cout << "      error  = " << format_error( error, error / norm_X ) << std::endl;
        // }

        {
            auto  error = blas::tucker_error( X, G, X0, X1, X2 );

            std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
            
            test_compression( X, G, X0, X1, X2 );
            return;
        }
        
        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;

        std::cout << "  "
                  << term::bullet << term::bold << "compression via "
                  << compress::provider << " + " << compress::aplr::provider
                  << term::reset << std::endl;

        Z.compress( relative_prec( cmdline::eps ) );

        error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }

    //
    // Hierarchical HOSVD
    //

    if ( contains( tapprox, "hhosvd" ) || contains( tapprox, "default" ) )
    {
        std::cout << term::bullet << term::bold << "Hierarchical HOSVD (" << "ntile = " << cmdline::ntile << ")" << term::reset << std::endl;

        auto  tol = cmdline::eps * norm_X / std::sqrt( double(X.size(0)) * double(X.size(1)) * double(X.size(2)) );
        auto  acc = local_accuracy( tol );
        // auto  acc = fixed_prec( cmdline::eps * norm_X / 3.0 );

        std::cout << "    tol    = " << boost::format( "%.4e" ) % tol << std::endl;
        
        tic = timer::now();
        
        auto  H   = impl::tensor::build_hierarchical_tucker( X, acc, apx, cmdline::ntile );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(H->byte_size()) ) << std::endl;

        if ( verbose(1) ) io::vtk::print( *H, "H" );

        auto  Y  = impl::tensor::to_dense( *H );
        auto  YT = Y->tensor();

        impl::blas::add( -1, X, YT );

        auto  error = impl::blas::norm_F( YT );
        
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        tic = timer::now();

        impl::tensor::compress( *H, Hpro::fixed_prec( cmdline::eps ) );

        toc = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(H->byte_size()) ) << std::endl;

        Y = impl::tensor::to_dense( *H );
        YT = Y->tensor();
        impl::blas::add( -1, X, YT );
        error = impl::blas::norm_F( YT );
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }

    //
    // cross approximation
    //

    if ( contains( tapprox, "tcafull" ) )
    {
        std::cout << term::bullet << term::bold << "TCA-Full" << " ( ε = " << cmdline::eps << " )" << term::reset << std::endl;
        
        auto  tol = cmdline::eps;
        auto  acc = relative_prec( tol );
        
        tic = timer::now();
        
        auto  [ G, X0, X1, X2 ] = blas::tca_full( X, acc, verbosity );
        
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;

        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
        
        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = blas::tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );

        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }// if
}

template < typename value_t >
void
test_tensors ()
{
    {
        int   n = 8;
        auto  X = blas::tensor4< value_t >( n, n, n, n );
        uint  val = 1;
        
        for ( size_t  i = 0; i < n*n*n*n; ++i )
            X.data()[i] = val++;

        std::cout << X << std::endl;

        // if ( false )
        // {
        //     auto  v_000 = X.fiber( 0, 0, 0, 0 );
        //     auto  v_100 = X.fiber( 0, 1, 0, 0 );
        //     auto  v_210 = X.fiber( 0, 2, 1, 0 );
        //     auto  v_231 = X.fiber( 0, 2, 2, 1 );

        //     std::cout << v_000 << std::endl;
        //     std::cout << v_100 << std::endl;
        //     std::cout << v_210 << std::endl;
        //     std::cout << v_231 << std::endl;
        // }// if

        // auto  M0 = X.unfold( 0 );
        // auto  M1 = X.unfold( 1 );
        // auto  M2 = X.unfold( 2 );
        // auto  M3 = X.unfold( 3 );

        // std::cout << M0 << std::endl;
        // std::cout << M1 << std::endl;
        // std::cout << M2 << std::endl;
        // std::cout << M3 << std::endl;

        auto  T                     = blas::copy( X );
        auto  apx                   = approx::SVD< value_t >();
        auto  [ G, U0, U1, U2, U3 ] = blas::hosvd( X, relative_prec( 1e-4 ), apx );

        std::cout << G.size(0) << " x " << G.size(1) << " x " << G.size(2) << " x " << G.size(3) << std::endl;
        std::cout << "error : " << format_error( blas::tucker_error( T, G, U0, U1, U2, U3 ), blas::tucker_error( T, G, U0, U1, U2, U3 ) / blas::norm_F( T ) ) << std::endl;
    }
    
    //
    // play around with slices and tensor_product
    //
        
    auto  X = blas::tensor3< value_t >( 3, 4, 2 );
    uint  val = 1;
        
    for ( size_t  i = 0; i < 3*4*2; ++i )
        X.data()[i] = val++;

    std::cout << X << std::endl;

    if ( true )
    {
        auto  v_z00 = X.fiber( 2, 0, 0 );
        auto  v_z10 = X.fiber( 2, 1, 0 );
        auto  v_z21 = X.fiber( 2, 2, 1 );

        std::cout << v_z00 << std::endl;
        std::cout << v_z10 << std::endl;
        std::cout << v_z21 << std::endl;
    }// if
    
    if ( false )
    {
        auto  U = blas::matrix< value_t >( 2, 3 );

        val = 1;

        for ( size_t  i = 0; i < 2*3; ++i )
            U.data()[i] = val++;
        
        std::cout << U << std::endl;

        auto  XU = blas::tensor_product( X, U, 0 );
            
        std::cout << XU << std::endl;
            
        auto  X0   = blas::copy( X.slice( 2, 0 ) );
        auto  X1   = blas::copy( X.slice( 2, 1 ) );
        auto  XU0  = blas::prod( U, X0 );
        auto  XU1  = blas::prod( U, X1 );
            
        std::cout << XU1 << std::endl;
        std::cout << XU0 << std::endl;
    }

    if ( false )
    {
        auto  U = blas::matrix< value_t >( 3, 4 );

        val = 1;

        for ( size_t  i = 0; i < 2*4; ++i )
            U.data()[i] = val++;
        
        std::cout << U << std::endl;

        auto  XU = blas::tensor_product( X, U, 1 );
            
        std::cout << XU << std::endl;
            
        auto  X0   = blas::copy( X.slice( 0, 0 ) );
        auto  X1   = blas::copy( X.slice( 0, 1 ) );
        auto  XU0  = blas::prod( U, X0 );
        auto  XU1  = blas::prod( U, X1 );
            
        std::cout << XU1 << std::endl;
        std::cout << XU0 << std::endl;
    }
        
    if ( false )
    {
        auto  U = blas::matrix< value_t >( 2, 2 );

        val = 1;

        for ( size_t  i = 0; i < 2*4; ++i )
            U.data()[i] = val++;
        
        std::cout << U << std::endl;

        auto  XU = blas::tensor_product( X, U, 2 );
            
        std::cout << XU << std::endl;
            
        auto  X0   = blas::copy( X.slice( 1, 0 ) );
        std::cout << X0 << std::endl;
        // auto  X1   = blas::copy( X.slice( 1, 1 ) );
        auto  XU0  = blas::prod( U, X0 );
        // auto  XU1  = blas::prod( X1, blas::transposed( U ) );
            
        // std::cout << XU1 << std::endl;
        std::cout << XU0 << std::endl;
    }
        
    // auto  X_1  = X.slice( 1, 2 );
        
    // std::cout << X_1 << std::endl;
        
    // auto  X_01 = X.fiber( 0, 0, 0 );

    // std::cout << X_01 << std::endl;
}
