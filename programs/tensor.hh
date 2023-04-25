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
// error of tucker decomposition
//
template < typename value_t >
Hpro::real_type_t< value_t >
tucker_error ( const blas::tensor3< value_t > &  X,
               const blas::tensor3< value_t > &  G,
               const blas::matrix< value_t > &   X0,
               const blas::matrix< value_t > &   X1,
               const blas::matrix< value_t > &   X2 )
{
    auto  T0 = blas::tensor_product( G,  X0, 0 );
    auto  T1 = blas::tensor_product( T0, X1, 1 );
    auto  Y  = blas::tensor_product( T1, X2, 2 );
        
    impl::blas::add( -1, X, Y );

    return impl::blas::norm_F( Y );
}

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
// tensor cross approximation
//
template < typename value_t >
void
tca_full ( blas::tensor3< value_t > &  X,
           const double                tol )
{
    auto    C    = std::list< value_t >();
    auto    V0   = std::list< blas::vector< value_t > >();
    auto    V1   = std::list< blas::vector< value_t > >();
    auto    V2   = std::list< blas::vector< value_t > >();
    size_t  step = 0;

    while ( true )
    {
        // std::cout << X << std::endl;
    
        //
        // determine maximal element in X
        //

        std::array< uint, 3 >  max_pos{ 0, 0, 0 };
        value_t                max_val = std::abs( X(0,0,0) );

        for ( uint  l = 0; l < X.size(2); ++l )
            for ( uint  j = 0; j < X.size(1); ++j )
                for ( uint  i = 0; i < X.size(0); ++i )
                {
                    const auto  X_ijl = std::abs( X(i,j,l) );

                    if ( X_ijl > max_val )
                    {
                        max_val = X_ijl;
                        max_pos = { i, j, l };
                    }// if
                }// for

        //
        // use fibers as next vectors
        //

        auto  v0 = blas::copy( X.fiber( 0, max_pos[1], max_pos[2] ) );
        auto  v1 = blas::copy( X.fiber( 1, max_pos[0], max_pos[2] ) );
        auto  v2 = blas::copy( X.fiber( 2, max_pos[0], max_pos[1] ) );

        // std::cout << v0 << std::endl;
        // std::cout << v1 << std::endl;
        // std::cout << v2 << std::endl;
        
        // auto  n0 = blas::norm2( v0 );
        // auto  n1 = blas::norm2( v1 );
        // auto  n2 = blas::norm2( v2 );
        // auto  c  = n0 * n1 * n2;
        
        if ( max_val < 1e-20 ) // just fail-safe
            break;

        blas::scale( 1.0 / max_val, v0 );
        blas::scale( 1.0 / max_val, v1 );
        blas::scale( 1.0 / max_val, v2 );
        
        // std::cout << v0 << std::endl;
        // std::cout << v1 << std::endl;
        // std::cout << v2 << std::endl;
        
        for ( uint  l = 0; l < X.size(2); ++l )
            for ( uint  j = 0; j < X.size(1); ++j )
                for ( uint  i = 0; i < X.size(0); ++i )
                    X(i,j,l) -= max_val * v0(i) * v1(j) * v2(l);

        const auto  norm_X = blas::norm_F( X );
        
        std::cout << step << " : " << format_norm( norm_X ) << " / " << max_val << std::endl;
        // std::cout << X << std::endl;
        
        if ( norm_X < tol )
            break;
        
        C.push_back( max_val );
        V0.push_back( std::move( v0 ) );
        V1.push_back( std::move( v1 ) );
        V2.push_back( std::move( v2 ) );

        step++;
    }// while

    std::cout << "rank : " << C.size() << std::endl;
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
        uint  val = 1;
        
        for ( size_t  i = 0; i < 3*4*2; ++i )
            X.data()[i] = val++;

        std::cout << X << std::endl;

        if ( true )
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

        return;
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

        switch ( 3 )
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
        
    if ( verbose(3) ) io::vtk::print( X, "X.vtk" );
    if ( verbose(2) ) io::hdf5::write( X, "X" );

    auto  tapprox = split( cmdline::tapprox, "," );
    
    //
    // cross approximation
    //

    if ( contains( tapprox, "tcafull" ) )
    {
        tca_full( X, cmdline::eps * norm_X );
    }// if
    
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

        auto  [ G, X0, X1, X2 ] = blas::hosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;
            
        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
            
        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = tucker_error( X, Z.G_decompressed(), Z.X_decompressed(0), Z.X_decompressed(1), Z.X_decompressed(2) );

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
        auto  error = tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
            
        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = tucker_error( X, Z.G_decompressed(), Z.X_decompressed(0), Z.X_decompressed(1), Z.X_decompressed(2) );

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
        
        auto  [ G, X0, X1, X2 ] = blas::greedy_hosvd( X, acc, apx );
            
        toc = timer::since( tic );
            
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    ranks  = " << term::bold << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << term::reset << std::endl;

        {
            std::cout << "  " << term::bullet << term::bold << "recompression" << term::reset << std::endl;

            std::cout << "      G  = " << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << std::endl;
            std::cout << "      X0 = " << X0.nrows() << " × " << X0.ncols() << std::endl;
            std::cout << "      X1 = " << X1.nrows() << " × " << X1.ncols() << std::endl;
            std::cout << "      X2 = " << X2.nrows() << " × " << X2.ncols() << std::endl;

            auto  acc2               = absolute_prec( Hpro::frobenius_norm, 10.0 * tol );
            auto  [ G2, W0, W1, W2 ] = blas::greedy_hosvd( G, acc2, apx );

            std::cout << "      G2 = " << G2.size(0) << " × " << G2.size(1) << " × " << G2.size(2) << std::endl;
            std::cout << "      W0 = " << W0.nrows() << " × " << W0.ncols() << std::endl;
            std::cout << "      W1 = " << W1.nrows() << " × " << W1.ncols() << std::endl;
            std::cout << "      W2 = " << W2.nrows() << " × " << W2.ncols() << std::endl;

            auto  V0 = blas::prod( X0, W0 );
            auto  V1 = blas::prod( X1, W1 );
            auto  V2 = blas::prod( X2, W2 );

            auto  error = tucker_error( X, G2, V0, V1, V2 );
            
            std::cout << "      error  = " << format_error( error, error / norm_X ) << std::endl;
        }
        
        auto  Z     = tensor::tucker_tensor3< value_t >( is( 0, X.size(0)-1 ), is( 0, X.size(1)-1 ), is( 0, X.size(2)-1 ),
                                                         std::move( G ),
                                                         std::move( X0 ),
                                                         std::move( X1 ),
                                                         std::move( X2 ) );
        auto  error = tucker_error( X, Z.G(), Z.X(0), Z.X(1), Z.X(2) );
            
        std::cout << "    mem    = " << format_mem( Z.byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(Z.byte_size()) ) << std::endl;
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        Z.compress( Hpro::fixed_prec( cmdline::eps ) );

        error = tucker_error( X, Z.G_decompressed(), Z.X_decompressed(0), Z.X_decompressed(1), Z.X_decompressed(2) );

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

        auto  Y = impl::tensor::to_dense( *H );

        impl::blas::add( -1, X, Y->tensor() );

        auto  error = impl::blas::norm_F( Y->tensor() );
        
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;

        std::cout << "  " << term::bullet << term::bold << "compression via " << compress::provider << term::reset << std::endl;

        tic = timer::now();

        impl::tensor::compress( *H, Hpro::fixed_prec( cmdline::eps ) );

        toc = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( H->byte_size() ) << std::endl;
        std::cout << "      rate = " << format_rate( double(X.byte_size()) / double(H->byte_size()) ) << std::endl;

        Y = impl::tensor::to_dense( *H );
        impl::blas::add( -1, X, Y->tensor() );
        error = impl::blas::norm_F( Y->tensor() );
        std::cout << "    error  = " << format_error( error, error / norm_X ) << std::endl;
    }
}
