//
// Project     : HLR
// Module      : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hlr/matrix/radial.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename radialfn_t >
struct radial_kernel
{
    using  value_t = typename radialfn_t::value_t;
    
    const radialfn_t *  radialfn;

    radial_kernel ( const radialfn_t &  rfn )
            : radialfn( & rfn )
    {}

    double operator () ( const double *  x,
                         const double *  y ) const
    {
        const auto  r = math::sqrt( math::abs2(x[0] - y[0]) +
                                    math::abs2(x[1] - y[1]) +
                                    math::abs2(x[2] - y[2]) );

        return  (*radialfn)( r );
    }
};

template < typename  kernelfn_t >
std::tuple< blas::matrix< typename kernelfn_t::value_t >,
            blas::matrix< typename kernelfn_t::value_t >,
            blas::matrix< typename kernelfn_t::value_t >,
            blas::matrix< typename kernelfn_t::value_t >,
            blas::matrix< typename kernelfn_t::value_t > >
build_blocks ( const kernelfn_t &              kernel,
               const size_t                    n3,
               const blas::matrix< double > &  C0,
               const blas::matrix< double > &  C1,
               const blas::matrix< double > &  C2,
               const blas::matrix< double > &  C3,
               const blas::matrix< double > &  C4 )
{
    using  value_t = typename kernelfn_t::value_t;
    
    auto  M0 = blas::matrix< value_t >( n3, n3 );
    auto  M1 = blas::matrix< value_t >( n3, n3 );
    auto  M2 = blas::matrix< value_t >( n3, n3 );
    auto  M3 = blas::matrix< value_t >( n3, n3 );
    auto  M4 = blas::matrix< value_t >( n3, n3 );

    ::tbb::parallel_invoke(
        [&,n3] ()
        {
            for ( size_t  j = 0; j < n3; ++j )
                for ( size_t  i = 0; i < n3; ++i )
                    M0(i,j) = kernel( C0.ptr( 0, i ), C0.ptr( 0, j ) );

            std::cout << "0, " << std::flush;
        },
    
        [&,n3] ()
        {
            for ( size_t  j = 0; j < n3; ++j )
                for ( size_t  i = 0; i < n3; ++i )
                    M1(i,j) = kernel( C0.ptr( 0, i ), C1.ptr( 0, j ) );

            std::cout << "1, " << std::flush;
        },
    
        [&,n3] ()
        {
            for ( size_t  j = 0; j < n3; ++j )
                for ( size_t  i = 0; i < n3; ++i )
                    M2(i,j) = kernel( C0.ptr( 0, i ), C2.ptr( 0, j ) );

            std::cout << "2, " << std::flush;
        },
    
        [&,n3] ()
        {
            for ( size_t  j = 0; j < n3; ++j )
                for ( size_t  i = 0; i < n3; ++i )
                    M3(i,j) = kernel( C0.ptr( 0, i ), C3.ptr( 0, j ) );

            std::cout << "3, " << std::flush;
        },
        
        [&,n3] ()
        {
            for ( size_t  j = 0; j < n3; ++j )
                for ( size_t  i = 0; i < n3; ++i )
                    M4(i,j) = kernel( C0.ptr( 0, i ), C4.ptr( 0, j ) );
            
            std::cout << "4, " << std::flush;
        } );

    return { std::move( M0 ),
             std::move( M1 ),
             std::move( M2 ),
             std::move( M3 ),
             std::move( M4 ) };
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    auto  runtime = std::vector< double >();
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );

    //
    // base 1D coordinates
    //

    auto  n  = cmdline::n;
    auto  h  = 1.0 / n;
    auto  h2 = h / 2.0;
    auto  X  = blas::vector< double >( n );

    for ( size_t  i = 0; i < n; ++i )
        X(i) = h2 + i * h;

    //
    // 3D coordinates
    //

    auto  n3  = n * n * n;
    
    std::cout << "setting up " << n3 << " coordinates ... " << std::flush;
    
    tic = timer::now();
    
    auto  C0  = blas::matrix< double >( 3, n3 );  // [0,1] × [0,1] × [0,1]
    auto  C1  = blas::matrix< double >( 3, n3 );  // [1,2] × [0,1] × [0,1]
    auto  C2  = blas::matrix< double >( 3, n3 );  // [1,2] × [1,2] × [0,1]
    auto  C3  = blas::matrix< double >( 3, n3 );  // [1,2] × [1,2] × [1,2]
    auto  C4  = blas::matrix< double >( 3, n3 );  // [2,3] × [0,1] × [0,1]
    auto  pos = size_t(0);
    
    for ( size_t  i = 0; i < n; ++i )
    {
        for ( size_t  j = 0; j < n; ++j )
        {
            for ( size_t  k = 0; k < n; ++k )
            {
                C0(0,pos) = X(i);
                C0(1,pos) = X(j);
                C0(2,pos) = X(k);
                     
                C1(0,pos) = X(i) + 1.0;
                C1(1,pos) = X(j);
                C1(2,pos) = X(k);
                     
                C2(0,pos) = X(i) + 1.0;
                C2(1,pos) = X(j) + 1.0;
                C2(2,pos) = X(k);
                     
                C3(0,pos) = X(i) + 1.0;
                C3(1,pos) = X(j) + 1.0;
                C3(2,pos) = X(k) + 1.0;
                     
                C4(0,pos) = X(i) + 2.0;
                C4(1,pos) = X(j);
                C4(2,pos) = X(k);

                pos++;
            }// for
        }// for
    }// for

    toc = timer::since( tic );
    
    std::cout << "done in " << format_time( toc ) << std::endl;
    
    //
    // build matrices
    //

    std::cout << "setting up matrices ... " << std::flush;
    
    tic = timer::now();

    auto  logr      = matrix::log_function< value_t >();
    auto  newton    = matrix::newton_function< value_t >();
    auto  exp       = matrix::exponential_function< value_t >( value_t(1) );
    auto  gaussian  = matrix::gaussian_function< value_t >( value_t(1) );
    auto  mquadric  = matrix::multiquadric_function< value_t >( value_t(1) );
    auto  imquadric = matrix::inverse_multiquadric_function< value_t >( value_t(1) );
    auto  tps       = matrix::thin_plate_spline_function< value_t >( value_t(1) );
    auto  ratquad   = matrix::rational_quadratic_function< value_t >( value_t(1), value_t(1) );
    auto  matcov    = matrix::matern_covariance_function< value_t >( value_t(1), value_t(1.0/3.0), value_t(1) );

    auto  M0 = blas::matrix< value_t >();
    auto  M1 = blas::matrix< value_t >();
    auto  M2 = blas::matrix< value_t >();
    auto  M3 = blas::matrix< value_t >();
    auto  M4 = blas::matrix< value_t >();

    if ( cmdline::kernel == "log" )
    {
        auto  kernel = radial_kernel( logr );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "newton" )
    {
        auto  kernel = radial_kernel( newton );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "exp" )
    {
        auto  kernel = radial_kernel( exp );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "gaussian" )
    {
        auto  kernel = radial_kernel( gaussian );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "mquadric" )
    {
        auto  kernel = radial_kernel( mquadric );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "imquadric" )
    {
        auto  kernel = radial_kernel( imquadric );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "tps" )
    {
        auto  kernel = radial_kernel( tps );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "ratquad" )
    {
        auto  kernel = radial_kernel( ratquad );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else if ( cmdline::kernel == "matcov" )
    {
        auto  kernel = radial_kernel( matcov );
        
        std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    }// if
    else
        HLR_ERROR( "unsupported radial function : " + cmdline::kernel );
    
    toc = timer::since( tic );
    
    std::cout << "done in " << format_time( toc ) << std::endl;

    //
    // computing singular values
    //

    std::cout << "computing singular values ... " << std::flush;

    auto  S0 = blas::vector< real_t >();
    auto  S1 = blas::vector< real_t >();
    auto  S2 = blas::vector< real_t >();
    auto  S3 = blas::vector< real_t >();
    auto  S4 = blas::vector< real_t >();
    
    tic = timer::now();

    ::tbb::parallel_invoke(
        [&] () { S0 = std::move( blas::sv( M0 ) ); std::cout << "0, " << std::flush; },
        [&] () { S1 = std::move( blas::sv( M1 ) ); std::cout << "1, " << std::flush; },
        [&] () { S2 = std::move( blas::sv( M2 ) ); std::cout << "2, " << std::flush; },
        [&] () { S3 = std::move( blas::sv( M3 ) ); std::cout << "3, " << std::flush; },
        [&] () { S4 = std::move( blas::sv( M4 ) ); std::cout << "4, " << std::flush; }
    );
    
    toc = timer::since( tic );
    
    std::cout << "done in " << format_time( toc ) << std::endl;

    auto  acc = fixed_prec( cmdline::eps );

    std::cout << "ranks : " << acc.trunc_rank( S1 ) << " / " << acc.trunc_rank( S2 ) << " / " << acc.trunc_rank( S3 ) << " / " << acc.trunc_rank( S4 ) << std::endl;
    
    io::matlab::write( S0, cmdline::kernel + "_S0" );
    io::matlab::write( S1, cmdline::kernel + "_S1" );
    io::matlab::write( S2, cmdline::kernel + "_S2" );
    io::matlab::write( S3, cmdline::kernel + "_S3" );
    io::matlab::write( S4, cmdline::kernel + "_S4" );
}
