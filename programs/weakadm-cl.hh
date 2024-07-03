//
// Project     : HLR
// Module      : weakadm
// Description : program for testing weak admissibility
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hlr/matrix/radial.hh>
#include <hlr/apps/radial.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

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
    // 3D coordinates
    //

    auto  gridname = "tensorgrid-" + Hpro::to_string( cmdline::n );
    auto  vertices = apps::make_vertices( gridname );
    auto  coord    = Hpro::TCoordinate( vertices );
    
    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << kernel
              << Hpro::to_string( ", n = %d/%d", cmdline::n, coord->ncoords() )
              << ", ntile = " << ntile
              << ( eps > 0 ? Hpro::to_string( ", ε = %.2e", eps ) : Hpro::to_string( ", k = %d", k ) )
              << std::endl;
    
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
