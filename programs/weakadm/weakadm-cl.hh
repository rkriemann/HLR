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
#include <hlr/cluster/weakadm.hh>
#include <hlr/cluster/build.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// determine number of blocks with face/edge/vertex/strong admissibility
//
size_t
nadmblocks ( const uint              anooverlap,
             const cluster::block &  bc )
{
    if ( bc.is_adm() )
    {
        auto  rowcl = cptrcast( bc.rowcl(), Hpro::TGeomCluster );
        auto  colcl = cptrcast( bc.colcl(), Hpro::TGeomCluster );

        if ( rowcl == colcl )
            return 0;

        const uint  dim       = rowcl->bbox().min().dim();
        uint        ndisjoint = 0;
    
        const auto  rbbox = rowcl->bbox();
        const auto  cbbox = colcl->bbox();
        
        for ( uint  i = 0; i < dim; ++i )
        {
            if (( rbbox.max()[i] <= cbbox.min()[i] ) ||   // ├── τ ──┼── σ ──┤
                ( cbbox.max()[i] <= rbbox.min()[i] ))     // ├── σ ──┼── τ ──┤
                ndisjoint++;
        }// for

        if ( anooverlap <= dim )
        {
            if ( std::min( rbbox.diameter(), cbbox.diameter() ) <= ( 2.0 * rbbox.distance( cbbox ) ) )
                return 0;
            else if ( ndisjoint == anooverlap )
                return 1;
            else
                return 0;
        }// if
        else
        {
            if ( std::min( rbbox.diameter(), cbbox.diameter() ) <= ( 2.0 * rbbox.distance( cbbox ) ) )
                return 1;
            else
                return 0;
        }// else
    }// if
    else
    {
        size_t  n = 0;
        
        for ( uint  i = 0; i < bc.nsons(); ++i )
            n += nadmblocks( anooverlap, * bc.son(i) );

        return n;
    }// else
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
    // 3D coordinates
    //

    auto  gridname = "tensorcube-" + Hpro::to_string( cmdline::n );
    auto  vertices = apps::make_vertices( gridname );
    auto  coord    = cluster::coordinates( vertices );
    
    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << kernel
              << Hpro::to_string( ", n = %d/%d", cmdline::n, coord.ncoord() )
              << ", ntile = " << ntile
              << ( eps > 0 ? Hpro::to_string( ", ε = %.2e", eps ) : Hpro::to_string( ", k = %d", k ) )
              << std::endl;
    
    auto  part         = Hpro::TGeomBSPPartStrat( Hpro::adaptive_split_axis );
    auto  [ ct, pe2i ] = cluster::build_cluster_tree( coord, part, cmdline::ntile );

    auto  adm0    = cluster::weak_adm( 0 ); // off-diagonal admissibility
    auto  adm1    = cluster::weak_adm( 1 ); // face admissibility
    auto  adm2    = cluster::weak_adm( 2 ); // edge admissibility
    auto  adm3    = cluster::weak_adm( 3 ); // vertex admissibility
    auto  strong  = cluster::strong_adm();
    auto  bct0    = cluster::build_block_tree( *ct, *ct, adm0 );
    auto  bct1    = cluster::build_block_tree( *ct, *ct, adm1 );
    auto  bct2    = cluster::build_block_tree( *ct, *ct, adm2 );
    auto  bct3    = cluster::build_block_tree( *ct, *ct, adm3 );
    auto  bcts    = cluster::build_block_tree( *ct, *ct, strong );

    if ( hpro::verbose( 3 ) )
    {
        io::vtk::print( coord, "coord" );
        io::eps::print( *bct0, "bct0" );
        io::eps::print( *bct1, "bct1" );
        io::eps::print( *bct2, "bct2" );
        io::eps::print( *bct3, "bct3" );
        io::eps::print( *bcts, "bcts" );
    }// if

    // std::cout << term::bold << "  c_sp:" << term::reset << std::endl
    //           << "    adm0 : " << Hpro::compute_c_sp( *bct0, true, true ) << std::endl
    //           << "    adm1 : " << Hpro::compute_c_sp( *bct1, true, true ) << std::endl
    //           << "    adm2 : " << Hpro::compute_c_sp( *bct2, true, true ) << std::endl
    //           << "    adm3 : " << Hpro::compute_c_sp( *bct3, true, true ) << std::endl
    //           << "    std  : " << Hpro::compute_c_sp( *bcts, true, true ) << std::endl;

    nadmblocks( 1, *bct2 );
    
    std::cout << term::bold << "  adm. blocks:" << term::reset << std::endl
              << "         │    face │   edge  │   vtx   │   std   │ c_sp " << std::endl
              << "   ──────┼─────────┼─────────┼─────────┼─────────┼──────" << std::endl
              << "    adm0 │ "
              << boost::format( "%7d │ %7d │ %7d │ %7d │ %3d" ) % nadmblocks( 1, *bct0 ) % nadmblocks( 2, *bct0 ) % nadmblocks( 3, *bct0 ) % nadmblocks( 4, *bct0 ) % Hpro::compute_c_sp( *bct0, true, true ) << std::endl
              << "    adm1 │ "
              << boost::format( "%7d │ %7d │ %7d │ %7d │ %3d" ) % nadmblocks( 1, *bct1 ) % nadmblocks( 2, *bct1 ) % nadmblocks( 3, *bct1 ) % nadmblocks( 4, *bct1 ) % Hpro::compute_c_sp( *bct1, true, true ) << std::endl
              << "    adm2 │ "
              << boost::format( "%7d │ %7d │ %7d │ %7d │ %3d" ) % nadmblocks( 1, *bct2 ) % nadmblocks( 2, *bct2 ) % nadmblocks( 3, *bct2 ) % nadmblocks( 4, *bct2 ) % Hpro::compute_c_sp( *bct2, true, true )<< std::endl
              << "    adm3 │ "
              << boost::format( "%7d │ %7d │ %7d │ %7d │ %3d" ) % nadmblocks( 1, *bct3 ) % nadmblocks( 2, *bct3 ) % nadmblocks( 3, *bct3 ) % nadmblocks( 4, *bct3 ) % Hpro::compute_c_sp( *bct3, true, true )<< std::endl
              << "    std  │ "
              << boost::format( "%7d │ %7d │ %7d │ %7d │ %3d" ) % nadmblocks( 1, *bcts ) % nadmblocks( 2, *bcts ) % nadmblocks( 3, *bcts ) % nadmblocks( 4, *bcts ) % Hpro::compute_c_sp( *bcts, true, true ) << std::endl;
    
    // tic = timer::now();

    // auto  logr      = matrix::log_function< value_t >();
    // auto  newton    = matrix::newton_function< value_t >();
    // auto  exp       = matrix::exponential_function< value_t >( value_t(1) );
    // auto  gaussian  = matrix::gaussian_function< value_t >( value_t(1) );
    // auto  mquadric  = matrix::multiquadric_function< value_t >( value_t(1) );
    // auto  imquadric = matrix::inverse_multiquadric_function< value_t >( value_t(1) );
    // auto  tps       = matrix::thin_plate_spline_function< value_t >( value_t(1) );
    // auto  ratquad   = matrix::rational_quadratic_function< value_t >( value_t(1), value_t(1) );
    // auto  matcov    = matrix::matern_covariance_function< value_t >( value_t(1), value_t(1.0/3.0), value_t(1) );

    // if ( cmdline::kernel == "log" )
    // {
    //     auto  kernel = radial_kernel( logr );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "newton" )
    // {
    //     auto  kernel = radial_kernel( newton );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "exp" )
    // {
    //     auto  kernel = radial_kernel( exp );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "gaussian" )
    // {
    //     auto  kernel = radial_kernel( gaussian );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "mquadric" )
    // {
    //     auto  kernel = radial_kernel( mquadric );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "imquadric" )
    // {
    //     auto  kernel = radial_kernel( imquadric );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "tps" )
    // {
    //     auto  kernel = radial_kernel( tps );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "ratquad" )
    // {
    //     auto  kernel = radial_kernel( ratquad );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "matcov" )
    // {
    //     auto  kernel = radial_kernel( matcov );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else
    //     HLR_ERROR( "unsupported radial function : " + cmdline::kernel );
    
    // toc = timer::since( tic );
    
    // std::cout << "done in " << format_time( toc ) << std::endl;

    // //
    // // computing singular values
    // //

    // std::cout << "computing singular values ... " << std::flush;

    // auto  S0 = blas::vector< real_t >();
    // auto  S1 = blas::vector< real_t >();
    // auto  S2 = blas::vector< real_t >();
    // auto  S3 = blas::vector< real_t >();
    // auto  S4 = blas::vector< real_t >();
    
    // tic = timer::now();

    // ::tbb::parallel_invoke(
    //     [&] () { S0 = std::move( blas::sv( M0 ) ); std::cout << "0, " << std::flush; },
    //     [&] () { S1 = std::move( blas::sv( M1 ) ); std::cout << "1, " << std::flush; },
    //     [&] () { S2 = std::move( blas::sv( M2 ) ); std::cout << "2, " << std::flush; },
    //     [&] () { S3 = std::move( blas::sv( M3 ) ); std::cout << "3, " << std::flush; },
    //     [&] () { S4 = std::move( blas::sv( M4 ) ); std::cout << "4, " << std::flush; }
    // );
    
    // toc = timer::since( tic );
    
    // std::cout << "done in " << format_time( toc ) << std::endl;

    // auto  acc = fixed_prec( cmdline::eps );

    // std::cout << "ranks : " << acc.trunc_rank( S1 ) << " / " << acc.trunc_rank( S2 ) << " / " << acc.trunc_rank( S3 ) << " / " << acc.trunc_rank( S4 ) << std::endl;
    
    // io::matlab::write( S0, cmdline::kernel + "_S0" );
    // io::matlab::write( S1, cmdline::kernel + "_S1" );
    // io::matlab::write( S2, cmdline::kernel + "_S2" );
    // io::matlab::write( S3, cmdline::kernel + "_S3" );
    // io::matlab::write( S4, cmdline::kernel + "_S4" );
}
