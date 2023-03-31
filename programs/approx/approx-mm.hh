//
// Project     : HLR
// Program     : approx-mm
// Description : testing approximation algorithms for matrix multiplication
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <fstream>

#include <hlr/utils/likwid.hh>

#include <hpro/io/TCoordVis.hh>

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/lanczos.hh>
#include <hlr/approx/randlr.hh>
#include <hlr/utils/io.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

uint64_t
get_flops ( const std::string &  method );

template < typename value_t >
hpro::real_type_t< value_t >
mm_error ( const hpro::TLinearOperator< value_t > &  A,
           const hpro::TLinearOperator< value_t > &  B )
{
    auto  n = B.domain_dim();
    auto  T = blas::matrix< value_t >( n, 100 );
    auto  S = blas::matrix< value_t >( n, 100 );

    std::random_device          rd{};
    std::mt19937                generator{ rd() };
    std::normal_distribution<>  distr{ 0, 1 };
    auto                        fill_rand = [&] () { return distr( generator ); };

    blas::fill_fn( T, fill_rand );
    blas::scale( value_t(0), S );
    
    auto  diff = matrix::sum( 1, A, -1, B );

    diff.apply_add( value_t(1), T, S, apply_normal );

    return hlr::norm::spectral( S ) / hlr::norm::spectral( T );
}
    
//
// standard mat-mul
//
template < typename value_t,
           typename approx_t >
void
mm_std ( const hpro::TMatrix< value_t >  &  A,
         const hpro::TTruncAcc &            acc,
         const std::string &                apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic      = timer::now();
    auto  toc      = timer::since( tic );
    
    auto  AxA      = matrix::product( A, A );
    auto  norm_AxA = norm::spectral( *AxA );
    auto  C        = impl::matrix::copy( A );
    auto  tstart   = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        C->scale( 0 );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hmmstd" );
            
        impl::multiply< value_t >( value_t(1), hpro::apply_normal, A, hpro::apply_normal, A, *C, acc, approx );

        LIKWID_MARKER_STOP( "hmmstd" );
            
        toc = timer::since( tic );
        std::cout << "      mult in  " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "mm" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  diff = matrix::sum( 1, *AxA, -1, *C );

    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
}

//
// accumulator based mat-mul
//
template < typename value_t,
           typename approx_t >
void
mm_accu ( const hpro::TMatrix< value_t > &  A,
          const hpro::TTruncAcc &           acc,
          const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic      = timer::now();
    auto  toc      = timer::since( tic );
    
    auto  AxA      = matrix::product( A, A );
    auto  norm_AxA = hlr::norm::spectral( *AxA );
    auto  C        = impl::matrix::copy( A );
    auto  tstart   = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        C->scale( 0 );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hmmaccu" );
            
        impl::accu::multiply< value_t >( value_t(1), hpro::apply_normal, A, hpro::apply_normal, A, *C, acc, approx );

        LIKWID_MARKER_STOP( "hmmaccu" );
            
        toc = timer::since( tic );
        std::cout << "      mult in  " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "mm" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "    runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  diff = matrix::sum( 1, *AxA, -1, *C );

    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
}

//
// lazy mat-mul
//
template < typename value_t,
           typename approx_t >
void
mm_lazy ( const hpro::TMatrix< value_t > &  A,
          const hpro::TTruncAcc &           acc,
          const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic      = timer::now();
    auto  toc      = timer::since( tic );
    
    auto  AxA      = matrix::product( A, A );
    auto  norm_AxA = hlr::norm::spectral( *AxA );
    auto  C        = impl::matrix::copy( A );
    auto  tstart   = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        C->scale( 0 );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hmmaccu" );
            
        impl::lazy::multiply< value_t >( value_t(1), hpro::apply_normal, A, hpro::apply_normal, A, *C, acc, approx );

        LIKWID_MARKER_STOP( "hmmaccu" );
            
        toc = timer::since( tic );
        std::cout << "      mult in  " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "mm" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for

    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "    runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  diff = matrix::sum( 1, *AxA, -1, *C );

    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    LIKWID_MARKER_INIT;

    using value_t = typename problem_t::value_t;

    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;

        print_vtk( coord.get(), "coord" );
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if

    blas::reset_flops();
    
    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );

    LIKWID_MARKER_START( "build" );
            
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    // auto  A      = io::hpro::read( "A" );
    
    LIKWID_MARKER_STOP( "build" );
    
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    // std::cout << "    flops = " << format_flops( get_flops( "build" ), toc.seconds() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,norank,nosize" );

    //////////////////////////////////////////////////////////////////////
    //
    // matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "Matrix Multiplication ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    // exact representation
    // auto  AxA      = matrix::product( *A, *A );
    // auto  norm_AxA = hlr::norm::spectral( *AxA );

    if ( cmdline::arith == "std" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "standard" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) mm_std< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if

    //
    // using accumulators
    //

    if ( cmdline::arith == "accu" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulator" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) mm_accu< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if

    //
    // using lazy evaluation
    //

    if ( cmdline::arith == "lazy" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "lazy" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) mm_lazy< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
    
    LIKWID_MARKER_CLOSE;
}

//
// return FLOPs for standard settings
//
uint64_t
get_flops ( const std::string &  method )
{
    #if HLIB_COUNT_FLOPS == 1

    return blas::get_flops();

    #else

    if ( ntile == 128 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 455151893464;   // 515345354964;
            if ( gridfile == "sphere-6" ) return 2749530544148;  // 3622694502712;
            if ( gridfile == "sphere-7" ) return 12122134505132; // 21122045509696;
            if ( gridfile == "sphere-8" ) return 118075035109436;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 124087920212;  // 122140965488;
            if ( gridfile == "sphere-6" ) return 881254402164;  // 832636379560;
            if ( gridfile == "sphere-7" ) return 5442869949704; // 5113133279628;
            if ( gridfile == "sphere-8" ) return 30466486574184;
        }// if
    }// if
    else if ( ntile == 64 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 362295459228;  // 362301558484;
            if ( gridfile == "sphere-6" ) return 2254979752712; // 2364851019180;
            if ( gridfile == "sphere-7" ) return 9888495763740; // 10305554560228;
            if ( gridfile == "sphere-8" ) return 119869484219652;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 111349327848; // 111663294708;
            if ( gridfile == "sphere-6" ) return 912967909892; // 936010549040;
            if ( gridfile == "sphere-7" ) return 6025437614656; // 6205509061236;
            if ( gridfile == "sphere-8" ) return 33396933144996;
        }// if
    }// if

    #endif

    return 0;
}
