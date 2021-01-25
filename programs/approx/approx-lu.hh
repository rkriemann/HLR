//
// Project     : HLR
// Program     : approx-lu
// Description : testing approximation algorithms for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <fstream>

#include <hlr/utils/likwid.hh>

#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/lanczos.hh>
#include <hlr/approx/randlr.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

uint64_t
get_flops ( const std::string &  method );

//
// standard LU
//
template < typename approx_t >
void
lu_std ( const hpro::TMatrix &    A,
         const hpro::TTruncAcc &  acc,
         const std::string &      apx_name )
{
    using  value_t = typename approx_t::value_t;

    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  C   = impl::matrix::copy( A );
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hlustd" );
            
        impl::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hlustd" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( nbench > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( inv_approx_2( & A, & A_inv ) ) << std::endl;
}

//
// accumulator base LU
//
template < typename approx_t >
void
lu_accu ( const hpro::TMatrix &    A,
          const hpro::TTruncAcc &  acc,
          const std::string &      apx_name )
{
    using  value_t = typename approx_t::value_t;

    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  C   = impl::matrix::copy( A );
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hluaccu" );
            
        impl::accu::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hluaccu" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( nbench > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( inv_approx_2( & A, & A_inv ) ) << std::endl;
}

//
// accumulator base LU
//
template < typename approx_t >
void
lu_lazy ( const hpro::TMatrix &    A,
          const hpro::TTruncAcc &  acc,
          const std::string &      apx_name )
{
    using  value_t = typename approx_t::value_t;

    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  C   = impl::matrix::copy( A );
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hluaccu" );
            
        impl::lazy::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hluaccu" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );
    }// for

    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( nbench > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( inv_approx_2( & A, & A_inv ) ) << std::endl;
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
        
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if

    blas::reset_flops();
    
    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );

    LIKWID_MARKER_START( "build" );
            
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    LIKWID_MARKER_STOP( "build" );
    
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    // std::cout << "    flops = " << format_flops( get_flops( "build" ), toc.seconds() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A" );

    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "LU factorization ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    //
    // standard recursion with immediate updates
    //

    if ( cmdline::arith == "std" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "standard" << term::reset << std::endl;
    
        if ( cmdline::approx == "hpro" || cmdline::approx == "all" )
        {
            std::cout << "    " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

            std::vector< double >  runtime, flops;

            auto  C = impl::matrix::copy( *A );
        
            for ( int i = 0; i < nbench; ++i )
            {
                impl::matrix::copy_to( *A, *C );
            
                blas::reset_flops();

                tic = timer::now();
        
                LIKWID_MARKER_START( "hlustd" );
            
                hpro::LU::factorise_rec( C.get(), acc );

                LIKWID_MARKER_STOP( "hlustd" );
            
                toc = timer::since( tic );
                std::cout << "      LU in    " << format_time( toc ) << std::endl;

                flops.push_back( get_flops( "lu" ) );
                runtime.push_back( toc.seconds() );
            }// for
        
            // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

            if ( nbench > 1 )
                std::cout << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
            std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
            std::cout << "      error  = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << std::endl;
        }// if
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_std< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_std< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_std< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if (( cmdline::approx == "randlr"  || cmdline::approx == "all" ) && ( A->nrows() <= 20000 )) lu_std< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_std< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_std< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
    
    //
    // using accumulators
    //

    if ( cmdline::arith == "accu" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulator" << term::reset << std::endl;
    
        if ( cmdline::approx == "hpro" || cmdline::approx == "all" )
        {
            std::cout << "    " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

            std::vector< double >  runtime, flops;
            auto                   old_config = hpro::CFG::Arith::use_accu;

            hpro::CFG::Arith::use_accu = true;
        
            auto  C = impl::matrix::copy( *A );
        
            for ( int i = 0; i < nbench; ++i )
            {
                impl::matrix::copy_to( *A, *C );
            
                blas::reset_flops();

                tic = timer::now();
        
                LIKWID_MARKER_START( "hluaccu" );
            
                hpro::LU::factorise_rec( C.get(), acc );

                LIKWID_MARKER_STOP( "hluaccu" );
            
                toc = timer::since( tic );
                std::cout << "      LU in    " << format_time( toc ) << std::endl;

                flops.push_back( get_flops( "lu" ) );
                runtime.push_back( toc.seconds() );
            }// for

            hpro::CFG::Arith::use_accu = old_config;
        
            // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

            if ( nbench > 1 )
                std::cout << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
            std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
            std::cout << "      error  = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << std::endl;
        }// if
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_accu< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_accu< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_accu< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if (( cmdline::approx == "randlr"  || cmdline::approx == "all" ) && ( A->nrows() <= 20000 )) lu_accu< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_accu< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_accu< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
    
    //
    // lazy evaluation
    //

    if ( cmdline::arith == "lazy" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "lazy" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_lazy< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_lazy< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_lazy< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if (( cmdline::approx == "randlr"  || cmdline::approx == "all" ) && ( A->nrows() <= 20000 )) lu_lazy< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_lazy< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_lazy< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
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
