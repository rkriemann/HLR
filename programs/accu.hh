//
// Project     : HLR
// Program     : accu
// Description : testing accumulator arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/utils/likwid.hh>

#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "hlr/seq/norm.hh"
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
// standard mat-mul
//
template < typename approx_t >
void
mm_std ( const hpro::TMatrix &    A,
         const hpro::TTruncAcc &  acc,
         const std::string &      apx_name )
{
    using  value_t = typename approx_t::value_t;

    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic      = timer::now();
    auto  toc      = timer::since( tic );
    
    auto  AxA      = hpro::matrix_product( &A, &A );
    auto  norm_AxA = hlr::seq::norm::norm_2( *AxA );
    auto  C        = impl::matrix::copy( A );
        
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
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( nbench > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), C.get() );

    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hlr::seq::norm::norm_2( *diff ) / norm_AxA ) << std::endl;
}

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
// accumulator based mat-mul
//
template < typename approx_t >
void
mm_accu ( const hpro::TMatrix &    A,
          const hpro::TTruncAcc &  acc,
          const std::string &      apx_name )
{
    using  value_t = typename approx_t::value_t;

    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic      = timer::now();
    auto  toc      = timer::since( tic );
    
    auto  AxA      = hpro::matrix_product( &A, &A );
    auto  norm_AxA = hlr::seq::norm::norm_2( *AxA );
    auto  C        = impl::matrix::copy( A );
        
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
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( nbench > 1 )
        std::cout << "    runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), C.get() );

    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hlr::seq::norm::norm_2( *diff ) / norm_AxA ) << std::endl;
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
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    // std::cout << "    flops = " << format_flops( get_flops( "build" ), toc.seconds() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A" );

    //////////////////////////////////////////////////////////////////////
    //
    // matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "Matrix Multiplication ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    // exact representation
    auto  AxA      = hpro::matrix_product( A.get(), A.get() );
    auto  norm_AxA = hlr::seq::norm::norm_2( *AxA );

    std::cout << "  " << term::bullet << term::bold << "standard" << term::reset << std::endl;
    
    //
    // reference: Hpro
    //

    if ( true )
    {
        std::cout << "    " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

        std::vector< double >  runtime, flops;

        auto  C   = impl::matrix::copy( *A );
        
        for ( int i = 0; i < nbench; ++i )
        {
            C->scale( 0 );
            
            blas::reset_flops();

            tic = timer::now();
        
            LIKWID_MARKER_START( "hmm" );
            
            hpro::multiply( value_t(1), apply_normal, A.get(), apply_normal, A.get(), value_t(1), C.get(), acc );

            LIKWID_MARKER_STOP( "hmm" );
            
            toc = timer::since( tic );
            std::cout << "      mult in  " << format_time( toc ) << std::endl;

            flops.push_back( get_flops( "mm" ) );
            runtime.push_back( toc.seconds() );
        }// for
        
        // std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "    runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), C.get() );

        std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "      error  = " << format_error( hlr::seq::norm::norm_2( *diff ) / norm_AxA ) << std::endl;
    }

    //
    // standard recursion with immediate updates
    //

    if ( true ) mm_std< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
    if ( true ) mm_std< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
    if ( true ) mm_std< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
    if ( true ) mm_std< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
    if ( true ) mm_std< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
    if ( true ) mm_std< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );

    //
    // using accumulators
    //

    std::cout << "  " << term::bullet << term::bold << "accumulator" << term::reset << std::endl;
    
    if ( true ) mm_accu< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
    if ( true ) mm_accu< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
    if ( true ) mm_accu< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
    if ( true ) mm_accu< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
    if ( true ) mm_accu< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
    if ( true ) mm_accu< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );

    LIKWID_MARKER_CLOSE;

    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "LU factorization ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    std::cout << "  " << term::bullet << term::bold << "standard" << term::reset << std::endl;
    
    //
    // standard recursion with immediate updates
    //

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

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
    
    if ( true ) lu_std< hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
    if ( true ) lu_std< hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
    if ( true ) lu_std< hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
    if ( true ) lu_std< hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
    if ( true ) lu_std< hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
    if ( true ) lu_std< hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );

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
