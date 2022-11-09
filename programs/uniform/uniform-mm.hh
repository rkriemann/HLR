//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/print.hh>
#include <hlr/bem/aca.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>

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

    auto  runtime = std::vector< double >();
    auto  prnopt  = "noid";
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *ct->root(), "ct" );
        io::eps::print( *bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );

    auto  tic    = timer::now();
    auto  A      = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    // auto  A      = io::hpro::read< value_t >( "A.hm" );
    auto  toc    = timer::since( tic );

    // io::hpro::write( *A, "A.hm" );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A, "A", prnopt );
        io::eps::print_mem( *A, "Am" );
        io::eps::print_lvl( *A, "L" );
    }// if

    const auto  normA = hlr::norm::spectral( *A, 1e-4 );
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "uniform matrix" << term::reset << std::endl;

    auto  apx = approx::SVD< value_t >();
    
    tic = timer::now();
    
    auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A, apx, acc );

    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

    {
        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = hlr::seq::norm::spectral( *diff, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
    }

    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A2, "A2", prnopt );
        io::eps::print_mem( *A2, "A2m", prnopt );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // H matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "H-Multiplication" << term::reset << std::endl;

    auto  AxA      = matrix::product( *A, *A );
    auto  norm_AxA = norm::spectral( *AxA );
    auto  mmapx    = approx::SVD< value_t >();
    
    if ( false )
    {
        std::cout << "  " << term::bullet << "eager" << term::reset << std::endl;
            
        auto  B = impl::matrix::copy( *A );

        B->scale( 0 );

        tic = timer::now();
                
        impl::multiply( value_t(1), apply_normal, *A, apply_normal, *A, *B, acc, mmapx );
            
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

        auto  diff = matrix::sum( 1.0, *AxA, -1.0, *B );

        std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

        if ( false )
        {
            std::cout << "    " << term::bullet << term::bold << "testing original bases" << term::reset << std::endl;

            auto  B2    = impl::matrix::copy_uniform< value_t >( *B, *rowcb, *colcb );
            auto  diff2 = matrix::sum( 1.0, *AxA, -1.0, *B2 );
                    
            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff2 ) / norm_AxA ) << std::endl;
        }// if
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << "accumulator" << term::reset << std::endl;
            
        auto  B = impl::matrix::copy( *A );

        B->scale( 0 );

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
                
            impl::accu::multiply( value_t(1), apply_normal, *A, apply_normal, *A, *B, acc, mmapx );
            
            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );

            std::cout << "    done in  " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                B->scale( 0 );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

        auto  diff = matrix::sum( 1.0, *AxA, -1.0, *B );

        std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

        if ( false )
        {
            std::cout << "    " << term::bullet << term::bold << "testing original bases" << term::reset << std::endl;

            auto  B2    = impl::matrix::copy_uniform< value_t >( *B, *rowcb, *colcb );
            auto  diff2 = matrix::sum( value_t(1), *AxA, value_t(-1), *B2 );
                    
            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff2 ) / norm_AxA ) << std::endl;
        }// if

        if ( false )
        {
            std::cout << "    " << term::bullet << term::bold << "converting to uniform-H" << term::reset << std::endl;
    
            auto  [ rowcb, colcb, B2 ] = impl::matrix::build_uniform_rec( *B, mmapx, acc, nseq );
            auto  diff2                = matrix::sum( value_t(1), *AxA, value_t(-1), *B2 );
        
            std::cout << "      mem    = " << format_mem( B2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;
            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff2 ) / norm_AxA ) << std::endl;
        }// if

        if ( hpro::verbose( 3 ) )
            io::eps::print( *B, "B", prnopt );

        runtime.clear();
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // Uniform H matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "Uniform H-Multiplication" << term::reset << std::endl;
    
    {
        // if ( false )
        // {
        //     std::cout << "  " << term::bullet << "eager" << term::reset << std::endl;

        //     auto  B      = impl::matrix::copy( *A2 );
        //     auto  rowcb2 = rowcb->copy();
        //     auto  colcb2 = colcb->copy();

        //     matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
        //     B->scale( 0 );
            
        //     tic = timer::now();
                
        //     impl::uniform::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, mmapx );
            
        //     toc = timer::since( tic );
        //     std::cout << "    done in  " << format_time( toc ) << std::endl;
        //     std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

        //     auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

        //     std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
        // }// if

        if ( false )
        {
            std::cout << "  " << term::bullet << "accumulator" << term::reset << std::endl;

            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );
            
            tic = timer::now();
                
            impl::uniform::accu::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, mmapx );
            
            toc = timer::since( tic );
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( B->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;

            auto  diff = matrix::sum( value_t(1), *AxA, value_t(-1), *B );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

            if ( hpro::verbose( 3 ) )
            {
                io::eps::print( *B, "B2", prnopt );
                io::eps::print( *rowcb2, "rowcb2", "mem" );
                io::eps::print( *colcb2, "colcb2", "mem" );
            }// if
        }// if

        if ( true )
        {
            std::cout << "  " << term::bullet << "accumulator (cached)" << term::reset << std::endl;

            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
                
                impl::uniform::accu::multiply_cached( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, mmapx );
            
                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
                
                std::cout << "    done in  " << format_time( toc ) << std::endl;
                if ( i < nbench-1 )
                    B->scale( 0 );
            }// for
            
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
            
            std::cout << "    mem    = " << format_mem( B->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;

            auto  diff = matrix::sum( value_t(1), *AxA, value_t(-1), *B );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

            if ( hpro::verbose( 3 ) )
            {
                io::eps::print( *B, "B3", prnopt );
                io::eps::print( *rowcb2, "rowcb3", "mem" );
                io::eps::print( *colcb2, "colcb3", "mem" );
            }// if

            runtime.clear();
        }// if
    }// else
}
