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

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

namespace hlr { namespace uniform { namespace accu {
double  t_basis       = 0.0;
double  t_apply       = 0.0;
double  t_apply_uni   = 0.0;
double  t_apply_dense = 0.0;
double  t_eval        = 0.0;
double  t_eval_uni    = 0.0;
double  t_eval_rest   = 0.0;
double  t_eval_rec    = 0.0;
double  t_add_accu    = 0.0;
size_t  n_inner       = 0;
size_t  n_prodA       = 0;
size_t  n_prodB       = 0;
}}}

namespace hlr { namespace uniform { namespace accu3 {
double  t_basis      = 0.0;
double  t_apply      = 0.0;
double  t_eval       = 0.0;
double  t_eval_uni   = 0.0;
double  t_eval_rest  = 0.0;
double  t_eval_rec   = 0.0;
}}}

namespace hlr { namespace seq { namespace accu {
double  t_apply = 0.0;
double  t_eval  = 0.0;
}}}

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
    auto  tic     = timer::now();
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
    auto  A      = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    // auto  A      = io::hpro::read( "A.hm" );
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

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "uniform matrix" << term::reset << std::endl;

    std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
    tic = timer::now();
    
    auto  [ rowcb, colcb ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *A, acc );

    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( rowcb->byte_size(), colcb->byte_size() ) << std::endl;


    std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

    tic = timer::now();
    
    auto  A2 = impl::matrix::copy_uniform< value_t >( *A, *rowcb, *colcb );
    
    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;

    {
        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = hlr::seq::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error ) << std::endl;
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

    auto  AxA      = hpro::matrix_product( A.get(), A.get() );
    auto  norm_AxA = hlr::norm::spectral( *AxA );
    
    if ( false )
    {
        std::cout << "  " << term::bullet << "eager" << term::reset << std::endl;
            
        auto  apx = approx::SVD< value_t >();
        auto  B   = impl::matrix::copy( *A );

        B->scale( 0 );

        tic = timer::now();
                
        impl::multiply( value_t(1), apply_normal, *A, apply_normal, *A, *B, acc, apx );
            
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

        auto  diff = matrix::sum( 1.0, *AxA, -1.0, *B );

        std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

        {
            std::cout << "    " << term::bullet << term::bold << "testing original bases" << term::reset << std::endl;

            auto  B2    = impl::matrix::copy_uniform< value_t >( *B, *rowcb, *colcb );
            auto  diff2 = matrix::sum( 1.0, *AxA, -1.0, *B2 );
                    
            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff2 ) / norm_AxA ) << std::endl;
        }
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << "accumulator" << term::reset << std::endl;
            
        auto  apx = approx::SVD< value_t >();
        auto  B   = impl::matrix::copy( *A );

        B->scale( 0 );

        tic = timer::now();
                
        impl::accu::multiply( value_t(1), apply_normal, *A, apply_normal, *A, *B, acc, apx );
            
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

        auto  diff = matrix::sum( 1.0, *AxA, -1.0, *B );

        std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

        {
            std::cout << "    " << term::bullet << term::bold << "testing original bases" << term::reset << std::endl;

            auto  B2    = impl::matrix::copy_uniform< value_t >( *B, *rowcb, *colcb );
            auto  diff2 = matrix::sum( 1.0, *AxA, -1.0, *B2 );
                    
            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff2 ) / norm_AxA ) << std::endl;
        }

        io::eps::print( *B, "B", prnopt );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // Uniform H matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "Uniform H-Multiplication" << term::reset << std::endl;
    
    auto  apx = approx::SVD< value_t >();
    
    if ( cmdline::cluster == "tlr" )
    {
        if ( true )
        {
            std::cout << "  " << term::bullet << "eager" << term::reset << std::endl;
            
            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );

            tic = timer::now();
                
            impl::uniform::tlr::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, apx );
            
            toc = timer::since( tic );
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
        }// if
    }// if
    else
    {
        if ( false )
        {
            std::cout << "  " << term::bullet << "eager" << term::reset << std::endl;

            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );
            
            tic = timer::now();
                
            impl::uniform::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, apx );
            
            toc = timer::since( tic );
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
        }// if

        if ( true )
        {
            std::cout << "  " << term::bullet << "accumulator" << term::reset << std::endl;

            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );
            
            tic = timer::now();
                
            impl::uniform::accu::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, apx );
            
            toc = timer::since( tic );
            std::cout << "    done in   " << format_time( toc ) << std::endl;
            std::cout << "      basis   " << format_time( hlr::uniform::accu::t_basis ) << std::endl;
            std::cout << "      apply   " << format_time( hlr::uniform::accu::t_apply ) << std::endl;
            std::cout << "        uni   " << format_time( hlr::uniform::accu::t_apply_uni ) << std::endl;
            std::cout << "        dense " << format_time( hlr::uniform::accu::t_apply_dense ) << std::endl;
            std::cout << "      eval    " << format_time( hlr::uniform::accu::t_eval ) << std::endl;
            std::cout << "        uni   " << format_time( hlr::uniform::accu::t_eval_uni ) << std::endl;
            std::cout << "        rest  " << format_time( hlr::uniform::accu::t_eval_rest ) << std::endl;
            std::cout << "        rec   " << format_time( hlr::uniform::accu::t_eval_rec ) << std::endl;
            std::cout << "        add   " << format_time( hlr::uniform::accu::t_add_accu ) << std::endl;
            std::cout << "    mem    =  " << format_mem( B->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;

            hlr::uniform::accu::t_basis = 0;
            hlr::uniform::accu::t_apply = 0;
            hlr::uniform::accu::t_apply_uni = 0;
            hlr::uniform::accu::t_apply_dense = 0;
            hlr::uniform::accu::t_eval = 0;
            hlr::uniform::accu::t_eval_uni = 0;
            hlr::uniform::accu::t_eval_rest = 0;
            hlr::uniform::accu::t_eval_rec = 0;
            hlr::uniform::accu::t_add_accu = 0;
            
            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

            if ( hpro::verbose( 3 ) )
            {
                io::eps::print( *rowcb2, "rowcb", "mem" );
                io::eps::print( *colcb2, "colcb", "mem" );
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
            
            tic = timer::now();
                
            impl::uniform::accu::multiply_cached( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, apx );
            
            toc = timer::since( tic );
            std::cout << "    done in   " << format_time( toc ) << std::endl;
            std::cout << "      basis   " << format_time( hlr::uniform::accu::t_basis ) << std::endl;
            std::cout << "      apply   " << format_time( hlr::uniform::accu::t_apply ) << std::endl;
            std::cout << "        uni   " << format_time( hlr::uniform::accu::t_apply_uni ) << std::endl;
            std::cout << "        dense " << format_time( hlr::uniform::accu::t_apply_dense ) << std::endl;
            std::cout << "      eval    " << format_time( hlr::uniform::accu::t_eval ) << std::endl;
            std::cout << "        uni   " << format_time( hlr::uniform::accu::t_eval_uni ) << std::endl;
            std::cout << "        rest  " << format_time( hlr::uniform::accu::t_eval_rest ) << std::endl;
            std::cout << "        rec   " << format_time( hlr::uniform::accu::t_eval_rec ) << std::endl;
            std::cout << "    hits      " << std::endl;
            std::cout << "      inner   " << hlr::uniform::accu::n_inner << std::endl;
            std::cout << "      prodA   " << hlr::uniform::accu::n_prodA << std::endl;
            std::cout << "      prodB   " << hlr::uniform::accu::n_prodB << std::endl;
            std::cout << "    mem    =  " << format_mem( B->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;

            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "    error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;

            io::eps::print( *B, "B2", prnopt );
        }// if
    }// else
}
