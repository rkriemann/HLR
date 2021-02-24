//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/TMatrixProduct.hh>
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>
#include <hlr/bem/aca.hh>

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
    auto  tic     = timer::now();
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis  bc_vis;
        
        print_ps( ct->root(), "ct" );
        bc_vis.id( false ).print( bct->root(), "bct" );
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
        matrix::print_eps( *A, "A" );
        matrix::print_lvl_eps( *A, "L" );
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
        auto  diff  = hpro::matrix_sum( value_t(1), A.get(), value_t(-1), A2.get() );
        auto  error = hlr::seq::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error ) << std::endl;
    }
    
    if ( hpro::verbose( 3 ) )
        matrix::print_eps( *A2, "A2" );

    //////////////////////////////////////////////////////////////////////
    //
    // matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << "  " << term::bullet << term::bold << "Multiplication" << term::reset << std::endl;

    auto  AxA      = hpro::matrix_product( A.get(), A.get() );
    auto  norm_AxA = hlr::norm::spectral( *AxA );
    
    if ( true )
    {
        std::cout << "    " << term::bullet << "H" << term::reset << std::endl;
            
        auto  apx = approx::SVD< value_t >();
        auto  B   = impl::matrix::copy( *A );

        B->scale( 0 );

        tic = timer::now();
                
        impl::multiply( value_t(1), apply_normal, *A, apply_normal, *A, *B, acc, apx );
            
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( B->byte_size() ) << std::endl;

        auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

        std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
    }// if

    if ( cmdline::cluster == "tlr" )
    {
        if ( true )
        {
            std::cout << "    " << term::bullet << "H²" << term::reset << std::endl;
            
            auto  B     = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );

            tic = timer::now();
                
            impl::uniform::tlr::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc );
            
            toc = timer::since( tic );
            std::cout << "      done in  " << format_time( toc ) << std::endl;
            std::cout << "      mem    = " << format_mem( B->byte_size() ) << std::endl;

            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
        }// if
    }// if
    else
    {
        if ( true )
        {
            std::cout << "    " << term::bullet << "H²" << term::reset << std::endl;

            auto  apx    = approx::SVD< value_t >();
            auto  B      = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();

            matrix::replace_cluster_basis( *B, *rowcb2, *colcb2 );
            
            B->scale( 0 );
            
            tic = timer::now();
                
            impl::uniform::multiply( value_t(1), apply_normal, *A2, apply_normal, *A2, *B, acc, apx );
            
            toc = timer::since( tic );
            std::cout << "      done in  " << format_time( toc ) << std::endl;
            std::cout << "      mem    = " << format_mem( B->byte_size() ) << std::endl;

            auto  diff = hpro::matrix_sum( hpro::real(1.0), AxA.get(), hpro::real(-1.0), B.get() );

            std::cout << "      error  = " << format_error( hlr::norm::spectral( *diff ) / norm_AxA ) << std::endl;
        }// if
    }// else
}
