//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/approx/accuracy.hh"
#include <hlr/bem/hca.hh>
#include "hlr/bem/aca.hh"
#include "hlr/bem/dense.hh"

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

    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    auto  runtime = std::vector< double >();

    blas::reset_flops();
    
    auto    acc   = gen_accuracy();
    size_t  mem_A = 0;

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        
    tic = timer::now();
        
    if ( cmdline::capprox == "hca" )
    {
        if constexpr ( problem_t::supports_hca )
        {
            std::cout << "    using HCA"
                      << " (" << hlr::compress::provider << " + " << hlr::compress::valr::provider << ")"
                      << std::endl;
                
            auto  hcagen = problem->hca_gen_func( *ct );
            auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
            auto  hcalr  = bem::hca_lrapx( hca );
                
            mem_A = impl::matrix::mem_mixedprec( bct->root(), pcoeff, hcalr, acc, nseq );
        }// if
        else
            cmdline::capprox = "default";
    }// if

    if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
    {
        std::cout << "    using ACA" 
                  << " (" << hlr::compress::provider << " + " << hlr::compress::valr::provider << ")"
                  << std::endl;

        auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );

        mem_A = impl::matrix::mem_mixedprec( bct->root(), pcoeff, acalr, acc, nseq );
    }// else
        
    if ( cmdline::capprox == "dense" )
    {
        std::cout << "    using dense"
                  << " (" << hlr::compress::provider << " + " << hlr::compress::valr::provider << ")"
                  << std::endl;

        auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
        mem_A = impl::matrix::mem_mixedprec( bct->root(), pcoeff, dense, acc, nseq );
    }// else
        
    toc = timer::since( tic );

    const auto  nrows = ct->root()->size();
    
    std::cout << "    dims  = " << nrows << " × " << nrows << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / double(nrows) ) << std::endl;
}
