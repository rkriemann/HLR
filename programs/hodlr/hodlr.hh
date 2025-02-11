//
// Project     : HLR
// Module      : hodlr.hh
// Description : generic code for HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>
#include <hlr/cluster/hodlr.hh>
#include <hlr/bem/aca.hh>
#include <hlr/matrix/luinv_eval.hh>

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
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  part    = gen_part_strat();
    auto  ct      = cluster::hodlr::cluster( *coord, *part, ntile );
    auto  bc      = cluster::hodlr::blockcluster( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
        io::eps::print( * bc->root(), "bt" );
    
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx   = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
    auto  acc     = gen_accuracy();
    auto  A       = impl::matrix::build( bc->root(), pcoeff, lrapx, acc );
    auto  toc     = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );
    
    {
        std::cout << term::bullet << term::bold << "LU ( HODLR )" << term::reset << std::endl;

        auto  apx = approx::SVD< value_t >();
        auto  LU  = impl::matrix::copy( *A );

        std::vector< double >  runtime;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            impl::hodlr::lu< value_t >( *LU, acc, apx );
            
            toc = timer::since( tic );

            std::cout << "    done in " << format_time( toc ) << std::endl;
            
            runtime.push_back( toc.seconds() );

            if ( i < nbench-1 )
                impl::matrix::copy_to( *A, *LU );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        auto  A_inv = matrix::luinv_eval( *LU );
        
        std::cout << "    mem   = " << format_mem( LU->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }
}
