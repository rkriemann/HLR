//
// Project     : HLR
// Program     : single-lu
// Description : LU factorization with single precision
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/arith/norm.hh>
#include <hlr/bem/aca.hh>
#include "hlr/dag/lu.hh"
#include <hlr/matrix/luinv_eval.hh>

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

    auto  prnopt  = "noinnerid";
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    auto  runtime = std::vector< double >();

    blas::reset_flops();
    
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = std::make_unique< Hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx   = std::make_unique< bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > > >( *pcoeff );
    auto  A       = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_A = A->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *A, "A", "norank,nosize" );

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-LU
    //
    //////////////////////////////////////////////////////////////////////

    auto  comp_lu =
        [&] ( const auto  apx, const std::string &  apxname )
        {
            size_t  mem_LU = 0;

            // define methods to execute
            const auto methods = std::set< std::string >{
                // "H+rec",
                // "H+rec+accu",
                "H+dag",
                "H+dag+accu+lazy",
                // "H+dag+accu+eager"
            };

            std::cout << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
        
            if ( methods.contains( "H+rec" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (recursive)" << term::reset << std::endl;

                auto  B  = impl::matrix::convert< float >( *A );
                auto  LU = impl::matrix::copy( *B );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::lu( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *B, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU", prnopt );

                auto  LU2   = impl::matrix::convert< value_t >( *LU );
                auto  A_inv = matrix::luinv_eval( *LU2 );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+rec+accu" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator)" << term::reset << std::endl;

                auto  B  = impl::matrix::convert< float >( *A );
                auto  LU = impl::matrix::copy( *B );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::accu::lu( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *B, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU", prnopt );
                
                auto  LU2   = impl::matrix::convert< value_t >( *LU );
                auto  A_inv = matrix::luinv_eval( *LU2 );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (DAG)" << term::reset << std::endl;

                auto  B   = impl::matrix::convert< float >( *A );
                auto  LU  = impl::matrix::copy( *B );
                auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

                // io::dot::print( dag, "LU.dot" );

                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();

                    impl::dag::run( dag, acc );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *B, *LU );
                }// for

                // io::hpro::write( *LU, "LU.hm" );
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;

                mem_LU = LU->byte_size();
        
                std::cout << "    mem    = " << format_mem( mem_LU ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  LU2   = impl::matrix::convert< value_t >( *LU );
                auto  A_inv = matrix::luinv_eval( *LU2 );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag+accu+lazy" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator, DAG, lazy)" << term::reset << std::endl;

                auto  B  = impl::matrix::convert< float >( *A );
                auto  LU = impl::matrix::copy( *B );

                // io::dot::print( dag, "LUa.dot" );

                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    // regenerate DAG to start with new accumulators
                    auto  [ dag, amap, amtx ] = hlr::dag::gen_dag_lu_accu_lazy( *LU, nseq, impl::dag::refine, apx );
            
                    tic = timer::now();

                    impl::dag::run( dag, acc );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *B, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  LU2   = impl::matrix::convert< value_t >( *LU );
                auto  A_inv = matrix::luinv_eval( *LU2 );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag+accu+eager" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator, DAG, eager)" << term::reset << std::endl;

                auto  B  = impl::matrix::convert< float >( *A );
                auto  LU = impl::matrix::copy( *B );

                // io::dot::print( dag, "LUa.dot" );

                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    // regenerate DAG to start with new accumulators
                    auto  [ dag, amap, amtx ] = hlr::dag::gen_dag_lu_accu_eager( *LU, nseq, impl::dag::refine, apx );
            
                    tic = timer::now();

                    impl::dag::run( dag, acc );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *B, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  LU2   = impl::matrix::convert< value_t >( *LU );
                auto  A_inv = matrix::luinv_eval( *LU2 );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }
        };

    if      ( cmdline::approx == "svd"     ) comp_lu( approx::SVD< float >(),  "SVD" );
    else if ( cmdline::approx == "rrqr"    ) comp_lu( approx::RRQR< float >(), "RRQR" );
    else if ( cmdline::approx == "randsvd" ) comp_lu( approx::RandSVD< float >(), "RandSVD" );
}
    
