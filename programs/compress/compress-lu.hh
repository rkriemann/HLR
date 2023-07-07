//
// Project     : HLR
// Program     : compress-lu
// Description : LU factorization with compressed matrix blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/arith/norm.hh>
#include <hlr/bem/aca.hh>
#include <hlr/dag/lu.hh>
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

    auto  prnopt  = "";
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
    auto  A       = std::unique_ptr< Hpro::TMatrix< value_t > >();

    if ( matrixfile == "" )
    {
        tic = timer::now();
        A   = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
        toc = timer::since( tic );

        // io::hpro::write< value_t >( *A, "A.hm" );
    }// if
    else
    {
        A = io::hpro::read< value_t >( matrixfile );
    }// else
    
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_A = A->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *A, "A", prnopt );

    // {
    //     auto  D = coeff->build( A->row_is(), A->col_is() );

    //     io::matlab::write( *D, "D" );
    // }
    
    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    //////////////////////////////////////////////////////////////////////
    //
    // further compress matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto        zA     = impl::matrix::copy_compressible( *A );
    auto        norm_A = impl::norm::frobenius( *A );
    const auto  delta  = cmdline::eps; // norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );
    // auto        lacc   = local_accuracy( delta );
    auto        lacc   = absolute_prec( delta );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", "
              << hlr::compress::provider << ')'
              << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    tic = timer::now();

    impl::matrix::compress( *zA, lacc );

    toc = timer::since( tic );
    runtime.push_back( toc.seconds() );
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_zA = zA->byte_size();
    
    std::cout << "    mem   = " << format_mem( zA->byte_size() ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *zA, "zA", prnopt );
    
    auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    auto  error = norm::spectral( impl::arithmetic, *diff );
    
    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // H-LU
    //
    //////////////////////////////////////////////////////////////////////

    auto  comp_lu =
        [&] ( const auto  apx, const std::string &  apxname )
        {
            auto  mem_LU  = std::unordered_map< std::string, size_t >();
            auto  time_LU = std::unordered_map< std::string, double >();

            // define methods to execute
            const auto methods = std::set< std::string >{
                "H+rec",
                "H+rec+accu",
                "H+dag",
                // "H+dag+accu+lazy",
                // "H+dag+accu+eager",
                "zH+rec",
                "zH+rec+accu",
                "zH+dag",
                // "zH+dag+accu+lazy"
            };

            std::cout << term::bullet << term::bold << "H-LU (" << apxname << ")" << term::reset << std::endl;
        
            if ( methods.contains( "H+rec" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (recursive)" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *A );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::lu< value_t >( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *A, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                mem_LU["H+rec"]  = LU->byte_size();
                time_LU["H+rec"] = min(runtime);
                std::cout << "    mem    = " << format_mem( mem_LU["H+rec"] ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU", prnopt );

                auto  A_inv = matrix::luinv_eval( *LU );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+rec+accu" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator)" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *A );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::accu::lu< value_t >( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *A, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                mem_LU["H+rec+accu"]  = LU->byte_size();
                time_LU["H+rec+accu"] = min(runtime);
                std::cout << "    mem    = " << format_mem( mem_LU["H+rec+accu"] ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (DAG)" << term::reset << std::endl;

                auto  LU  = impl::matrix::copy( *A );
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
                        impl::matrix::copy_to( *A, *LU );
                }// for

                // io::hpro::write( *LU, "LU.hm" );
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;

                mem_LU["H+dag"]  = LU->byte_size();
                time_LU["H+dag"] = min(runtime);
                std::cout << "    mem    = " << format_mem( mem_LU["H+dag"] ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag+accu+lazy" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator, DAG, lazy)" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *A );

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
                        impl::matrix::copy_to( *A, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                mem_LU["H+dag+accy+lazy"]  = LU->byte_size();
                time_LU["H+dag+accy+lazy"] = min(runtime);
                std::cout << "    mem    = " << format_mem( mem_LU["H+dag+accy+lazy"] ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "H+dag+accu+eager" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator, DAG, eager)" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *A );

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
                        impl::matrix::copy_to( *A, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                mem_LU["H+dag+accy+eager"]  = LU->byte_size();
                time_LU["H+dag+accy+eager"] = min(runtime);
                std::cout << "    mem    = " << format_mem( mem_LU["H+dag+accy+eager"] ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "HLU2", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "zH+rec" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "compressed (recursive, " << hlr::compress::provider << ")" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *zA );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::lu< value_t >( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *zA, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                if ( time_LU.contains( "H+rec" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(min(runtime)) / double(time_LU["H+rec"]) ) << std::endl;
        
                const auto  mem_zLU = LU->byte_size();

                std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

                if ( mem_LU.contains( "H+rec" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+rec"]) ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "zHLU", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                impl::matrix::decompress( *LU );
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "zH+rec+accu" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "compressed (accumulator, " << hlr::compress::provider << ")" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *zA );
                
                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();
                
                    impl::accu::lu< value_t >( *LU, acc, apx );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *zA, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                if ( time_LU.contains( "H+rec+accu" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(min(runtime)) / double(time_LU["H+rec+accu"]) ) << std::endl;
        
                const auto  mem_zLU = LU->byte_size();

                std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

                if ( mem_LU.contains( "H+rec+accu" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+rec+accu"]) ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "zHLU", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                impl::matrix::decompress( *LU );
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "zH+dag" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "compressed (DAG, " << hlr::compress::provider << ")" << term::reset << std::endl;

                auto  LU  = impl::matrix::copy( *zA );
                auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

                // io::dot::print( dag, "zLU.dot" );

                runtime.clear();

                for ( int i = 0; i < nbench; ++i )
                {
                    tic = timer::now();

                    impl::dag::run( dag, acc );
                
                    toc = timer::since( tic );
                    runtime.push_back( toc.seconds() );
            
                    std::cout << "    done in  " << format_time( toc ) << std::endl;

                    if ( i < nbench-1 )
                        impl::matrix::copy_to( *zA, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;

                if ( time_LU.contains( "H+dag" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(min(runtime)) / double(time_LU["H+dag"]) ) << std::endl;
        
                const auto  mem_zLU = LU->byte_size();

                std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

                if ( mem_LU.contains( "H+dag" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+dag"]) ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "zHLU2", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                impl::matrix::decompress( *LU );
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }

            if ( methods.contains( "zH+dag+accu+lazy" ) )
            {
                std::cout << "  " << term::bullet << term::bold << "compressed (accumulator, DAG, lazy, " << hlr::compress::provider << ")" << term::reset << std::endl;

                auto  LU = impl::matrix::copy( *zA );

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
                        impl::matrix::copy_to( *zA, *LU );
                }// for
        
                if ( nbench > 1 )
                    std::cout << "  runtime  = "
                              << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                              << std::endl;
        
                if ( time_LU.contains( "H+dag+accu+lazy" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(min(runtime)) / double(time_LU["H+dag+accu+lazy"]) ) << std::endl;
        
                const auto  mem_zLU = LU->byte_size();

                std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

                if ( mem_LU.contains( "H+dag+accu+lazy" ) )
                    std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+dag+accu+lazy"]) ) << std::endl;

                if ( hpro::verbose( 3 ) )
                    io::eps::print( *LU, "zHLU2", prnopt );
                
                auto  A_inv = matrix::luinv_eval( *LU );
                    
                impl::matrix::decompress( *LU );
                std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
            }
        };

    if      ( cmdline::approx == "default" ) comp_lu( approx::SVD< value_t >(),  "SVD" );
    else if ( cmdline::approx == "svd"     ) comp_lu( approx::SVD< value_t >(),  "SVD" );
    // else if ( cmdline::approx == "rrqr"    ) comp_lu( approx::RRQR< value_t >(), "RRQR" );
    // else if ( cmdline::approx == "randsvd" ) comp_lu( approx::RandSVD< value_t >(), "RandSVD" );
}
    
