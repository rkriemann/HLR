//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include "hlr/arith/norm.hh"
#include "hlr/approx/accuracy.hh"
#include "hlr/bem/aca.hh"
#include <hlr/dag/lu.hh>
#include <hlr/matrix/print.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/matrix/info.hh>
#include <hlr/utils/io.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

struct local_accuracy : public hpro::TTruncAcc
{
    local_accuracy ( const double  abs_eps )
            : hpro::TTruncAcc( 0.0, abs_eps )
    {}
    
    virtual const TTruncAcc  acc ( const indexset &  rowis,
                                   const indexset &  colis ) const
    {
        return hpro::absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
    }
};

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
    
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< Hpro::TMatrix< value_t > >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();

        if ( verbose( 3 ) )
            io::vtk::print( *coord, "coord" );

        auto  ct      = gen_ct( *coord );
        auto  bt      = gen_bct( *ct, *ct );

        if ( verbose( 3 ) )
            io::eps::print( *bt->root(), "bt" );

        auto  coeff   = problem->coeff_func();
        auto  pcoeff  = std::make_unique< Hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx   = std::make_unique< bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > > >( *pcoeff );
        
        tic = timer::now();
        A   = impl::matrix::build( bt->root(), *pcoeff, *lrapx, acc, nseq );
        toc = timer::since( tic );

        // if ( verbose( 2 ) )
        //     io::hpro::write< value_t >( *A, "A.hm" );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        // A = io::hpro::read< value_t >( matrixfile );
    }// else
    
    const auto  mem_A  = matrix::data_byte_size( *A );
    const auto  norm_A = impl::norm::frobenius( *A );
    // const auto  delta  = norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );
    const auto  delta  = cmdline::eps;
        
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( impl::norm::frobenius( *A ) ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    //////////////////////////////////////////////////////////////////////
    //
    // convert to mixed precision format
    //
    //////////////////////////////////////////////////////////////////////

    const auto  compstr = std::string( hlr::compress::provider ) + " + " + hlr::compress::aplr::provider;
    auto        zA      = impl::matrix::copy_mixedprec( *A );

    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", " << compstr << ")"
              << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    // auto  lacc = local_accuracy( delta );
    // auto  lacc = relative_prec( delta );
    auto  lacc = relative_prec( Hpro::frobenius_norm, delta );
        
    impl::matrix::compress( *zA, lacc );

    const auto  mem_zA = matrix::data_byte_size( *zA );
    
    std::cout << "    mem   = " << format_mem( mem_zA ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );
    
    auto  error  = impl::norm::frobenius( 1, *A, -1, *zA );

    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-LU factorization
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
    
    auto  apx     = approx::SVD< value_t >();
    auto  mem_LU  = std::unordered_map< std::string, size_t >();
    auto  time_LU = std::unordered_map< std::string, double >();
    auto  luacc   = relative_prec( aeps );

    // define methods to execute
    const auto methods = cmdline::arith;
    // const auto methods = std::set< std::string >{
    //     // "H+rec",
    //     // "H+rec+accu",
    //     // "H+dag",
    //     "H+dag+accu+lazy",
    //     // "H+dag+accu+eager",
    //     // "zH+rec",
    //     // "zH+rec+accu",
    //     // "zH+dag",
    //     "zH+dag+accu+lazy"
    // };
    
    if ( contains( methods, "help" ) )
    {
        std::cout << "H-LU versions: H+rec, H+rec+accu, H+dag, H+dag+accu+lazy, H+dag+accu+eager" << std::endl;
        return;
    }// if
            
    if ( contains( methods, "H+rec" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "uncompressed (recursive)" << term::reset << std::endl;

        auto  LU = impl::matrix::copy( *A );
                
        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
                
            impl::lu< value_t >( *LU, luacc, apx );
                
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
        
        mem_LU["H+rec"]  = matrix::data_byte_size( *LU );
        time_LU["H+rec"] = min( runtime );
        std::cout << "    mem    = " << format_mem( mem_LU["H+rec"] ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU", prnopt );

        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if

    if ( contains( methods, "H+rec+accu" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator)" << term::reset << std::endl;

        auto  LU = impl::matrix::copy( *A );
                
        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
                
            impl::accu::lu< value_t >( *LU, luacc, apx );
                
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
        
        mem_LU["H+rec+accu"]  = matrix::data_byte_size( *LU );
        time_LU["H+rec+accu"] = min(runtime);
        std::cout << "    mem    = " << format_mem( mem_LU["H+rec+accu"] ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if

    if ( contains( methods, "H+dag" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "uncompressed (DAG)" << term::reset << std::endl;

        auto  LU  = impl::matrix::copy( *A );
        auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

        // io::dot::print( dag, "LU.dot" );

        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();

            impl::dag::run( dag, luacc );
                
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

        mem_LU["H+dag"]  = matrix::data_byte_size( *LU );
        time_LU["H+dag"] = min(runtime);
        std::cout << "    mem    = " << format_mem( mem_LU["H+dag"] ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU2", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if
    
    if ( contains( methods, "H+dag+accu+lazy" ) )
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

            impl::dag::run( dag, luacc );
                
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
        
        mem_LU["H+dag+accy+lazy"]  = matrix::data_byte_size( *LU );
        time_LU["H+dag+accy+lazy"] = min(runtime);
        std::cout << "    mem    = " << format_mem( mem_LU["H+dag+accy+lazy"] ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU2", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }
    
    if ( contains( methods, "zH+rec" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "compressed (recursive, " << compstr << ")" << term::reset << std::endl;

        auto  LU = impl::matrix::copy( *zA );
                
        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
                
            impl::lu< value_t >( *LU, luacc, apx );
                
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
        
        const auto  mem_zLU = matrix::data_byte_size( *LU );

        std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

        if ( mem_LU.contains( "H+rec" ) )
            std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+rec"]) ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "zHLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        impl::matrix::decompress( *LU );
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if

    if ( contains( methods, "zH+rec+accu" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "compressed (accumulator, " << compstr << ")" << term::reset << std::endl;

        auto  LU = impl::matrix::copy( *zA );
                
        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
                
            impl::accu::lu< value_t >( *LU, luacc, apx );
                
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
        
        const auto  mem_zLU = matrix::data_byte_size( *LU );

        std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

        if ( mem_LU.contains( "H+rec+accu" ) )
            std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+rec+accu"]) ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "zHLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        impl::matrix::decompress( *LU );
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if

    if ( contains( methods, "zH+dag" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "compressed (DAG, " << compstr << ")" << term::reset << std::endl;

        auto  LU  = impl::matrix::copy( *zA );
        auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

        // io::dot::print( dag, "zLU.dot" );

        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();

            impl::dag::run( dag, luacc );
                
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
        
        const auto  mem_zLU = matrix::data_byte_size( *LU );

        std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

        if ( mem_LU.contains( "H+dag" ) )
            std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+dag"]) ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "zHLU2", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        impl::matrix::decompress( *LU );
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }// if

    if ( contains( methods, "zH+dag+accu+lazy" ) )
    {
        std::cout << "  " << term::bullet << term::bold << "compressed (accumulator, DAG, lazy, " << hlr::compress::provider << ")" << term::reset << std::endl;

        auto  LU = impl::matrix::copy( *zA );

        runtime.clear();

        for ( int i = 0; i < nbench; ++i )
        {
            // regenerate DAG to start with new accumulators
            auto  [ dag, amap, amtx ] = hlr::dag::gen_dag_lu_accu_lazy( *LU, nseq, impl::dag::refine, apx );
            
            tic = timer::now();

            impl::dag::run( dag, luacc );
                
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
        
        const auto  mem_zLU = matrix::data_byte_size( *LU );

        std::cout << "    mem    = " << format_mem( mem_zLU ) << std::endl;

        if ( mem_LU.contains( "H+dag+accu+lazy" ) )
            std::cout << "      vs H   " << boost::format( "%.3f" ) % ( double(mem_zLU) / double(mem_LU["H+dag+accu+lazy"]) ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "zHLU2", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        impl::matrix::decompress( *LU );
        std::cout << "    error  = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
    }
}
