//
// Project     : HLR
// Program     : compress-lu
// Description : LU factorization with compressed matrix blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/arith/norm.hh>
#include <hlr/bem/aca.hh>
#include "hlr/dag/lu.hh"
#include <hlr/matrix/luinv_eval.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

using indexset = Hpro::TIndexSet;

struct local_accuracy : public Hpro::TTruncAcc
{
    local_accuracy ( const double  abs_eps )
            : Hpro::TTruncAcc( 0.0, abs_eps )
    {}
    
    virtual const TTruncAcc  acc ( const indexset &  rowis,
                                   const indexset &  colis ) const
    {
        return Hpro::absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
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
        matrix::print_eps( *A, "A", "noid,norank,nosize" );

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    //////////////////////////////////////////////////////////////////////
    //
    // further compress matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto  zA     = impl::matrix::copy_compressible( *A );
    auto  norm_A = norm::spectral( impl::arithmetic, *A );
    
    std::cout << "  " << term::bullet << term::bold << "compression via "
              << hlr::compress::provider
              << ", ε = " << boost::format( "%.2e" ) % cmdline::eps << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    tic = timer::now();

    impl::matrix::compress( *zA, Hpro::fixed_prec( acc.rel_eps() ) );

    toc = timer::since( tic );
    runtime.push_back( toc.seconds() );
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_zA = zA->byte_size();
    
    std::cout << "    mem   = " << format_mem( zA->byte_size() ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );
    
    auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    auto  error = norm::spectral( impl::arithmetic, *diff );
    
    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // uncompressed H-LU
    //
    //////////////////////////////////////////////////////////////////////

    auto  apx = approx::SVD< value_t >();

    std::cout << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
        
    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator)" << term::reset << std::endl;

        auto  LU = seq::matrix::copy( *A );
                
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
        
        std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
    }

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "uncompressed (accumulator, DAG)" << term::reset << std::endl;

        auto  LU                = seq::matrix::copy( *A );
        auto  [ dag, accu_map ] = hlr::dag::gen_dag_lu_accu( *LU, nseq, impl::dag::refine, apx );

        std::cout << accu_map->contains( 0 ) << std::endl;
        
        dag.print_dot( "LU.dot" );

        for ( const auto &  [ key, value ] : *accu_map )
            std::cout << key << std::endl;
        
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
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "HLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
    }
    
    {
        std::cout << "  " << term::bullet << term::bold << "compressed (accumulator)" << term::reset << std::endl;

        auto  LU = seq::matrix::copy( *zA );
                
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
        
        std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *LU, "zHLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
    }
}
    
