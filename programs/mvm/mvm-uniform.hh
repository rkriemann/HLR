//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <fstream>
#include <limits>

#include <hlr/bem/hca.hh>

#include "hlr/arith/norm.hh"
#include "hlr/approx/accuracy.hh"
#include "hlr/bem/aca.hh"
#include "hlr/bem/dense.hh"
#include <hlr/matrix/print.hh>
#include <hlr/utils/io.hh>

#include <hlr/utils/eps_printer.hh>

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
    
    auto  acc   = gen_accuracy();
    auto  A     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    auto  rowcb = std::unique_ptr< matrix::shared_cluster_basis< value_t > >();
    auto  colcb = std::unique_ptr< matrix::shared_cluster_basis< value_t > >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  cbapx   = approx::SVD< value_t >();
        
    tic = timer::now();
        
    if ( cmdline::capprox == "hca" )
    {
        if constexpr ( problem_t::supports_hca )
        {
            std::cout << "    using HCA"
                      << " (" << hlr::compress::provider << " + " << hlr::compress::aplr::provider << ")"
                      << std::endl;
                
            auto  hcagen = problem->hca_gen_func( *ct );
            auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
            auto  hcalr  = bem::hca_lrapx( hca );

            if ( sep_coup )
                std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl_sep( bct->root(), pcoeff, hcalr, cbapx, acc, cmdline::compress );
            else
                std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl( bct->root(), pcoeff, hcalr, cbapx, acc, cmdline::compress );
        }// if
        else
            cmdline::capprox = "default";
    }// if

    if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
    {
        std::cout << "    using ACA" 
                  << " (" << hlr::compress::provider << " + " << hlr::compress::aplr::provider << ")"
                  << std::endl;

        auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );

        if ( sep_coup )
            std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl_sep( bct->root(), pcoeff, acalr, cbapx, acc, cmdline::compress );
        else
            std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl( bct->root(), pcoeff, acalr, cbapx, acc, cmdline::compress );
    }// else
        
    if ( cmdline::capprox == "dense" )
    {
        std::cout << "    using dense"
                  << " (" << hlr::compress::provider << " + " << hlr::compress::aplr::provider << ")"
                  << std::endl;

        auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
        if ( sep_coup )
            std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl_sep( bct->root(), pcoeff, dense, cbapx, acc, cmdline::compress );
        else
            std::tie( rowcb, colcb, A ) = impl::matrix::build_uniform_lvl( bct->root(), pcoeff, dense, cbapx, acc, cmdline::compress );
    }// else
        
    toc = timer::since( tic );
    
    const auto  mem_A  = A->byte_size();
    const auto  norm_A = impl::norm::frobenius( *A );
        
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( rowcb->byte_size(), colcb->byte_size(), mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( norm_A ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    //////////////////////////////////////////////////////////////////////
    //
    // matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////

    if ( nbench > 0 )
    {
        std::cout << term::bullet << term::bold
                  << "mat-vec"
                  << term::reset << std::endl;

        const uint  nmvm     = 50;
        const auto  flops_uh = nmvm * hlr::uniform::mul_vec_flops( apply_normal, *A, *rowcb, *colcb );
        const auto  bytes_uh = nmvm * hlr::uniform::mul_vec_datasize( apply_normal, *A, *rowcb, *colcb );
    
        std::cout << "  " << term::bullet << term::bold << "FLOPs/byte " << term::reset() << std::endl;
        std::cout << "    Uni-H  = " << format_flops( flops_uh ) << ", " << flops_uh / bytes_uh << std::endl;

        double  t_ref = 0.0;
        auto    y_ref = std::unique_ptr< vector::scalar_vector< value_t > >();

        {
            // generate reference solution
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );

            y_ref = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            x->fill( 1 );
            impl::uniform::mul_vec< value_t >( nmvm * 2.0, Hpro::apply_normal, *A, *x, *y_ref, *rowcb, *colcb );
        }
        
        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "mutex" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::uniform::mul_vec_mtx< value_t >( 2.0, Hpro::apply_normal, *A, *x, *y, *rowcb, *colcb );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;

            std::cout << "      flops   = " << format_flops( flops_uh, min( runtime ) ) << std::endl;
            
            t_ref = min( runtime );
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "row wise" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::uniform::mul_vec_row< value_t >( 2.0, Hpro::apply_normal, *A, *x, *y, *rowcb, *colcb );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;

            std::cout << "      flops   = " << format_flops( flops_uh, min( runtime ) ) << std::endl;
            
            t_ref = min( runtime );
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "row wise (id based)" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            auto  cbmap    = impl::uniform::build_id2cb( *colcb );
            auto  blockmap = impl::uniform::build_id2blocks( *rowcb, *A, false );
            
            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::uniform::mul_vec_row< value_t >( 2.0, Hpro::apply_normal, *x, *y, *rowcb, *colcb, cbmap, blockmap );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;

            std::cout << "      flops   = " << format_flops( flops_uh, min( runtime ) ) << std::endl;
            
            t_ref = min( runtime );
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "level" << term::reset << std::endl;
        
            auto  A_hier   = matrix::build_level_hierarchy( *A );
            auto  rcb_hier = matrix::build_level_hierarchy( *rowcb );
            auto  ccb_hier = matrix::build_level_hierarchy( *colcb );

            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::uniform::mul_vec_hier( value_t(2), hpro::apply_normal, A_hier, *x, *y, rcb_hier, ccb_hier );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;
        
            std::cout << "      ratio   = " << boost::format( "%.02f" ) % ( min( runtime ) / t_ref ) << std::endl;
            std::cout << "      flops   = " << format_flops( flops_uh, min( runtime ) ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
