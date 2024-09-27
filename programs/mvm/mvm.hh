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
    
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< Hpro::TMatrix< value_t > >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );

    io::eps::print( bct->root(), "bt" );
    
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        
    tic = timer::now();
        
    if ( cmdline::capprox == "hca" )
    {
        if constexpr ( problem_t::supports_hca )
        {
            std::cout << "    using HCA"
                      << ( cmdline::compress ? std::string( " (" ) + hlr::compress::provider + " + " + hlr::compress::aplr::provider + ")" : "" )
                      << std::endl;
                
            auto  hcagen = problem->hca_gen_func( *ct );
            auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
            auto  hcalr  = bem::hca_lrapx( hca );
                
            A = impl::matrix::build_sv( bct->root(), pcoeff, hcalr, acc, cmdline::compress, nseq );
        }// if
        else
            cmdline::capprox = "default";
    }// if

    if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
    {
        std::cout << "    using ACA" 
                  << ( cmdline::compress ? std::string( " (" ) + hlr::compress::provider + " + " + hlr::compress::aplr::provider + ")" : "" )
                  << std::endl;

        auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );

        A = impl::matrix::build_sv( bct->root(), pcoeff, acalr, acc, cmdline::compress, nseq );
    }// else
        
    if ( cmdline::capprox == "dense" )
    {
        std::cout << "    using dense"
                  << ( cmdline::compress ? std::string( " (" ) + hlr::compress::provider + " + " + hlr::compress::aplr::provider + ")" : "" )
                  << std::endl;

        auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
        A = impl::matrix::build_sv( bct->root(), pcoeff, dense, acc, cmdline::compress, nseq );
    }// else
        
    toc = timer::since( tic );
    
    const auto  mem_A  = A->byte_size();
    const auto  norm_A = impl::norm::frobenius( *A );
        
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( norm_A ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////

    if ( nbench > 0 )
    {
        std::cout << term::bullet << term::bold
                  << "mat-vec"
                  << term::reset << std::endl;

        const uint  nmvm    = 50;
        const auto  flops_h = nmvm * hlr::mul_vec_flops( apply_normal, *A );
        const auto  bytes_h = nmvm * hlr::mul_vec_datasize( apply_normal, *A );
    
        std::cout << "  " << term::bullet << term::bold << "FLOPs/byte " << term::reset() << std::endl;
        std::cout << "    H    = " << format_flops( flops_h ) << ", " << flops_h / bytes_h << std::endl;
    
        double  t_ref = 0.0;
        auto    y_ref = std::unique_ptr< vector::scalar_vector< value_t > >();

        {
            // generate reference solution
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );

            y_ref = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            x->fill( 1 );
            impl::mul_vec_chunk< value_t >( nmvm * value_t(2), apply_normal, *A, *x, *y_ref );
        }
        
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "chunk locks" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_chunk< value_t >( value_t(2), apply_normal, *A, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            t_ref = min( runtime );
            
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        if ( true )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "row cluster lists" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_blocks< value_t >( apply_normal, *A );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            std::cout << "      ratio   = " << boost::format( "%.02f" ) % ( min( runtime ) / t_ref ) << std::endl;
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        if ( false )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "row cluster lists (BtT)" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_blocks< value_t >( apply_normal, *A );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl2< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            std::cout << "      ratio   = " << boost::format( "%.02f" ) % ( min( runtime ) / t_ref ) << std::endl;
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        if ( true )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "joined lowrank blocks" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_matrix< value_t >( apply_normal, *A );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            std::cout << "      ratio   = " << boost::format( "%.02f" ) % ( min( runtime ) / t_ref ) << std::endl;
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }

        if ( true )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "thread local" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_ts< value_t >( value_t(2), apply_normal, *A, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            t_ref = min( runtime );
            
            std::cout << "      ratio   = " << boost::format( "%.02f" ) % ( min( runtime ) / t_ref ) << std::endl;
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "      error   = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
