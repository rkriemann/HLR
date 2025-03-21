//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <fstream>
#include <limits>

#include <hlr/bem/hca.hh>

#include "hlr/arith/norm.hh"
#include "hlr/approx/accuracy.hh"
#include "hlr/bem/aca.hh"
#include "hlr/bem/dense.hh"
#include <hlr/matrix/info.hh>
#include <hlr/matrix/print.hh>
#include <hlr/utils/io.hh>

#include <hlr/utils/eps_printer.hh>

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
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
        auto  coeff   = problem->coeff_func();
        auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        
        tic = timer::now();
        
        if ( cmdline::capprox == "hca" )
        {
            if constexpr ( problem_t::supports_hca )
            {
                std::cout << "    using HCA" << std::endl;
                
                auto  hcagen = problem->hca_gen_func( *ct );
                auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
                auto  hcalr  = bem::hca_lrapx( hca );
                
                A = impl::matrix::build_sv( bct->root(), pcoeff, hcalr, acc, nseq );
            }// if
            else
                cmdline::capprox = "default";
        }// if

        if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
        {
            std::cout << "    using ACA" << std::endl;

            auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build_sv( bct->root(), pcoeff, acalr, acc, nseq );
        }// else
        
        if ( cmdline::capprox == "dense" )
        {
            std::cout << "    using dense" << std::endl;

            auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build_sv( bct->root(), pcoeff, dense, acc, nseq );
        }// else
        
        toc = timer::since( tic );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        tic = timer::now();
        A   = io::hpro::read< value_t >( matrixfile );
        toc = timer::since( tic );
    }// else
    
    const auto  mem_A    = matrix::data_byte_size( *A );
    const auto  mem_A_d  = matrix::data_byte_size_dense( *A );
    const auto  mem_A_lr = matrix::data_byte_size_lowrank( *A );
    const auto  norm_A   = impl::norm::frobenius( *A );
        
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A, mem_A_d, mem_A_lr ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( norm_A ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    //////////////////////////////////////////////////////////////////////
    //
    // coarsen matrix
    //
    //////////////////////////////////////////////////////////////////////
    
    if ( cmdline::coarsen )
    {
        std::cout << term::bullet << term::bold << "coarsening" << term::reset << std::endl;
        
        auto  apx = approx::SVD< value_t >();

        tic = timer::now();
        
        auto  Ac = impl::matrix::coarsen( *A, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Ac = Ac->byte_size();
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Ac ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Ac) / double(mem_A) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Ac, "Ac", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *A, -1, *Ac );
        auto  norm_A = impl::norm::spectral( *A );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        A = std::move( Ac );
    }// if
    
    if ( cmdline::tohodlr )
    {
        std::cout << term::bullet << term::bold << "converting to HODLR" << term::reset << std::endl;
        
        auto  apx = approx::SVD< value_t >();

        tic = timer::now();
        
        auto  Ac = impl::matrix::convert_to_hodlr( *A, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Ac = Ac->byte_size();
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Ac ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Ac) / double(mem_A) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Ac, "Ac", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *A, -1, *Ac );
        auto  norm_A = impl::norm::spectral( *A );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        A = std::move( Ac );
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // convert to mixed precision format
    //
    //////////////////////////////////////////////////////////////////////

    auto        zA    = impl::matrix::copy_mixedprec( *A );
    const auto  delta = cmdline::eps; // norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );

    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", "
              << hlr::compress::provider << " + " << hlr::compress::valr::provider << ")"
              << term::reset << std::endl;

    {
        // auto  lacc = local_accuracy( delta );
        auto  lacc  = relative_prec( Hpro::frobenius_norm, delta );
        auto  niter = std::max( nbench, 1u );
        
        runtime.clear();
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            impl::matrix::compress( *zA, lacc );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << term::rollback << term::clearline << "      compressed in   " << format_time( toc ) << term::flush;

            if ( i < niter-1 )
            {
                zA.reset( nullptr );
                zA = std::move( impl::matrix::copy_mixedprec( *A ) );
            }// if
        }// for

        if ( nbench > 1 )
            std::cout << term::rollback << term::clearline << "    runtime = "
                      << format_time( min( runtime ), median( runtime ), max( runtime ) );
        std::cout << std::endl;
    }

    const auto  mem_zA    = matrix::data_byte_size( *zA );
    const auto  mem_zA_d  = matrix::data_byte_size_dense( *zA );
    const auto  mem_zA_lr = matrix::data_byte_size_lowrank( *zA );
    
    std::cout << "    mem     = " << format_mem( mem_zA, mem_zA_d, mem_zA_lr ) << std::endl;
    std::cout << "        vs H  "
              << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << " / "
              << boost::format( "%.3f" ) % ( double(mem_zA_d) / double(mem_A_d) ) << " / "
              << boost::format( "%.3f" ) % ( double(mem_zA_lr) / double(mem_A_lr) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );

    {
        auto  error = impl::norm::frobenius( 1, *A, -1, *zA );

        std::cout << "    error   = " << format_error( error, error / norm_A ) << std::endl;
    }
    
    std::cout << "  "
              << term::bullet << term::bold
              << "decompression "
              << term::reset << std::endl;

    {
        runtime.clear();
        
        auto  zB    = impl::matrix::copy( *zA );
        auto  niter = std::max( nbench, 1u );
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            impl::matrix::decompress( *zB );
            
            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << term::rollback << term::clearline << "      decompressed in   " << format_time( toc ) << term::flush;

            if ( i < niter-1 )
            {
                zB.reset( nullptr );
                zB = std::move( impl::matrix::copy( *zA ) );
            }// if
        }// for
        
        if ( nbench > 1 )
            std::cout << term::rollback << term::clearline << "    runtime = "
                      << format_time( min( runtime ), median( runtime ), max( runtime ) );
        std::cout << std::endl;

        auto  error = impl::norm::frobenius( 1, *A, -1, *zB );
        
        std::cout << "    error   = " << format_error( error, error / norm_A ) << std::endl;
    }

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
        const auto  bytes_z = nmvm * hlr::mul_vec_datasize( apply_normal, *zA );
    
        std::cout << "  " << term::bullet << term::bold << "FLOPs/byte " << term::reset() << std::endl;
        std::cout << "    H    = " << format_flops( flops_h ) << ", " << flops_h / bytes_h << std::endl;
        std::cout << "    zH   = " << format_flops( flops_h ) << ", " << flops_h / bytes_z << std::endl;
    
        double  t_ref = 0.0;
        auto    y_ref = std::unique_ptr< vector::scalar_vector< value_t > >();

        //
        // uncompressed
        //

        std::cout << "  " << term::bullet << term::bold << "uncompressed" << term::reset << std::endl;
        
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "standard" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec< value_t >( value_t(2), apply_normal, *A, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "      runtime = "
                          << format_time( min( runtime ), median( runtime ), max( runtime ) );
            std::cout << std::endl;

            t_ref = min( runtime );
            
            std::cout << "      flops   = " << format_flops( flops_h, min( runtime ) ) << std::endl;
            
            y_ref = std::move( y );
        }

        if ( false )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "row cluster lists (TtB)" << term::reset << std::endl;
        
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
                    y->fill( 1 );
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
                    y->fill( 1 );
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
                    y->fill( 1 );
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
            
            std::cout << "    " << term::bullet << term::bold << "level-wise" << term::reset << std::endl;
        
            auto  A_hier = matrix::build_level_hierarchy( *A );
            auto  x      = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y      = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_hier< value_t >( value_t(2), apply_normal, A_hier, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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

        //
        // compressed
        //
        std::cout << "  " << term::bullet << term::bold << "compressed"
                  #if defined(HLR_HAS_ZBLAS_APLR)
                  << " (zblas)"
                  #endif
                  << term::reset << std::endl;
        

        if ( true )
        {
            runtime.clear();
            
            std::cout << "    " << term::bullet << term::bold << "standard" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec< value_t >( value_t(2), apply_normal, *zA, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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
            
            std::cout << "    " << term::bullet << term::bold << "row cluster lists (TtB)" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_blocks< value_t >( apply_normal, *zA );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_blocks< value_t >( apply_normal, *zA );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl2< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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
            
            std::cout << "    " << term::bullet << term::bold << "joined lowrank blocks" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            auto  blocks = impl::build_cluster_matrix< value_t >( apply_normal, *zA );
            
            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_cl< value_t >( value_t(2), apply_normal, *blocks, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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
            
            std::cout << "    " << term::bullet << term::bold << "level-wise" << term::reset << std::endl;
        
            auto  A_hier = matrix::build_level_hierarchy( *zA );
            auto  x      = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y      = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::mul_vec_hier< value_t >( value_t(2), apply_normal, A_hier, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
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
    }// if
}
