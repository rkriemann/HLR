//
// Project     : HLR
// Program     : compress-h2
// Description : construction and MVM for H²-matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/arith/norm.hh>
#include <hlr/bem/aca.hh>
#include <hlr/bem/hca.hh>
#include <hlr/bem/dense.hh>

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
    auto  H   = std::unique_ptr< Hpro::TMatrix< value_t > >();

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
                
                H = impl::matrix::build( bct->root(), pcoeff, hcalr, acc, nseq );
            }// if
            else
                cmdline::capprox = "default";
        }// if

        if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
        {
            std::cout << "    using ACA" << std::endl;

            auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            H = impl::matrix::build( bct->root(), pcoeff, acalr, acc, nseq );
        }// else
        
        if ( cmdline::capprox == "dense" )
        {
            std::cout << "    using dense" << std::endl;

            auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            H = impl::matrix::build( bct->root(), pcoeff, dense, acc, nseq );
        }// else
        
        toc = timer::since( tic );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        tic = timer::now();
        H = io::hpro::read< value_t >( matrixfile );
        toc = timer::since( tic );
    }// else
    
    std::cout << "    dims  = " << H->nrows() << " × " << H->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_H  = H->byte_size();
    const auto  norm_H = impl::norm::frobenius( *H );
    
    std::cout << "    mem   = " << format_mem( mem_H ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *H, "H", "noid,nosize" );
    
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
        
        auto  Hc = impl::matrix::coarsen( *H, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Hc = Hc->byte_size();
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Hc ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Hc) / double(mem_H) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Hc, "Hc", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *H, -1, *Hc );
        auto  norm_A = impl::norm::spectral( *H );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        H = std::move( Hc );
    }// if
    
    if ( cmdline::tohodlr )
    {
        std::cout << term::bullet << term::bold << "converting to HODLR" << term::reset << std::endl;
        
        auto  apx = approx::SVD< value_t >();

        tic = timer::now();
        
        auto  Hc = impl::matrix::convert_to_hodlr( *H, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Hc = Hc->byte_size();
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Hc ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Hc) / double(mem_H) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Hc, "Hc", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *H, -1, *Hc );
        auto  norm_A = impl::norm::spectral( *H );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        H = std::move( Hc );
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // convert to H²
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

    auto  h2acc = ( cmdline::tohodlr ? fixed_prec( 0.1 * cmdline::eps ) : fixed_prec( 0.4 * cmdline::eps ) );
    auto  cbapx = approx::SVD< value_t >();

    tic = timer::now();

    auto  [ rowcb, colcb, A ] = impl::matrix::build_h2_rec( *H, cbapx, h2acc );
    
    toc = timer::since( tic );

    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_A   = A->byte_size();
    const auto  mem_rcb = rowcb->byte_size();
    const auto  mem_ccb = colcb->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_rcb, mem_ccb, mem_A, mem_rcb + mem_ccb + mem_A ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_rcb + mem_ccb + mem_A) / double(mem_H) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,nosize" );

    {
        auto  error = impl::norm::frobenius( 1, *H, -1, *A );

        std::cout << "    error = " << format_error( error, error / norm_H ) << std::endl;
    }
    
    // assign clusters since needed for cluster bases
    // seq::matrix::assign_cluster( *A, *bct->root() );
    
    //////////////////////////////////////////////////////////////////////
    //
    // further compress matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto  zA     = impl::matrix::copy( *A );
    auto  zrowcb = rowcb->copy();
    auto  zcolcb = colcb->copy();

    matrix::replace_cluster_basis( *zA, *zrowcb, *zcolcb );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "ε = " << boost::format( "%.2e" ) % cmdline::eps
              << ", "
              << hlr::compress::provider << " + " << hlr::compress::aplr::provider << ')'
              << term::reset << std::endl;

    {
        auto  lacc  = relative_prec( cmdline::eps );
        auto  niter = std::max( nbench, 1u );
        
        runtime.clear();
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            impl::matrix::compress< matrix::shared_cluster_basis< value_t > >( *zrowcb, lacc );
            impl::matrix::compress< matrix::shared_cluster_basis< value_t > >( *zcolcb, lacc );
            impl::matrix::compress( *zA, lacc );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << term::rollback << term::clearline << "      compressed in   " << format_time( toc ) << term::flush;

            if ( i < niter-1 )
            {
                zA.reset( nullptr );
                zrowcb.reset( nullptr );
                zcolcb.reset( nullptr );
                
                zA     = std::move( impl::matrix::copy( *A ) );
                zrowcb = std::move( rowcb->copy() );
                zcolcb = std::move( colcb->copy() );
        
                matrix::replace_cluster_basis( *zA, *zrowcb, *zcolcb );
            }// if
        }// for

        if ( nbench > 1 )
            std::cout << term::rollback << term::clearline << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
        std::cout << std::endl;
    }

    const auto  mem_zA   = zA->byte_size();
    const auto  mem_zrcb = zrowcb->byte_size();
    const auto  mem_zccb = zcolcb->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_zrcb, mem_zccb, mem_zA, mem_zrcb + mem_zccb + mem_zA ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zrcb + mem_zccb + mem_zA) / double(mem_H) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );
    
    // auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    // auto  error = norm::spectral( impl::arithmetic, *diff );

    // std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    {
        auto  error = impl::norm::frobenius( 1, *H, -1, *zA );

        std::cout << "    error = " << format_error( error, error / norm_H ) << std::endl;
    }

    std::cout << "  "
              << term::bullet << term::bold
              << "decompression "
              << term::reset << std::endl;

    {
        auto  niter = std::max( nbench, 1u );
        
        runtime.clear();
        
        auto  zA2     = impl::matrix::copy( *zA );
        auto  zrowcb2 = zrowcb->copy();
        auto  zcolcb2 = zcolcb->copy();
        
        matrix::replace_cluster_basis( *zA2, *zrowcb2, *zcolcb2 );
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            impl::matrix::decompress< matrix::shared_cluster_basis< value_t > >( *zrowcb2 );
            impl::matrix::decompress< matrix::shared_cluster_basis< value_t > >( *zcolcb2 );
            impl::matrix::decompress( *zA2 );
            
            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << term::rollback << term::clearline << "      decompressed in   " << format_time( toc ) << term::flush;

            if ( i < niter-1 )
            {
                zA2.reset( nullptr );
                zrowcb2.reset( nullptr );
                zcolcb2.reset( nullptr );

                zA2     = std::move( impl::matrix::copy( *zA ) );
                zrowcb2 = std::move( zrowcb->copy() );
                zcolcb2 = std::move( zcolcb->copy() );
            
                matrix::replace_cluster_basis( *zA2, *zrowcb2, *zcolcb2 );
            }// if
        }// for
        
        if ( nbench > 1 )
            std::cout << term::rollback << term::clearline << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
        std::cout << std::endl;

        // auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zA2 );

        {
            auto  error = impl::norm::frobenius( 1, *H, -1, *zA2 );

            std::cout << "    error = " << format_error( error, error / norm_H ) << std::endl;
        }
    }

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
        const auto  flops_h2 = nmvm * hlr::h2::mul_vec_flops( apply_normal, *A, *rowcb, *colcb );
        const auto  bytes_h2 = nmvm * hlr::h2::mul_vec_datasize( apply_normal, *A, *rowcb, *colcb );
        const auto  bytes_z2 = nmvm * hlr::h2::mul_vec_datasize( apply_normal, *zA, *zrowcb, *zcolcb );
    
        std::cout << "  " << term::bullet << term::bold << "FLOPs/byte " << term::reset() << std::endl;
        std::cout << "    H²   = " << format_flops( flops_h2 ) << ", " << flops_h2 / bytes_h2 << std::endl;
        std::cout << "    zH²  = " << format_flops( flops_h2 ) << ", " << flops_h2 / bytes_z2 << std::endl;

        double  t_orig       = 0.0;
        double  t_compressed = 0.0;
        auto    y_ref        = std::unique_ptr< vector::scalar_vector< value_t > >();
        
        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "uncompressed" << term::reset << std::endl;
            
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A, *x, *y, *rowcb, *colcb );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "    mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;

            std::cout << "    flops  = " << format_flops( flops_h2, min( runtime ) ) << std::endl;
            
            t_orig = min( runtime );
            
            y_ref = std::move( y );
        }

        {
            runtime.clear();
            
            std::cout << "  " << term::bullet << term::bold << "compressed" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
                    impl::h2::mul_vec< value_t >( 2.0, apply_normal, *zA, *x, *y, *zrowcb, *zcolcb );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << term::rollback << term::clearline << "    mvm in   " << format_time( toc ) << term::flush;

                if ( i < nbench-1 )
                    y->fill( 1 );
            }// for
        
            if ( nbench > 1 )
                std::cout << term::rollback << term::clearline << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime );
            std::cout << std::endl;
        
            std::cout << "    ratio  = " << boost::format( "%.02f" ) % ( min( runtime ) / t_orig ) << std::endl;
            std::cout << "    flops  = " << format_flops( flops_h2, min( runtime ) ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "    error  = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
    
