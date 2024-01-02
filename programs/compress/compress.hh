//
// Project     : HLR
// Program     : compress
// Description : construction and MVM with compressed data blocks
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

    // if ( false )
    // {
    //     auto  tic     = timer::now();
    //     auto  toc     = timer::since( tic );
        
    //     size_t  n = 4096;
    //     auto  M = blas::matrix< value_t >( n, n );

    //     for ( int i = 0; i < n*n; ++i )
    //         M.data()[i] = i+1;

    //     // for ( int i = 0; i < n; ++i )
    //     // {
    //     //     for ( int j = 0; j < n; ++j )
    //     //         std::cout << M(i,j) << ", ";
    //     //     std::cout << std::endl;
    //     // }// for
        
    //     auto  x = blas::vector< value_t >( M.ncols() );
    //     auto  y1 = blas::vector< value_t >( M.nrows() );
    //     auto  y2 = blas::vector< value_t >( M.nrows() );
    //     auto  y3 = blas::vector< value_t >( M.nrows() );

    //     for ( int i = 0; i < M.ncols(); ++i )
    //         x(i) = i+1;

    //     auto  zcfg = compress::get_config( 1e-4 );
    //     auto  zM   = compress::compress( zcfg, M.data(), M.nrows(), M.ncols() );

    //     tic = timer::now();
    //     blas::mulvec( value_t(1), blas::adjoint( M ), x, value_t(1), y1 );
    //     toc = timer::since( tic );
    //     std::cout << toc.seconds() << std::endl;
        
    //     std::cout << y1(0) << " / " << y1(1) << " / " << y1(2) << std::endl;
        
    //     tic = timer::now();
    //     compress::aflp::mulvec( M.nrows(), M.ncols(), apply_adjoint, 1.0, zM, x.data(), y2.data() );
    //     toc = timer::since( tic );
    //     std::cout << toc.seconds() << std::endl;

    //     std::cout << y2(0) << " / " << y2(1) << " / " << y2(2) << std::endl;

    //     return;
    // }
    
    // if ( false )
    // {
    //     auto  tic     = timer::now();
    //     auto  toc     = timer::since( tic );
        
    //     size_t  n = 4;
    //     size_t  k = 4;
    //     auto    M = blas::matrix< value_t >( n, k );

    //     for ( int i = 0; i < n*k; ++i )
    //         M.data()[i] = i+1;

    //     for ( int i = 0; i < M.nrows(); ++i )
    //     {
    //         for ( int j = 0; j < M.ncols(); ++j )
    //             std::cout << M(i,j) << ", ";
    //         std::cout << std::endl;
    //     }// for
        
    //     auto  x  = blas::vector< value_t >( M.ncols() );
    //     auto  y1 = blas::vector< value_t >( M.nrows() );
    //     auto  y2 = blas::vector< value_t >( M.nrows() );
    //     auto  y3 = blas::vector< value_t >( M.nrows() );

    //     for ( int i = 0; i < x.length(); ++i )
    //         x(i) = i+1;

    //     auto  S = blas::vector< value_t >( k );

    //     S(0) = 1e-8;
    //     S(1) = 1e-7;
    //     S(2) = 1e-6;
    //     S(3) = 1e-5;
        
    //     auto  zM   = compress::aflp::compress_lr( M, S );

    //     blas::mulvec( value_t(1), blas::adjoint( M ), x, value_t(1), y1 );

    //     // for ( uint  i = 0; i < k; ++i )
    //     //     y1(i) *= S(i);
        
    //     std::cout << y1(0) << " / " << y1(1) << " / " << y1(2) << std::endl;
        
    //     compress::aflp::mulvec_lr( M.nrows(), M.ncols(), apply_adjoint, 1.0, zM, x.data(), y2.data() );

    //     // for ( uint  i = 0; i < k; ++i )
    //     //     y2(i) *= S(i);

    //     std::cout << y2(0) << " / " << y2(1) << " / " << y2(2) << std::endl;

    //     return;
    // }
    
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
                
                A = impl::matrix::build( bct->root(), pcoeff, hcalr, acc, nseq );
            }// if
            else
                cmdline::capprox = "default";
        }// if

        if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
        {
            std::cout << "    using ACA" << std::endl;

            auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, acalr, acc, nseq );
        }// else
        
        if ( cmdline::capprox == "dense" )
        {
            std::cout << "    using dense" << std::endl;

            auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, dense, acc, nseq );
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

    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_A = A->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,nosize" );

    const auto  norm_A = impl::norm::frobenius( *A );
    
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

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
    // further compress matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto        zA     = impl::matrix::copy_compressible( *A );
    const auto  delta  = cmdline::eps; // norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", "
              << hlr::compress::provider << ')'
              << term::reset << std::endl;

    {
        // auto  lacc = local_accuracy( delta );
        auto  lacc  = relative_prec( delta );
        auto  niter = std::max( nbench, 1u );
        
        runtime.clear();
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            // impl::matrix::compress( *B, Hpro::fixed_prec( norm_A * acc.rel_eps() ) );
            impl::matrix::compress( *zA, lacc );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << "      compressed in   " << format_time( toc ) << std::endl;

            if ( i < niter-1 )
            {
                zA.reset( nullptr );
                zA = std::move( impl::matrix::copy_compressible( *A ) );
            }// if
        }// for

        if ( nbench > 1 )
            std::cout << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
    }

    const auto  mem_zA = zA->byte_size();
    
    std::cout << "    mem   = " << format_mem( zA->byte_size() ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );

    {
        auto  error = impl::norm::frobenius( 1, *A, -1, *zA );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
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
            std::cout << "      decompressed in   " << format_time( toc ) << std::endl;

            if ( i < niter-1 )
            {
                zB.reset( nullptr );
                zB = std::move( impl::matrix::copy( *zA ) );
            }// if
        }// for
        
        if ( nbench > 1 )
            std::cout << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        auto  error = impl::norm::frobenius( 1, *A, -1, *zB );
        
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
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

        double  t_orig       = 0.0;
        double  t_compressed = 0.0;
        auto    y_ref        = std::unique_ptr< vector::scalar_vector< value_t > >();
        
        {
            runtime.clear();
            
            std::cout << "  "
                      << term::bullet << term::bold
                      << "uncompressed"
                      << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < 50; ++j )
                    impl::mul_vec< value_t >( 2.0, Hpro::apply_normal, *A, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << "    mvm in   " << format_time( toc ) << std::endl;

                if ( i < nbench-1 )
                    y->fill( 1 );
            }// for
        
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            t_orig = min( runtime );
            
            y_ref = std::move( y );
        }

        {
            runtime.clear();
            
            std::cout << "  "
                      << term::bullet << term::bold
                      << "compressed"
                      << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < 50; ++j )
                    impl::mul_vec< value_t >( 2.0, Hpro::apply_normal, *zA, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << "    mvm in   " << format_time( toc ) << std::endl;

                if ( i < nbench-1 )
                    y->fill( 1 );
            }// for
        
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
        
            t_compressed = min( runtime );

            std::cout << "    ratio  = " << boost::format( "%.02f" ) % ( t_compressed / t_orig ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "    error  = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
    
