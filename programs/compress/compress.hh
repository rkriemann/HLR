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
        auto  pcoeff  = std::make_unique< Hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx   = std::make_unique< bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > > >( *pcoeff );
        
        tic = timer::now();
        A   = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
        toc = timer::since( tic );

        if ( verbose( 2 ) )
            io::hpro::write< value_t >( *A, "A.hm" );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        tic = timer::now();
        A = io::hpro::read< value_t >( matrixfile );
        toc = timer::since( tic );
    }// else
    
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_A = A->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,norank,nosize" );

    //////////////////////////////////////////////////////////////////////
    //
    // further compress matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto        zA     = impl::matrix::copy_compressible( *A );
    auto        norm_A = impl::norm::frobenius( *A );
    const auto  delta  = cmdline::eps; // norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", "
              << hlr::compress::provider << ')'
              << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    {
        // auto  lacc = local_accuracy( delta );
        auto  lacc = relative_prec( delta );
        
        runtime.clear();
        
        for ( uint  i = 0; i < std::max( nbench, 1u ); ++i )
        {
            auto  B = impl::matrix::copy( *zA );
        
            tic = timer::now();
    
            // impl::matrix::compress( *B, Hpro::fixed_prec( norm_A * acc.rel_eps() ) );
            impl::matrix::compress( *B, lacc );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << "      compressed in   " << format_time( toc ) << std::endl;

            if ( i == nbench-1 )
                zA = std::move( B );
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
        auto  B     = impl::matrix::convert_to_h( *zA );
        auto  error = impl::norm::frobenius( value_t(1), *A, value_t(-1), *B );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }

    std::cout << "  "
              << term::bullet << term::bold
              << "decompression "
              << term::reset << std::endl;

    {
        runtime.clear();
        
        auto  zB = impl::matrix::copy( *zA );
        
        for ( uint  i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            impl::matrix::decompress( *zB );
            
            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << "      decompressed in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                zB = std::move( impl::matrix::copy( *zA ) );
        }// for
        
        if ( nbench > 1 )
            std::cout << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        // auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zB );

        // error = norm::spectral( impl::arithmetic, *diffB );
        
        auto  error = impl::norm::frobenius( value_t(1), *A, value_t(-1), *zB );
        
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    //
    // test also for FP32
    //
    //////////////////////////////////////////////////////////////////////

    if ( false )
    {
        std::cout << "  "
                  << term::bullet << term::bold
                  << "FP32 compression ("
                  << "ε = " << boost::format( "%.2e" ) % cmdline::eps
                  << ", "
                  << hlr::compress::provider << ')'
                  << term::reset << std::endl;
        std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;
        
        auto  As      = impl::matrix::convert< float >( *A );
        auto  zAs     = impl::matrix::copy_compressible( *As );
        auto  norm_As = impl::norm::frobenius( *As );
        auto  mem_As  = As->byte_size();

        std::cout << "    mem   = " << format_mem( mem_As ) << std::endl;

        {
            auto  lacc = local_accuracy( delta );
            // auto  lacc = relative_prec( cmdline::eps );
        
            runtime.clear();
        
            for ( uint  i = 0; i < std::max( nbench, 1u ); ++i )
            {
                auto  B = impl::matrix::copy( *zAs );
        
                tic = timer::now();
    
                impl::matrix::compress( *B, lacc );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
                std::cout << "      compressed in   " << format_time( toc ) << std::endl;

                if ( i == nbench-1 )
                    zAs = std::move( B );
            }// for

            if ( nbench > 1 )
                std::cout << "    runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
        }

        const auto  mem_zAs = zAs->byte_size();
    
        std::cout << "    mem   = " << format_mem( mem_zAs ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zAs) / double(mem_As) ) << std::endl;

        auto  error = impl::norm::frobenius( value_t(1), *As, value_t(-1), *zAs );

        std::cout << "    error = " << format_error( error, error / norm_As ) << std::endl;

        std::cout << "  "
                  << term::bullet << term::bold
                  << "decompression "
                  << term::reset << std::endl;

        {
            runtime.clear();
        
            auto  zB = impl::matrix::copy( *zAs );
        
            for ( uint  i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                impl::matrix::decompress( *zB );
            
                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
                std::cout << "      decompressed in   " << format_time( toc ) << std::endl;

                if ( i < nbench-1 )
                    zB = std::move( impl::matrix::copy( *zAs ) );
            }// for
        
            if ( nbench > 1 )
                std::cout << "    runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            error = impl::norm::frobenius( value_t(1), *As, value_t(-1), *zB );
        
            std::cout << "    error = " << format_error( error, error / norm_As ) << std::endl;
        }
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
    
