//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/arith/norm.hh>
#include <hlr/utils/tensor.hh>
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

    {
        for ( uint  i = 0; i < std::max( nbench, 1 ); ++i )
        {
            auto  B = impl::matrix::copy( *zA );
        
            tic = timer::now();
    
            impl::matrix::compress( *B, Hpro::fixed_prec( acc.rel_eps() ) );

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

        runtime.clear();
    }

    const auto  mem_zA = zA->byte_size();
    
    std::cout << "    mem   = " << format_mem( zA->byte_size() ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zA) / double(mem_A) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );
    
    auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    auto  error = norm::spectral( impl::arithmetic, *diff );
    
    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    std::cout << "  " << term::bullet << term::bold << "decompression " << term::reset << std::endl;

    {
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

        auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zB );

        error = norm::spectral( impl::arithmetic, *diffB );
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // try uniform format
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "uniform H-matrix" << term::reset << std::endl;
    
    std::cout << "  " << term::bullet << term::bold << "construction" << term::reset << std::endl;
    
    auto  apx = approx::SVD< value_t >();
    
    tic = timer::now();
    
    auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A, apx, acc, nseq );

    toc = timer::since( tic );

    const auto  mem_A2 = A2->byte_size();
    const auto  mem_rb = rowcb->byte_size();
    const auto  mem_cb = colcb->byte_size();
    const auto  mem_U  = mem_A2 + mem_rb + mem_cb;
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A2, mem_rb, mem_cb ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_U) / double(mem_A) ) << std::endl;

    {
        auto  diff2 = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = norm::spectral( impl::arithmetic, *diff2 );
        
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }
    
    std::cout << "  " << term::bullet << term::bold << "compression via "
              << hlr::compress::provider
              << ", ε = " << boost::format( "%.2e" ) % cmdline::eps << term::reset << std::endl;
    
    {
        for ( uint  i = 0; i < std::max( nbench, 1 ); ++i )
        {
            auto  zA2    = impl::matrix::copy( *A2 );
            auto  zrowcb = rowcb->copy();
            auto  zcolcb = colcb->copy();
        
            matrix::replace_cluster_basis( *zA2, *zrowcb, *zcolcb );
            
            tic = timer::now();

            impl::matrix::compress( *zrowcb, Hpro::fixed_prec( acc.rel_eps() ) );
            impl::matrix::compress( *zcolcb, Hpro::fixed_prec( acc.rel_eps() ) );
            impl::matrix::compress( *zA2,    Hpro::fixed_prec( acc.rel_eps() ) );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            std::cout << "      compressed in   " << format_time( toc ) << std::endl;

            if ( i == nbench-1 )
            {
                A2    = std::move( zA2 );
                rowcb = std::move( zrowcb );
                colcb = std::move( zcolcb );
                // update pointers in A2
                // matrix::replace_cluster_basis( *A2, *rowcb, *colcb );
            }// if
        }// for

        if ( nbench > 1 )
            std::cout << "    runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        runtime.clear();
    }

    const auto  mem_zA2 = A2->byte_size();
    const auto  mem_zrb = rowcb->byte_size();
    const auto  mem_zcb = colcb->byte_size();
    const auto  mem_zU  = mem_zA2 + mem_zrb + mem_zcb;
    
    std::cout << "    mem   = " << format_mem( mem_zA2, mem_zrb, mem_zcb ) << std::endl;
    std::cout << "      vs U  " << boost::format( "%.3f" ) % ( double(mem_zU) / double(mem_U) ) << std::endl;
    std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_zU) / double(mem_A) ) << std::endl;
    std::cout << "      vs zH " << boost::format( "%.3f" ) % ( double(mem_zU) / double(mem_zA) ) << std::endl;

    {
        auto  diff2 = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = norm::spectral( impl::arithmetic, *diff2 );
        
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }

    std::cout << "  " << term::bullet << term::bold << "decompression " << term::reset << std::endl;

    {
        auto  zB     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *zB, *rowcb2, *colcb2 );
        
        tic = timer::now();
    
        impl::matrix::decompress( *rowcb2 );
        impl::matrix::decompress( *colcb2 );
        impl::matrix::decompress( *zB );

        toc = timer::since( tic );

        auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zB );
        
        std::cout << "    done in " << format_time( toc ) << std::endl;

        const auto  error = norm::spectral( impl::arithmetic, *diffB );
            
        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////

    if ( nbench > 0 )
    {
        std::cout << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;

        double  t_orig       = 0.0;
        double  t_compressed = 0.0;
        auto    y_ref        = std::unique_ptr< vector::scalar_vector< value_t > >();
        
        {
            std::cout << "  " << term::bullet << term::bold << "uncompressed" << term::reset << std::endl;
        
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
            
            runtime.clear();

            y_ref = std::move( y );
        }

        {
            std::cout << "  " << term::bullet << term::bold << "compressed" << term::reset << std::endl;
        
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
            
            runtime.clear();

            std::cout << "    ratio  = " << boost::format( "%.02f" ) % ( t_compressed / t_orig ) << std::endl;

            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "    error  = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
    
