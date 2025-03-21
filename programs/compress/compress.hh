//
// Project     : HLR
// Description : construction and MVM with compressed data blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/arith/norm.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/bem/aca.hh>
#include <hlr/bem/hca.hh>
#include <hlr/bem/dense.hh>
#include <hlr/matrix/info.hh>

#include "common.hh"
#include "common-main.hh"

#include <hpro/io/TGridIO.hh>

using namespace hlr;

using indexset = Hpro::TIndexSet;

struct local_accuracy : public accuracy
{
    local_accuracy ( const double  abs_eps )
            : accuracy( 0.0, abs_eps )
    {}
    
    virtual const accuracy  acc ( const indexset &  rowis,
                                  const indexset &  colis ) const
    {
        return Hpro::absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
    }
};

//
// compression tests
//
template < typename value_t >
void
test_compress ( const std::string &  filename )
{
    std::cout << std::endl << filename << std::endl;
    
    auto  D = io::matlab::read< value_t >( filename );
    
    auto  m1 = D.data_byte_size();
    auto  n1 = blas::norm_F( D );
    
    std::cout << "  mem:  " << m1 << std::endl;
    std::cout << "  norm: " << n1 << std::endl;

    //
    // dense compression
    //

    std::cout << "dense compression" << std::endl;
    
    for ( double  eps = 1e-3; eps >= 1e-10; eps = eps / 10.0 )
    {
        std::cout << "  eps: " << eps;

        auto  zconf = compress::get_config( relative_prec( eps ), D );
        auto  zD    = compress::compress( zconf, D );
        
        std::cout << " / " << compress::compressed_size( zD );

        auto  T     = blas::copy( D );

        compress::decompress( zD, T );

        blas::add( -1, D, T );
        
        std::cout << " / " << blas::norm_F( T ) / n1 << std::endl;
    }// for

    
    //
    // lowrank compress
    //

    std::cout << "lowrank compression" << std::endl;
    
    auto  [ UD, SD, VD ] = blas::svd( D );

    blas::prod_diag( UD, SD, UD.ncols() );
    
    for ( double  eps = 1e-3; eps >= 1e-10; eps = eps / 10.0 )
    {
        auto  acc = relative_prec( eps );
        auto  k   = acc.trunc_rank( SD );
        
        auto  Uk  = blas::matrix< value_t >( UD, blas::range::all, blas::range( 0, k-1 ) );
        auto  Vk  = blas::matrix< value_t >( VD, blas::range::all, blas::range( 0, k-1 ) );
        auto  U   = blas::copy( Uk );
        auto  V   = blas::copy( Vk );
        
        std::cout << "  eps / k / mem / zmem / zerror : " << eps
                  << " / " << k << " / " << U.data_byte_size() + V.data_byte_size();

        auto  zcfU = compress::get_config( relative_prec( eps ), U );
        auto  zcfV = compress::get_config( relative_prec( eps ), V );
        auto  zU   = compress::compress( zcfU, U );
        auto  zV   = compress::compress( zcfV, V );
        
        std::cout << " / " << compress::compressed_size( zU ) + compress::compressed_size( zV );

        auto  TU   = blas::copy( U );
        auto  TV   = blas::copy( V );

        compress::decompress( zU, TU );
        compress::decompress( zV, TV );

        auto  T    = blas::prod( TU, blas::adjoint( TV ) );
        
        blas::add( -1, D, T );
        
        std::cout << " / " << blas::norm_F( T ) / n1 << std::endl;
    }// for
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    // {
    //     // test_compress< value_t >( "D64.mat" );
    //     test_compress< value_t >( "D128.mat" );
    //     // test_compress< value_t >( "D256.mat" );
    //     // test_compress< value_t >( "D512.mat" );
    //     // test_compress< value_t >( "D1024.mat" );
    //     test_compress< value_t >( "D2048.mat" );
    //     // test_compress< value_t >( "D4096.mat" );
    //     // test_compress< value_t >( "D8192.mat" );
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

        // {
        //     //
        //     // test matrices from --config ../laplace.conf --grid sphere-7
        //     //
            
        //     {
        //         auto  D = pcoeff.build( is(99968,100031), is(105472,105727) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D64" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(11904,12031), is(74112,74239) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D128" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(104960,105215), is(105472,105727) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D256" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(112128,112639), is(59392,59903) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D512" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(106496,107519), is(59392,59903) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D1024" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(34816,36863), is(45056,47103) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D2048" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(90112,94207), is(28672,32767) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D4096" );
        //     }// if

        //     {
        //         auto D = pcoeff.build( is(0,8191), is(106496,114687) );
        //         io::matlab::write( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > )->blas_mat(), "D8192" );
        //     }// if
        // }
        
        tic = timer::now();

        if ( cmdline::capprox == "hca" )
        {
            if constexpr ( problem_t::supports_hca )
            {
                std::cout << "    using HCA" << std::endl;
                
                auto  hcagen = problem->hca_gen_func( *ct );
                auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
                auto  hcalr  = bem::hca_lrapx( hca );
                
                A = impl::matrix::build( bct->root(), pcoeff, hcalr, acc, false, nseq );
            }// if
            else
                cmdline::capprox = "default";
        }// if
        else if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
        {
            std::cout << "    using ACA" << std::endl;

            auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, acalr, acc, false, nseq );
        }// else
        else if ( cmdline::capprox == "dense" )
        {
            std::cout << "    using dense" << std::endl;

            auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, dense, acc, false, nseq );
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

        auto  mem_Ac = Ac->data_byte_size();
        
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

        auto  mem_Ac = Ac->data_byte_size();
        
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

    auto        zA    = impl::matrix::copy_compressible( *A );
    const auto  delta = cmdline::eps; // norm_A * cmdline::eps / std::sqrt( double(A->nrows()) * double(A->ncols()) );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "δ = " << boost::format( "%.2e" ) % delta
              << ", "
              << hlr::compress::provider << ')'
              << term::reset << std::endl;

    {
        // auto  lacc = local_accuracy( delta );
        auto  lacc  = relative_prec( Hpro::frobenius_norm, delta );
        auto  niter = std::max( nbench, 1u );
        
        runtime.clear();
        
        for ( uint  i = 0; i < niter; ++i )
        {
            tic = timer::now();
    
            seq::matrix::compress( *zA, lacc );
            // impl::matrix::compress( *zA, lacc );

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
    
            // impl::matrix::decompress( *zB );
            seq::matrix::decompress( *zB );
            
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
    // H-matrix vector multiplication
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
    
        double  t_orig       = 0.0;
        double  t_compressed = 0.0;
        auto    y_ref        = std::unique_ptr< vector::scalar_vector< value_t > >();

        //
        // uncompressed
        //
        
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
    
                for ( int j = 0; j < nmvm; ++j )
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
            
            std::cout << "    flops  = " << format_flops( flops_h, min( runtime ) ) << std::endl;
            
            y_ref = std::move( y );
        }

        //
        // compressed
        //
        {
            runtime.clear();
            
            std::cout << "  "
                      << term::bullet << term::bold
                      << "compressed"
                      #if defined(HLR_HAS_ZBLAS_DIRECT)
                      << " (zblas)"
                      #endif
                      << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( zA->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( zA->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < nmvm; ++j )
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

            std::cout << "    flops  = " << format_flops( flops_h, min( runtime ) ) << std::endl;
            
            auto  diff = y_ref->copy();

            diff->axpy( value_t(-1), y.get() );

            const auto  error = diff->norm2();
            
            std::cout << "    error  = " << format_error( error, error / y_ref->norm2() ) << std::endl;
        }
    }// if
}
    
