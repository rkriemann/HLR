//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <fstream>
#include <limits>

#include <half.hpp>

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/matrix/mplrmatrix.hh>
#include <hlr/utils/io.hh>

#include <hlr/utils/eps_printer.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print_prec ( const hpro::TMatrix< value_t > &  M,
             const double                      tol );

template < typename value_t >
std::tuple< size_t, size_t, size_t >
convert_mp ( const hpro::TMatrix< value_t > &  M,
             const double                      tol );

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
    
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );

    if ( hpro::verbose( 3 ) )
        io::eps::print( *bct->root(), "bct" );
    
    auto  acc     = gen_accuracy();
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

    // std::cout << "  " << term::bullet << term::bold << "nearfield" << term::reset << std::endl;
    
    // auto  A_nf    = impl::matrix::build_nearfield( bct->root(), pcoeff, nseq );
    
    // std::cout << "    done in " << format_time( toc ) << std::endl;
    // std::cout << "    dims   = " << A_nf->nrows() << " × " << A_nf->ncols() << std::endl;
    // std::cout << "    mem    = " << format_mem( A_nf->byte_size() ) << std::endl;

    // // auto  norm_nf  = norm::spectral( *A_nf );
    // auto  norm_nf  = norm::frobenius( *A_nf );

    // std::cout << "    |A_nf| = " << format_norm( norm_nf ) << std::endl;

    // auto  delta   = norm_nf * hlr::cmdline::eps / ( A_nf->nrows() / hlr::cmdline::ntile );
    // // auto  acc2    = hpro::absolute_prec( delta );
    // auto  acc2    = local_accuracy( delta );

    // std::cout << "  " << term::bullet << term::bold << "H-matrix, ε = " << delta << term::reset << std::endl;

    auto  lrapx   = bem::aca_lrapx< hpro::TPermCoeffFn< value_t > >( pcoeff );
    // auto  lrapx   = hpro::TDenseLRApx< value_t >( & pcoeff );

    tic = timer::now();
    
    auto  A       = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    // auto  A       = impl::matrix::build( bct->root(), pcoeff, lrapx, acc2, nseq );
    
    toc = timer::since( tic );

    auto  mem_A  = A->byte_size();
    auto  norm_A = impl::norm::frobenius( *A );

    // delta = norm_A * hlr::cmdline::eps / ( A_nf->nrows() / hlr::cmdline::ntile );
    // delta = hlr::cmdline::eps; //  * norm_A / (A_nf->nrows());
    auto  delta   = hlr::cmdline::eps * norm_A; // / (A->nrows());
        
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
    std::cout << "      idx  = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|    = " << format_norm( impl::norm::frobenius( *A ) ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    //////////////////////////////////////////////////////////////////////
    //
    // virtual mixedprec test (and output)
    //
    //////////////////////////////////////////////////////////////////////

    if ( false )
    {
        std::cout << "  "
                  << term::bullet << term::bold
                  << "compression ("
                  << "ε = " << boost::format( "%.2e" ) % delta
                  << ", "
                  << "mixedprec)"
                  << term::reset << std::endl;
        std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;
    
        if ( hpro::verbose( 2 ) )
            print_prec( *A, delta ); // acc2.abs_eps() );

        auto  Amp     = impl::matrix::copy( *A );
        auto  mem_mp  = convert_mp( *Amp, delta );
        auto  mem_Amp = std::get<0>( mem_mp ) + std::get<1>( mem_mp ) + std::get<2>( mem_mp );
    
        std::cout << "    mem   = " << format_mem( mem_Amp ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Amp) / double(mem_A) ) << std::endl;
        std::cout << "      FP16  " << boost::format( "%.3f" ) % ( double(std::get<0>( mem_mp )) / double(mem_A) ) << std::endl;
        std::cout << "      FP32  " << boost::format( "%.3f" ) % ( double(std::get<1>( mem_mp )) / double(mem_A) ) << std::endl;
        std::cout << "      FP64  " << boost::format( "%.3f" ) % ( double(std::get<2>( mem_mp )) / double(mem_A) ) << std::endl;
        std::cout << "    |Amp| = " << format_norm( impl::norm::frobenius( *Amp ) ) << std::endl;
        
        auto  error   = impl::norm::frobenius( 1.0, *A, -1.0, *Amp );
        
        std::cout << "    error = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    //
    // convert to mixed precision format
    //
    //////////////////////////////////////////////////////////////////////

    auto  zA = impl::matrix::copy_mixedprec( *A );

    norm_A = norm::spectral( impl::arithmetic, *A );
    
    std::cout << "  "
              << term::bullet << term::bold
              << "compression ("
              << "ε = " << boost::format( "%.2e" ) % delta
              << ", mixedprec )"
              << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    {
        runtime.clear();
        
        for ( uint  i = 0; i < std::max( nbench, 1 ); ++i )
        {
            auto  B = impl::matrix::copy( *zA );
        
            tic = timer::now();
    
            impl::matrix::compress( *B, Hpro::fixed_prec( delta ) );

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
    
    auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    auto  error = norm::spectral( impl::arithmetic, *diff );

    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    norm_A = impl::norm::frobenius( *A );
    error  = impl::norm::frobenius( 1, *A, -1, *zA );

    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;
    
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

        auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zB );

        error = norm::spectral( impl::arithmetic, *diffB );
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
    
    // std::cout << "  " << term::bullet << term::bold << "exact matrix" << term::reset << std::endl;

    // auto  acc3    = hpro::fixed_prec( 1e-12 );
    // // auto  exact   = std::make_unique< hpro::TSVDLRApx< value_t > >( & pcoeff );
    // auto  exact   = std::make_unique< hpro::TDenseLRApx< value_t > >( & pcoeff );

    // tic = timer::now();
    
    // auto  A_full  = impl::matrix::build( bct->root(), pcoeff, *exact, acc3, nseq );

    // toc = timer::since( tic );
    
    // std::cout << "    done in " << format_time( toc ) << std::endl;
    // std::cout << "    mem    = " << format_mem( A_full->byte_size() ) << std::endl;

    // auto  norm_A  = norm::spectral( *A_full );
    // auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), *A_full );
    // auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );

    // std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;

    // // auto  norm_A = hlr::norm::spectral( *A, true, 1e-4 );

    // std::cout << "    |A|    = " << format_norm( norm::frobenius( *A_full ) ) << std::endl;
    // std::cout << "    |A|_2  = " << format_norm( norm_A ) << std::endl;


    // //
    // // standard single and half compression
    // //
    
    // {
    //     std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

    //     using single_t = math::decrease_precision_t< value_t >;

    //     auto  A2   = impl::matrix::copy( *A );
    //     auto  mem2 = impl::matrix::convert_prec< single_t, value_t >( *A2 );
            
    //     std::cout << "      mem    = " << format_mem( mem2 ) << std::endl;
            
    //     auto  diff  = matrix::sum( value_t(1), *A_full, value_t(-1), *A2 );
    //     auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
    //     std::cout << "      error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    // }
}

//
// actual print function
//
template < typename value_t >
void
print_prec ( const hpro::TMatrix< value_t > &  M,
             eps_printer &                     prn,
             const double                      tol )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    const uint  col_fp8[3]  = { 255,255,255 }; // white
    const uint  col_fp16[3] = { 252,233,79 }; // yellow
    const uint  col_bf16[3] = { 252,233,79 }; // yellow
    const uint  col_tf32[3] = { 252,233,79 };  // yellow
    const uint  col_fp32[3] = { 114,159,207 }; // blue
    const uint  col_fp64[3] = { 239,41,41 };   // red
    
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_prec( * B->block( i, j ), prn, tol );
            }// for
        }// for
    }// if
    else
    {
        auto  norm_M = norm::frobenius( M );
        auto  S      = blas::vector< real_t >( std::min( M.nrows(), M.ncols() ) );

        if ( is_dense( M ) )
        {
            auto  D = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
            auto  A = blas::copy( blas::mat( D ) );
            
            blas::sv( A, S );
            for ( uint  i = 0; i < S.length(); ++i ) S(i) = 1e10;
            // for ( uint  i = 1; i < S.length(); ++i ) S(i) = 0.0;
        }// if
        else if ( is_lowrank( M ) )
        {
            auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
            auto  U = blas::copy( blas::mat_U( R ) );
            auto  V = blas::copy( blas::mat_V( R ) );

            blas::sv( U, V, S );
        }// if

        const uint  rank = S.length();
        
        #if 0

        // if      ( S(rank-1) / S(0) >= 6.2e-2 ) { prn.set_rgb( col_fp8[0],  col_fp8[1],  col_fp8[2]  ); std::cout << "FP8"  << std::endl; }
        // else if ( S(rank-1) / S(0) >= 3.9e-3 ) { prn.set_rgb( col_bf16[0], col_bf16[1], col_bf16[2] ); std::cout << "BF16" << std::endl; }
        // else if ( S(rank-1) / S(0) >= 4.9e-4 ) { prn.set_rgb( col_tf32[0], col_tf32[1], col_tf32[2] ); std::cout << "TF32" << std::endl; }
        // else if ( S(rank-1) / S(0) >= 6.0e-8 ) { prn.set_rgb( col_fp32[0], col_fp32[1], col_fp32[2] ); std::cout << "FP32" << std::endl; }
        // else                                   { prn.set_rgb( col_fp64[0], col_fp64[1], col_fp64[2] ); std::cout << "FP64" << std::endl; }

        std::cout << norm_M << " / " << norm_M * 4.9e-4 << " / " << norm_M * 6.0e-8 << " / " << tol << std::endl;
        
        if      ( norm_M * 4.9e-4 < tol ) { prn.set_rgb( col_bf16[0], col_bf16[1], col_bf16[2] ); }
        else if ( norm_M * 6.0e-8 < tol ) { prn.set_rgb( col_fp32[0], col_fp32[1], col_fp32[2] ); }
        else                              { prn.set_rgb( col_fp64[0], col_fp64[1], col_fp64[2] ); }
            
        // if ( is_dense( M ) )
        // { prn.set_rgb( col_fp64[0], col_fp64[1], col_fp64[2] ); std::cout << "FP64" << std::endl; }
        // else if ( std::max( M.row_is().first(), M.col_is().first() ) - std::min( M.row_is().first(), M.col_is().first() ) > 1000 )
        // { prn.set_rgb( col_bf16[0], col_bf16[1], col_bf16[2] ); std::cout << "BF16" << std::endl; }
        // else if ( std::max( M.row_is().first(), M.col_is().first() ) - std::min( M.row_is().first(), M.col_is().first() ) > 350 )
        // { prn.set_rgb( col_fp32[0], col_fp32[1], col_fp32[2] ); std::cout << "FP32" << std::endl; }
        // else
        // { prn.set_rgb( col_fp64[0], col_fp64[1], col_fp64[2] ); std::cout << "FP64" << std::endl; }
        
        #else
        
        int  i = rank-1;

        auto  test_prec = [&i,&S,tol] ( double  u )
        {
            uint  nprec = 0;
            
            while ( i >= 0 )
            {
                if ( S(i) <= tol / u ) nprec++;
                else                   break;
                --i;
            }// while

            return nprec;
        };
            
        // const uint  n_fp8  = test_prec( 6.2e-2 );
        // const uint  n_bf16 = test_prec( 3.9e-3 );
        // const uint  n_tf32 = test_prec( 4.9e-4 );
        const uint  n_fp16 = test_prec( 4.9e-4 );
        const uint  n_fp32 = test_prec( 6.0e-8 );
        const uint  n_fp64 = i+1; // std::max< int >( i, 0 );

        if ( hpro::verbose( 3 ) && is_lowrank( M ) )
            std::cout << norm_M << " / " << n_fp16 << " / " << n_fp32 << " / " << n_fp64 << std::endl;
            // std::cout << n_fp8 << " / " << n_bf16 << " / " << n_tf32 << " / " << n_fp32 << " / " << n_fp64 << std::endl;

        uint    col_bg[3]   = { 0, 0, 0 };

        // if      ( n_tf32 + n_bf16 + n_fp8 > n_fp32 ) { prn.set_rgb( col_bf16[0], col_bf16[1], col_bf16[2] ); }
        // else if ( n_fp32 > n_fp64 )                  { prn.set_rgb( col_fp32[0], col_fp32[1], col_fp32[2] ); }
        // else                                         { prn.set_rgb( col_fp64[0], col_fp64[1], col_fp64[2] ); }
        
        for ( int  c = 0; c < 3; ++c )
            col_bg[c] = std::min< uint >( 255, uint( ( n_fp16 * col_fp16[c] +
                                                       n_fp32 * col_fp32[c] +
                                                       n_fp64 * col_fp64[c] ) / double(rank) ) );
            // col_bg[c] = std::min< uint >( 255, uint( ( n_fp8  * col_fp8[c]  +
            //                                            n_bf16 * col_bf16[c] +
            //                                            n_tf32 * col_tf32[c] +
            //                                            n_fp32 * col_fp32[c] +
            //                                            n_fp64 * col_fp64[c] ) / double(rank) ) );

        prn.set_rgb( col_bg[0], col_bg[1], col_bg[2] );

        // if      ( n_bf16 >  0 && n_tf32 >  0 && n_fp32 >  0 && n_fp64 >  0 ) prn.set_rgb( 206,92,0 );
        // else if ( n_bf16 >  0 && n_tf32 >  0 && n_fp32 >  0                ) prn.set_rgb( 196,160,0 );
        // else if ( n_bf16 >  0 && n_tf32 >  0                               ) prn.set_rgb( 252,175,62 );
        // else if ( n_bf16 >  0                                              ) prn.set_rgb( 252,233,79 );
        // else if ( n_bf16 == 0 && n_tf32 >  0 && n_fp32 >  0 && n_fp64 >  0 ) prn.set_rgb( 173,127,168 );
        // else if ( n_bf16 == 0 && n_tf32 >  0 && n_fp32 >  0                ) prn.set_rgb( 117,80,123 );
        // else if ( n_bf16 == 0 && n_tf32 >  0                               ) prn.set_rgb( 173,127,168 );
        // else if ( n_bf16 == 0 && n_tf32 == 0 && n_fp32 >  0 && n_fp64 >  0 ) prn.set_rgb( 78,154,6 );
        // else if ( n_bf16 == 0 && n_tf32 == 0 && n_fp32 >  0                ) prn.set_rgb( 138,226,52 );
        // else                                                                 prn.set_rgb( 204,0,0 );

        #endif
        
        prn.fill_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );

        // // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// if
}

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print_prec ( const hpro::TMatrix< value_t > &  M,
             const double                      tol )
{
    std::ofstream  out( "prec.eps" );
    eps_printer    prn( out );

    const auto   max_size = std::max( std::max( M.nrows(), M.ncols() ), size_t(1) );
    const auto   min_size = std::max( std::min( M.nrows(), M.ncols() ), size_t(1) );
    const auto   width    = ( M.ncols() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    const auto   height   = ( M.nrows() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    
    prn.begin( width, height );
    prn.scale( double(width)  / double(M.ncols()),
               double(height) / double(M.nrows()) );
    prn.translate( - double(M.col_ofs()),
                   - double(M.row_ofs()) );
    prn.set_line_width( 0.1 );
    print_prec( M, prn, tol );
    prn.end();
}
    
//
// return memory of mixed precision storage
//
template < typename value_t >
std::tuple< size_t, size_t, size_t >
convert_mp ( const hpro::TMatrix< value_t > &  M,
             const double                      tol )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    if ( is_blocked( M ) )
    {
        auto    B = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto    s = std::tuple< size_t, size_t, size_t >( 0, 0, 0 );
        // size_t  s = 4+8+8+8 + sizeof(Hpro::TProcSet) + sizeof(Hpro::matform_t);

        // s += 8 + 8 + 4 + 4 + sizeof( std::vector< Hpro::TMatrix< value_t > * > );
        // s += B->nblock_rows() * B->nblock_cols() * sizeof(Hpro::TMatrix< value_t >*);
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                {
                    auto  s_ij = convert_mp( * B->block( i, j ), tol );

                    std::get<0>( s ) += std::get<0>( s_ij );
                    std::get<1>( s ) += std::get<1>( s_ij );
                    std::get<2>( s ) += std::get<2>( s_ij );
                }// if
            }// for
        }// for

        return s;
    }// if
    else
    {
        auto  S  = blas::vector< real_t >( std::min( M.nrows(), M.ncols() ) );
        auto  RU = blas::matrix< value_t >();
        auto  RV = blas::matrix< value_t >();

        if ( is_dense( M ) )
        {
            return { 0, 0, sizeof(double) * M.nrows() * M.ncols() };
            // return M.byte_size();
        }// if
        else if ( is_lowrank( M ) )
        {
            auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
            auto  U = blas::copy( blas::mat_U( R ) );
            auto  V = blas::copy( blas::mat_V( R ) );

            RU = blas::mat_U( R );
            RV = blas::mat_V( R );
            
            blas::sv( U, V, S );
        }// if

        const uint  rank = S.length();
        
        #if 0

        if      ( norm_M * 4.9e-4 < tol ) { s /= 4; }
        else if ( norm_M * 6.0e-8 < tol ) { s /= 2; }
            
        #else
        
        int  i = rank-1;

        auto  test_prec = [&i,&S,tol] ( double  u )
        {
            uint  nprec = 0;
            
            while ( i >= 0 )
            {
                if ( S(i) <= tol / u ) nprec++;
                else                   break;
                --i;
            }// while

            return nprec;
        };
            
        const uint  n_fp16 = test_prec( 4.9e-4 );
        const uint  n_fp32 = test_prec( 6.0e-8 );
        const uint  n_fp64 = i+1; // std::max< int >( i+1, 0 );
        size_t      s      = 0;

        HLR_ASSERT( n_fp64 >= 0 );
        HLR_ASSERT( n_fp16 + n_fp32 + n_fp64 == rank );
        
        // s  = M.byte_size();
        // s -= sizeof(double) * rank * ( M.nrows() + M.ncols() );
        // s += 2 * n_fp16 * ( M.nrows() + M.ncols() );
        // s += 4 * n_fp32 * ( M.nrows() + M.ncols() );
        // s += 8 * n_fp64 * ( M.nrows() + M.ncols() );
        
        auto  nrows = M.nrows();
        auto  ncols = M.ncols();

        using half = half_float::half;

        i = n_fp64;

        if constexpr ( ! Hpro::is_complex_type_v< value_t > )
        {
            for ( uint  pos = 0; pos < n_fp32; ++pos, ++i )
            {
                for ( uint  j = 0; j < nrows; ++j )
                    RU(j,i) = double( float( RU(j,i) ) );
                for ( uint  j = 0; j < ncols; ++j )
                    RV(j,i) = double( float( RV(j,i) ) );
            }// for
                  
            for ( uint  pos = 0; pos < n_fp16; ++pos, ++i )
            {
                for ( uint  j = 0; j < nrows; ++j )
                    RU(j,i) = double( half( RU(j,i) ) );
                for ( uint  j = 0; j < ncols; ++j )
                    RV(j,i) = double( half( RV(j,i) ) );
            }// for
        }// if

        return { 2 * n_fp16 * ( M.nrows() + M.ncols() ),
                 4 * n_fp32 * ( M.nrows() + M.ncols() ),
                 8 * n_fp64 * ( M.nrows() + M.ncols() ) };

        // return s;
        
        #endif
    }// if

    return { 0, 0, 0 };
}

