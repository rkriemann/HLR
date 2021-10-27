//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <fstream>
#include <limits>

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/utils/io.hh>

#include <hlr/utils/eps_printer.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

#if defined(HAS_UNIVERSAL)
template < uint  bitsize,
           uint  expsize >
void
run_posit ( hpro::TMatrix &  A,
            hpro::TMatrix &  A_full,
            const double     norm_A )
{
    auto  A_posit   = impl::matrix::copy( A );
    auto  mem_posit = impl::matrix::convert_posit< bitsize, expsize >( *A_posit );
    auto  diff      = matrix::sum( hpro::real(1), A_full, hpro::real(-1), *A_posit );
    auto  error     = hlr::norm::spectral( *diff, true, 1e-4 );
    
    std::cout << "      " << bitsize << "/" << expsize << " : "
              << format_error( error ) << " / "
              << format_error( error / norm_A ) << " / "
              << format_mem( mem_posit ) << std::endl;
}
#endif

//
// replace leaves by generic versions
//
void
convert_generic ( hpro::TMatrix &  M );

#if defined(HAS_ZFP)
//
// compress data in generic matrices
//
void
compress ( hpro::TMatrix &     M,
           const zfp_config &  config );
#endif

//
// print matrix <M> to file <filename>
//
void
print_prec ( const hpro::TMatrix &  M,
             const double           tol );


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

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    auto  acc     = gen_accuracy();
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

    std::cout << "  " << term::bullet << term::bold << "nearfield" << term::reset << std::endl;
    
    auto  tic     = timer::now();
    auto  A_nf    = impl::matrix::build_nearfield( bct->root(), pcoeff, nseq );
    auto  toc     = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A_nf->nrows() << " × " << A_nf->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A_nf->byte_size() ) << std::endl;

    // auto  norm_nf  = norm::spectral( *A_nf );
    auto  norm_nf  = norm::frobenius( *A_nf );

    std::cout << "    |A_nf| = " << format_norm( norm_nf ) << std::endl;

    auto  delta   = norm_nf * hlr::cmdline::eps / A_nf->nrows();
    // auto  acc2    = hpro::absolute_prec( delta );
    auto  acc2    = local_accuracy( delta );

    std::cout << "  " << term::bullet << term::bold << "H-matrix, ε = " << delta << term::reset << std::endl;
    
    auto  lrapx   = bem::aca_lrapx< hpro::TPermCoeffFn< value_t > >( pcoeff );
    // auto  lrapx   = hpro::TDenseLRApx< value_t >( & pcoeff );

    tic = timer::now();
    
    auto  A       = impl::matrix::build( bct->root(), pcoeff, lrapx, acc2, nseq );
    
    toc = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A ) ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );
    
    print_prec( *A, acc2.abs_eps() );

    std::cout << "  " << term::bullet << term::bold << "exact matrix" << term::reset << std::endl;

    auto  acc3    = hpro::fixed_prec( 1e-12 );
    // auto  exact   = std::make_unique< hpro::TSVDLRApx< value_t > >( & pcoeff );
    auto  exact   = std::make_unique< hpro::TDenseLRApx< value_t > >( & pcoeff );

    tic = timer::now();
    
    auto  A_full  = impl::matrix::build( bct->root(), pcoeff, *exact, acc3, nseq );

    toc = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A_full->byte_size() ) << std::endl;

    auto  norm_A  = norm::spectral( *A_full );
    auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), *A_full );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );

    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;

    convert_generic( *A );

    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    // impl::matrix::convert_prec< float, value_t >( *A );
    
    // std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    // auto  norm_A = hlr::norm::spectral( *A, true, 1e-4 );

    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A_full ) ) << std::endl;
    std::cout << "    |A|_2  = " << format_norm( norm_A ) << std::endl;


    //
    // standard single and half compression
    //
    
    {
        std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

        using single_t = math::decrease_precision_t< value_t >;

        auto  A2   = impl::matrix::copy( *A );
        auto  mem2 = impl::matrix::convert_prec< single_t, value_t >( *A2 );
            
        std::cout << "      mem    = " << format_mem( mem2 ) << std::endl;
            
        auto  diff  = matrix::sum( value_t(1), *A_full, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "      error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    }

    #if defined(HAS_HALF)
    {
        std::cout << "    " << term::bullet << term::bold << "half precision" << term::reset << std::endl;

        using single_t = math::decrease_precision_t< value_t >;
        using half_t   = math::decrease_precision_t< single_t >;

        auto  A2   = impl::matrix::copy( *A );
        auto  mem2 = impl::matrix::convert_prec< half_t, value_t >( *A2 );
            
        std::cout << "      mem    = " << format_mem( mem2 ) << std::endl;
            
        auto  diff  = matrix::sum( value_t(1), *A_full, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "      error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    }
    #endif

    //
    // ZFP compression
    //

    #if defined(HAS_ZFP)
    
    std::cout << "    " << term::bullet << term::bold << "ZFP compression" << term::reset << std::endl;

    for ( uint  rate = 32; rate >= 8; rate -= 4 )
    // for ( double  rate = 1e-20; rate <= 1e-2; rate *= 10 )
    {
        auto  A_zfp   = impl::matrix::copy( *A );
        // auto  config  = zfp_config_reversible();
        auto  config  = zfp_config_rate( rate, false );
        // auto  config  = zfp_config_precision( rate );
        // auto  config  = zfp_config_accuracy( rate );

        auto  mem_zfp = impl::matrix::convert_zfp< value_t >( *A_zfp, config, 0 );
        auto  diff    = matrix::sum( value_t(1), *A_full, value_t(-1), *A_zfp );
        auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    
        std::cout << "      " << boost::format( "%2d" ) % rate << " / "
            // std::cout << "      " << boost::format( "%.1e" ) % rate << " / "
                  << format_error( error ) << " / "
                  << format_error( error / norm_A ) << " / "
                  << format_mem( mem_zfp ) << std::endl;
    }// for

    for ( uint  rate = 32; rate >= 8; rate -= 4 )
    {
        auto  A_zfp   = impl::matrix::copy( *A );
        auto  config  = zfp_config_rate( rate, false );

        compress( *A_zfp, config );
        
        auto  mem_zfp = A_zfp->byte_size();
        auto  diff    = matrix::sum( value_t(1), *A_full, value_t(-1), *A_zfp );
        auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    
        std::cout << "      " << boost::format( "%2d" ) % rate << " / "
                  << format_error( error ) << " / "
                  << format_error( error / norm_A ) << " / "
                  << format_mem( mem_zfp ) << std::endl;

        // std::cout << "      " << boost::format( "%2d" ) % rate << " / "
        //           << format_mem( mem_zfp ) << std::endl;

        // auto  x  = A_zfp->col_vector();
        // auto  y1 = A_zfp->row_vector();
        // auto  y2 = A_zfp->row_vector();

        // x->fill( 1.0 );

        // A_zfp->apply( x.get(), y1.get() );
        // A->apply( x.get(), y2.get() );

        // y1->axpy( -1.0, y2.get() );
        // std::cout << format_error( y1->norm2() ) << std::endl;
    }// for
    
    #endif

    //
    // posit number format
    //

    #if defined(HAS_UNIVERSAL)
    
    // std::cout << "    " << term::bullet << term::bold << "Posit format" << term::reset << std::endl;

    // run_posit< 64, 3 >( *A, *A_full, norm_A );
    // run_posit< 60, 3 >( *A, *A_full, norm_A );
    // run_posit< 56, 3 >( *A, *A_full, norm_A );
    // run_posit< 52, 3 >( *A, *A_full, norm_A );
    // run_posit< 48, 3 >( *A, *A_full, norm_A );
    // run_posit< 44, 3 >( *A, *A_full, norm_A );
    // run_posit< 40, 2 >( *A, *A_full, norm_A );
    // run_posit< 36, 2 >( *A, *A_full, norm_A );
    // run_posit< 32, 2 >( *A, *A_full, norm_A );
    // run_posit< 28, 2 >( *A, *A_full, norm_A );
    // run_posit< 24, 2 >( *A, *A_full, norm_A );
    // run_posit< 20, 2 >( *A, *A_full, norm_A );
    // run_posit< 16, 1 >( *A, *A_full, norm_A );
    // run_posit< 12, 1 >( *A, *A_full, norm_A );
    // run_posit<  8, 0 >( *A, *A_full, norm_A );
    
    #endif
}

//
// replace TRkMatrix by lrmatrix
//
void
convert_generic ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    convert_generic( *B->block( i, j ) );
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = ptrcast( &M, hpro::TDenseMatrix );
        auto  D  = std::make_unique< matrix::dense_matrix >( M.row_is(), M.col_is() );

        if ( M.is_complex() )
            D->set_matrix( std::move( blas::mat< hpro::complex >( *DM ) ) );
        else
            D->set_matrix( std::move( blas::mat< hpro::real >( *DM ) ) );

        DM->parent()->replace_block( DM, D.release() );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  RM = ptrcast( &M, hpro::TRkMatrix );
        auto  R  = std::make_unique< matrix::lrmatrix >( M.row_is(), M.col_is() );

        if ( M.is_complex() )
            R->set_lrmat( std::move( blas::mat_U< hpro::complex >( *RM ) ),
                          std::move( blas::mat_V< hpro::complex >( *RM ) ) );
        else
            R->set_lrmat( std::move( blas::mat_U< hpro::real >( *RM ) ),
                          std::move( blas::mat_V< hpro::real >( *RM ) ) );

        RM->parent()->replace_block( RM, R.release() );
    }// if
}

#if defined(HAS_ZFP)
//
// compress data in generic matrices
//
void
compress ( hpro::TMatrix &     M,
           const zfp_config &  config )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    compress( *B->block( i, j ), config );
            }// for
        }// for
    }// if
    else if ( matrix::is_generic_dense( M ) )
    {
        auto  D = ptrcast( &M, matrix::dense_matrix );

        D->compress( config );
    }// if
    else if ( matrix::is_generic_lowrank( M ) )
    {
        auto  R = ptrcast( &M, matrix::lrmatrix );

        R->compress( config );
    }// if
}
#endif

//
// actual print function
//
void
print_prec ( const hpro::TMatrix &  M,
             eps_printer &          prn,
             const double           tol )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_prec( * B->block( i, j ), prn, tol );
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        prn.set_rgb( 85,87,83 );
        
        prn.fill_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// if
    else
    {
        // auto  norm_M = norm::spectral( M );
        auto  norm_M = norm::frobenius( M );

        if      ( norm_M <= tol / 4e-3 )   prn.set_rgb(  52, 101, 164 ); // bfloat16
        else if ( norm_M <= tol / 5e-4 )   prn.set_rgb(  15, 210,  22 ); // fp16
        else if ( norm_M <= tol / 6e-8 )   prn.set_rgb( 252, 175,  62 ); // fp32
        else if ( norm_M <= tol / 1e-16 )  prn.set_rgb( 239,  41,  41 ); // fp64
        else                               prn.set_rgb( 164,   0,   0 ); // fp128
        
        prn.fill_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// else
}

//
// print matrix <M> to file <filename>
//
void
print_prec ( const hpro::TMatrix &  M,
             const double           tol )
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
    
