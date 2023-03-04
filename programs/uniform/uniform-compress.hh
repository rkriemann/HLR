//
// Project     : HLR
// Module      : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlib-config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#endif

#if defined( HAS_H2 )
#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/algebra/mat_conv.hh>
#endif

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/info.hh>
#include <hlr/bem/aca.hh>
#include <hlr/approx/randsvd.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

bool within ( double  arg, double ul, double ll )
{
    return ( arg >= ll ) && ( arg <= ul );
}

#if defined(HAS_UNIVERSAL)
template < uint bitsize,
           uint expsize >
void
run_posit ( hpro::TMatrix &  A_ref,
            hpro::TMatrix &  A,
            const double     norm_A )
{
    auto  A_posit = impl::matrix::copy( A );
    auto  mem_A   = impl::matrix::convert_posit< bitsize, expsize >( *A_posit );
    auto  diff    = matrix::sum( hpro::real(1), A_ref, hpro::real(-1), *A_posit );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    
    std::cout << "      "
              << boost::format( "%2d" ) % bitsize << "/" << expsize << " : "
              << format_error( error, error / norm_A ) << " / "
              << format_mem( mem_A ) << std::endl;
}

template < uint bitsize,
           uint expsize,
           typename value_t >
void
run_posit ( hpro::TMatrix &                     A_ref,
            hpro::TMatrix &                     A,
            matrix::cluster_basis< value_t > &  rowcb,
            matrix::cluster_basis< value_t > &  colcb,
            const double                        norm_A )
{
    auto  A_posit     = impl::matrix::copy( A );
    auto  rowcb_posit = matrix::copy< value_t >( rowcb );
    auto  colcb_posit = matrix::copy< value_t >( colcb );

    matrix::replace_cluster_basis( *A_posit, *rowcb_posit, *colcb_posit );
    
    auto  mem_A     = impl::matrix::convert_posit< bitsize, expsize >( *A_posit );
    auto  mem_rowcb = impl::matrix::convert_posit< bitsize, expsize, value_t >( *rowcb_posit );
    auto  mem_colcb = impl::matrix::convert_posit< bitsize, expsize, value_t >( *colcb_posit );
    
    auto  diff      = matrix::sum( hpro::real(1), A_ref, hpro::real(-1), *A_posit );
    auto  error     = hlr::norm::spectral( *diff, true, 1e-4 );
    
    std::cout << "      "
              << boost::format( "%2d" ) % bitsize << "/" << expsize << " : "
              << format_error( error, error / norm_A ) << " / "
              << format_mem( mem_A, mem_rowcb, mem_colcb ) << std::endl;
}

template < uint bitsize,
           uint expsize,
           typename value_t >
void
run_posit ( hpro::TMatrix &                   A_ref,
            hpro::TMatrix &                   A,
            hpro::TClusterBasis< value_t > &  rowcb,
            hpro::TClusterBasis< value_t > &  colcb,
            const double                      norm_A )
{
    auto  A_posit     = impl::matrix::copy( A );
    auto  rowcb_posit = rowcb.copy();
    auto  colcb_posit = colcb.copy();
    
    hpro::replace_cluster_basis( *A_posit, *rowcb_posit, *colcb_posit );
    
    auto  mem_A       = impl::matrix::convert_posit< bitsize, expsize >( *A_posit );
    auto  mem_rowcb   = impl::matrix::convert_posit< bitsize, expsize, value_t >( *rowcb_posit );
    auto  mem_colcb   = impl::matrix::convert_posit< bitsize, expsize, value_t >( *colcb_posit );
    
    auto  diff        = matrix::sum( hpro::real(1), A_ref, hpro::real(-1), *A_posit );
    auto  error       = hlr::norm::spectral( *diff, true, 1e-4 );
    
    std::cout << "      "
              << boost::format( "%2d" ) % bitsize << "/" << expsize << " : "
              << format_error( error, error / norm_A ) << " / "
              << format_mem( mem_A, mem_rowcb, mem_colcb ) << std::endl;
}
#endif

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    auto  runtime = std::vector< double >();
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    
    auto  acc_ref = hpro::fixed_prec( 1e-12 );
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );

    auto  A_ref  = std::unique_ptr< hpro::TMatrix >();
    
    tic = timer::now();

    if ( cmdline::matrixfile != "" )
    {
        A_ref = io::hpro::read( cmdline::matrixfile );
    }// if
    else
    {
        A_ref = impl::matrix::build( bct->root(), pcoeff, lrapx, acc_ref, nseq );
        // io::hpro::write( *A, "A.hm" );
    }// else
    
    toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << term::bold << A_ref->nrows() << " × " << A_ref->ncols() << term::reset << std::endl;
    std::cout << "    mem    = " << format_mem( A_ref->byte_size() ) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::eps::print( *A_ref, "A_ref", "noid" );

    const auto  normA = hlr::norm::spectral( *A_ref, true, 1e-4 );

    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A_ref ) ) << std::endl;

    for ( double  eps = 1e-4 ; eps >= 1e-11 ; eps = eps / 100 )
    {
        std::cout << std::endl
                  << "─────────────── " << term::bold << "ε = " << boost::format( "%.2e" ) % eps << term::reset << " ────────────────" << std::endl
                  << std::endl;

        auto  acc = hpro::fixed_prec( eps );
        
        #if defined(HAS_ZFP)
        const uint  zfp_ub = (   eps <= 1e-10 ? 40
                               : eps <= 1e-8 ? 34
                               : eps <= 1e-6 ? 28
                               : eps <= 1e-4 ? 20
                               : 20 );
        
        const uint  zfp_lb = (   eps <= 1e-10 ? 32
                               : eps <= 1e-8 ? 26
                               : eps <= 1e-6 ? 20
                               : eps <= 1e-4 ? 14
                               : 8 );
        #endif
        
        //
        // H-matrix approximation
        //

        if ( false )
        {
            std::cout << term::bullet << term::bold << "truncated H-matrix" << term::reset << std::endl;
    
            auto  A = impl::matrix::copy( *A_ref, acc );

            std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

            if ( hpro::verbose( 3 ) )
                io::eps::print( *A, "A", "noid" );
    
            {
                auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *A );
                auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
    
                std::cout << "    error  = " << format_error( error, error / normA ) << std::endl;
            }
    
            // assign clusters since needed for cluster bases
            seq::matrix::assign_cluster( *A, *bct->root() );
    
            if ( true )
            {
                std::cout << "  " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

                using single_t = math::decrease_precision_t< value_t >;

                auto  As     = impl::matrix::copy( *A );
                auto  mem_A  = impl::matrix::convert_prec< single_t, value_t >( *As );
            
                std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
            
                {
                    auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *As );
                    auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                    std::cout << "    error  = " << format_error( error, error / normA ) << std::endl;
                }
            }// if
    
            #if defined(HAS_ZFP)
            const uint  zfp_ub = (   eps <= 1e-10 ? 40
                                     : eps <= 1e-8 ? 34
                                     : eps <= 1e-6 ? 28
                                     : eps <= 1e-4 ? 20
                                     : 20 );
        
            const uint  zfp_lb = (   eps <= 1e-10 ? 32
                                     : eps <= 1e-8 ? 26
                                     : eps <= 1e-6 ? 20
                                     : eps <= 1e-4 ? 14
                                     : 8 );
        
            if ( true )
            {
                std::cout << "  " << term::bullet << term::bold << "ZFP compression" << term::reset << std::endl;

                for ( uint  rate = zfp_ub; rate >= zfp_lb; rate -= 2 )
                {
                    auto  A_zfp  = impl::matrix::copy( *A );
                    auto  config = zfp_config_rate( rate, false );
                    auto  mem_A  = impl::matrix::convert_zfp< value_t >( *A_zfp, config );
                
                    auto  diff      = matrix::sum( value_t(1), *A_ref, value_t(-1), *A_zfp );
                    auto  error     = hlr::norm::spectral( *diff, true, 1e-4 );
    
                    std::cout << "      " << boost::format( "%2d" ) % rate << " / "
                              << format_error( error ) << " / "
                              << format_error( error / normA ) << " / "
                              << format_mem( mem_A ) << std::endl;
                }// for
            }// if
            #endif
        
            #if defined(HAS_UNIVERSAL)
            if ( true )
            {
                std::cout << "  " << term::bullet << term::bold << "using posits" << term::reset << std::endl;

                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 46, 3 >( *A_ref, *A, normA );
                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 44, 3 >( *A_ref, *A, normA );
                if ( within( eps, 1e-10, 1e-12 ) ) run_posit< 42, 3 >( *A_ref, *A, normA );
                if ( within( eps, 1e-10, 1e-11 ) ) run_posit< 40, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-9,  1e-11 ) ) run_posit< 38, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-8,  1e-10 ) ) run_posit< 36, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 34, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 32, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 30, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 28, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-6,  1e-7  ) ) run_posit< 26, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-5,  1e-6  ) ) run_posit< 24, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-4,  1e-6  ) ) run_posit< 22, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-4,  1e-5  ) ) run_posit< 20, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-3,  1e-4  ) ) run_posit< 18, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-2,  1e-4  ) ) run_posit< 16, 2 >( *A_ref, *A, normA );
                if ( within( eps, 1e-1,  1e-4  ) ) run_posit< 16, 1 >( *A_ref, *A, normA );
                if ( within( eps, 1e-0,  1e-3  ) ) run_posit< 14, 1 >( *A_ref, *A, normA );
            }// if
            #endif
        }// if

        //////////////////////////////////////////////////////////////////////
        //
        // directly build uniform matrix
        //
        //////////////////////////////////////////////////////////////////////

        auto  apx = approx::SVD< value_t >();

        {
            std::cout << term::bullet << term::bold << "uniform H-matrix (rec)" << term::reset << std::endl;
    
            tic = timer::now();
    
            auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A_ref, apx, acc, nseq );

            toc = timer::since( tic );
            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

            if ( hpro::verbose( 3 ) )
            {
                io::eps::print( *A2, "A2", "noid" );
                io::eps::print( *rowcb, "rowcb2" );
                io::eps::print( *colcb, "colcb2" );
            }// if
        
            {
                auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2 );
                auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                std::cout << "    error  = " << format_error( error, error / normA ) << std::endl;
            }

            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

                using single_t = math::decrease_precision_t< value_t >;

                auto  rowcbs = matrix::copy< single_t >( *rowcb );
                auto  colcbs = matrix::copy< single_t >( *colcb );
            
                auto  rowcbv = matrix::copy< value_t >( *rowcbs );
                auto  colcbv = matrix::copy< value_t >( *colcbs );

                auto  A3     = impl::matrix::copy( *A2 );
                auto  mem_A  = impl::matrix::convert_prec< single_t, value_t >( *A3 );
            
                std::cout << "      mem    = " << format_mem( mem_A, rowcbs->byte_size(), colcbs->byte_size() ) << std::endl;
            
                matrix::replace_cluster_basis( *A3, *rowcbv, *colcbv );
            
                {
                    auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2 );
                    auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                    std::cout << "      error  = " << format_error( error, error / normA ) << std::endl;
                }
            }

            #if defined(HAS_HALF)
            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "half precision" << term::reset << std::endl;

                using single_t = math::decrease_precision_t< value_t >;
                using half_t   = math::decrease_precision_t< single_t >;
            
                auto  rowcbh = matrix::copy< half_t >( *rowcb );
                auto  colcbh = matrix::copy< half_t >( *colcb );
            
                auto  rowcbs = matrix::copy< single_t >( *rowcbh );
                auto  colcbs = matrix::copy< single_t >( *colcbh );

                auto  rowcbv = matrix::copy< value_t >( *rowcbs );
                auto  colcbv = matrix::copy< value_t >( *colcbs );

                std::cout << "      mem    = " << format_mem( A2->byte_size(), rowcbh->byte_size(), colcbh->byte_size() ) << std::endl;
            
                matrix::replace_cluster_basis( *A2, *rowcbv, *colcbv );
            
                {
                    auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2 );
                    auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                    std::cout << "      error  = " << format_error( error, error / normA ) << std::endl;
                }
            }
            #endif

            #if defined(HAS_ZFP)
            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "ZFP compression" << term::reset << std::endl;

                for ( uint  rate = zfp_ub; rate >= zfp_lb; rate -= 2 )
                {
                    auto  A2_zfp    = impl::matrix::copy( *A2 );
                    auto  rowcb_zfp = matrix::copy< value_t >( *rowcb );
                    auto  colcb_zfp = matrix::copy< value_t >( *colcb );
                    auto  config    = zfp_config_rate( rate, false );

                    matrix::replace_cluster_basis( *A2_zfp, *rowcb_zfp, *colcb_zfp );
                
                    auto  mem_A     = impl::matrix::convert_zfp< value_t >( *A2_zfp, config );
                    auto  mem_rowcb = impl::matrix::convert_zfp< value_t >( *rowcb_zfp, config );
                    auto  mem_colcb = impl::matrix::convert_zfp< value_t >( *colcb_zfp, config );
                
                    auto  diff      = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2_zfp );
                    auto  error     = hlr::norm::spectral( *diff, true, 1e-4 );
    
                    std::cout << "      " << boost::format( "%2d" ) % rate << " / "
                              << format_error( error, error / normA ) << " / "
                              << format_mem( mem_A, mem_rowcb, mem_colcb ) << std::endl;
                }// for
            }// if
            #endif
        
            #if defined(HAS_UNIVERSAL)
            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "using posits" << term::reset << std::endl;

                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 46, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 44, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-10, 1e-12 ) ) run_posit< 42, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-10, 1e-11 ) ) run_posit< 40, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-9,  1e-11 ) ) run_posit< 38, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-10 ) ) run_posit< 36, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 34, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 32, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 30, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 28, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-7  ) ) run_posit< 26, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-5,  1e-6  ) ) run_posit< 24, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-4,  1e-6  ) ) run_posit< 22, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-4,  1e-5  ) ) run_posit< 20, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-3,  1e-4  ) ) run_posit< 18, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-2,  1e-4  ) ) run_posit< 16, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-1,  1e-4  ) ) run_posit< 16, 1 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-0,  1e-3  ) ) run_posit< 14, 1 >( *A_ref, *A2, *rowcb, *colcb, normA );
            }// if
            #endif
        }

        //////////////////////////////////////////////////////////////////////
        //
        // conversion to H²
        //
        //////////////////////////////////////////////////////////////////////

        #if defined( HAS_H2 )
    
        if ( false )
        {
            std::cout << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

            std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
            hpro::THClusterBasisBuilder< value_t >  bbuilder;

            tic = timer::now();
    
            auto [ rowcb, colcb ] = bbuilder.build( ct->root(), ct->root(), A_ref.get(), acc );

            toc = timer::since( tic );

            std::cout << "    done in  " << format_time( toc ) << std::endl;

            if ( verbose( 3 ) )
            {
                io::eps::print( *rowcb, "rowcb" );
                io::eps::print( *colcb, "colcb" );
            }// if

            std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

            tic = timer::now();
    
            auto  A2 = to_h2( A_ref.get(), rowcb.get(), colcb.get() );
    
            toc = timer::since( tic );

            std::cout << "    done in  " << format_time( toc ) << std::endl;
            std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

            auto  diff  = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2 );
            auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
            std::cout << "    error  = " << format_error( error, error / normA ) << std::endl;
        
            if ( hpro::verbose( 3 ) )
                io::eps::print( *A2, "A2", "noid" );

            //
            // try lower precisions
            //
        
            #if defined(HAS_ZFP)
            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "ZFP compression" << term::reset << std::endl;

                for ( uint  rate = zfp_ub; rate >= zfp_lb; rate -= 2 )
                {
                    auto  A2_zfp    = impl::matrix::copy( *A2 );
                    auto  rowcb_zfp = rowcb->copy();
                    auto  colcb_zfp = colcb->copy();
                    auto  config    = zfp_config_rate( rate, false );

                    hpro::replace_cluster_basis( *A2_zfp, *rowcb_zfp, *colcb_zfp );
                
                    auto  mem_A     = impl::matrix::convert_zfp< value_t >( *A2_zfp, config );
                    auto  mem_rowcb = impl::matrix::convert_zfp< value_t >( *rowcb_zfp, config );
                    auto  mem_colcb = impl::matrix::convert_zfp< value_t >( *colcb_zfp, config );
                
                    auto  diff      = matrix::sum( value_t(1), *A_ref, value_t(-1), *A2_zfp );
                    auto  error     = hlr::norm::spectral( *diff, true, 1e-4 );
    
                    std::cout << "      " << boost::format( "%2d" ) % rate << " / "
                              << format_error( error ) << " / "
                              << format_error( error / normA ) << " / "
                              << format_mem( mem_A, mem_rowcb, mem_colcb ) << std::endl;
                }// for
            }// if
            #endif
        
            #if defined(HAS_UNIVERSAL)
            if ( true )
            {
                std::cout << "    " << term::bullet << term::bold << "using posits" << term::reset << std::endl;

                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 46, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-11, 1e-12 ) ) run_posit< 44, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-10, 1e-12 ) ) run_posit< 42, 3 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-10, 1e-11 ) ) run_posit< 40, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-9,  1e-11 ) ) run_posit< 38, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-10 ) ) run_posit< 36, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 34, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-8,  1e-9  ) ) run_posit< 32, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 30, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-8  ) ) run_posit< 28, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-6,  1e-7  ) ) run_posit< 26, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-5,  1e-6  ) ) run_posit< 24, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-4,  1e-6  ) ) run_posit< 22, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-4,  1e-5  ) ) run_posit< 20, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-3,  1e-4  ) ) run_posit< 18, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-2,  1e-4  ) ) run_posit< 16, 2 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-1,  1e-4  ) ) run_posit< 16, 1 >( *A_ref, *A2, *rowcb, *colcb, normA );
                if ( within( eps, 1e-0,  1e-3  ) ) run_posit< 14, 1 >( *A_ref, *A2, *rowcb, *colcb, normA );
            }// if
            #endif
        }// if

        #endif
    }// for
}
