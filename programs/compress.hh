//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#if defined(HAS_UNIVERSAL)

#include <universal/number/posit/posit.hpp>

namespace std
{

template <> struct __is_floating_point_helper< sw::universal::posit< 24, 2 > > : public true_type { };

sw::universal::posit< 24, 2 > real ( sw::universal::posit< 24, 2 >  f ) { return f; }

};

#endif

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

using indexset = hpro::TIndexSet;

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
    
    if ( verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;

        print_vtk( coord.get(), "coord" );
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if

    blas::reset_flops();
    
    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );

    auto  tic    = timer::now();
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;

    const auto  mem_A = A->byte_size();
    
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,norank,nosize" );
    
    //
    // further compress matrix
    //

    auto  zA     = impl::matrix::copy( *A );
    auto  norm_A = norm::frobenius( *A );
    
    std::cout << "  " << term::bullet << term::bold << "compression via "
              << hlr::compress::provider
              << ", ε = " << cmdline::eps << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    tic = timer::now();
    
    impl::matrix::compress( *zA, Hpro::fixed_prec( acc.rel_eps() ) );
    // impl::matrix::compress( *zA, local_accuracy( acc.rel_eps() * norm_A / A->nrows() ) );

    toc = timer::since( tic );

    std::cout << "    done in " << format_time( toc ) << std::endl;

    const auto  mem_zA = zA->byte_size();
    
    std::cout << "    mem   = " << format_mem( zA->byte_size() ) << std::endl;
    std::cout << "      rate  " << boost::format( "%.2f" ) % ( double(mem_A) / double(mem_zA) ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *zA, "zA", "noid,norank,nosize" );
    
    auto  diff = matrix::sum( value_t(1), *A, value_t(-1), *zA );
    auto  error = norm::spectral( impl::arithmetic, *diff );
    
    std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

    std::cout << "  " << term::bullet << term::bold << "uncompression " << term::reset << std::endl;

    {
        auto  zB = impl::matrix::copy( *zA );
        
        tic = timer::now();
    
        impl::matrix::uncompress( *zB );

        toc = timer::since( tic );

        auto  diffB = matrix::sum( value_t(1), *A, value_t(-1), *zB );
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    error = " << format_error( norm::spectral( impl::arithmetic, *diffB ) ) << std::endl;
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////

    if ( nbench > 0 )
    {
        auto  runtime = std::vector< double >();
    
        std::cout << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;

        double  t_orig       = 0.0;
        double  t_compressed = 0.0;
        
        {
            std::cout << "  " << term::bullet << term::bold << "uncompressed" << term::reset << std::endl;
        
            auto  x = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            x->fill( 1 );

            for ( int i = 0; i < nbench; ++i )
            {
                tic = timer::now();
    
                for ( int j = 0; j < 50; ++j )
                    impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x, *y );

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
                    impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *zA, *x, *y );

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
        }
    }// if
}
    
