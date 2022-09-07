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
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,norank,nosize" );
    
    //
    // further compress matrix
    //

    auto  B      = impl::matrix::copy( *A );
    auto  norm_A = norm::frobenius( *A );
    
    std::cout << "  " << term::bullet << term::bold << "compression, "
              << cmdline::approx << ", ε = " << cmdline::eps << term::reset << std::endl;
    std::cout << "    norm  = " << format_norm( norm_A ) << std::endl;

    tic = timer::now();
    
    impl::matrix::compress( *B, Hpro::fixed_prec( acc.rel_eps() * 10.0 ) );
    // impl::matrix::compress( *B, local_accuracy( acc.rel_eps() * norm_A / A->nrows() ) );

    toc = timer::since( tic );

    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( B->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *B, "B", "noid,norank,nosize" );
    
    auto  diff = matrix::sum( value_t(1), *A, value_t(-1), *B );

    std::cout << "    error = " << format_error( norm::spectral( impl::arithmetic, *diff ) ) << std::endl;

    std::cout << "  " << term::bullet << term::bold << "uncompression " << term::reset << std::endl;
    
    tic = timer::now();
    
    impl::matrix::uncompress( *B );

    toc = timer::since( tic );

    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    error = " << format_error( norm::spectral( impl::arithmetic, *diff ) ) << std::endl;


    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    auto  runtime = std::vector< double >();
    
    std::cout << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;

    if ( true )
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
        
        runtime.clear();
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "compressed" << term::reset << std::endl;
        
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( B->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( B->row_is() );

        x->fill( 1 );

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *B, *x, *y );

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
        
        runtime.clear();
    }// if
}
    
