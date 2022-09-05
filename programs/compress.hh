//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <filesystem>

// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range2d.h>

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

//#include <hlr/arith/cuda.hh>

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
    // std::cout << "    flops = " << format_flops( get_flops( "build" ), toc.seconds() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A", "noid,norank,nosize" );
        
    //
    // further compress matrix
    //

    auto  B      = impl::matrix::copy( *A );
    auto  mem_A  = A->byte_size();
    auto  norm_A = norm::frobenius( *A );
    auto  delta  = norm_A * hlr::cmdline::eps / A->nrows();
    
    std::cout << "  " << term::bullet << term::bold << "compression, "
              << cmdline::approx << ", ε = " << cmdline::eps << ", δ = " << delta << term::reset << std::endl;

    tic = timer::now();
    
    matrix::compress( *B, acc );

    toc = timer::since( tic );

    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    matrix::uncompress( *B );
    
    auto  diff = matrix::sum( value_t(1), *A, value_t(-1), *B );

    std::cout << "    error = " << format_error( norm::spectral( *diff ) ) << std::endl;
}
    
