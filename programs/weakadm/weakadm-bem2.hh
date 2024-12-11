//
// Project     : HLR
// Module      : weakadm
// Description : program for testing weak admissibility
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hpro/config.h>
#include <hpro/bem/TBFCoeffFn.hh>

#include <hlr/bem/aca.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/cluster/build.hh>
#include <hlr/cluster/weakadm.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t   = typename problem_t::value_t;
    using real_t    = Hpro::real_type_t< value_t >;
    using fnspace_t = Hpro::TLinearFnSpace< value_t >;

    auto  runtime      = std::vector< double >();
    auto  tic          = timer::now();
    auto  toc          = timer::since( tic );

    auto  grid         = Hpro::make_grid( cmdline::gridfile );
    auto  fnspace      = fnspace_t( grid.get() );
    auto  coord        = fnspace.build_coord();
    auto  n            = coord->ncoord();
    auto  h            = 1.0 / ( std::sqrt( n ) - 1 );
    
    if ( verbose(3) )
        io::vtk::print( *coord, "coord" );

    std::cout << "    dims   = " << n << std::endl;
    std::cout << "    h      = " << h << std::endl;

    auto  part         = Hpro::TGeomBSPPartStrat( Hpro::adaptive_split_axis );
    auto  [ ct, pe2i ] = cluster::build_cluster_tree( *coord, part, cmdline::ntile );
    auto  pi2e         = pe2i->inverse();

    std::cout << term::bullet << term::bold << "reference matrix" << term::reset << std::endl;
    
    auto  bf           = Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t >( & fnspace, & fnspace, 6u );
    auto  coeff        = Hpro::TBFCoeffFn< decltype( bf ) >( & bf ); 
    auto  pcoeff       = hpro::TPermCoeffFn< value_t >( &coeff, &pi2e, &pi2e );
    auto  acalr        = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
    auto  acc          = relative_prec( cmdline::eps );
    auto  REF          = pcoeff.build( is( 0, n-1 ), is( 0, n-1 ) );
    auto  norm_REF     = impl::norm::spectral( *REF );
    
    for ( uint  codim = 2; codim <= 3; ++codim )
    {
        std::cout << term::bullet << term::bold << "codim " << codim << term::reset << std::endl;
        
        auto  adm      = cluster::weak_adm_cond( codim, 1.5 * h ); // vertex admissibility
        auto  strong   = cluster::strong_adm_cond();
        auto  bt       = cluster::build_block_tree( *ct, *ct, adm );

        if ( verbose(3) )
            io::eps::print( *bt, "bt" );

        std::cout << "  nnodes : " << bt->nnodes() << std::endl
                  << "  c_sp   : " << Hpro::compute_c_sp( *bt ) << std::endl;
        
        auto  A        = impl::matrix::build( bt.get(), pcoeff, acalr, acc, false );
        
        if ( verbose(3) )
            io::eps::print( *A, "A" );
        
        auto  diff     = matrix::sum( 1, *REF, -1, *A );
        auto  error    = impl::norm::spectral( *diff );
        
        std::cout << "  error  : " << format_error( error, error / norm_REF ) << std::endl;
    }// if
}
