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
    
    if ( verbose(3) )
        io::vtk::print( *coord, "coord" );

    std::cout << "    dims   = " << coord->ncoord() << std::endl;

    auto  part         = Hpro::TGeomBSPPartStrat( Hpro::adaptive_split_axis );
    auto  [ ct, pe2i ] = cluster::build_cluster_tree( *coord, part, cmdline::ntile );
    auto  pi2e         = pe2i->inverse();

    auto  adm1         = cluster::weak_adm_cond( 1 ); // face admissibility
    auto  adm2         = cluster::weak_adm_cond( 2 ); // edge admissibility
    auto  adm3         = cluster::weak_adm_cond( 3 ); // vertex admissibility
    auto  strong       = cluster::strong_adm_cond();
    auto  bt           = cluster::build_block_tree( *ct, *ct, adm1 );

    if ( verbose(3) )
        io::eps::print( *bt, "bt" );
    
    auto  bf           = Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t >( & fnspace, & fnspace, 6u );
    auto  coeff        = Hpro::TBFCoeffFn< decltype( bf ) >( & bf ); 
    auto  pcoeff       = hpro::TPermCoeffFn< value_t >( &coeff, &pi2e, &pi2e );
    auto  acalr        = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
    auto  acc          = relative_prec( cmdline::eps );
        
    auto  A            = impl::matrix::build( bt.get(), pcoeff, acalr, acc, false );

    if ( verbose(3) )
        io::eps::print( *A, "A" );
}
