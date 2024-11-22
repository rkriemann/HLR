//
// Project     : HLR
// Module      : Laplace.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cassert>

#include <hpro/bem/TLaplaceBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/io/TGridIO.hh>

#include "hlr/utils/log.hh"

#include "hlr/apps/laplace.hh"

namespace hlr { namespace apps {

//
// ctor
//
laplace_slp::laplace_slp ( const std::string &  grid,
                           const double         quad_error )
{
    _grid = Hpro::make_grid( grid );
        
    auto  fnspace = std::make_unique< fnspace_t >( _grid.get() );
    auto  bf      = std::make_unique< Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t > >( fnspace.get(), fnspace.get(), quad_error );
    // auto  bf      = std::make_unique< Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t > >( fnspace.get(), fnspace.get(), uint(3) );

    log( 1, Hpro::to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< Hpro::TCoordinate >
laplace_slp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< Hpro::TCoeffFn< laplace_slp::value_t > >
laplace_slp::coeff_func () const
{
    using  bf_t = Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t, value_t >;
    
    return std::make_unique< Hpro::TBFCoeffFn< bf_t > >( static_cast< bf_t * >( _bf.get() ) );
}
    
}}// namespace hlr::apps
