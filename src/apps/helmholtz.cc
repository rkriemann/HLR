//
// Project     : HLib
// File        : Helmholtz.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <hpro/bem/THelmholtzBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/io/TGridIO.hh>

#include "hlr/utils/log.hh"

#include "hlr/apps/helmholtz.hh"

namespace hlr { namespace apps {

//
// ctor
//
helmholtz_slp::helmholtz_slp ( const std::complex< double >  kappa,
                               const std::string &           grid )
{
    _grid = Hpro::make_grid( grid );
        
    auto  fnspace = std::make_unique< Hpro::TConstFnSpace< double > >( _grid.get() );
    auto  bf      = std::make_unique< Hpro::THelmholtzSLPBF< Hpro::TConstFnSpace< double >, Hpro::TConstFnSpace< double >, value_t > >( kappa, fnspace.get(), fnspace.get(), 5 );

    log( 1, Hpro::to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< Hpro::TCoordinate >
helmholtz_slp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< Hpro::TCoeffFn< helmholtz_slp::value_t > >
helmholtz_slp::coeff_func () const
{
    using  bf_t = Hpro::THelmholtzSLPBF< Hpro::TConstFnSpace< double >, Hpro::TConstFnSpace< double >, value_t >;
    
    return std::make_unique< Hpro::TBFCoeffFn< bf_t > >( static_cast< bf_t * >( _bf.get() ) );
}
    
}}// namespace hlr::apps
