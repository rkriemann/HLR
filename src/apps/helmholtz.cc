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


using namespace HLIB;

namespace hlr
{

namespace apps
{

//
// ctor
//
helmholtz_slp::helmholtz_slp ( const hpro::complex  kappa,
                               const std::string &  grid )
{
    _grid = hpro::make_grid( grid );
        
    auto  fnspace = std::make_unique< TConstFnSpace >( _grid.get() );
    auto  bf      = std::make_unique< THelmholtzSLPBF< TConstFnSpace, TConstFnSpace > >( kappa, fnspace.get(), fnspace.get() );

    log( 1, HLIB::to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
helmholtz_slp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< HLIB::TCoeffFn< helmholtz_slp::value_t > >
helmholtz_slp::coeff_func () const
{
    return std::make_unique<
        TBFCoeffFn< THelmholtzSLPBF< TConstFnSpace,
                                     TConstFnSpace > > >( static_cast< THelmholtzSLPBF< TConstFnSpace, TConstFnSpace > * >( _bf.get() ) );
}
    
}// namespace Helmholtz

}// namespace HLR
