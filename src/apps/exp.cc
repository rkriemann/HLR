//
// Project     : HLib
// File        : Exp.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>

#include <hpro/bem/TExpBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/io/TGridIO.hh>

#include "hlr/utils/log.hh"

#include "hlr/apps/exp.hh"

namespace hpro = HLIB;

namespace hlr
{

namespace apps
{

using namespace hpro;

//
// ctor
//
exp::exp ( const std::string &  grid )
{
    _grid = make_grid( grid );
        
    auto  fnspace = std::make_unique< TConstFnSpace >( _grid.get() );
    auto  bf      = std::make_unique< TExpBF< TConstFnSpace, TConstFnSpace > >( fnspace.get(), fnspace.get(), 5 );

    log( 1, to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
exp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< TCoeffFn< exp::value_t > >
exp::coeff_func () const
{
    return std::make_unique< TBFCoeffFn< TExpBF< TConstFnSpace,
                                                 TConstFnSpace > > >( static_cast< TExpBF< TConstFnSpace, TConstFnSpace > * >( _bf.get() ) );
}
    
}// namespace Exp

}// namespace HLR
