//
// Project     : HLib
// File        : Laplace.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <hpro/bem/TLaplaceBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/io/TGridIO.hh>

#include "hlr/utils/log.hh"

#include "hlr/apps/laplace.hh"

namespace hpro = HLIB;

namespace hlr
{

namespace apps
{

using namespace hpro;

//
// ctor
//
laplace_slp::laplace_slp ( const std::string &  grid )
{
    _grid = make_grid( grid );
        
    auto  fnspace = std::make_unique< TConstFnSpace >( _grid.get() );
    auto  bf      = std::make_unique< TLaplaceSLPBF< TConstFnSpace, TConstFnSpace > >( fnspace.get(), fnspace.get(), 5 );

    log( 1, to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
laplace_slp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< TCoeffFn< laplace_slp::value_t > >
laplace_slp::coeff_func () const
{
    return std::make_unique< TBFCoeffFn< TLaplaceSLPBF< TConstFnSpace,
                                                        TConstFnSpace > > >( static_cast< TLaplaceSLPBF< TConstFnSpace, TConstFnSpace > * >( _bf.get() ) );
}
    
}// namespace Laplace

}// namespace HLR
