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

namespace hlr { namespace apps {

//
// ctor
//
exp::exp ( const std::string &  grid )
{
    _grid = Hpro::make_grid( grid );
        
    auto  fnspace = std::make_unique< Hpro::TConstFnSpace< double > >( _grid.get() );
    auto  bf      = std::make_unique< Hpro::TExpBF< Hpro::TConstFnSpace< double >, Hpro::TConstFnSpace< double >, value_t > >( fnspace.get(), fnspace.get(), 5 );

    log( 1, Hpro::to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
    _fnspace = std::move( fnspace );
    _bf      = std::move( bf );
}

//
// set up coordinates
//
std::unique_ptr< Hpro::TCoordinate >
exp::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< Hpro::TCoeffFn< exp::value_t > >
exp::coeff_func () const
{
    using  bf_t = Hpro::TExpBF< Hpro::TConstFnSpace< double >, Hpro::TConstFnSpace< double >, value_t >;
                                
    return std::make_unique< Hpro::TBFCoeffFn< bf_t > >( static_cast< bf_t * >( _bf.get() ) );
}
    
}}// namespace hlr::apps
