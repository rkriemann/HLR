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


using namespace HLIB;

namespace hlr
{

namespace apps
{

namespace
{

std::unique_ptr< TGrid >
make_hlib_grid ( const std::string &  name,
                 const uint           lvl )
{
    assert( ( name == "sphere"  ) ||
            ( name == "sphere2" ) ||
            ( name == "cube"    ) ||
            ( name == "square"  ) );

    std::unique_ptr< TRefinableGrid >  grid;
    
    if      ( name == "sphere"  ) grid = make_sphere();
    else if ( name == "sphere2" ) grid = make_sphere2();
    else if ( name == "cube"    ) grid = make_cube();
    else if ( name == "square"  ) grid = make_square();

    for ( uint  i = 0; i < lvl; ++i )
    {
        auto  rgrid = grid->refine();
        
        grid = std::move( rgrid );
    }// while

    return grid;
}

}// namespace anonymous

std::unique_ptr< TGrid >
make_grid ( const std::string &  grid )
{
    const auto  dashpos = grid.find( '-' );

    if ( dashpos != std::string::npos )
    {
        const auto  basename = grid.substr( 0, dashpos );
        const auto  lvl      = grid.substr( dashpos+1, grid.length() );

        if (( basename == "sphere" ) || ( basename == "sphere2" ) || ( basename == "cube" ) || ( basename == "square" ))
            return make_hlib_grid( basename, atoi( lvl.c_str() ) );
        else
            return read_grid( grid );
    }// if
    else
    {
        if (( grid == "sphere" ) || ( grid == "sphere2" ) || ( grid == "cube" ) || ( grid == "square" ))
            return make_hlib_grid( grid, 0 );
        else
            return read_grid( grid );
    }// else
}

//
// ctor
//
laplace_slp::laplace_slp ( const std::string &  grid )
{
    _grid = make_grid( grid );
        
    auto  fnspace = std::make_unique< TConstFnSpace >( _grid.get() );
    auto  bf      = std::make_unique< TLaplaceSLPBF< TConstFnSpace, TConstFnSpace > >( fnspace.get(), fnspace.get() );

    log( 1, HLIB::to_string( "    no. of indices = %d", fnspace->n_indices() ) );
    
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
std::unique_ptr< HLIB::TCoeffFn< laplace_slp::value_t > >
laplace_slp::coeff_func () const
{
    return std::make_unique< TBFCoeffFn< TLaplaceSLPBF< TConstFnSpace,
                                                        TConstFnSpace > > >( static_cast< TLaplaceSLPBF< TConstFnSpace, TConstFnSpace > * >( _bf.get() ) );
}
    
}// namespace Laplace

}// namespace HLR
