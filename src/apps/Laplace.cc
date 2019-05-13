//
// Project     : HLib
// File        : Laplace.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <bem/TLaplaceBF.hh>
#include <bem/TRefinableGrid.hh>
#include <bem/TBFCoeffFn.hh>
#include <io/TGridIO.hh>

#include "utils/log.hh"

#include "apps/Laplace.hh"


using namespace HLIB;

namespace HLR
{

namespace Apps
{

namespace
{

std::unique_ptr< TGrid >
make_grid ( const size_t         n,
            const std::string &  name )
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

    while ( grid->n_triangles() < n )
    {
        auto  rgrid = grid->refine();

        // do not exceed given upper limit
        if ( rgrid->n_triangles() > n )
            break;
        
        grid = std::move( rgrid );
    }// while

    return grid;
}

}// namespace anonymous

//
// ctor
//
LaplaceSLP::LaplaceSLP ( const size_t         n,
                         const std::string &  grid )
{
    if (( grid == "sphere" ) || ( grid == "sphere2" ) || ( grid == "cube" ) || ( grid == "square" ))
        _grid = make_grid( n, grid );
    else
        _grid = read_grid( grid );

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
LaplaceSLP::coordinates () const
{
    return _fnspace->build_coord();
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< HLIB::TCoeffFn< LaplaceSLP::value_t > >
LaplaceSLP::coeff_func () const
{
    return std::make_unique< TBFCoeffFn< TLaplaceSLPBF< TConstFnSpace,
                                                        TConstFnSpace > > >( static_cast< TLaplaceSLPBF< TConstFnSpace, TConstFnSpace > * >( _bf.get() ) );
}
    
}// namespace Laplace

}// namespace HLR
