//
// Project     : HLib
// File        : matern.cc
// Description : use matern kernel to fill matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cmath>
#include <vector>
#include <random>

using namespace std;

#include <hpro/misc/TMaternCovCoeffFn.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/io/TGridIO.hh>

#include "hlr/apps/matern_cov.hh"

namespace hlr
{

using namespace HLIB;

namespace apps
{

using  point_t = T3Point;

//
// ctor
//
matern_cov::matern_cov ( const std::string &  geometry,
                         const size_t         n )
        : _n( n )
{
    if ( geometry == "randcube" )
    {
        std::mt19937_64                   generator{ 1 };
        std::uniform_real_distribution<>  distr{ 0, 1 };

        _vertices.reserve( n );
    
        for ( size_t i = 0; i < n; i++ )
            _vertices.push_back( point_t( distr( generator ), distr( generator ), distr( generator ) ) );
    }// if
    else if ( geometry == "randsphere" )
    {
        std::mt19937_64                   generator{ 1 };
        std::uniform_real_distribution<>  distr{ 0, 1 };

        _vertices.reserve( n );
    
        for ( size_t i = 0; i < n; i++ )
            _vertices.push_back( spherical( 2.0 * M_PI * distr( generator ),
                                            2.0 * M_PI * distr( generator ) - M_PI,
                                            1.0 ) );
    }// if
    else
        throw std::runtime_error( "unknown geometry : " + geometry );
}

//
// ctor
//
matern_cov::matern_cov ( const std::string &  gridfile )
{
    //
    // read grid and copy coordinates
    //
        
    auto  grid = make_grid( gridfile );
        
    _n = grid->n_vertices();
    _vertices.reserve( _n );
        
    for ( size_t  i = 0; i < _n; ++i )
    {
        _vertices.push_back( grid->vertex( i ) );
    }// for
}

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
matern_cov::coordinates () const
{
    return  std::make_unique< TCoordinate >( _vertices );
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< TCoeffFn< matern_cov::value_t > >
matern_cov::coeff_func () const
{
    return std::make_unique< TMaternCovCoeffFn< point_t > >( 1.0, 1.0, 0.5, _vertices );
}
    
}// namespace apps

}// namespace hlr
