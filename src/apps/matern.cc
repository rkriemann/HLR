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

#include <misc/TMaternCovCoeffFn.hh>
#include <io/TGridIO.hh>

#include "apps/matern.hh"

namespace HLR
{

namespace Apps
{

using namespace HLIB;

using  point_t = T3Point;

// external grid generation function (see Laplace.cc)
std::unique_ptr< TGrid >
make_grid ( const std::string &  grid );

//
// ctor
//
MaternCov::MaternCov ( const size_t  n )
        : _n( n )
{
    //
    // build vertices
    //
    
    // std::random_device                rd{};
    // std::mt19937                      generator{ rd() };
    std::mt19937_64                   generator{ 1 };
    std::uniform_real_distribution<>  distr{ 0, 1 };

    _vertices.reserve( n );
    
    for ( size_t i = 0; i < n; i++ )
        _vertices.push_back( spherical( 2.0 * M_PI * distr( generator ),
                                        2.0 * M_PI * distr( generator ) - M_PI,
                                        1.0 ) ); // point_t( distr( generator ), distr( generator ) );
}

//
// ctor
//
MaternCov::MaternCov ( const std::string &  gridfile )
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
MaternCov::coordinates () const
{
    return  std::make_unique< TCoordinate >( _vertices );
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< TCoeffFn< MaternCov::value_t > >
MaternCov::coeff_func () const
{
    return std::make_unique< TMaternCovCoeffFn< point_t > >( 1.0, 1.29, 0.325, _vertices );
    
    // return std::make_unique< TPermCoeffFn< real_t >( matern_coeff, bct->row_ct()->perm_i2e(), bct->row_ct()->perm_i2e() );
}
    
// //
// // build matrix
// //
// std::unique_ptr< TMatrix >
// MaternCov::build_matrix ( const TBlockClusterTree *  bct,
//                           const TTruncAcc &          acc )
// {
//     // unique_ptr< TProgressBar >    progress( verbose(2) ? new TConsoleProgressBar( std::cout ) : nullptr );
//     TMaternCovCoeffFn< point_t >  matern_coeff( 1.0, 1.29, 0.325, vertices );
//     TPermCoeffFn< real_t >        coefffn( & matern_coeff, bct->row_ct()->perm_i2e(), bct->row_ct()->perm_i2e() );
//     TACAPlus< real_t >            aca( & coefffn );
//     TDenseMBuilder< real_t >      h_builder( & coefffn, & aca );
    
//     h_builder.set_build_ghosts( true );
    
//     return h_builder.build( bct, unsymmetric, acc );
// }

}// namespace Matern

}// namespace HLR
