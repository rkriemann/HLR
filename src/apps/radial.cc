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

#include <hpro/io/TGridIO.hh>

#include "hlr/apps/radial.hh"

namespace hlr
{

using namespace HLIB;

namespace apps
{

using  point_t = T3Point;

//
// return vertices of given grid
//
std::vector< point_t >
make_vertices ( const std::string &  gridfile )
{
    const auto  dashpos = gridfile.find( '-' );

    if ( dashpos != std::string::npos )
    {
        const auto    geometry = gridfile.substr( 0, dashpos );
        const auto    size     = gridfile.substr( dashpos+1, gridfile.length() );
        const size_t  nsize    = std::atoi( size.c_str() );
        auto          vertices = std::vector< point_t >();

        if ( geometry == "randcube" )
        {
            std::mt19937_64                   generator{ 1 };
            std::uniform_real_distribution<>  distr{ 0, 1 };

            vertices.reserve( nsize );
    
            for ( size_t i = 0; i < nsize; i++ )
                vertices.push_back( point_t( distr( generator ), distr( generator ), distr( generator ) ) );
        }// if
        else if ( geometry == "randsphere" )
        {
            std::mt19937_64                   generator{ 1 };
            std::uniform_real_distribution<>  distr{ -1, 1 };

            vertices.reserve( nsize );
    
            for ( size_t i = 0; i < nsize; i++ )
            {
                // map cube to sphere
                // - slightly more points along mapped cube edges!
                auto  p = point_t( distr( generator ), distr( generator ), distr( generator ) );

                p *= double(1) / p.norm2();
                
                vertices.push_back( p );

                // "spherical" produces too many points near poles
                // vertices.push_back( spherical( 2.0 * M_PI * distr( generator ),
                //                                2.0 * M_PI * distr( generator ) - M_PI,
                //                                1.0 ) );
            }// for
        }// if
        else if ( geometry == "randball" )
        {
            std::mt19937_64                   generator{ 1 };
            std::uniform_real_distribution<>  distr{ -1, 1 };

            vertices.reserve( nsize );
    
            while ( vertices.size() < nsize )
            {
                // map cube to sphere
                // - slightly more points along mapped cube edges!
                auto  p = point_t( distr( generator ), distr( generator ), distr( generator ) );

                // only permit values in ball
                if ( p.norm2() <= 1.0 )
                    vertices.push_back( p );
            }// for
        }// if
        else if ( geometry == "tensorcube" )
        {
            vertices.reserve( nsize * nsize * nsize );
            
            const double  h = 1.0 / double(nsize);

            for ( size_t  x = 0; x < nsize; ++x )
                for ( size_t  y = 0; y < nsize; ++y )
                    for ( size_t  z = 0; z < nsize; ++z )
                        vertices.push_back( point_t( x * h, y * h, z * h ) );
        }// if
        else if ( geometry == "jittercube" )
        {
            vertices.reserve( nsize * nsize * nsize );
            
            const double                      h = 1.0 / double(nsize);
            std::mt19937_64                   generator{ 1 };
            std::uniform_real_distribution<>  distr{ -h/2.0, h/2.0 };

            for ( size_t  x = 0; x < nsize; ++x )
                for ( size_t  y = 0; y < nsize; ++y )
                    for ( size_t  z = 0; z < nsize; ++z )
                        vertices.push_back( point_t( x * h + distr( generator ),
                                                     y * h + distr( generator ),
                                                     z * h + distr( generator ) ) );
        }// if
        else
            throw std::runtime_error( "unknown geometry : " + geometry );

        return vertices;
    }// if

    //
    // generate/read standard grid and copy vertices
    //

    {
        auto  grid     = hpro::make_grid( gridfile );
        auto  vertices = std::vector< point_t >();
        auto  n        = grid->n_vertices();

        vertices.reserve( n );
        
        for ( size_t  i = 0; i < n; ++i )
            vertices.push_back( grid->vertex( i ) );

        return vertices;
    }
}
    
}// namespace apps

}// namespace hlr
