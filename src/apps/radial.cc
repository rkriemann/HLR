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
#include <filesystem>

#include <hpro/io/TGridIO.hh>
#include <hpro/io/TCoordIO.hh>

#include "hlr/apps/radial.hh"
#include "hlr/utils/log.hh"

namespace hlr { namespace apps {

using  point_t = Hpro::T3Point;

//
// return vertices of given grid
//
std::vector< point_t >
make_vertices ( const std::string &  gridfile )
{
    const auto  dashpos      = gridfile.find( '-' );
    bool        ignore_coord = false;

    if ( ! std::filesystem::exists( gridfile ) && ( dashpos != std::string::npos ))
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
        else if ( geometry == "randcylinder" )
        {
            std::mt19937_64                   generator{ 1 };
            std::uniform_real_distribution<>  distr{ -1, 1 };

            vertices.reserve( nsize );
    
            while ( vertices.size() < nsize )
            {
                auto  p = point_t( distr( generator ), distr( generator ), 0 );

                // only permit values in circle
                if ( p.norm2() <= 1.0 )
                    vertices.push_back( point_t( p.x(), p.y(), 2 * distr( generator ) ) );
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
        else if (( geometry == "sphere" ) || ( geometry == "cube" ) || ( geometry == "square" ))
        {
            // ignore as generated by Hpro::make_grid
            ignore_coord = true;
        }// if

        if ( vertices.size() > 0 )
            return vertices;
    }// if

    //
    // try to read coordinates from file
    //

    if ( ! ignore_coord )
    {
        try
        {
            auto  cio      = Hpro::TAutoCoordIO();
            auto  coord    = cio.read( gridfile );
            auto  n        = coord->ncoord();
            auto  vertices = std::vector< point_t >();
        
            if ( coord->dim() == 3 )
            {
                vertices.reserve( n );
                
                for ( size_t  i = 0; i < n; ++i )
                {
                    auto  v_i = coord->coord( i );
                    
                    vertices.push_back( point_t( v_i[0], v_i[1], v_i[2] ) );
                }// for
                
                return vertices;
            }// if
        }// try
        catch ( ... ) {}
    }// if

    //
    // generate/read standard grid and copy vertices
    //

    try
    {
        auto  grid     = Hpro::make_grid( gridfile );
        auto  vertices = std::vector< point_t >();
        auto  n        = grid->n_vertices();

        vertices.reserve( n );
        
        for ( size_t  i = 0; i < n; ++i )
            vertices.push_back( grid->vertex( i ) );

        return vertices;
    }// try
    catch ( ... ) {}

    HLR_ERROR( "unknown geometry/grid or failed to read file \"" + gridfile + "\"" );
}
    
}}// namespace hlr::apps
