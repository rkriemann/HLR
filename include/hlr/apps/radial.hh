#ifndef __HLR_APPS_RADIAL_HH
#define __HLR_APPS_MATERN_HH
//
// Project     : HLib
// Module      : apps/radial
// Description : functions for radial functions (Matérn, Gaussian, etc.)
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/base/TPoint.hh>
#include <hpro/bem/TRefinableGrid.hh>

#include <hlr/apps/application.hh>
#include <hlr/matrix/radial.hh>

namespace hlr { namespace apps {

namespace hpro = HLIB;

//
// return vertices of given grid
//
std::vector< hpro::T3Point >
make_vertices ( const std::string &  gridname );

//
// application for radial functions, e.g. Matérn covariance, etc.
//
template < typename kernel_t >
struct radial : public application< typename kernel_t::value_t >
{
    using  value_t = hpro::real;

    const kernel_t                kernel;
    std::vector< hpro::T3Point >  vertices;
    
    // ctor: use coordinates from given grid
    radial ( const kernel_t &     akernel,
             const std::string &  agrid )
            : kernel( akernel )
            , vertices( make_vertices( agrid ) )
    {}

    // set up coordinates
    std::unique_ptr< hpro::TCoordinate >
    coordinates () const
    {
        return std::make_unique< hpro::TCoordinate >( vertices );
    }
    
    // return coefficient function to evaluate matrix entries
    std::unique_ptr< hpro::TCoeffFn< value_t > >
    coeff_func () const
    {
        return std::make_unique< matrix::radial_function< kernel_t, std::vector< hpro::T3Point > > >( kernel, vertices, vertices );
    }
};
    
struct matern_covariance : public radial< matrix::matern_covariance_function< hpro::real > >
{
    using  value_t = hpro::real;

    matern_covariance ( const hpro::real     sigma,
                        const std::string &  grid )
            : radial( matrix::matern_covariance_function< hpro::real >( sigma, 0.5, 1.0 ),
                      grid )
    {}
};

struct gaussian : public radial< matrix::gaussian_function< hpro::real > >
{
    using  value_t = hpro::real;

    gaussian ( const hpro::real     sigma,
               const std::string &  grid )
            : radial( matrix::gaussian_function< hpro::real >( sigma * sigma ), grid )
    {}
};

}}// namespace hlr::apps

#endif  // __HLR_APPS_MATERN_HH
