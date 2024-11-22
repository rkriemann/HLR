#ifndef __HLR_APPS_RADIAL_HH
#define __HLR_APPS_RADIAL_HH
//
// Project     : HLR
// Module      : apps/radial
// Description : functions for radial functions (Matérn, Gaussian, etc.)
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/base/TPoint.hh>
#include <hpro/bem/TRefinableGrid.hh>

#include <hlr/apps/application.hh>
#include <hlr/matrix/radial.hh>

namespace hlr { namespace apps {

//
// return vertices of given grid
//
std::vector< Hpro::T3Point >
make_vertices ( const std::string &  gridname );

//
// application for radial functions, e.g. Matérn covariance, etc.
//
template < typename kernel_t >
struct radial : public application< typename kernel_t::value_t >
{
    using  value_t = double;

    const kernel_t                kernel;
    std::vector< Hpro::T3Point >  vertices;
    
    // ctor: use coordinates from given grid
    radial ( const kernel_t &     akernel,
             const std::string &  agrid )
            : kernel( akernel )
            , vertices( make_vertices( agrid ) )
    {}

    // set up coordinates
    std::unique_ptr< Hpro::TCoordinate >
    coordinates () const
    {
        return std::make_unique< Hpro::TCoordinate >( vertices );
    }
    
    // return coefficient function to evaluate matrix entries
    std::unique_ptr< Hpro::TCoeffFn< value_t > >
    coeff_func () const
    {
        return std::make_unique< matrix::radial_function< kernel_t, std::vector< Hpro::T3Point > > >( kernel, vertices, vertices );
    }
};
    
struct matern_covariance : public radial< matrix::matern_covariance_function< double > >
{
    using  value_t = double;

    matern_covariance ( const double         sigma,
                        const double         nu,
                        const double         l,
                        const std::string &  grid )
            : radial( matrix::matern_covariance_function< double >( sigma, nu, l ),
                      grid )
    {}
};

struct gaussian : public radial< matrix::gaussian_function< double > >
{
    using  value_t = double;

    gaussian ( const double         sigma,
               const std::string &  grid )
            : radial( matrix::gaussian_function< double >( sigma * sigma ), grid )
    {}
};

}}// namespace hlr::apps

#endif  // __HLR_APPS_RADIAL_HH
