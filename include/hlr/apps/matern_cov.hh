#ifndef __HLR_APPS_MATERN_HH
#define __HLR_APPS_MATERN_HH
//
// Project     : HLib
// File        : matern.hh
// Description : functions for Matern covariance function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/apps/application.hh"

namespace hlr
{

namespace apps
{

class matern_cov : public application< hpro::real >
{
public:
    //
    // public types
    //
    
    using  value_t = hpro::real;

private:
    // problem size
    size_t                        _n;
    
    // coordinates
    std::vector< hpro::T3Point >  _vertices;

public:
    //
    // ctor: generate random coordinates with specific geometry
    //
    matern_cov ( const std::string &  geometry,
                 const size_t         n );

    //
    // ctor: use coordinates from given grid
    //
    matern_cov ( const std::string &  grid );

    //
    // dtor
    //
    
    virtual ~matern_cov () {}
    
    //
    // set up coordinates
    //
    std::unique_ptr< hpro::TCoordinate >
    coordinates () const;
    
    //
    // return coefficient function to evaluate matrix entries
    //
    std::unique_ptr< hpro::TCoeffFn< value_t > >
    coeff_func () const;
    
    // //
    // // build matrix
    // //
    // std::unique_ptr< hpro::TMatrix >
    // build_matrix ( const hpro::TBlockClusterTree *  bct,
    //                const hpro::TTruncAcc &          acc );
};

}// namespace apps

}// namespace hlr

#endif  // __HLR_APPS_MATERN_HH
