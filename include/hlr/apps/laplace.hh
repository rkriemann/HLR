#ifndef __HLR_APPS_LAPLACE_HH
#define __HLR_APPS_LAPLACE_HH
//
// Project     : HLib
// Module      : apps/laplace
// Description : functions for Laplace SLP/DLP BEM application
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <memory>

#include <hpro/bem/TGrid.hh>
#include <hpro/bem/TFnSpace.hh>
#include <hpro/bem/TBEMBF.hh>

#include "hlr/apps/application.hh"

namespace hlr { namespace apps {

class laplace_slp : public application< double >
{
public:
    // public types
    using  value_t = double;

private:
    // BEM data
    std::unique_ptr< Hpro::TGrid >                     _grid;
    std::unique_ptr< Hpro::TFnSpace< double > >        _fnspace;
    std::unique_ptr< Hpro::TBilinearForm< value_t > >  _bf;

public:
    // ctor with grid name (plus refinement levels)
    laplace_slp ( const std::string &  grid );
    
    // dtor
    virtual ~laplace_slp () {}
    
    // return coordinates for problem indices
    std::unique_ptr< Hpro::TCoordinate >
    coordinates () const;
    
    // return coefficient function to evaluate matrix entries
    std::unique_ptr< Hpro::TCoeffFn< value_t > >
    coeff_func  () const;
};

}}//namespace hlr::apps

#endif  // __HLR_APPS_LAPLACE_HH
