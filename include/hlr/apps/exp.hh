#ifndef __HLR_APPS_EXP_HH
#define __HLR_APPS_EXP_HH
//
// Project     : HLib
// File        : Exp.hh
// Description : functions for Exp BEM application
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <memory>

#include <hpro/bem/TGrid.hh>
#include <hpro/bem/TFnSpace.hh>
#include <hpro/bem/TBEMBF.hh>

#include "hlr/apps/application.hh"

namespace hlr
{

namespace hpro = HLIB;

namespace apps
{

class exp : public application< hpro::real >
{
public:
    //
    // public types
    //
    
    using  value_t = hpro::real;

private:
    // BEM data
    std::unique_ptr< hpro::TGrid >                     _grid;
    std::unique_ptr< hpro::TFnSpace >                  _fnspace;
    std::unique_ptr< hpro::TBilinearForm< value_t > >  _bf;

public:
    //
    // ctor with grid name (plus refinement levels)
    //
    exp ( const std::string &  grid );
    
    //
    // dtor
    //
    
    virtual ~exp () {}
    
    //
    // return coordinates for problem indices
    //
    std::unique_ptr< hpro::TCoordinate >
    coordinates () const;
    
    //
    // return coefficient function to evaluate matrix entries
    //
    std::unique_ptr< hpro::TCoeffFn< value_t > >
    coeff_func  () const;
};

}//namespace apps

}//namespace hlr

#endif  // __HLR_APPS_EXP_HH
