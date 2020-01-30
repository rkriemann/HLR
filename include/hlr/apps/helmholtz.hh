#ifndef __HLR_APPS_HELMHOLTZ_HH
#define __HLR_APPS_HELMHOLTZ_HH
//
// Project     : HLib
// File        : Helmholtz.hh
// Description : functions for Helmholtz SLP/DLP BEM application
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
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

class helmholtz_slp : public application< hpro::complex >
{
public:
    //
    // public types
    //
    
    using  value_t = hpro::complex;

private:
    // BEM data
    std::unique_ptr< hpro::TGrid >                     _grid;
    std::unique_ptr< hpro::TFnSpace >                  _fnspace;
    std::unique_ptr< hpro::TBilinearForm< value_t > >  _bf;

public:
    //
    // ctor with grid name (plus refinement levels)
    //
    helmholtz_slp ( const hpro::complex  kappa,
                    const std::string &  grid );
    
    //
    // dtor
    //
    
    virtual ~helmholtz_slp () {}
    
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

#endif  // __HLR_APPS_HELMHOLTZ_HH
