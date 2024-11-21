#ifndef __HLR_APPS_EXP_HH
#define __HLR_APPS_EXP_HH
//
// Project     : HLR
// Module      : Exp.hh
// Description : functions for Exp BEM application
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <memory>

#include <hpro/bem/TGrid.hh>
#include <hpro/bem/TFnSpace.hh>
#include <hpro/bem/TBEMBF.hh>

#include "hlr/apps/application.hh"

namespace hlr
{

namespace apps
{

class exp : public application< double >
{
public:
    //
    // public types
    //
    
    using  value_t = double;

    // signal support for HCA based construction
    static constexpr bool supports_hca = false;
    
private:
    // BEM data
    std::unique_ptr< Hpro::TGrid >                     _grid;
    std::unique_ptr< Hpro::TFnSpace< double > >        _fnspace;
    std::unique_ptr< Hpro::TBilinearForm< value_t > >  _bf;

public:
    //
    // ctor with grid name (plus refinement levels)
    //
    exp ( const std::string &  grid,
          const double         quad_error );
    
    //
    // dtor
    //
    
    virtual ~exp () {}
    
    //
    // return coordinates for problem indices
    //
    std::unique_ptr< Hpro::TCoordinate >
    coordinates () const;
    
    //
    // return coefficient function to evaluate matrix entries
    //
    std::unique_ptr< Hpro::TCoeffFn< value_t > >
    coeff_func  () const;
};

}//namespace apps

}//namespace hlr

#endif  // __HLR_APPS_EXP_HH
