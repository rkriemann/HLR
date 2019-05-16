#ifndef __HLR_APPS_LAPLACE_HH
#define __HLR_APPS_LAPLACE_HH
//
// Project     : HLib
// File        : Laplace.hh
// Description : functions for Laplace SLP/DLP BEM application
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include <memory>

#include <bem/TGrid.hh>
#include <bem/TFnSpace.hh>
#include <bem/TBEMBF.hh>

#include "apps/problem.hh"

namespace HLR
{

using namespace HLIB;

namespace Apps
{

class LaplaceSLP : public Application< HLIB::real >
{
public:
    //
    // public types
    //
    
    using  value_t = HLIB::real;

private:
    // BEM data
    std::unique_ptr< TGrid >                     _grid;
    std::unique_ptr< TFnSpace >                  _fnspace;
    std::unique_ptr< TBilinearForm< value_t > >  _bf;

public:
    //
    // ctor with grid name (plus refinement levels)
    //
    LaplaceSLP ( const std::string &  grid );
    
    //
    // dtor
    //
    
    virtual ~LaplaceSLP () {}
    
    //
    // return coordinates for problem indices
    //
    std::unique_ptr< HLIB::TCoordinate >
    coordinates () const;
    
    //
    // return coefficient function to evaluate matrix entries
    //
    std::unique_ptr< HLIB::TCoeffFn< value_t > >
    coeff_func  () const;
};

}//namespace Apps

}//namespace HLR

#endif  // __HLR_APPS_LAPLACE_HH
