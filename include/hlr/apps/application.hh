#ifndef __HLR_APPS_PROBLEM_HH
#define __HLR_APPS_PROBLEM_HH
//
// Project     : HLib
// File        : problem.hh
// Description : basic interface for applications
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/matrix/TCoeffFn.hh>

namespace hlr
{

namespace hpro = HLIB;

namespace apps
{

template < typename T_value >
class application
{
public:
    using  value_t = T_value;

    // return true if problem is real/complex valued
    bool  is_real_valued    () const { return ! hpro::is_complex_type< value_t >::value; }
    bool  is_complex_valued () const { return   hpro::is_complex_type< value_t >::value; }
    
    //
    // set up coordinates
    //
    virtual
    std::unique_ptr< hpro::TCoordinate >
    coordinates () const = 0;

    //
    // return coefficient function to evaluate matrix entries
    //
    virtual
    std::unique_ptr< hpro::TCoeffFn< value_t > >
    coeff_func  () const = 0;
};

}// namespace apps

}// namespace hlr

#endif // __HLR_APPS_PROBLEM_HH
