#ifndef __HLR_APPS_PROBLEM_HH
#define __HLR_APPS_PROBLEM_HH
//
// Project     : HLR
// Module      : problem.hh
// Description : basic interface for applications
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TCoordinate.hh>
#include <hpro/matrix/TCoeffFn.hh>

namespace hlr { namespace apps {

template < typename T_value >
class application
{
public:
    using  value_t = T_value;

    // signal support for HCA based construction
    static constexpr bool supports_hca = false;
    
    // return true if problem is real/complex valued
    bool  is_real_valued    () const { return ! Hpro::is_complex_type_v< value_t >; }
    bool  is_complex_valued () const { return   Hpro::is_complex_type_v< value_t >; }
    
    //
    // set up coordinates
    //
    virtual
    std::unique_ptr< Hpro::TCoordinate >
    coordinates () const = 0;

    //
    // return coefficient function to evaluate matrix entries
    //
    virtual
    std::unique_ptr< Hpro::TCoeffFn< value_t > >
    coeff_func  () const = 0;
};

}// namespace apps

}// namespace hlr

#endif // __HLR_APPS_PROBLEM_HH
