#ifndef __HLR_APPS_LOG_KERNEL_HH
#define __HLR_APPS_LOG_KERNEL_HH
//
// Project     : HLR
// Module      : apps/log_kernel
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/apps/application.hh"

namespace hlr { namespace apps {

class log_kernel : public application< double >
{
public:
    // public types
    using  value_t = double;

private:
    // problem size
    const size_t   _n;
    
    // step width
    const value_t  _h;

public:
    // ctor
    log_kernel ( const size_t  n );

    // dtor
    virtual ~log_kernel () {}
    
    // return coordinates for problem indices
    std::unique_ptr< Hpro::TCoordinate >
    coordinates () const;
    
    // return coefficient function to evaluate matrix entries
    std::unique_ptr< Hpro::TCoeffFn< value_t > >
    coeff_func  () const;
};

}}// namespace hlr::apps

#endif  // __HLR_APPS_LOGKERNEL_HH
