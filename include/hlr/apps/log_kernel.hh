#ifndef __HLR_APPS_LOG_KERNEL_HH
#define __HLR_APPS_LOG_KERNEL_HH
//
// Project     : HLib
// File        : log_kernel.hh
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include "hlr/apps/application.hh"

namespace hlr
{

namespace apps
{

class log_kernel : public application< hpro::real >
{
public:
    //
    // public types
    //
    
    using  value_t = hpro::real;

private:
    // problem size
    const size_t   _n;
    
    // step width
    const value_t  _h;

public:
    //
    // ctor
    //
    log_kernel ( const size_t  n );

    //
    // dtor
    //
    
    virtual ~log_kernel () {}
    
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
    
    // //
    // // build matrix
    // //
    // std::unique_ptr< hpro::TMatrix >
    // build_matrix ( const hpro::TBlockClusterTree *  bct,
    //                const hpro::TTruncAcc &          acc );
};

}//namespace apps

}//namespace hlr

#endif  // __HLR_APPS_LOGKERNEL_HH
