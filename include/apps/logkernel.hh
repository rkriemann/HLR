#ifndef __HLR_APPS_LOGKERNEL_HH
#define __HLR_APPS_LOGKERNEL_HH
//
// Project     : HLib
// File        : logkernel.hh
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include "apps/problem.hh"

namespace HLR
{

namespace Apps
{

class LogKernel : public Application< HLIB::real >
{
public:
    //
    // public types
    //
    
    using  value_t = HLIB::real;

private:
    // problem size
    const size_t   _n;
    
    // step width
    const value_t  _h;

public:
    //
    // ctor
    //
    LogKernel ( const size_t  n );
    
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
    
    // //
    // // build matrix
    // //
    // std::unique_ptr< HLIB::TMatrix >
    // build_matrix ( const HLIB::TBlockClusterTree *  bct,
    //                const HLIB::TTruncAcc &          acc );
};

}//namespace Apps

}//namespace HLR

#endif  // __HLR_APPS_LOGKERNEL_HH
