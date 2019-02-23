#ifndef __HLR_APPS_PROBLEM_HH
#define __HLR_APPS_PROBLEM_HH
//
// Project     : HLib
// File        : problem.hh
// Description : basic interface for applications
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cluster/TCoordinate.hh>
#include <matrix/TCoeffFn.hh>

namespace HLR
{

namespace Apps
{

template < typename T_value >
class Application
{
public:
    using  value_t = T_value;
    
    //
    // set up coordinates
    //
    virtual
    std::unique_ptr< HLIB::TCoordinate >
    coordinates () const = 0;

    //
    // return coefficient function to evaluate matrix entries
    //
    virtual
    std::unique_ptr< HLIB::TCoeffFn< value_t > >
    coeff_func  () const = 0;
    
    // //
    // // build matrix
    // //
    // virtual
    // std::unique_ptr< HLIB::TMatrix >
    // build_matrix ( const HLIB::TBlockClusterTree *  bct,
    //                const HLIB::TTruncAcc &          acc ) = 0;
};

}// namespace Apps

}// namespace HLR

#endif // __HLR_APPS_PROBLEM_HH
