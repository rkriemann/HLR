#ifndef __HLR_LOGKERNEL_HH
#define __HLR_LOGKERNEL_HH
//
// Project     : HLib
// File        : hodlr-lu.hh
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include "problem.hh"

namespace LogKernel
{

struct Problem : public ProblemBase
{
    // step width
    double  h;
    
    //
    // set up coordinates
    //
    std::unique_ptr< HLIB::TCoordinate >
    build_coord ( const size_t  n );
    
    //
    // build matrix
    //
    std::unique_ptr< HLIB::TMatrix >
    build_matrix ( const HLIB::TBlockClusterTree *  bct,
                   const HLIB::TTruncAcc &          acc );
};

}//namespace LogKernel

#endif  // __HLR_LOGKERNEL_HH
