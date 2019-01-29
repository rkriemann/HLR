#ifndef __HLR_MATERN_HH
#define __HLR_MATERN_HH
//
// Project     : HLib
// File        : matern.hh
// Description : functions for Matern covariance function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "problem.hh"

namespace Matern
{

struct Problem : public ProblemBase
{
    // coordinates
    std::vector< HLIB::T3Point >  vertices;
    
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

}//namespace Matern

#endif  // __HLR_MATERN_HH
