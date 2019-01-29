#ifndef __PROBLEM_HH
#define __PROBLEM_HH
//
// Project     : HLib
// File        : problem.hh
// Description : basic interface for applications
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

struct ProblemBase
{
    //
    // set up coordinates
    //
    virtual
    std::unique_ptr< HLIB::TCoordinate >
    build_coord ( const size_t  n ) = 0;

    //
    // build matrix
    //
    virtual
    std::unique_ptr< HLIB::TMatrix >
    build_matrix ( const HLIB::TBlockClusterTree *  bct,
                   const HLIB::TTruncAcc &          acc ) = 0;
};

#endif // __PROBLEM_HH
