#ifndef __HLR_DISTR_HH
#define __HLR_DISTR_HH
//
// Project     : HLR-HPC
// File        : distr.cc
// Description : cluster tree distribution functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cluster/TBlockCluster.hh>

namespace distribution
{

//
// assigns 2d cyclic distribution
//
void
cyclic_2d ( const uint             nprocs,
            HLIB::TBlockCluster *  bct );

//
// assigns shifted 1d cyclic distribution
//
void
shifted_cyclic_1d ( const uint             nprocs,
                    HLIB::TBlockCluster *  bct );

}// namespace distribution

#endif  // __HLR_DISTR_HH
