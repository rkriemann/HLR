#ifndef __HLR_CLUSTER_DISTR_HH
#define __HLR_CLUSTER_DISTR_HH
//
// Project     : HLR
// Module      : distr.cc
// Description : cluster tree distribution functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBlockCluster.hh>

namespace hlr { namespace cluster { namespace distribution {

//
// assigns 2d cyclic distribution
//
void
cyclic_2d ( const uint             nprocs,
            Hpro::TBlockCluster *  bct );

//
// assigns 1d cyclic distribution
//
void
cyclic_1d ( const uint             nprocs,
            Hpro::TBlockCluster *  bct );

//
// assigns shifted 1d cyclic distribution
//
void
shifted_cyclic_1d ( const uint             nprocs,
                    Hpro::TBlockCluster *  bct );

}}}// namespace hlr::cluster::distribution

#endif  // __HLR_DISTR_HH
