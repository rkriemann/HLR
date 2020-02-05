//
// Project     : HLib
// File        : H.cc
// Description : H specific clustering functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/h.hh"

namespace hlr { namespace cluster { namespace h {

using namespace hpro;

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< cluster_tree >
cluster ( coordinates &  coords,
          const size_t   ntile )
{
    TCardBSPPartStrat  part_strat( adaptive_split_axis );
    TBSPCTBuilder      ct_builder( & part_strat, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< blockcluster_tree >
blockcluster ( cluster_tree &  rowct,
               cluster_tree &  colct )
{
    TStdGeomAdmCond  adm_cond( 2.0, use_min_diam );
    TBCBuilder       bct_builder;

    return bct_builder.build( & rowct, & colct, & adm_cond );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< blockcluster_tree >
blockcluster ( cluster_tree &         rowct,
               cluster_tree &         colct,
               const admissibility &  adm )
{
    TBCBuilder  bct_builder;

    return bct_builder.build( & rowct, & colct, & adm );
}

}}}// namespace hlr::cluster::h
