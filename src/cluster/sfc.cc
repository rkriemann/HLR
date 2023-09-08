//
// Project     : HLR
// Module      : sfc.cc
// Description : Hilbert curve based clustering
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBSPPartStrat.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include "hlr/cluster/sfc.hh"

namespace hlr { namespace cluster { namespace sfc {

//
// cluster set of coordinates with minimal block size <ntile>
//
std::unique_ptr< cluster_tree >
cluster ( const part_type_t  part_type,
          coordinates &      coords,
          const size_t       ntile )
{
    Hpro::TSFCCTBuilder  ct_builder( part_type, ntile );

    return ct_builder.build( & coords );
}

//
// build block cluster tree based on given row/column cluster trees
//
std::unique_ptr< blockcluster_tree >
blockcluster ( cluster_tree &  rowct,
               cluster_tree &  colct )
{
    Hpro::TStdGeomAdmCond  adm_cond( 2.0, Hpro::use_min_diam );
    Hpro::TBCBuilder       bct_builder;

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
    Hpro::TBCBuilder  bct_builder;

    return bct_builder.build( & rowct, & colct, & adm );
}

}}}// namespace hlr::cluster::sfc
