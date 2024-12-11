#ifndef __HLR_CLUSTER_BUILD_HH
#define __HLR_CLUSTER_BUILD_HH
//
// Project     : HLR
// Module      : cluster/build
// Description : cluster tree and block tree constructions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBSPCTBuilder.hh>
#include <hpro/cluster/TBCBuilder.hh>

namespace hlr { namespace cluster {

using coordinates           = Hpro::TCoordinate;
using permutation           = Hpro::TPermutation;
using cluster               = Hpro::TCluster;
using block                 = Hpro::TBlockCluster;
using partitioning_strategy = Hpro::TBSPPartStrat;
using admissibility         = Hpro::TAdmCondition;

//
// build cluster tree
//
std::tuple< std::unique_ptr< cluster >, std::unique_ptr< permutation > >
build_cluster_tree ( const coordinates &            coord,
                     const partitioning_strategy &  part,
                     const uint                     ntile )
{
    auto  builder = Hpro::TBSPCTBuilder( &part, ntile );
    auto  ct      = builder.build( & coord );
    auto  root    = ct->root();
    auto  pe2i    = ct->perm_e2i();
    auto  pi2e    = ct->perm_i2e();

    ct->release_data();

    return { std::unique_ptr< cluster >( const_cast< cluster * >( root ) ),
             std::unique_ptr< permutation >( const_cast< permutation * >( pe2i ) ) };
}

//
// lay tensor grid over bbox of coordinates and split into cubes
// of size |bbox| / lpart per axis creating single level clustering
//
std::tuple< std::unique_ptr< cluster >, std::unique_ptr< permutation > >
build_cluster_tree_tensor ( const coordinates &  coord,
                            const uint           lpart )
{
    auto  dim = coord.dim();

    HLR_ASSERT( dim == 3 );

    //
    // compute bbox
    //

    auto  ncoord = coord.ncoord();
    auto  bbmin  = Hpro::T3Point();
    auto  bbmax  = Hpro::T3Point();
    bool  init   = false;

    for ( size_t  i = 0; i < ncoord; ++i )
    {
        auto  x_i = coord.coord(i);

        if ( ! init )
        {
            bbmax[0] = bbmin[0] = x_i[0];
            bbmax[1] = bbmin[1] = x_i[1];
            bbmax[2] = bbmin[2] = x_i[2];
            init = true;
        }// if
        else
        {
            bbmin[0] = std::min( bbmin[0], x_i[0] );
            bbmin[1] = std::min( bbmin[1], x_i[1] );
            bbmin[2] = std::min( bbmin[2], x_i[2] );

            bbmax[0] = std::max( bbmax[0], x_i[0] );
            bbmax[1] = std::max( bbmax[1], x_i[1] );
            bbmax[2] = std::max( bbmax[2], x_i[2] );
        }// else
    }// for

    //
    // split into cubes and set up leaf clusters
    //

    auto   pe2i  = std::make_unique< Hpro::TPermutation >( ncoord );
    auto   ct    = std::make_unique< Hpro::TGeomCluster >( 0, idx_t(ncoord)-1, Hpro::TBBox( bbmin, bbmax ) );
    auto   h     = Hpro::T3Point( ( bbmax[0] - bbmin[0] ) / lpart,
                                  ( bbmax[1] - bbmin[1] ) / lpart,
                                  ( bbmax[2] - bbmin[2] ) / lpart );
    idx_t  idx   = 0;
    uint   sonid = 0;

    ct->set_nsons( lpart * lpart * lpart );

    auto  within = [] ( const double  lb,
                        const double  x,
                        const double  ub )
    {
        return (( x >= lb - 1e-8 ) && ( x < ub ));
    };
        
    for ( uint  z = 0; z < lpart; ++z )
    {
        const auto  zlb = z * h[2];
        const auto  zub = ( z == lpart-1 ? bbmax[2] + 1e-6 : (z+1) * h[2] );
        
        for ( uint  y = 0; y < lpart; ++y )
        {
            const auto  ylb = y     * h[1];
            const auto  yub = ( y == lpart-1 ? bbmax[1] + 1e-6 : (y+1) * h[1] );
        
            for ( uint  x = 0; x < lpart; ++x )
            {
                const auto  xlb = x     * h[0];
                const auto  xub = ( x == lpart-1 ? bbmax[0] + 1e-6 : (x+1) * h[0] );

                //
                // sub cube is now fixed, go over coordinates and collect
                // (not very efficient!!!)
                //

                auto  first = idx;
                auto  bmin  = Hpro::T3Point();
                auto  bmax  = Hpro::T3Point();
                
                for ( size_t  i = 0; i < coord.ncoord(); ++i )
                {
                    auto  x_i = coord.coord(i);

                    if ( within( xlb, x_i[0], xub ) &&
                         within( ylb, x_i[1], yub ) &&
                         within( zlb, x_i[2], zub ) )
                    {
                        if ( idx == first )
                        {
                            bmax[0] = bmin[0] = x_i[0];
                            bmax[1] = bmin[1] = x_i[1];
                            bmax[2] = bmin[2] = x_i[2];
                        }// if
                        else
                        {
                            bmin[0] = std::min( bmin[0], x_i[0] );
                            bmin[1] = std::min( bmin[1], x_i[1] );
                            bmin[2] = std::min( bmin[2], x_i[2] );

                            bmax[0] = std::max( bmax[0], x_i[0] );
                            bmax[1] = std::max( bmax[1], x_i[1] );
                            bmax[2] = std::max( bmax[2], x_i[2] );
                        }// else
                        
                        (*pe2i)[i] = idx++;
                    }// if
                }// for

                HLR_ASSERT( first != idx );

                // std::cout << sonid << " : " << idx - first << std::endl;
                
                auto  son = std::make_unique< Hpro::TGeomCluster >( first, idx-1, Hpro::TBBox( bmin, bmax ) );

                ct->set_son( sonid++, son.release() );
            }// for
        }// for
    }// for

    HLR_ASSERT( idx == ncoord );

    return { std::move( ct ), std::move( pe2i ) };
}
    
//
// build block tree
//
std::unique_ptr< block >
build_block_tree ( const cluster &        rowcl,
                   const cluster &        colcl,
                   const admissibility &  adm )
{
    auto  builder = Hpro::TBCBuilder();

    return builder.build( & rowcl, & colcl, & adm );
}
    
}}// hlr::cluster

#endif  // __HLR_CLUSTER_BUILD_HH
