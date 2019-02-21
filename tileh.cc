//
// Project     : HLib
// File        : tileh.cc
// Description : common Tile-H functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <cluster/TBSPCTBuilder.hh>
#include <cluster/TBSPPartStrat.hh>
#include <cluster/TBCBuilder.hh>
#include <cluster/TGeomAdmCond.hh>

#include "tileh.hh"

namespace TileH
{

using namespace HLIB;

//
// flatten top <n> levels of cluster tree
//
void
flatten ( TCluster *  cl,
          const uint  nlevel )
{
    if (( cl == nullptr ) || ( cl->is_leaf() ))
        return;

    //
    // collect nodes on level <nlevel> 
    //

    std::list< TCluster * >  lvl_nodes;
    std::list< TCluster * >  del_nodes;
    uint                     lvl = 0;

    lvl_nodes.push_back( cl );
    
    while ( lvl < nlevel )
    {
        std::list< TCluster * >  sons;

        for ( auto &&  node : lvl_nodes )
        {
            // ensure that we do not have leaves before <nlevel>
            assert( ! node->is_leaf() );

            // remember node for later removal
            if ( node != cl )
                del_nodes.push_back( node );
            
            for ( uint  i = 0; i < node->nsons(); ++i )
            {
                auto  son_i = node->son(i);
        
                if ( son_i != nullptr )
                    sons.push_back( son_i );
            }// for
        }// for

        lvl_nodes = std::move( sons );
        ++lvl;
    }// while

    //
    // remove inbetween nodes
    //

    for ( auto &&  node : del_nodes )
    {
        // reset son pointers
        for ( uint  i = 0; i < node->nsons(); ++i )
            node->set_son( i, nullptr, false );

        delete node;
    }// for

    // also reset root son pointers
    for ( uint  i = 0; i < cl->nsons(); ++i )
        cl->set_son( i, nullptr, false );
    
    //
    // set collected sons as sons of root node
    //

    size_t  pos = 0;
    
    cl->set_nsons( lvl_nodes.size() );

    for ( auto &&  node : lvl_nodes )
        cl->set_son( pos++, node );
}

//
// set up cluster and block cluster tree
//
std::pair< std::unique_ptr< TClusterTree >,
           std::unique_ptr< TBlockClusterTree > >
cluster ( TCoordinate *  coords,
          const size_t   ntile,
          const int      nprocs )
{
    TCardBSPPartStrat    part_strat;
    TBSPCTBuilder        ct_builder( & part_strat, ntile );

    auto  ct = ct_builder.build( coords );

    // flatten top levels to set up Tile-H
    flatten( ct->root(), std::max<uint>( 3, std::ceil( std::log2( nprocs )+1 ) ) );
    
    TWeakStdGeomAdmCond  adm_cond;
    TBCBuilder           bct_builder;

    auto  bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

    return { std::move( ct ), std::move( bct ) };
}

}// namespace TileH
