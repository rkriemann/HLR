//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>
#include <unordered_map>
#include <deque>
#include <cassert>

#include "utils/tools.hh"
#include "utils/log.hh"

#include "omp/dag.hh"

using namespace HLIB;

namespace HLR 
{

namespace DAG
{

namespace OMP
{

//
// construct DAG using refinement of given node
//
Graph
refine ( Node *  root )
{
    assert( root != nullptr );
    
    std::deque< Node * >  nodes;
    std::list< Node * >   tasks, start, end;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( Node * node )
        {
            const bool  node_changed = node->refine_deps();

            if ( node->is_refined() )       // node was refined; collect all sub nodes
            {
                for ( auto  sub : node->sub_nodes() )
                    subnodes.push_back( sub );
                    
                del_nodes.push_back( node );
            }// if
            else if ( node_changed )        // node was not refined but dependencies were
            {
                subnodes.push_back( node );
            }// if
            else                            // neither node nor dependencies changed: reached final state
            {
                tasks.push_back( node );

                // adjust dependency counter of successors (which were NOT refined!)
                for ( auto  succ : node->successors() )
                    succ->inc_dep_cnt();
            }// else
        };

        // first refine nodes
        std::for_each( nodes.begin(), nodes.end(),
                       [] ( Node * node ) { node->refine(); } );

        // then refine dependencies and collect new nodes
        std::for_each( nodes.begin(), nodes.end(),
                       node_dep_refine );

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        std::for_each( del_nodes.begin(), del_nodes.end(),
                       [] ( Node * node ) { delete node; } );
        
        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
    // for ( auto  t : tasks )
    std::for_each( tasks.begin(), tasks.end(),
                   [&] ( Node * node )
                   {
                       if ( node->dep_cnt() == 0 )
                           start.push_back( node );
                       
                       if ( node->successors().empty() )
                           end.push_back( node );
                   } );

    return Graph( tasks, start, end );
}

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc )
{
    auto tic = Time::Wall::now();

    // keep track of dependencies for a node
    using  dep_list_t = std::list< Node * >;
    using  dep_vec_t  = std::vector< Node * >;

    std::unordered_map< Node *, dep_list_t >  dep_lists;
    std::unordered_map< Node *, dep_vec_t  >  dep_vecs;

    //
    // add node to dependency list of successors
    //
    
    for ( auto  node : dag.nodes() )
        for ( auto  succ : node->successors() )
            dep_lists[ succ ].push_back( node );

    //
    // convert lists to vectors
    //

    for ( auto  node : dag.nodes() )
    {
        dep_list_t &  dep_list = dep_lists[ node ];

        dep_vecs[ node ].reserve( dep_list.size() );

        for ( auto  dep : dep_list )
            dep_vecs[ node ].push_back( dep );
    }// for
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // loop through nodes and create OpenMP task with dependencies from dep_vecs
    //

    tic = Time::Wall::now();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for ( auto  node : dag.nodes() )
            {
                dep_vec_t &  deps  = dep_vecs[ node ];
                auto         cdeps = & deps[0];
                const auto   ndeps = deps.size();
                
                #pragma omp task depend( in : cdeps[0:ndeps] ) depend( out : node )
                {
                    node->run( acc );
                }// omp task
            }// for
        }// omp single
    }// omp parallel

    toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace OMP

}// namespace DAG

}// namespace HLR
