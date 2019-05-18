//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG sequentially
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <deque>

#include "utils/tools.hh"
#include "utils/log.hh"

#include "seq/dag.hh"

namespace HLR
{

namespace DAG
{

namespace Seq
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
    // adjust dependency counter
    //
    
    // for ( auto  t : tasks )
    // {
    //     for ( auto  succ : t->successors() )
    //         succ->inc_dep_cnt();
    // }// for

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
    // holds pending tasks
    std::list< DAG::Node * >  worklist;

    for ( auto  t : dag.start() )
        worklist.push_back( t );
    
    while ( ! worklist.empty() )
    {
        auto  t = behead( worklist );

        log( 4, t->to_string() );
        
        t->run( acc );

        for ( auto  succ : t->successors() )
        {
            auto  deps = succ->dec_dep_cnt();

            assert( deps >= 0 );
            
            if ( deps == 0 )
                worklist.push_front( succ );
        }// for
    }// while
}

}// namespace Seq

}// namespace DAG

}// namespace DAG
