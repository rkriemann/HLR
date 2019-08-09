//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG sequentially
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <deque>

#include "hlr/utils/tools.hh"
#include "hlr/utils/log.hh"

#include "hlr/seq/dag.hh"

namespace hlr
{

namespace seq
{

namespace dag
{

using hlr::dag::graph;
using hlr::dag::node;

//
// construct DAG using refinement of given node
//
graph
refine ( node *        root,
         const size_t  min_size )
{
    assert( root != nullptr );
    
    std::deque< node * >  nodes{ root };
    std::list< node * >   tasks, start, end;
    const bool            output_inter = HLIB::verbose( 4 );
    uint                  step         = 0;

    while ( ! nodes.empty() )
    {
        std::deque< node * >  subnodes, del_nodes;
        bool                  any_changed = false;

        HLR_LOG( 4, HLIB::to_string( "no. of nodes in refinement step    = %d", nodes.size() ) );

        // first refine nodes
        std::for_each( nodes.begin(), nodes.end(),
                       [=,&any_changed] ( node *  node )
                       {
                           node->refine( min_size );

                           if ( node->is_refined() )
                               any_changed = true;
                       } );

        // then refine dependencies and collect new nodes
        if ( any_changed )
        {
            std::for_each( nodes.begin(), nodes.end(),
                           [&] ( node *  node )
                           {
                               const bool  node_changed = node->refine_deps( false );

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
                           } );

            // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
            std::for_each( del_nodes.begin(), del_nodes.end(),
                           [] ( node *  node )
                           {
                               delete node;
                           } );

            nodes = std::move( subnodes );
        }// if
        else
        {
            std::for_each( nodes.begin(), nodes.end(),
                           [&] ( node *  node )
                           {
                               tasks.push_back( node );

                               // adjust dependency counter of successors (which were NOT refined!)
                               for ( auto  succ : node->successors() )
                                   succ->inc_dep_cnt();
                           } );

            nodes.clear();
        }// else

        if ( output_inter )
        {
            std::list< node * >  ltasks, lstart, lend;
            
            for ( auto  n : nodes )
                ltasks.push_back( n );

            for ( auto  n : tasks )
                ltasks.push_back( n );
            
            graph  dag( ltasks, lstart, lend, hlr::dag::use_multiple_end_nodes );

            dag.print_dot( HLIB::to_string( "dag_%03d.dot", step ) );
        }// if

        ++step;
    }// while

    //
    // collect start and end nodes
    //
    
    std::for_each( tasks.begin(), tasks.end(),
                   [&] ( node *  node )
                   {
                       node->finalize();
                       
                       if ( node->dep_cnt() == 0 )
                           start.push_back( node );
                       
                       if ( node->successors().empty() )
                           end.push_back( node );
                   } );

    return graph( std::move( tasks ), std::move( start ), std::move( end ), hlr::dag::use_multiple_end_nodes );
}

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    // uint    max_deps = 0;
    // uint    avg_deps = 0;
    // node *  max_node = nullptr;
    
    // for ( auto  t : dag.nodes() )
    // {
    //     const auto ndeps = t->dep_cnt();
        
    //     avg_deps += ndeps;

    //     if ( ndeps > max_deps )
    //     {
    //         max_deps = ndeps;
    //         max_node = t;
    //     }// if
    // }// for

    // log( 0, HLIB::to_string( "max dependencies = %d", max_deps ) );
    // log( 0, HLIB::to_string( "avg dependencies = %.1f", double(avg_deps) / double(dag.nnodes()) ) );
    // log( 0, "max node : " + max_node->to_string() );
    
    // holds pending tasks
    std::list< dag::node * >  worklist;

    for ( auto  t : dag.start() )
        worklist.push_back( t );
    
    while ( ! worklist.empty() )
    {
        auto  t = behead( worklist );

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

}// namespace dag

}// namespace seq

}// namespace HLR
