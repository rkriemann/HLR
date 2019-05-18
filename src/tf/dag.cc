//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>
#include <unordered_map>
#include <cassert>
#include <mutex>

#include <taskflow/taskflow.hpp>

#include "utils/log.hh"
#include "tf/dag.hh"

using namespace HLIB;

namespace HLR
{

namespace DAG
{

namespace TF
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
    std::mutex            mtx;
    tf::Taskflow          tf;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( Node * node )
        {
            const bool  node_changed = node->refine_deps();

            if ( node->is_refined() )       // node was refined; collect all sub nodes
            {
                std::scoped_lock  lock( mtx );
                    
                for ( auto  sub : node->sub_nodes() )
                    subnodes.push_back( sub );
                    
                del_nodes.push_back( node );
            }// if
            else if ( node_changed )        // node was not refined but dependencies were
            {
                std::scoped_lock  lock( mtx );
                    
                subnodes.push_back( node );
            }// if
            else                            // neither node nor dependencies changed: reached final state
            {
                {
                    std::scoped_lock  lock( mtx );
                    
                    tasks.push_back( node );
                }

                // adjust dependency counter of successors (which were NOT refined!)
                for ( auto  succ : node->successors() )
                    succ->inc_dep_cnt();
            }// else
        };

        // first refine nodes
        tf.parallel_for( nodes.begin(), nodes.end(),
                         [] ( Node * node ) { node->refine(); } );
        tf.wait_for_all();

        // then refine dependencies and collect new nodes
        tf.parallel_for( nodes.begin(), nodes.end(),
                         node_dep_refine );
        tf.wait_for_all();

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        tf.parallel_for( del_nodes.begin(), del_nodes.end(),
                         [] ( Node * node ) { delete node; } );
        tf.wait_for_all();
        
        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
    std::for_each( tasks.begin(), tasks.end(),
                   [&] ( Node * node )
                   {
                       if ( node->dep_cnt() == 0 )
                       {
                           std::scoped_lock  lock( mtx );
                              
                           start.push_back( node );
                       }// if
                          
                       if ( node->successors().empty() )
                       {
                           std::scoped_lock  lock( mtx );
                              
                           end.push_back( node );
                       }// if
                   } );

    return Graph( tasks, start, end );
}

// mapping of Node to TF task
using  taskmap_t = std::unordered_map< Node *, tf::Task >;

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc )
{
    //
    // TF needs single end node
    //
    
    auto          tic = Time::Wall::now();
    taskmap_t     taskmap;
    tf::Taskflow  tf;

    // auto          observer = tf.share_executor()->make_observer< tf::ExecutorObserver >();
    
    // create TF tasks for all nodes
    for ( auto  node : dag.nodes() )
        taskmap[ node ] = tf.silent_emplace( [node,&acc] () { node->run( acc ); } );
    
    // set up dependencies
    for ( auto  node : dag.nodes() )
        for ( auto  succ : node->successors() )
            taskmap[ node ].precede( taskmap[ succ ] );
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for TF DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // run tasks
    //
    
    tic = Time::Wall::now();
    
    tf.wait_for_all();

    toc = Time::Wall::since( tic );

    log( 2, "time for TF DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    // std::ofstream  ofs( "timestamps.json" );
    
    // observer->dump( ofs );
}

}// namespace TF

}// namespace DAG

}// namespace HLR
