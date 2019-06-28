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

#include "hlr/utils/log.hh"
#include "hlr/utils/term.hh" // DEBUG
#include "hlr/tf/dag.hh"

using namespace HLIB;

namespace hlr
{

namespace tf
{

namespace dag
{

using hlr::dag::node;
using hlr::dag::graph;

//
// construct DAG using refinement of given node
//
graph
refine ( node *  root )
{
    assert( root != nullptr );
    
    std::deque< node * >  nodes{ root };
    std::list< node * >   tasks, start, end;
    std::mutex            mtx;
    ::tf::Executor        executor;
    ::tf::Taskflow        tf;

    while ( ! nodes.empty() )
    {
        std::deque< node * >  subnodes, del_nodes;

        // first refine nodes
        tf.parallel_for( nodes.begin(), nodes.end(),
                         [] ( node * node )
                         {
                             node->refine();
                         } );
        executor.run( tf ).wait();
        tf.clear();

        // then refine dependencies and collect new nodes
        tf.parallel_for( nodes.begin(), nodes.end(),
                         [&] ( node * node )
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
                                 // adjust dependency counter of successors (which were NOT refined!)
                                 for ( auto  succ : node->successors() )
                                     succ->inc_dep_cnt();
                                     
                                 std::scoped_lock  lock( mtx );
                                     
                                 tasks.push_back( node );
                             }// else
                         } );
        executor.run( tf ).wait();
        tf.clear();
        
        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        tf.parallel_for( del_nodes.begin(), del_nodes.end(),
                         [] ( node * node )
                         {
                             delete node;
                         } );
        executor.run( tf ).wait();
        tf.clear();
        
        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
    std::for_each( tasks.begin(), tasks.end(),
                   [&] ( node * node )
                   {
                       if ( node->dep_cnt() == 0 )
                           start.push_back( node );
                          
                       if ( node->successors().empty() )
                           end.push_back( node );
                   } );

    return graph( tasks, start, end );
}

// mapping of node to TF task
using  taskmap_t = std::unordered_map< node *, ::tf::Task >;

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    //
    // TF needs single end node
    //
    
    auto            tic = Time::Wall::now();
    taskmap_t       taskmap;
    ::tf::Taskflow  tf;

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

    ::tf::Executor  executor;
    
    tic = Time::Wall::now();
    
    executor.run( tf ).wait();

    toc = Time::Wall::since( tic );

    log( 2, "time for TF DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    // std::ofstream  ofs( "timestamps.json" );
    
    // observer->dump( ofs );
}

}// namespace TF

}// namespace DAG

}// namespace HLR
