//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <unordered_map>
#include <cassert>

#include <taskflow/taskflow.hpp>

#include "utils/log.hh"
#include "tf/dag.hh"

using namespace HLIB;

namespace HLR
{

namespace DAG
{

namespace
{

// mapping of Node to TF task
using  taskmap_t = std::unordered_map< Node *, tf::Task >;

}// namespace anonymous

namespace TF
{

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
    
    auto          tic          = Time::Wall::now();
    taskmap_t     taskmap;
    tf::Taskflow  tf;
    
    // create TF tasks for all nodes
    for ( auto  node : dag.nodes() )
        taskmap[ node ] = tf.silent_emplace( [node,&acc] () { node->run( acc ); } );
    
    // set up dependencies
    for ( auto  node : dag.nodes() )
        for ( auto  succ : node->successors() )
            taskmap[ node ].precede( taskmap[ succ ] );
    
    auto  toc = Time::Wall::since( tic );

    log( 3, "time for TF DAG runtime = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    // run tasks
    tf.wait_for_all();
}

}// namespace TF

}// namespace DAG

}// namespace HLR
