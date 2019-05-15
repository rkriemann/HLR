//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>
#include <unordered_map>

#include <hpx/async.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/unwrap.hpp>

#include "utils/tools.hh"
#include "utils/log.hh"

#include "hpx/dag.hh"

using namespace HLIB;

// HPX types for tasks and dependencies
using  task_t         = hpx::shared_future< void >;
using  dependencies_t = std::list< hpx::shared_future< void > >;

namespace HLR
{

namespace DAG
{

namespace
{

//
// execute node without dependencies
//
void
run_node ( Node *             node,
           const TTruncAcc &  acc )
{
    log( 4, "run_node : " + node->to_string() );
    
    node->run( acc );
}

//
// execute node with dependencies
//
void
run_node_dep ( Node *             node,
               const TTruncAcc &  acc,
               dependencies_t     dep )
{
    log( 4, "run_node_dep : " + node->to_string() );
    
    node->run( acc );
}

}// namespace anonymous

namespace HPX
{

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc )
{
    using hpx::async;
    using hpx::dataflow;
    using hpx::when_all;
    using hpx::util::unwrapping;
    
    //
    // use single end node to not wait sequentially for all
    // original end nodes (and purely use HPX framework)
    //

    auto         tic          = Time::Wall::now();
    DAG::Node *  final        = nullptr;
    bool         multiple_end = false;

    if ( dag.end().size() > 1 )
    {
        log( 5, "DAG::HPX::run : multiple end nodes" );

        multiple_end = true;
        
        //
        // create single special end node
        //

        final = new DAG::EmptyNode();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() );
    }// if

    // map of DAG nodes to tasks
    std::unordered_map< Node *, task_t >          taskmap;

    // keep track of dependencies for a node
    std::unordered_map< Node *, dependencies_t >  nodedeps;

    //
    // Go through DAG, decrement dependency counter for each successor of
    // current node and if this reaches zero, add node to list of nodes
    // to be visited. Since now all dependencies are met, all tasks for
    // nodes exist and "when_all( dependency-set )" can be constructed.
    //
    // For start nodes, dependency set is empty, so use "async".
    //
    
    std::list< Node * >  nodes;

    for ( auto  node : dag.start() )
    {
        log( 4, "async( " + node->to_string() + " )" );

        task_t  task = async( run_node, node, acc );
        
        taskmap[ node ] = task;

        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( task );

            if ( succ->dec_dep_cnt() == 0 )
                nodes.push_back( succ );
        }// for
    }// for

    while ( ! nodes.empty() )
    {
        auto  node = behead( nodes );
        
        log( 4, "dataflow( " + node->to_string() + " )" );
        
        task_t  task = hpx::dataflow( unwrapping( run_node_dep ), node, acc, when_all( nodedeps[ node ] ) );
        
        taskmap[ node ] = task;
        
        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( task );
            
            if ( succ->dec_dep_cnt() == 0 )
                nodes.push_back( succ );
        }// for
    }// while

    // if DAG originally had single end node, get pointer to it
    if ( final == nullptr )
        final = dag.end().front();
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for HPX DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // start execution by requesting future result for end node
    //

    tic = Time::Wall::now();
    
    taskmap[ final ].get();

    toc = Time::Wall::since( tic );

    log( 2, "time for HPX DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // remove auxiliary end node from DAG
    //
    
    if ( multiple_end )
    {
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if
}

}// namespace HPX

}// namespace DAG

}// namespace HLR
