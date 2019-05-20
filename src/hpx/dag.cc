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
#include <hpx/include/parallel_for_each.hpp>

#include "hlr/utils/tools.hh"
#include "hlr/utils/log.hh"

#include "hlr/hpx/dag.hh"

namespace hlr
{

using namespace HLIB;

namespace hpx
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
    
    std::deque< node * >  nodes;
    std::list< node * >   tasks, start, end;
    std::mutex            mtx;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( node * node )
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
        ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                 nodes.begin(), nodes.end(),
                                 [] ( node * node ) { node->refine(); } );

        // then refine dependencies and collect new nodes
        ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                 nodes.begin(), nodes.end(),
                                 node_dep_refine );

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                 del_nodes.begin(), del_nodes.end(),
                                 [] ( node * node ) { delete node; } );
        
        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
    std::for_each( tasks.begin(), tasks.end(),
        //tbb::parallel_do( tasks,
                   [&] ( node * node )
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

    return graph( tasks, start, end );
}

namespace
{

// HPX types for tasks and dependencies
using  task_t         = ::hpx::shared_future< void >;
using  dependencies_t = std::list< ::hpx::shared_future< void > >;

//
// execute node without dependencies
//
void
run_node ( node *             node,
           const TTruncAcc &  acc )
{
    log( 4, "run_node : " + node->to_string() );
    
    node->run( acc );
}

//
// execute node with dependencies
//
void
run_node_dep ( node *             node,
               const TTruncAcc &  acc,
               dependencies_t     dep )
{
    log( 4, "run_node_dep : " + node->to_string() );
    
    node->run( acc );
}

}// namespace anonymous

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    using ::hpx::async;
    using ::hpx::dataflow;
    using ::hpx::when_all;
    using ::hpx::util::unwrapping;
    
    //
    // use single end node to not wait sequentially for all
    // original end nodes (and purely use HPX framework)
    //

    auto    tic          = Time::Wall::now();
    node *  final        = nullptr;
    bool    multiple_end = false;

    if ( dag.end().size() > 1 )
    {
        log( 5, "dag::hpx::run : multiple end nodes" );

        multiple_end = true;
        
        //
        // create single special end node
        //

        final = new hlr::dag::empty_node();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() );
    }// if
    else
        final = dag.end().front();

    //
    // Go through DAG, decrement dependency counter for each successor of
    // current node and if this reaches zero, add node to list of nodes
    // to be visited. Since now all dependencies are met, all tasks for
    // nodes exist and "when_all( dependency-set )" can be constructed.
    //
    // For start nodes, dependency set is empty, so use "async".
    //
    
    // map of DAG nodes to tasks
    std::unordered_map< node *, task_t >          taskmap;

    // keep track of dependencies for a node
    std::unordered_map< node *, dependencies_t >  nodedeps;

    // list of "active" nodes
    std::list< node * >                           nodes;

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
        
        task_t  task = ::hpx::dataflow( unwrapping( run_node_dep ), node, acc, when_all( nodedeps[ node ] ) );
        
        taskmap[ node ] = task;
        
        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( task );
            
            if ( succ->dec_dep_cnt() == 0 )
                nodes.push_back( succ );
        }// for
    }// while

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
