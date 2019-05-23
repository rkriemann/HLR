//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <unordered_map>
#include <cassert>
#include <deque>

#include <tbb/task.h>
#include <tbb/parallel_for_each.h>
#include <tbb/mutex.h>

#include "hlr/utils/log.hh"
#include "hlr/tbb/dag.hh"

using namespace HLIB;

namespace hlr
{

namespace tbb
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
    
    std::deque< node * >   nodes{ root };
    std::list< node * >    tasks;
    ::tbb::mutex           mtx;
    
    while ( ! nodes.empty() )
    {
        std::deque< node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( node * node )
        {
            const bool  node_changed = node->refine_deps();

            if ( node->is_refined() )       // node was refined; collect all sub nodes
            {
                ::tbb::mutex::scoped_lock  lock( mtx );
                    
                for ( auto  sub : node->sub_nodes() )
                    subnodes.push_back( sub );
                    
                del_nodes.push_back( node );
            }// if
            else if ( node_changed )        // node was not refined but dependencies were
            {
                ::tbb::mutex::scoped_lock  lock( mtx );
                    
                subnodes.push_back( node );
            }// if
            else                            // neither node nor dependencies changed: reached final state
            {
                // adjust dependency counter of successors (which were NOT refined!)
                for ( auto  succ : node->successors() )
                    succ->inc_dep_cnt();

                ::tbb::mutex::scoped_lock  lock( mtx );
                    
                tasks.push_back( node );
            }// else
        };

        // first refine nodes
        ::tbb::parallel_for_each( nodes.begin(), nodes.end(),
                                  [] ( node * node ) { node->refine(); } );

        // then refine dependencies and collect new nodes
        ::tbb::parallel_for_each( nodes.begin(), nodes.end(),
                                  node_dep_refine );

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        std::for_each( del_nodes.begin(), del_nodes.end(),
                       [] ( node * node ) { delete node; } );

        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
    std::list< node * >   start, end;
    
    // for ( auto  t : tasks )
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

namespace
{

#if 1

//
// helper class for executing node via TBB
//
class runtime_task : public ::tbb::task
{
private:
    node *             _node;
    const TTruncAcc &  _acc;
    node *             _final;
    ::tbb::task *      _final_task;
    
public:
    runtime_task ( node *             anode,
                   const TTruncAcc &  aacc,
                   node *             afinal,
                   ::tbb::task *      afinal_task )
            : _node( anode )
            , _acc( aacc )
            , _final( afinal )
            , _final_task( afinal_task )
    {}

    ::tbb::task *  execute ()
    {
        _node->run( _acc );

        for ( auto  succ : _node->successors() )
        {
            if ( succ == _final )
            {
                succ->dec_dep_cnt();
                if ( _final_task->decrement_ref_count() == 0 )
                    spawn( * _final_task );
            }// if
            else if ( succ->dec_dep_cnt() == 0 )
            {
                spawn( * new ( ::tbb::task::allocate_root() ) runtime_task( succ, _acc, _final, _final_task ) );
            }// if
        }// for

        return nullptr;
    }
};

}// namespace anonymous

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    auto       tic          = Time::Wall::now();

    //
    // ensure only single end node
    //
    
    node *         final        = nullptr;
    bool           multiple_end = false;
    ::tbb::task *  final_task   = nullptr;
    
    if ( dag.end().size() > 1 )
    {
        log( 5, "tbb::dag::run : multiple end nodes" );

        multiple_end = true;
        
        final = new hlr::dag::empty_node();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() );

        final_task = new ( ::tbb::task::allocate_root() ) runtime_task( final, acc, final, nullptr );
        final_task->set_ref_count( final->dep_cnt() );
    }// if
    else
    {
        final      = dag.end().front();
        final_task = new ( ::tbb::task::allocate_root() ) runtime_task( final, acc, final, nullptr );
        final_task->set_ref_count( final->dep_cnt() );
    }// else

    //
    // set up start tasks
    //
    
    ::tbb::task_list  work_queue;
    
    for ( auto  node : dag.start() )
    {
        if ( node != final )
        {
            auto  task = new ( ::tbb::task::allocate_root() ) runtime_task( node, acc, final, final_task );
            
            work_queue.push_back( * task );
        }// if
    }// for

    auto  toc = Time::Wall::since( tic );

    log( 2, "time for TBB DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // run DAG
    //
    
    tic = Time::Wall::now();
    
    final_task->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final_task->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final_task->execute();                            // and the final node explicitly
    ::tbb::task::destroy( * final_task );             // not done by TBB since executed manually

    toc = Time::Wall::since( tic );

    log( 2, "time for TBB DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // remove auxiliary node from DAG
    //
        
    if ( multiple_end )
    {
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if
}

#else

// mapping of node to TBB task
using  taskmap_t = std::unordered_map< node *, ::tbb::task * >;

//
// helper class for executing node via TBB
//
class runtime_task : public ::tbb::task
{
private:
    node *             _node;
    const TTruncAcc &  _acc;
    taskmap_t &        _taskmap;
    
public:
    runtime_task ( node *             anode,
                  const TTruncAcc &  aacc,
                  taskmap_t &        ataskmap )
            : _node( anode )
            , _acc( aacc )
            , _taskmap( ataskmap )
    {
        set_ref_count( _node->dep_cnt() );
    }

    ::tbb::task *  execute ()
    {
        _node->run( _acc );

        for ( auto  succ : _node->successors() )
        {
            auto  succ_task = _taskmap[ succ ];

            assert( succ_task != nullptr );
            
            if ( succ_task->decrement_ref_count() == 0 )
                spawn( * succ_task );
        }// for

        return nullptr;
    }
};

}// namespace anonymous

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    auto       tic          = Time::Wall::now();

    //
    // ensure only single end node
    //
    
    node *     final        = nullptr;
    bool       multiple_end = false;
    taskmap_t  taskmap;
    
    if ( dag.end().size() > 1 )
    {
        log( 5, "tbb::dag::run : multiple end nodes" );

        multiple_end = true;
        
        final = new hlr::dag::empty_node();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() );

        taskmap[ final ] = new ( ::tbb::task::allocate_root() ) runtime_task( final, acc, taskmap );
    }// if
    else
        final = dag.end().front();

    //
    // create tbb tasks for all nodes
    //
    
    for ( auto  node : dag.nodes() )
        taskmap[ node ] = new ( ::tbb::task::allocate_root() ) runtime_task( node, acc, taskmap );

    //
    // set up start tasks
    //
    
    ::tbb::task_list  work_queue;
    
    for ( auto  node : dag.start() )
    {
        if ( node != final )
        {
            auto  task = taskmap[ node ];

            assert( task != nullptr );
            
            work_queue.push_back( * task );
        }// if
    }// for

    auto  toc = Time::Wall::since( tic );

    log( 2, "time for TBB DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // run DAG
    //
    
    tic = Time::Wall::now();
    
    auto  final_task = taskmap[ final ];
    
    assert( final_task != nullptr );
    
    final_task->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final_task->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final_task->execute();                            // and the final node explicitly
    ::tbb::task::destroy( * final_task );             // not done by TBB since executed manually

    toc = Time::Wall::since( tic );

    log( 2, "time for TBB DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // remove auxiliary node from DAG
    //
        
    if ( multiple_end )
    {
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if
}

#endif

}// namespace dag

}// namespace tbb

}// namespace hlr
