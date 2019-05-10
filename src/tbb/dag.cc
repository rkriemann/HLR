//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <unordered_map>
#include <cassert>

#include <tbb/task.h>

#include "tbb/dag.hh"

using namespace HLIB;

// enables some debug output
#define  log( lvl, msg )  if ( HLIB::verbose( lvl ) ) DBG::print( msg )

namespace HLR
{

namespace DAG
{

namespace
{

class RuntimeTask;

// mapping of Node to TBB task
using  taskmap_t = std::unordered_map< Node *, tbb::task * >;

//
// helper class for executing Node via TBB
//
class RuntimeTask : public tbb::task
{
private:
    DAG::Node *        _node;
    const TTruncAcc &  _acc;
    taskmap_t &        _taskmap;
    
public:
    RuntimeTask ( DAG::Node *        anode,
                  const TTruncAcc &  aacc,
                  taskmap_t &        ataskmap )
            : _node( anode )
            , _acc( aacc )
            , _taskmap( ataskmap )
    {
        set_ref_count( _node->dep_cnt() );
    }

    tbb::task *  execute ()
    {
        _node->run( _acc );

        for ( auto  succ : _node->successors() )
        {
            // auto  succ_task = static_cast< tbb::task * >( succ->task() );
            auto  succ_task = _taskmap[ succ ];

            assert( succ_task != nullptr );
            
            if ( succ_task->decrement_ref_count() == 0 )
                spawn( * succ_task );
        }// for

        return nullptr;
    }
};

}// namespace anonymous

namespace TBB
{

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc )
{
    //
    // TBB needs single end node
    //
    
    auto         tic          = Time::Wall::now();
    DAG::Node *  final        = nullptr;
    bool         multiple_end = false;
    taskmap_t    taskmap;

    if ( dag.end().size() > 1 )
    {
        log( 5, "DAG::TBB::run : multiple end nodes" );

        multiple_end = true;
        
        //
        // create single special end node
        //

        final = new DAG::EmptyNode();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() ); // final->in.size();

        taskmap[ final ] = new ( tbb::task::allocate_root() ) DAG::RuntimeTask( final, acc, taskmap );
    }// if

    // create tbb tasks for all nodes
    for ( auto  node : dag.nodes() )
        taskmap[ node ] = new ( tbb::task::allocate_root() ) DAG::RuntimeTask( node, acc, taskmap );
    
    // if DAG has single end node, get pointer to it
    if ( final == nullptr )
        final = dag.end().front();
    
    tbb::task_list  work_queue;
    
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

    log( 3, "time for TBB DAG runtime = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    auto  final_task = taskmap[ final ];
    
    assert( final_task != nullptr );
    
    final_task->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final_task->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final_task->execute();                            // and the final node explicitly
    tbb::task::destroy( * final_task );               // not done by TBB since executed manually

    if ( multiple_end )
    {
        //
        // remove node from DAG
        //
        
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if
}

}// namespace TBB

}// namespace DAG

}// namespace HLR
