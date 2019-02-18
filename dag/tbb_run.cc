//
// Project     : HLib
// File        : tbb_run.cc
// Description : execute DAG using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/task.h>

#include "tbb_run.hh"

using namespace HLIB;

namespace DAG
{

//
// helper class for executing Node via TBB
//
class RuntimeTask : public tbb::task
{
private:
    Node *             _node;
    const TTruncAcc &  _acc;
    
public:
    RuntimeTask ( Node *             anode,
                  const TTruncAcc &  aacc )
            : _node( anode )
            , _acc( aacc )
    {
        set_ref_count( _node->dep_cnt() );
    }

    tbb::task *  execute ()
    {
        _node->run( _acc );

        for ( auto  succ : _node->successors() )
        {
            auto  succ_task = static_cast< tbb::task * >( succ->task() );
            
            if ( succ_task->decrement_ref_count() == 0 )
                spawn( * succ_task );
        }// for

        return nullptr;
    }
};

namespace TBB
{

// enables some debug output
#define  LOG( lvl, msg )  if ( HLIB::verbose( lvl ) ) DBG::print( msg )

//////////////////////////////////////////////
//
// node for collecting dependencies
// without any computation
//
//////////////////////////////////////////////

struct EmptyNode : public Node
{
    // return text version of node
    virtual std::string  to_string () const { return "Empty"; }

    // (optional) color for DAG visualization (format: RRGGBB)
    virtual std::string  color     () const { return "888A85"; }

private:

    virtual void  run_    ( const TTruncAcc & ) {}
    virtual void  refine_ ( node_list_t & )     {}
};
    
//
// execute DAG <dag>
//
void
run ( Graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    //
    // TBB needs single end node
    //
    
    Node *  final        = nullptr;
    bool    multiple_end = false;

    if ( dag.end().size() > 1 )
    {
        LOG( 5, "DAG::TBB::run : multiple end nodes" );

        multiple_end = true;
        
        //
        // create single special end node
        //

        final = new EmptyNode();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() ); // final->in.size();
        final->set_task( new ( tbb::task::allocate_root() ) RuntimeTask( final, acc ) );
    }// if

    // create tbb tasks for all nodes
    for ( auto  node : dag.nodes() )
        node->set_task( new ( tbb::task::allocate_root() ) RuntimeTask( node, acc ) );
    
    // if DAG has single end node, get pointer to it
    if ( final == nullptr )
        final = dag.end().front();
    
    tbb::task_list  work_queue;
    
    for ( auto  node : dag.start() )
    {
        if ( node != final )
            work_queue.push_back( * node->task() );
    }// for
    
    final->task()->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final->task()->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final->task()->execute();                            // and the final node explicitly
    tbb::task::destroy( * static_cast< tbb::task * >( final->task() ) ); // not done by TBB since executed manually

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
