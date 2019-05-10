//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>

#include <hpx/async.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/unwrap.hpp>

#include "utils/tools.hh"

#include "hpx/dag.hh"

using namespace HLIB;

// enables some debug output
#define  log( lvl, msg )  if ( HLIB::verbose( lvl ) ) DBG::print( msg )

// HPX types for tasks and dependencies
using  task_t         = hpx::shared_future< void >;
using  dependencies_t = std::vector< hpx::shared_future< void > >;

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
               task_t             dep )
{
    log( 4, "run_node_dep : " + node->to_string() );
    
    node->run( acc );
}

//
// return vector with tasks corresponding to dependencies
//
dependencies_t
gen_dependencies ( Node *                                   node,
                   std::map< Node *, task_t >               taskmap,
                   std::map< Node *, std::list< Node * > >  nodedeps )
{
    dependencies_t  deps;
    
    deps.reserve( nodedeps[ node ].size() );
    
    for ( auto  dep : nodedeps[ node ] )
        deps.push_back( taskmap[ dep ] );
    
    // not needed anymore
    nodedeps[ node ].clear();

    return deps;
}

// enables some debug output
#define  log( lvl, msg )  if ( HLIB::verbose( lvl ) ) DBG::print( msg )

//////////////////////////////////////////////
//
// node for collecting dependencies
// without any computation
//
//////////////////////////////////////////////

struct EmptyNode : public DAG::Node
{
    // return text version of node
    virtual std::string  to_string () const { return "Empty"; }

    // (optional) color for DAG visualization (format: RRGGBB)
    virtual std::string  color     () const { return "888A85"; }

private:

    virtual void        run_    ( const TTruncAcc & ) {}
    virtual LocalGraph  refine_ ()                    { return {}; }
};

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
    std::map< Node *, task_t >               taskmap;

    // keep track of dependencies for a node
    std::map< Node *, std::list< Node * > >  nodedeps;
    
    // go in BFS style through dag and create the tasks with dependencies
    std::list< Node * >  nodes;

    // start nodes do not have dependencies, therefore create futures using "async"
    for ( auto  node : dag.start() )
    {
        taskmap[ node ] = async( run_node, node, acc );

        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( node );
            nodes.push_back( succ );
        }// for
    }// for

    while ( ! nodes.empty() )
    {
        std::list< Node * >  succs;

        while ( ! nodes.empty() )
        {
            auto  node = behead( nodes );
            auto  deps = gen_dependencies( node, taskmap, nodedeps );
                
            taskmap[ node ] = hpx::dataflow( unwrapping( run_node_dep ), node, acc, when_all( deps ) );

            for ( auto  succ : node->successors() )
            {
                nodedeps[ succ ].push_back( node );
                succs.push_back( succ );
            }// for
        }// while

        nodes = std::move( succs );
    }// while

    //
    // if DAG originally had single end node, get pointer to it
    // otherwise set up task (if not a start node [not possible, or?])
    //
    
    if ( final == nullptr )
        final = dag.end().front();
    else if ( final->dep_cnt() > 0 )
    {
        auto  deps = gen_dependencies( final, taskmap, nodedeps );
        
        taskmap[ final ] = dataflow( unwrapping( run_node_dep ), final, acc, when_all( deps ) );
    }// else
    
    //
    // start execution by requesting future result for end node
    //

    taskmap[ final ].get();

    //
    // remove auxiliary ende node from DAG
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
