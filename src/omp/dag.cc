//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>
#include <unordered_map>
#include <deque>
#include <cassert>

#include "hlr/utils/tools.hh"
#include "hlr/utils/log.hh"

#include "hlr/omp/dag.hh"

namespace hlr 
{

using namespace HLIB;

namespace omp
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
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( node * node )
        {
            const bool  node_changed = node->refine_deps();

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
        };

        // first refine nodes
        std::for_each( nodes.begin(), nodes.end(),
                       [] ( node * node ) { node->refine(); } );

        // then refine dependencies and collect new nodes
        std::for_each( nodes.begin(), nodes.end(),
                       node_dep_refine );

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        std::for_each( del_nodes.begin(), del_nodes.end(),
                       [] ( node * node ) { delete node; } );
        
        nodes = std::move( subnodes );
    }// while

    //
    // collect start and end nodes
    //
    
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

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    const int  MAX_DEPS = 15;
    
    auto tic = Time::Wall::now();

    //
    // associate each task with an array position
    //

    const size_t                       nnodes = dag.nnodes();
    std::vector< int >                 task_no( nnodes );  // just dummy array
    int *                              d   = task_no.data(); // C-style and short name
    std::unordered_map< node *, int >  taskmap;
    size_t                             pos = 0;
        
    for ( auto  node : dag.nodes() )
    {
        log( 0, HLIB::to_string( "%d : ", pos ) + node->to_string() );
        d[ pos ] = pos;
        taskmap[ node ] = pos++;
    }// for
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // loop through nodes and create OpenMP task with dependencies from dep_vecs
    //

    tic = Time::Wall::now();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for ( auto  node : dag.nodes() )
            {
                //
                // fill dependency positions in above array
                //

                const int  task_pos = taskmap[ node ];
                int        s[ MAX_DEPS ]; // also short name!
                int        dpos = 0;

                for ( auto  succ : node->successors() )
                {
                    assert( dpos < MAX_DEPS );
                    s[ dpos++ ] = taskmap[ succ ];
                }// for

                if ( dpos == 0 )
                {
                    #pragma omp task depend( in : d[ task_pos ] )
                    {
                        node->run( acc );
                    }// omp task
                }// if
                else if ( dpos <= 5 ) // should cover most cases
                {
                    for ( int  i = dpos; i < 5; ++i )
                        s[ i ] = s[ i-1 ];
                    
                    log( 0, node->to_string() );
                    for ( int  i = 0; i < dpos; ++i )
                        log( 0, HLIB::to_string( "%d -> %d", task_pos, s[i] ) );
                    
                    #pragma omp task depend( in : d[ task_pos ] ) depend( out : d[s[0]], d[s[1]], d[s[2]], d[s[3]], d[s[4]] )
                    {
                        node->run( acc );
                    }// omp task
                }// else
                else 
                {
                    for ( int  i = dpos; i < MAX_DEPS; ++i )
                        s[ i ] = s[ i-1 ];
                    
                    #pragma omp task depend( in : d[ task_pos ] ) depend( out : d[s[0]], d[s[1]], d[s[2]], d[s[3]], d[s[4]], d[s[5]], d[s[6]], d[s[7]], d[s[8]], d[s[10]] )
                    {
                        node->run( acc );
                    }// omp task
                }// else
            }// for
        }// omp single
    }// omp parallel

    toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace dag

}// namespace omp

}// namespace hlr
