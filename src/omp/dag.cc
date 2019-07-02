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
#include <mutex>
#include <condition_variable>

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
    
    std::deque< node * >  nodes{ root };
    std::list< node * >   tasks, start, end;
    std::mutex            mtx;

    #pragma omp parallel
    {
        #pragma omp single
        {
            while ( ! nodes.empty() )
            {
                std::deque< node * >  subnodes, del_nodes;

                // first refine nodes
                #pragma omp taskloop shared( nodes )
                for ( size_t  i = 0; i < nodes.size(); ++i )
                {
                    nodes[i]->refine();
                }// for
                #pragma omp taskwait

                // then refine dependencies and collect new nodes
                #pragma omp taskloop shared( nodes, subnodes, tasks, del_nodes, mtx )
                for ( size_t  i = 0; i < nodes.size(); ++i )
                {
                    auto        node         = nodes[i];
                    const bool  node_changed = node->refine_deps( true );
                        
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
                }// for
                #pragma omp taskwait

                // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
                std::for_each( del_nodes.begin(), del_nodes.end(),
                               [] ( node * node )
                               {
                                   delete node;
                               } );

                nodes = std::move( subnodes );
            }// while
        }// omp single
    }// omp parallel

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

    return graph( tasks, start, end, hlr::dag::use_single_end_node );
}

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    assert( dag.end().size() == 1 );

    //
    // begin with start nodes and proceed until final node was reached
    //

    auto                 tic = Time::Wall::now();
    std::list< node * >  workset;
    
    for ( auto  node : dag.start() )
        workset.push_back( node );

    bool                     reached_final = false;
    std::mutex               wmtx, cmtx;
    std::condition_variable  cv;
    auto                     final = dag.end().front();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            while ( ! reached_final )
            {
                node *  task = nullptr;

                {
                    std::scoped_lock  wlock( wmtx );
                        
                    if ( ! workset.empty() )
                        task = behead( workset );
                }

                if ( task == nullptr )
                {
                    std::unique_lock  lock( cmtx );

                    cv.wait_for( lock, std::chrono::microseconds( 10 ) );
                }// if

                if ( task == nullptr )
                    continue;
                
                if ( task == final )
                    reached_final = true;

                #pragma omp task firstprivate( task ) shared( wmtx, workset )
                {
                    task->run( acc );

                    for ( auto  succ : task->successors() )
                    {
                        if ( succ->dec_dep_cnt() == 0 )
                        {
                            std::scoped_lock  lock( wmtx );

                            workset.push_back( succ );
                            cv.notify_one();
                        }// if
                    }// for
                }// omp task
            }// while
        }// omp single
    }// omp parallel
        
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for OpenMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace dag

}// namespace omp

}// namespace hlr
