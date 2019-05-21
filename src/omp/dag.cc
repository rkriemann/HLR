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
    
    std::deque< node * >  nodes;
    std::list< node * >   tasks, start, end;
    std::mutex            mtx;
    
    nodes.push_back( root );

    #pragma omp parallel
    {
        #pragma omp single
        {
            while ( ! nodes.empty() )
            {
                std::deque< node * >  subnodes, del_nodes;

                auto  node_dep_refine =
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
                std::for_each( nodes.begin(), nodes.end(),
                               [] ( node * node )
                               {
                                   #pragma omp task firstprivate( node )
                                   {
                                       node->refine();
                                   }// omp task
                               } );

                #pragma omp taskwait

                // then refine dependencies and collect new nodes
                std::for_each( nodes.begin(), nodes.end(),
                               [node_dep_refine] ( node * node )
                               {
                                   #pragma omp task firstprivate( node )
                                   {
                                       node_dep_refine( node );
                                   }// omp task
                               } );

                #pragma omp taskwait

                // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
                std::for_each( del_nodes.begin(), del_nodes.end(),
                               [] ( node * node )
                               {
                                   #pragma omp task firstprivate( node )
                                   {
                                       delete node;
                                   }// omp task
                               } );
        
                #pragma omp taskwait

                nodes = std::move( subnodes );
            }// while
        }// omp single
    }// omp parallel

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
    auto tic = Time::Wall::now();

    //
    // ensure only single end node
    //

    node *  final        = nullptr;
    bool    multiple_end = false;
    
    if ( dag.end().size() > 1 )
    {
        log( 5, "omp::dag::run : multiple end nodes" );

        multiple_end = true;
        
        final = new hlr::dag::empty_node();

        for ( auto  node : dag.end() )
            final->after( node );

        final->set_dep_cnt( dag.end().size() );
    }// if
    else
        final = dag.end().front();

    auto  toc = Time::Wall::since( tic );

    log( 2, "time for OpenMP DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // begin with start nodes and proceed until final node was reached
    //

    tic = Time::Wall::now();

    std::list< node * >  workset;
    
    for ( auto  node : dag.start() )
        workset.push_back( node );

    bool                     reached_final = false;
    std::mutex               wmtx, cmtx;
    std::condition_variable  cv;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            while ( ! reached_final )
            {
                node *  task = nullptr;

                {
                    std::unique_lock  lock( cmtx );

                    cv.wait_for( lock, std::chrono::microseconds( 10 ) );

                    {
                        std::scoped_lock  wlock( wmtx );
                        
                        if ( ! workset.empty() )
                            task = behead( workset );
                    }
                }

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
        
    toc = Time::Wall::since( tic );

    log( 2, "time for OpenMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // remove auxiliary node from DAG
    //
        
    if ( multiple_end )
    {
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if

    // //
    // // associate each task with an array position
    // //

    // const size_t                       nnodes = dag.nnodes();
    // std::vector< int >                 task_no( nnodes );  // just dummy array
    // int *                              d   = task_no.data(); // C-style and short name
    // std::unordered_map< node *, int >  taskmap;
    // size_t                             pos = 0;
        
    // for ( auto  node : dag.nodes() )
    // {
    //     log( 0, HLIB::to_string( "%d : ", pos ) + node->to_string() );
    //     d[ pos ] = pos;
    //     taskmap[ node ] = pos++;
    // }// for
    
    // auto  toc = Time::Wall::since( tic );

    // log( 2, "time for OMP DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    // //
    // // loop through nodes and create OpenMP task with dependencies from dep_vecs
    // //

    // const int  MAX_DEPS = 15;
    
    // tic = Time::Wall::now();
    
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     {
    //         for ( auto  node : dag.nodes() )
    //         {
    //             //
    //             // fill dependency positions in above array
    //             //

    //             const int  task_pos = taskmap[ node ];
    //             int        s[ MAX_DEPS ]; // also short name!
    //             int        dpos = 0;

    //             for ( auto  succ : node->successors() )
    //             {
    //                 assert( dpos < MAX_DEPS );
    //                 s[ dpos++ ] = taskmap[ succ ];
    //             }// for

    //             if ( dpos == 0 )
    //             {
    //                 #pragma omp task depend( in : d[ task_pos ] )
    //                 {
    //                     node->run( acc );
    //                 }// omp task
    //             }// if
    //             else if ( dpos <= 5 ) // should cover most cases
    //             {
    //                 for ( int  i = dpos; i < 5; ++i )
    //                     s[ i ] = s[ i-1 ];
                    
    //                 log( 0, node->to_string() );
    //                 for ( int  i = 0; i < dpos; ++i )
    //                     log( 0, HLIB::to_string( "%d -> %d", task_pos, s[i] ) );
                    
    //                 #pragma omp task depend( in : d[ task_pos ] ) depend( out : d[s[0]], d[s[1]], d[s[2]], d[s[3]], d[s[4]] )
    //                 {
    //                     node->run( acc );
    //                 }// omp task
    //             }// else
    //             else 
    //             {
    //                 for ( int  i = dpos; i < MAX_DEPS; ++i )
    //                     s[ i ] = s[ i-1 ];
                    
    //                 #pragma omp task depend( in : d[ task_pos ] ) depend( out : d[s[0]], d[s[1]], d[s[2]], d[s[3]], d[s[4]], d[s[5]], d[s[6]], d[s[7]], d[s[8]], d[s[10]] )
    //                 {
    //                     node->run( acc );
    //                 }// omp task
    //             }// else
    //         }// for
    //     }// omp single
    // }// omp parallel

    // toc = Time::Wall::since( tic );

    // log( 2, "time for OMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace dag

}// namespace omp

}// namespace hlr
