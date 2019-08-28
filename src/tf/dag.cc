//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>
#include <unordered_map>
#include <cassert>
#include <mutex>

#include <taskflow/taskflow.hpp>

#include "hlr/utils/log.hh"
#include "hlr/utils/term.hh" // DEBUG
#include "hlr/tf/dag.hh"

using namespace HLIB;

namespace hlr
{

namespace tf
{

namespace dag
{

using hlr::dag::node;
using hlr::dag::graph;

//
// construct DAG using refinement of given node
//
graph
refine ( node *        root,
         const size_t  min_size )
{
    assert( root != nullptr );
    
    std::deque< std::deque< node * > >  node_sets{ { root } };
    std::list< node * >                 tasks;
    std::mutex                          mtx;
    std::list< node * >                 end;
    const bool                          do_lock = (( hlr::dag::sparsify_mode != hlr::dag::sparsify_none  ) &&
                                                   ( hlr::dag::sparsify_mode != hlr::dag::sparsify_local ));
    ::tf::Executor                      executor;
    ::tf::Taskflow                      tf;
    
    while ( node_sets.size() > 0 )
    {
        std::vector< std::deque< node * > >  subnodes( node_sets.size() );
        std::vector< std::deque< node * > >  delnodes( node_sets.size() );
        std::atomic< bool >                  any_changed = false;

        // first refine nodes
        tf.parallel_for( node_sets.begin(), node_sets.end(),
                         [&,min_size] ( const auto &  nset )
                         {
                             bool  any_chg_loc = false;
                                           
                             for ( auto  node : nset )
                             {
                                 node->refine( min_size );
                                 
                                 if ( node->is_refined() )
                                     any_chg_loc = true;
                             }// for
                             
                             if ( any_chg_loc )
                                 any_changed = true;
                         } );
        executor.run( tf ).wait();
        tf.clear();

        if ( any_changed )
        {
            // then refine dependencies and collect new nodes
            tf.parallel_for( size_t(0), node_sets.size(), size_t(1),
                             [&,do_lock] ( const auto  i )
                             {
                                 const auto &  nset = node_sets[i];
                                 
                                 for ( auto  node : nset )
                                 {
                                     const bool  node_changed = node->refine_deps( do_lock );
                                     
                                     if ( node->is_refined() )       // node was refined; collect all sub nodes
                                     {
                                         for ( auto  sub : node->sub_nodes() )
                                             subnodes[i].push_back( sub );
                                         
                                         delnodes[i].push_back( node );
                                     }// if
                                     else if ( node_changed )        // node was not refined but dependencies were
                                     {
                                         subnodes[i].push_back( node );
                                     }// if
                                     else                            // neither node nor dependencies changed: reached final state
                                     {
                                         // adjust dependency counter of successors (which were NOT refined!)
                                         for ( auto  succ : node->successors() )
                                             succ->inc_dep_cnt();
                                         
                                         {
                                             std::scoped_lock  lock( mtx );
                                             
                                             tasks.push_back( node );
                                             
                                             if ( node->successors().empty() )
                                                 end.push_back( node );
                                         }
                                     }// else
                                 }// for
                             } );
            executor.run( tf ).wait();
            tf.clear();
            
            // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
            tf.parallel_for( delnodes.begin(), delnodes.end(),
                             [&] ( const auto &  nset )
                             {
                                 for ( auto  node : nset )
                                     delete node;
                             } );
            executor.run( tf ).wait();
            tf.clear();
            
            //
            // split node sets if too large (increase parallelity)
            //

            std::deque< std::deque< node * > >  new_sets;

            constexpr size_t  max_size = 1000;

            for ( size_t  i = 0; i < subnodes.size(); ++i )
            {
                const auto  snsize = subnodes[i].size();
            
                if ( snsize >= max_size )
                {
                    //
                    // split into chunks of size <max_size>
                    //
                
                    size_t  pos = 0;

                    while ( pos < snsize )
                    {
                        std::deque< node * >  sset;
                        size_t                nsset = 0;

                        for ( ; ( nsset < max_size ) && ( pos < snsize ); ++nsset )
                            sset.push_back( subnodes[i][pos++] );

                        // put rest into last set if not much left
                        if ( snsize - pos < max_size / 2 )
                        {
                            while ( pos < snsize )
                                sset.push_back( subnodes[i][pos++] );
                        }// if

                        new_sets.push_back( std::move( sset ) );
                    }// while
                }// if
                else if (( snsize > 0 ) && ( snsize < max_size / 10 ))
                {
                    //
                    // merge with next non-empty set
                    //
                
                    std::deque< node * >  mset = std::move( subnodes[i] );

                    size_t  j = i+1;
                
                    while (( j < subnodes.size() ) && ( subnodes[j].size() == 0 ))
                        ++j;

                    if ( j < subnodes.size() )
                    {
                        for ( auto  n : subnodes[j] )
                            mset.push_back( n );
                    }// if
                
                    new_sets.push_back( std::move( mset ) );
                    i = j;
                }// if
                else if ( snsize > 0 )
                {
                    // log( 0, HLIB::to_string( "copying %d", nset.size() ) );
                    new_sets.push_back( std::move( subnodes[i] ) );
                }// if
            }// for
        
            node_sets = std::move( new_sets );
        }// if
        else
        {
            std::for_each( node_sets.begin(), node_sets.end(),
                           [&] ( std::deque< node * > &  nset )
                           {
                               for ( auto  node : nset )
                               {
                                   tasks.push_back( node );

                                   // adjust dependency counter of successors (which were NOT refined!)
                                   for ( auto  succ : node->successors() )
                                       succ->inc_dep_cnt();

                                   if ( node->successors().empty() )
                                       end.push_back( node );
                               }// for
                           } );

            node_sets.clear();
        }// else
    }// while

    //
    // collect start nodes
    //
    
    std::list< node * >   start;
    
    std::for_each( tasks.begin(), tasks.end(),
                   [&] ( node * node )
                   {
                       node->finalize();
                       
                       if ( node->dep_cnt() == 0 )
                           start.push_back( node );
                   } );

    return graph( std::move( tasks ), std::move( start ), std::move( end ), hlr::dag::use_single_end_node );
}

// mapping of node to TF task
using  taskmap_t = std::unordered_map< node *, ::tf::Task >;

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    //
    // TF needs single end node
    //
    
    auto            tic = Time::Wall::now();
    taskmap_t       taskmap;
    ::tf::Taskflow  tf;

    // auto          observer = tf.share_executor()->make_observer< tf::ExecutorObserver >();
    
    // create TF tasks for all nodes
    for ( auto  node : dag.nodes() )
        taskmap[ node ] = tf.silent_emplace( [node,&acc] () { node->run( acc ); } );
    
    // set up dependencies
    for ( auto  node : dag.nodes() )
        for ( auto  succ : node->successors() )
            taskmap[ node ].precede( taskmap[ succ ] );
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for TF DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    //
    // run tasks
    //

    ::tf::Executor  executor;
    
    tic = Time::Wall::now();
    
    executor.run( tf ).wait();

    toc = Time::Wall::since( tic );

    log( 2, "time for TF DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );

    // std::ofstream  ofs( "timestamps.json" );
    
    // observer->dump( ofs );
}

}// namespace TF

}// namespace DAG

}// namespace HLR
