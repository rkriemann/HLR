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
refine ( node *                            root,
         const size_t                      min_size,
         const hlr::dag::end_nodes_mode_t  end_mode )
{
    assert( root != nullptr );
    
    std::deque< std::deque< node * > >  node_sets{ { root } };
    std::list< node * >                 tasks;
    std::mutex                          mtx;
    std::list< node * >                 end;
    const bool                          do_lock = (( hlr::dag::sparsify_mode != hlr::dag::sparsify_none  ) &&
                                                   ( hlr::dag::sparsify_mode != hlr::dag::sparsify_local ));
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            while ( node_sets.size() > 0 )
            {
                std::vector< std::deque< node * > >  subnodes( node_sets.size() );
                std::vector< std::deque< node * > >  delnodes( node_sets.size() );
                std::atomic< bool >                  any_changed = false;

                // first refine nodes
                #pragma omp taskloop shared( node_sets, any_changed )
                for ( size_t  i = 0; i < node_sets.size(); ++i )
                {
                    const auto &  nset        = node_sets[i];
                    bool          any_chg_loc = false;
                    
                    for ( auto  node : nset )
                    {
                        node->refine( min_size );
                        
                        if ( node->is_refined() )
                            any_chg_loc = true;
                    }// for
                    
                    if ( any_chg_loc )
                        any_changed = true;
                }// for
                #pragma omp taskwait

                if ( any_changed )
                {
                    // then refine dependencies and collect new nodes
                    #pragma omp taskloop shared( node_sets, subnodes, tasks, delnodes, mtx )
                    for ( size_t  i = 0; i < node_sets.size(); ++i )
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
                    }// for
                    #pragma omp taskwait

                    // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
                    #pragma omp taskloop shared( delnodes )
                    for ( size_t  i = 0; i < delnodes.size(); ++i )
                    {
                        const auto &  nset = delnodes[i];
                        
                        for ( auto  node : nset )
                            delete node;
                    }// for
                    #pragma omp taskwait
            
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
        }// omp single
    }// omp parallel

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

    return graph( std::move( tasks ), std::move( start ), std::move( end ), end_mode );
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
}

}// namespace dag

}// namespace omp

}// namespace hlr
