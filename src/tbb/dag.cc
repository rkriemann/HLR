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
#include <tbb/parallel_for.h>
#include "tbb/partitioner.h"

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
    
    while ( node_sets.size() > 0 )
    {
        std::vector< std::deque< node * > >  subnodes( node_sets.size() );
        std::vector< std::deque< node * > >  delnodes( node_sets.size() );
        std::atomic< bool >                  any_changed = false;

        // first refine nodes
        ::tbb::parallel_for< size_t >( 0, node_sets.size(),
                                       [&,min_size] ( const size_t  i )
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
                                       } );

        if ( any_changed )
        {
            // then refine dependencies and collect new nodes
            ::tbb::parallel_for< size_t >( 0, node_sets.size(),
                                           [&,do_lock] ( const size_t  i )
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

            // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
            ::tbb::parallel_for< size_t >( 0, node_sets.size(),
                                           [&] ( const size_t  i )
                                           {
                                               const auto &  nset = delnodes[i];

                                               for ( auto  node : nset )
                                                   delete node;
                                           } );
            
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

namespace
{

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
    assert( dag.end().size() == 1 );
    
    //
    // create task for end node since needed by all other tasks
    //
    
    auto           final = dag.end().front();
    ::tbb::task *  final_task = new ( ::tbb::task::allocate_root() ) runtime_task( final, acc, final, nullptr );
    
    final_task->set_ref_count( final->dep_cnt() );

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

    //
    // run DAG
    //
    
    final_task->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final_task->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final_task->execute();                            // and the final node explicitly
    ::tbb::task::destroy( * final_task );             // not done by TBB since executed manually
}

}// namespace dag

}// namespace tbb

}// namespace hlr
