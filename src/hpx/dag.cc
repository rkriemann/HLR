//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>
#include <unordered_map>

#include <hpx/async.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/unwrap.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/range/irange.hpp>

#include "hlr/utils/tools.hh"
#include "hlr/utils/log.hh"

#include "hlr/hpx/dag.hh"

namespace hlr
{

using namespace HLIB;

namespace hpx
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
    
    while ( node_sets.size() > 0 )
    {
        std::vector< std::deque< node * > >  subnodes( node_sets.size() );
        std::vector< std::deque< node * > >  delnodes( node_sets.size() );
        std::atomic< bool >                  any_changed = false;

        // first refine nodes
        ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                   node_sets.begin(), node_sets.end(),
                                   [&,min_size] ( const auto &  nset )
                                   {
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
            auto  set_range = boost::irange( size_t(0), node_sets.size() );
            
            // then refine dependencies and collect new nodes
            ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                       set_range.begin(), set_range.end(),
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

            // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
            ::hpx::parallel::for_each( ::hpx::parallel::execution::par,
                                       delnodes.begin(), delnodes.end(),
                                       [&] ( const auto &  nset )
                                       {
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

    return graph( std::move( tasks ), std::move( start ), std::move( end ), end_mode );
}

namespace
{

// HPX types for tasks and dependencies
using  task_t         = ::hpx::shared_future< void >;
using  dependencies_t = std::list< ::hpx::shared_future< void > >;

//
// execute node without dependencies
//
void
run_node ( node *             node,
           const TTruncAcc &  acc )
{
    log( 4, "run_node : " + node->to_string() );
    
    node->run( acc );
}

//
// execute node with dependencies
//
void
run_node_dep ( node *             node,
               const TTruncAcc &  acc,
               dependencies_t )
{
    log( 4, "run_node_dep : " + node->to_string() );
    
    node->run( acc );
}

}// namespace anonymous

//
// execute DAG <dag>
//
void
run ( graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    using ::hpx::async;
    using ::hpx::dataflow;
    using ::hpx::when_all;
    using ::hpx::util::unwrapping;
    
    assert( dag.end().size() == 1 );
    
    auto  tic = Time::Wall::now();

    //
    // Go through DAG, decrement dependency counter for each successor of
    // current node and if this reaches zero, add node to list of nodes
    // to be visited. Since now all dependencies are met, all tasks for
    // nodes exist and "when_all( dependency-set )" can be constructed.
    //
    // For start nodes, dependency set is empty, so use "async".
    //
    
    // map of DAG nodes to tasks
    std::unordered_map< node *, task_t >          taskmap;

    // keep track of dependencies for a node
    std::unordered_map< node *, dependencies_t >  nodedeps;

    // list of "active" nodes
    std::list< node * >                           nodes;

    for ( auto  node : dag.start() )
    {
        log( 4, "async( " + node->to_string() + " )" );

        task_t  task = async( run_node, node, acc );
        
        taskmap[ node ] = task;

        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( task );

            if ( succ->dec_dep_cnt() == 0 )
                nodes.push_back( succ );
        }// for
    }// for

    while ( ! nodes.empty() )
    {
        auto  node = behead( nodes );
        
        log( 4, "dataflow( " + node->to_string() + " )" );
        
        task_t  task = ::hpx::dataflow( unwrapping( run_node_dep ), node, acc, when_all( nodedeps[ node ] ) );
        
        taskmap[ node ] = task;
        
        for ( auto  succ : node->successors() )
        {
            nodedeps[ succ ].push_back( task );
            
            if ( succ->dec_dep_cnt() == 0 )
                nodes.push_back( succ );
        }// for
    }// while

    auto  toc = Time::Wall::since( tic );

    log( 2, "time for HPX DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // start execution by requesting future result for end node
    //

    auto  final = dag.end().front();

    tic = Time::Wall::now();
    
    taskmap[ final ].get();

    toc = Time::Wall::since( tic );

    log( 2, "time for HPX DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace HPX

}// namespace DAG

}// namespace HLR
