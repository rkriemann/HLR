//
// Project     : HLib
// File        : node.cc
// Description : node in a compute DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <deque>
#include <unordered_set>


#include "hlr/utils/log.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/node.hh"

namespace hlr
{

namespace dag
{

using namespace HLIB;

// controls edge sparsification
constexpr bool         sparsify     = true;

// default maximal path distance in reachability test
constexpr int          def_path_len = 2;
    
// controls edge sparsification
constexpr bool         lock_nodes   = true;

// activates collision counting
constexpr bool         count_coll   = true;

// counter for lock collisions
std::atomic< size_t >  collisions;

//////////////////////////////////////////////
//
// node
//
//////////////////////////////////////////////

//
// per node initialization
//
void
node::init ()
{
    _in_blk_deps  = in_blocks_();
    _out_blk_deps = out_blocks_();

    log( 5, to_string() );
}
    
//
// handles execution of node code and spawning of successor nodes
//
void
node::run ( const TTruncAcc & acc )
{
    log( 4, "run( " + this->to_string() + " )" );
    
    run_( acc );
}

//
// split node into subnodes and update dependencies
// if retval is empty, no refinement was done
//
void
node::refine ()
{
    HLR_LOG( 5, "refine( " + to_string() + " )" );

    //
    // create subnodes
    //

    auto  g = refine_();

    g.set_dependencies();

    // g.print_dot( "g.dot" );
        
    //
    // copy nodes to local array
    //

    _sub_nodes.reserve( g.size() );

    for ( auto  n : g )
        _sub_nodes.push_back( n );
}

namespace
{

//
// return set of nodes reachable by <steps> steps
// - if <neighbourhood> is non-empty, search is restricted to nodes
//   in <neighbourhood>
//
std::unordered_set< node * >
reachable_indirect ( node *                                root,
                     const std::unordered_set< node * > &  neighbourhood = {},
                     const uint                            steps         = def_path_len )
{
    const bool                    no_neigh = ( neighbourhood.size() == 0 );
    std::deque< node * >          nodes;
    std::unordered_set< node * >  descendants;
    uint                          step  = 1;

    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< node * >  sons;

        for ( auto  node : nodes )
        {
            for ( auto  out : node->successors() )
            {
                if ( no_neigh || ( neighbourhood.find( out ) != neighbourhood.end() ))
                {
                    sons.push_back( out );

                    if ( step > 1 )
                        descendants.insert( out );
                }// if
            }// for
        }// for

        nodes = std::move( sons );

        if ( ++step > steps )
            break;
    }// while

    return descendants;
}

//
// compute reachable nodes from <n> in <neighbourhood> and
// remove edges (n,m) if m is reachable otherwise
//
void
remove_redundant ( node *                                n,
                   const std::unordered_set< node * > &  neighbourhood )
{
    auto  descendants = reachable_indirect( n, neighbourhood );
    
    // if a direct edge to otherwise reachable node exists, remove it
    for ( auto  succ_iter = n->successors().begin(); succ_iter != n->successors().end(); )
    {
        if ( descendants.find( *succ_iter ) != descendants.end() )
        {
            HLR_LOG( 6, "  removing " + n->to_string() + " → " + (*succ_iter)->to_string() + " from " + n->to_string() );
            succ_iter = n->successors().erase( succ_iter );
        }// if
        else
            ++succ_iter;
    }// for
}

}// namespace anonymous

//
// refine dependencies of local node or of sub nodes
//
bool
node::refine_deps ()
{
    log( 5, "refine_deps( " + to_string() + " )" );

    //
    // first lock all nodes
    //

    node_list_t  locked;
    
    if ( lock_nodes )
    {
        locked.push_back( this );

        for ( auto  sub : _sub_nodes )
            locked.push_back( sub );
    
        for ( auto  succ : successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                    locked.push_back( succ_sub );
            }// if
            else
                locked.push_back( succ );
        }// for

        // remove duplicates
        locked.sort();
        locked.unique();
    
        for ( auto  n : locked )
        {
            if ( count_coll )
            {
                if ( ! n->try_lock() )
                {
                    ++collisions;
                    n->lock();
                }// if
            }// if
            else
                n->lock();
            
            HLR_LOG( 7, "locked: " + n->to_string() + " by " + this->to_string() );
        }// for
    }// if

    //
    // decide to refine local node or sub node dependencies
    //

    bool  changed = false;

    if ( is_refined() )
        refine_sub_deps();
    else
        changed = refine_loc_deps();
    
    //
    // finally unlock nodes
    //
    
    if ( lock_nodes )
    {
         for ( auto  n : locked )
         {
             n->unlock();
             HLR_LOG( 7, "unlocked: " + n->to_string() + " by " + this->to_string() );
         }// for
    }// if
         
    return changed;
}

//
// refine dependencies of local node
//
bool
node::refine_loc_deps ()
{
    log( 5, "refine_loc_deps( " + to_string() + " )" );

    //
    // replace successor by refined successors if available
    //
    
    bool  changed = false;

    {
        node_list_t  new_out;
        auto         succ = successors().begin();
            
        while ( succ != successors().end() )
        {
            if ( (*succ)->is_refined() )
            {
                changed = true;
                
                // insert succendencies for subnodes (intersection test neccessary???)
                for ( auto  succ_sub : (*succ)->sub_nodes() )
                {
                    HLR_LOG( 6, to_string() + " ⟶ " + succ_sub->to_string() );
                    new_out.push_back( succ_sub );
                }// for

                // remove previous succendency
                succ = successors().erase( succ );
            }// if
            else
                ++succ;
        }// for

        successors().splice( successors().end(), new_out );

        // remove duplicates
        successors().sort();
        successors().unique();
    }
    
    //
    // remove direct outgoing edges if reachable otherwise
    //
    
    if ( changed && sparsify )
    {
        // restrict search to local nodes and neighbouring nodes
        std::unordered_set< node * >  neighbourhood{ this };
        
        for ( auto  succ : successors() )
            neighbourhood.insert( succ );

        remove_redundant( this, neighbourhood );
    }// if
    
    return changed;
}

//
// refine dependencies of sub nodes
//
void
node::refine_sub_deps ()
{
    log( 5, "refine_sub_deps( " + to_string() + " )" );
    
    if ( _sub_nodes.size() == 0 )
        return;

    //
    // add dependencies for all sub nodes to old or refined successors
    //
    
    for ( auto  sub : _sub_nodes )
    {
        for ( auto  succ : successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                {
                    if ( is_intersecting( sub->out_blocks(), succ_sub->in_blocks() ) )
                    {
                        HLR_LOG( 6, sub->to_string() + " ⟶ " + succ_sub->to_string() );
                        sub->successors().push_back( succ_sub );
                    }// if
                }// for
            }// if
            else
            {
                if ( is_intersecting( sub->out_blocks(), succ->in_blocks() ) )
                {
                    HLR_LOG( 6, sub->to_string() + " → " + succ->to_string() );
                    sub->successors().push_back( succ );
                }// if
            }// for
        }// else
    }// for

    //
    // remove direct outgoing edges if reachable via path
    //
    
    if ( sparsify )
    {
        #if 0

        //
        // search for each sub node individually (less code, slightly slower)
        //
        
        for ( auto  sub : _sub_nodes )
        {
            std::unordered_set< node * >  neighbourhood;

            neighbourhood.insert( sub );

            for ( auto  succ : sub->successors() )
                neighbourhood.insert( succ );

            remove_redundant( sub, neighbourhood );
        }// for
        
        #else
        
        //
        // build common neighbourhood and then sparsify for each node (more code, slightly faster)
        //
        
        std::unordered_set< node * >  neighbourhood;

        for ( auto  sub : _sub_nodes )
            neighbourhood.insert( sub );

        // go through local successors and their sub nodes instead of
        // successors of own sub nodes for efficiency (successor sets 
        // of sub nodes are not disjoint)
        for ( auto  succ : _successors )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                    neighbourhood.insert( succ_sub );
            }// if
            else
                neighbourhood.insert( succ );
        }// for

        //
        // now remove redundant edges
        //
        
        for ( auto  sub : _sub_nodes )
            remove_redundant( sub, neighbourhood );
        
        #endif
    }// if

    //
    // remove duplicates
    //

    for ( auto  sub : _sub_nodes )
    {
        sub->successors().sort();
        sub->successors().unique();
    }// for
}
    
//
// print node with full edge information
//
void
node::print () const
{
    std::cout << to_string() << std::endl;
    std::cout << "   #deps : " << _dep_cnt << std::endl;

    std::cout << "   succ  : " << successors().size() << std::endl;
    for ( auto  succ : successors() )
        std::cout << "      " << succ->to_string() << std::endl;
}
    
}// namespace dag

}// namespace hlr
