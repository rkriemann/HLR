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
#include <set>


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
    
// controls node locking during refinement
bool                   lock_nodes   = true;

// activates collision counting
constexpr bool         count_coll   = false;

// counter for lock collisions
std::atomic< size_t >  collisions;

// type for a node set
using  node_set_t = std::vector< node * >;

//////////////////////////////////////////////
//
// auxiliary functions
//
//////////////////////////////////////////////

inline
void
insert ( node *                   n,
         std::vector< node * > &  v )
{
    v.push_back( n );
}

inline
void
insert ( node *                n,
         std::set< node * > &  s )
{
    s.insert( n );
}

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

    HLR_LOG( 5, to_string() );
}
    
//
// handles execution of node code and spawning of successor nodes
//
void
node::run ( const TTruncAcc & acc )
{
    HLR_LOG( 4, "run( " + this->to_string() + " )" );
    
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

    _sub_nodes = std::move( g );
}

namespace
{

//
// return set of nodes reachable by <steps> steps
// - if <neighbourhood> is non-empty, search is restricted to nodes
//   in <neighbourhood>
//
node_set_t
reachable_indirect ( node *              root,
                     const node_set_t &  neighbourhood = {},
                     const uint          steps         = def_path_len )
{
    const bool            no_neigh = ( neighbourhood.size() == 0 );
    std::deque< node * >  nodes;
    node_set_t            descendants;
    uint                  step  = 1;

    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< node * >  sons;

        for ( auto  node : nodes )
        {
            for ( auto  out : node->successors() )
            {
                if ( no_neigh || contains( neighbourhood, out ) )
                {
                    sons.push_back( out );

                    if ( step > 1 )
                        insert( out, descendants );
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
node_vec_t
remove_redundant ( node *              n,
                   const node_set_t &  neighbourhood )
{
    auto        descendants = reachable_indirect( n, neighbourhood );
    node_vec_t  new_out;

    new_out.reserve( n->successors().size() );
    
    // if a direct edge to otherwise reachable node exists, remove it
    for ( auto  succ : n->successors() )
    {
        if ( contains( descendants, succ ) )
            HLR_LOG( 6, "  removing " + n->to_string() + " → " + succ->to_string() + " from " + n->to_string() );
        else
            new_out.push_back( succ );
    }// for

    return new_out;
}

}// namespace anonymous

//
// refine dependencies of local node or of sub nodes
//
bool
node::refine_deps ()
{
    HLR_LOG( 5, "refine_deps( " + to_string() + " )" );

    //
    // first lock all nodes
    //

    std::set< node * >  locked;
    
    if ( lock_nodes )
    {
        insert( this, locked );

        for ( auto  sub : _sub_nodes )
            insert( sub, locked );
    
        for ( auto  succ : successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                    insert( succ_sub, locked );
            }// if
            else
                insert( succ, locked );
        }// for

        // // remove duplicates
        // locked.sort();
        // locked.unique();
    
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
    HLR_LOG( 5, "refine_loc_deps( " + to_string() + " )" );

    //
    // replace successor by refined successors if available
    //
    
    bool  changed = false;

    {
        node_vec_t  new_out;
        auto        succ = successors().begin();
            
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

                ++succ;
            }// if
            else
            {
                new_out.push_back( *succ );
                ++succ;
            }// else
        }// for

        _successors = std::move( new_out );
    }
    
    //
    // remove direct outgoing edges if reachable otherwise
    //
    
    if ( changed && sparsify )
    {
        // restrict search to local nodes and neighbouring nodes
        node_set_t  neighbourhood;

        neighbourhood.reserve( 1 + successors().size() );

        insert( this, neighbourhood );
        
        for ( auto  succ : successors() )
            insert( succ, neighbourhood );

        _successors = remove_redundant( this, neighbourhood );
    }// if
    
    return changed;
}

//
// refine dependencies of sub nodes
//
void
node::refine_sub_deps ()
{
    HLR_LOG( 5, "refine_sub_deps( " + to_string() + " )" );
    
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
        for ( auto  sub : _sub_nodes )
        {
            node_set_t  neighbourhood;

            neighbourhood.reserve( 1 + sub->successors().size() );
            
            insert( sub, neighbourhood );

            for ( auto  succ : sub->successors() )
                insert( succ, neighbourhood );

            sub->_successors = remove_redundant( sub, neighbourhood );
        }// for
    }// if
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
