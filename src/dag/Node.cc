//
// Project     : HLib
// File        : Node.cc
// Description : Node in a compute DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <deque>
#include <unordered_set>

#include "utils/log.hh"
#include "dag/Node.hh"

namespace HLR
{

namespace DAG
{

using namespace HLIB;

// controls edge sparsification
const bool  sparsify = true;

//////////////////////////////////////////////
//
// Node
//
//////////////////////////////////////////////

//
// per node initialization
//
void
Node::init ()
{
    _in_blk_deps  = in_blocks_();
    _out_blk_deps = out_blocks_();

    log( 5, to_string() );
}
    
//
// handles execution of node code and spawning of successor nodes
//
void
Node::run ( const TTruncAcc & acc )
{
    log( 4, "run( " + this->to_string() + " )" );
    
    run_( acc );
}

//
// split node into subnodes and update dependencies
// if retval is empty, no refinement was done
//
void
Node::refine ()
{
    log( 5, "refine( " + to_string() + " )" );

    //
    // create subnodes
    //

    auto  g = refine_();

    g.set_dependencies();

    g.print_dot( "g.dot" );
        
    //
    // copy nodes to local array
    //

    _sub_nodes.reserve( g.size() );

    for ( auto  node : g )
        _sub_nodes.push_back( node );
}

namespace
{

//
// return set of nodes reachable by <steps> steps
// - if <neighbourhood> is non-empty, search is restricted to nodes
//   in <neighbourhood>
//
std::unordered_set< Node * >
reachable_indirect ( Node *                                root,
                     const std::unordered_set< Node * > &  neighbourhood = {},
                     const uint                            steps         = 2 )
{
    const bool                    no_neigh = ( neighbourhood.size() == 0 );
    std::deque< Node * >          nodes;
    std::unordered_set< Node * >  descendants;
    uint                          step  = 1;

    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  sons;

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
                     
}// namespace anonymous

//
// refine dependencies for sub nodes
//
void
Node::refine_sub_deps ()
{
    log( 5, "refine_sub_deps( " + to_string() + " )" );
    
    if ( _sub_nodes.size() == 0 )
        return;

    //
    // first lock all nodes
    //

    node_list_t  locked;
    
    locked.push_back( this );

    for ( auto  node : _sub_nodes )
        locked.push_back( node );
    
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
    
    for ( auto  node : locked )
    {
        node->lock();
        log( 6, "locked: " + node->to_string() + " by " + this->to_string() );
    }// for
                
    //
    // add dependencies for all sub nodes to old or refined successors
    //
    
    for ( auto  succ : successors() )
    {
        if ( succ->is_refined() )
        {
            for ( auto  succ_sub : succ->sub_nodes() )
            {
                for ( auto  node : _sub_nodes )
                {
                    if ( is_intersecting( node->out_blocks(), succ_sub->in_blocks() ) )
                    {
                        log( 6, node->to_string() + " ⟶ " + succ_sub->to_string() );
                        node->successors().push_back( succ_sub );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            for ( auto  node : _sub_nodes )
            {
                if ( is_intersecting( node->out_blocks(), succ->in_blocks() ) )
                {
                    log( 6, node->to_string() + " ⟶ " + succ->to_string() );
                    node->successors().push_back( succ );
                }// if
            }// for
        }// else
    }// for

    //
    // remove direct outgoing edges if reachable via path
    //
    
    if ( sparsify )
    {
        std::unordered_set< Node * >  neighbourhood;

        // restrict search to local nodes and neighbouring nodes
        for ( auto  node : _sub_nodes )
            neighbourhood.insert( node );
        
        for ( auto  node : _successors )
        {
            if ( node->is_refined() )
            {
                for ( auto  node_sub : node->sub_nodes() )
                    neighbourhood.insert( node_sub );
            }// if
            else
                neighbourhood.insert( node );
        }// for
        
        // now test reachability
        for ( auto  node : _sub_nodes )
        {
            auto  descendants = reachable_indirect( node, neighbourhood );

            // if a direct edge to otherwise reachable node exists, remove it
            for ( auto  succ_iter = node->successors().begin(); succ_iter != node->successors().end(); )
            {
                if ( descendants.find( *succ_iter ) != descendants.end() )
                {
                    log( 6, "  removing " + node->to_string() + " ⟶ " + (*succ_iter)->to_string() + " from " + node->to_string() );
                    succ_iter = node->successors().erase( succ_iter );
                    // node->print();
                }// if
                else
                    ++succ_iter;
            }// for
        }// for
    }// if

    //
    // remove duplicates
    //

    for ( auto  node : _sub_nodes )
    {
        node->successors().sort();
        node->successors().unique();
    }// for

    //
    // finally unlock nodes
    //
    
    for ( auto  node : locked )
    {
        node->unlock();
        log( 6, "unlocked: " + node->to_string() + " by " + this->to_string() );
    }// for
}
    
//
// check local dependencies for refinement
//
bool
Node::refine_deps ()
{
    log( 5, "refine_deps( " + to_string() + " )" );

    //
    // first lock all nodes
    //

    node_list_t  locked;
    
    locked.push_back( this );

    for ( auto  node : _sub_nodes )
        locked.push_back( node );
    
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
    
    for ( auto  node : locked )
    {
        node->lock();
        log( 6, "locked: " + node->to_string() + " by " + this->to_string() );
    }// for

    //
    // replace successor by refined successors if available
    //
    
    bool         changed = false;
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
                log( 6, to_string() + " ⟶ " + succ_sub->to_string() );
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
    
    //
    // remove direct outgoing edges if reachable otherwise
    //
    
    if ( changed && sparsify )
    {
        // restrict search to local nodes and neighbouring nodes
        std::unordered_set< Node * >  neighbourhood{ this };
        
        for ( auto  node : successors() )
            neighbourhood.insert( node );
        
        // now test reachability
        auto  descendants = reachable_indirect( this, neighbourhood );

        // if a direct edge to otherwise reachable node exists, remove it
        for ( auto  succ_iter = successors().begin(); succ_iter != successors().end(); )
        {
            if ( descendants.find( *succ_iter ) != descendants.end() )
            {
                log( 6, "  removing " + to_string() + " ⟶ " + (*succ_iter)->to_string() + " from " + to_string() );
                succ_iter = successors().erase( succ_iter );
                // node->print();
            }// if
            else
                ++succ_iter;
        }// for
    }// if
    
    //
    // finally unlock nodes
    //
    
    for ( auto  node : locked )
    {
        node->unlock();
        log( 6, "unlocked: " + node->to_string() + " by " + this->to_string() );
    }// for
    
    return changed;
}

//
// print node with full edge information
//
void
Node::print () const
{
    std::cout << to_string() << std::endl;
    std::cout << "   #deps : " << _dep_cnt << std::endl;

    std::cout << "   succ  : " << successors().size() << std::endl;
    for ( auto  succ : successors() )
        std::cout << "      " << succ->to_string() << std::endl;
}
    
}// namespace DAG

}// namespace HLR
