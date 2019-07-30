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

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/tools.hh"
#include "hlr/utils/term.hh"
#include "hlr/dag/node.hh"

namespace hlr
{

namespace dag
{

using namespace HLIB;

// enable edge sparsification after edge refinement
constexpr bool         sparsify       = true;

// enable edge sparsification after node refinement (within local_graph)
constexpr bool         local_sparsify = true;

// default maximal path distance in reachability test
constexpr int          def_path_len   = 2;
    
// activates collision counting
constexpr bool         count_coll     = false;

// counter for lock collisions
std::atomic< size_t >  collisions;

// type for a node set
using  node_set_t = std::vector< node * >;

// indentation offset
constexpr char         indent[] = "    ";

//////////////////////////////////////////////
//
// auxiliary functions
//
//////////////////////////////////////////////

namespace
{

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
    const bool            no_neigh = ( neighbourhood.empty() );
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

node_set_t
reachable_indirect ( node *      root,
                     const uint  steps = def_path_len )
{
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
                sons.push_back( out );
                
                if ( step > 1 )
                    insert( out, descendants );
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
                   const node_set_t &  neighbourhood,
                   const uint          steps = def_path_len )
{
    auto        descendants = ( neighbourhood.empty()
                                ? reachable_indirect( n, steps )
                                : reachable_indirect( n, neighbourhood, steps ) );
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

//
// refine dependencies of local node
//
bool
refine_loc_deps ( node *  node )
{
    assert( ! is_null( node ) );
    
    HLR_LOG( 5, term::magenta( term::bold( "refine_loc_deps" ) ) + "( " + node->to_string() + " )" );

    //
    // replace successor by refined successors if available
    //
    
    bool  changed = false;

    {
        node_vec_t  new_out;
        auto        succ = node->successors().begin();
            
        while ( succ != node->successors().end() )
        {
            if ( (*succ)->is_refined() )
            {
                changed = true;
                
                // insert succendencies for subnodes (intersection test neccessary???)
                for ( auto  succ_sub : (*succ)->sub_nodes() )
                {
                    HLR_LOG( 6, indent + node->to_string() + " ⟶ " + succ_sub->to_string() );
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

        node->successors() = std::move( new_out );
    }
    
    //
    // remove direct outgoing edges if reachable otherwise
    //
    
    if ( changed && sparsify )
    {
        // restrict search to local nodes and neighbouring nodes
        node_set_t  neighbourhood;

        neighbourhood.reserve( 1 + node->successors().size() );

        insert( node, neighbourhood );
        
        for ( auto  succ : node->successors() )
            insert( succ, neighbourhood );

        node->successors() = remove_redundant( node, neighbourhood );
    }// if
    
    return changed;
}

//
// refine dependencies of sub nodes
//
void
refine_sub_deps ( node *  node )
{
    assert( ! is_null( node ) );
    
    HLR_LOG( 5, term::magenta( term::bold( "refine_sub_deps" ) ) + "( " + node->to_string() + " )" );
    
    if ( node->sub_nodes().size() == 0 )
        return;

    //
    // add dependencies for all sub nodes to old or refined successors
    //
    
    for ( auto  sub : node->sub_nodes() )
    {
        for ( auto  succ : node->successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                {
                    if ( is_intersecting( sub->out_blocks(), succ_sub->in_blocks() ) )
                    {
                        HLR_LOG( 6, indent + sub->to_string() + " ⟶ " + succ_sub->to_string() );
                        sub->successors().push_back( succ_sub );
                    }// if
                }// for
            }// if
            else
            {
                if ( is_intersecting( sub->out_blocks(), succ->in_blocks() ) )
                {
                    HLR_LOG( 6, indent + sub->to_string() + " → " + succ->to_string() );
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
        for ( auto  sub : node->sub_nodes() )
        {
            node_set_t  neighbourhood;

            neighbourhood.reserve( 1 + sub->successors().size() );
            
            insert( sub, neighbourhood );

            for ( auto  succ : sub->successors() )
                insert( succ, neighbourhood );

            sub->successors() = remove_redundant( sub, neighbourhood );
        }// for
    }// if
}

}// namespace anonymous

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

    HLR_LOG( 5, indent + to_string() );
}
    
//
// handles execution of node code and spawning of successor nodes
//
void
node::run ( const TTruncAcc & acc )
{
    HLR_LOG( 4, term::bold( term::green( "run( " ) ) + term::bold( this->to_string() ) + term::bold( term::green( " )" ) ) );
    
    run_( acc );
    reset_dep_cnt();
}

//
// split node into subnodes and update dependencies
// if retval is empty, no refinement was done
//
void
node::refine ( const size_t  min_size )
{
    HLR_LOG( 5, term::cyan( term::bold( "refine" ) ) + "( " + to_string() + " )" );

    //
    // create subnodes
    //

    auto  g = refine_( min_size );

    //
    // set dependencies in sub graph and sparsify edges
    //

    if ( ! g.is_finalized() )
    {
        g.set_dependencies();

        if ( sparsify && local_sparsify )
        {
            for ( auto  sub : g )
                sub->successors() = remove_redundant( sub, {}, g.size()-1 );
        }// if
    }// if
        
    //
    // copy nodes to local array
    //

    _sub_nodes = std::move( g );
}

//
// finalize node data (if internal data will not change)
//
void
node::finalize ()
{
    _ndeps = _dep_cnt;
    
    _in_blk_deps.resize( 0 );
    _out_blk_deps.resize( 0 );
}

//
// print node with full edge information
//
void
node::print () const
{
    std::cout << to_string() << std::endl;

    std::cout << "   in blks  : " << std::endl;
    for ( auto  b : in_blocks() )
        std::cout << "       " << b.id << " " << b.is.to_string() << std::endl;
    
    std::cout << "   out blks : " << std::endl;
    for ( auto  b : out_blocks() )
        std::cout << "       " << b.id << " " << b.is.to_string() << std::endl;
    
    std::cout << "   #deps    : " << _dep_cnt << std::endl;

    std::cout << "   succ     : " << successors().size() << std::endl;
    for ( auto  succ : successors() )
        std::cout << "       " << succ->to_string() << std::endl;
}

//
// refine dependencies of local node or of sub nodes
//
bool
node::refine_deps ( const bool  do_lock )
{
    // HLR_LOG( 5, "refine_deps( " + to_string() + " )" );

    //
    // first lock all nodes
    //

    std::set< node * >  locked;
    
    if ( do_lock )
    {
        insert( this, locked );

        for ( auto  sub : sub_nodes() )
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
    // decide to refine node or sub node dependencies
    //

    bool  changed = false;

    if ( is_refined() )
        refine_sub_deps( this );
    else
        changed = refine_loc_deps( this );
    
    //
    // finally unlock nodes
    //
    
    if ( do_lock )
    {
         for ( auto  n : locked )
         {
             n->unlock();
             HLR_LOG( 7, "unlocked: " + n->to_string() + " by " + this->to_string() );
         }// for
    }// if
         
    return changed;
}

}// namespace dag

}// namespace hlr
