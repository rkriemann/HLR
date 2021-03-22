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
sparsify_mode_t        sparsify_mode  = sparsify_all;

// default maximal path distance in reachability test
int                    def_path_len   = 10;
    
// activates collision counting
bool                   count_coll     = false;

// counter for lock collisions
std::atomic< size_t >  collisions;

// indentation offset
constexpr char         indent[]       = "    ";

// type for a node set
using  node_set_t = std::vector< node * >;

//////////////////////////////////////////////
//
// auxiliary functions
//
//////////////////////////////////////////////

namespace
{

//
// wrappers to permit switch between std::vector and std::set
//
inline void insert  ( std::vector< node * > &  v, node *  n ) { v.push_back( n ); }
inline void insert  ( std::set< node * > &     s, node *  n ) { s.insert( n ); }

inline void reserve ( std::vector< node * > &  v, const size_t  n ) { v.reserve( n ); }
inline void reserve ( std::set< node * > &,       const size_t    ) {}

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
    uint                  step      = 1;
    const uint            max_steps = ( no_neigh ? steps : std::min< uint >( steps, neighbourhood.size() - 1 ) );

    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< node * >  sons;

        if ( no_neigh )
        {
            for ( auto  node : nodes )
            {
                for ( auto  out : node->successors() )
                {
                    sons.push_back( out );
                        
                    if ( step > 1 )
                        insert( descendants, out );
                }// for
            }// fo
        }// if
        else
        {
            for ( auto  node : nodes )
            {
                for ( auto  out : node->successors() )
                {
                    if ( contains( neighbourhood, out ) )
                    {
                        sons.push_back( out );
                        
                        if ( step > 1 )
                            insert( descendants, out );
                    }// if
                }// for
            }// for
        }// else

        nodes = std::move( sons );

        if ( ++step > max_steps )
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
    const auto  descendants = reachable_indirect( n, neighbourhood, steps );
    node_vec_t  new_out;

    new_out.reserve( n->successors().size() );
    
    // if a direct edge to otherwise reachable node exists, remove it
    for ( auto  succ : n->successors() )
    {
        if ( contains( descendants, succ ) )
        {
            HLR_LOG( 6, "  removing " + n->to_string() + " → " + succ->to_string() + " from " + n->to_string() );
        }// if
        else
            new_out.push_back( succ );
    }// for

    return new_out;
}

node_vec_t
remove_redundant ( node *              n,
                   const node_set_t &  neighbourhood,
                   const node_vec_t &  keep,
                   const uint          steps = def_path_len )
{
    const auto  descendants = reachable_indirect( n, neighbourhood, steps );
    node_vec_t  new_out;

    new_out.reserve( n->successors().size() );
    
    // if a direct edge to otherwise reachable node exists, remove it
    for ( auto  succ : n->successors() )
    {
        if ( contains( descendants, succ ) && ! contains( keep, succ ) )
        {
            HLR_LOG( 6, "  removing " + n->to_string() + " → " + succ->to_string() + " from " + n->to_string() );
        }// if
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
                
                // insert succendencies for subnodes
                for ( auto  succ_sub : (*succ)->sub_nodes() )
                {
                    // test only needed if sub nodes do not have data dependencies corresponding to actual sub blocks
                    // (example: lu_accu with shift/apply sub nodes not acting on ID_A/L/U)
                    if ( is_intersecting( node->out_blocks(), succ_sub->in_blocks() ) )
                    {
                        HLR_LOG( 6, indent + node->to_string() + " ⟶ " + succ_sub->to_string() );
                        new_out.push_back( succ_sub );
                    }// if
                    else
                    {
                        HLR_LOG( 6, indent + term::red( "NOT " ) + node->to_string() + " ⟶ " + succ_sub->to_string() );
                    }// else
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
    
    if ( changed && ( sparsify_mode & ( sparsify_node_succ | sparsify_sub_succ | sparsify_sub_all | sparsify_sub_all_ext ) ))
    {
        // restrict search to local nodes and neighbouring nodes
        node_set_t  neighbourhood;

        reserve( neighbourhood, 1 + node->successors().size() );

        insert( neighbourhood, node );
        
        for ( auto  succ : node->successors() )
            insert( neighbourhood, succ );

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
                    else
                    {
                        HLR_LOG( 6, indent + term::red( "NOT " ) + sub->to_string() + " ⟶ " + succ_sub->to_string() );
                    }// else
                }// for
            }// if
            else
            {
                if ( is_intersecting( sub->out_blocks(), succ->in_blocks() ) )
                {
                    HLR_LOG( 6, indent + sub->to_string() + " → " + succ->to_string() );
                    sub->successors().push_back( succ );
                }// if
                else
                {
                    HLR_LOG( 6, indent + term::red( "NOT " ) + sub->to_string() + " ⟶ " + succ->to_string() );
                }// else
            }// for
        }// else
    }// for

    //
    // remove direct outgoing edges if reachable via path
    //
    
    if ( sparsify_mode & sparsify_node_succ )
    {
        //
        // look for paths in successor set of each sub node
        //
        
        for ( auto  sub : node->sub_nodes() )
        {
            node_set_t  neighbourhood;

            reserve( neighbourhood, 1 + sub->successors().size() );

            insert( neighbourhood, sub );

            for ( auto  succ : sub->successors() )
                insert( neighbourhood, succ );

            sub->successors() = remove_redundant( sub, neighbourhood );
        }// for
    }// if
    else if ( sparsify_mode & sparsify_sub_succ )
    {
        //
        // look for paths in successor set of all sub nodes
        //
        
        node_set_t  neighbourhood;

        for ( auto  sub : node->sub_nodes() )
        {
            insert( neighbourhood, sub );

            for ( auto  succ : sub->successors() )
                insert( neighbourhood, succ );
        }// for

        for ( auto  sub : node->sub_nodes() )
            sub->successors() = remove_redundant( sub, neighbourhood );
    }// if
    else if ( sparsify_mode & sparsify_sub_all )
    {
        //
        // look for paths in set of all sub nodes of all successors
        //
        
        node_set_t  neighbourhood;

        for ( auto  sub : node->sub_nodes() )
            insert( neighbourhood, sub );

        for ( auto  succ : node->successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  sub : succ->sub_nodes() )
                    insert( neighbourhood, sub );
            }// if
            else
                insert( neighbourhood, succ );
        }// for

        for ( auto  sub : node->sub_nodes() )
            sub->successors() = remove_redundant( sub, neighbourhood );
    }// if
    else if ( sparsify_mode & sparsify_sub_all_ext )
    {
        //
        // look for path in set of all sub nodes of all successors
        // but only eliminate edges to external nodes (not within
        // nodes sub nodes)
        //
        
        node_set_t  neighbourhood;

        for ( auto  sub : node->sub_nodes() )
            insert( neighbourhood, sub );

        for ( auto  succ : node->successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  sub : succ->sub_nodes() )
                    insert( neighbourhood, sub );
            }// if
            else
                insert( neighbourhood, succ );
        }// for

        for ( auto  sub : node->sub_nodes() )
            sub->successors() = remove_redundant( sub, neighbourhood, node->sub_nodes() );
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

        if ( sparsify_mode & sparsify_local )
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
        insert( locked, this );

        for ( auto  sub : sub_nodes() )
            insert( locked, sub );
    
        for ( auto  succ : successors() )
        {
            if ( succ->is_refined() )
            {
                for ( auto  succ_sub : succ->sub_nodes() )
                    insert( locked, succ_sub );
            }// if
            else
                insert( locked, succ );
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

//
// remove direct edges of node to descendants if path of length <max_path_len> exists
//
void
node::sparsify ( const uint  max_path_len )
{
    node_set_t  neighbourhood;

    reserve( neighbourhood, 1 + successors().size() );

    insert( neighbourhood, this );
        
    for ( auto  succ : successors() )
        insert( neighbourhood, succ );

    successors() = remove_redundant( this, neighbourhood, max_path_len );
}
    
}// namespace dag

}// namespace hlr
