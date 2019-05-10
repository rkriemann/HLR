#ifndef __HLR_DAG_NODE_HH
#define __HLR_DAG_NODE_HH
//
// Project     : HLib
// File        : Node.hh
// Description : node in a compute DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <vector>
#include <list>
#include <string>

#include <tbb/spin_mutex.h>

#include <cluster/TIndexSet.hh>
#include <base/TTruncAcc.hh>

#include "dag/LocalGraph.hh"

namespace HLR
{

namespace DAG
{

//
// defines a memory block by an Id and a block index set
// (memory block as used within a matrix)
//
struct mem_block_t
{
    HLIB::id_t            id;
    HLIB::TBlockIndexSet  is;
};

// list of memory blocks
using  block_list_t = std::vector< mem_block_t >;

//
// return true if any of the indexsets in vis0 intersects
// with any of the indexsets in vis1
//
template < typename T_container >
bool
is_intersecting ( const T_container &  vblk0,
                  const T_container &  vblk1 )
{
    for ( auto &  blk0 : vblk0 )
        for ( auto &  blk1 : vblk1 )
            if (( blk0.id == blk1.id ) && is_intersecting( blk0.is, blk1.is ) )
                return true;

    return false;
}

//!
//! @class Node
//!
//! @brief Represents a node in a DAG
//!
//! A node in a DAG with incoming and outgoing edges (dependencies)
//! - data dependencies are described in the form of mem_block_t lists (in/out)
//! - also implements actual actions to perform per node
//! - user defined functions are "run_", "in_blocks_" and "out_blocks_"
//!
class Node
{
private:
    // successor nodes in DAG
    node_list_t            _successors;

    // dependency counter (#incoming edges)
    int                    _dep_cnt;

    // block index set dependencies for automatic dependency refinement
    block_list_t           _in_blk_deps;
    block_list_t           _out_blk_deps;

    // set of sub nodes
    std::vector< Node * >  _sub_nodes;

    // mutex to handle concurrent access to internal data
    tbb::spin_mutex        _mutex;

public:
    // ctor
    Node ()
            : _dep_cnt( 0 )
    {}

    // dtor
    virtual ~Node () {}

    // per node initialization (e.g., data dependencies)
    void  init  ();
    
    // handles execution of node code and spawning of successor nodes
    void  run   ( const HLIB::TTruncAcc & acc );

    // givess access to successor nodes
    node_list_t &        successors ()       { return _successors; }
    const node_list_t &  successors () const { return _successors; }

    //
    // task dependencies
    //
    
    // run <this> before <t>
    void  before   ( Node *  t ) { _successors.push_back( t ); }
    
    // run <this> after <t>
    void  after    ( Node *  t ) { t->_successors.push_back( this ); }

    // return dependency counter
    int   dep_cnt      () const   { return _dep_cnt; }
    
    // increase task dependency counter
    int   inc_dep_cnt  ()         { return ++_dep_cnt; }
    
    // decrease task dependency counter
    int   dec_dep_cnt  ()         { return --_dep_cnt; }
    
    // set task dependency counter
    void  set_dep_cnt  ( int  d ) { _dep_cnt = d; }
    
    //
    // data dependencies
    //
    
    // return local list of block index sets for input dependencies
    const block_list_t &  in_blocks  () const { return _in_blk_deps; }
    
    // return local list of block index sets for output dependencies
    const block_list_t &  out_blocks () const { return _out_blk_deps; }

    //
    // refinement
    //
    
    // return true if node is refined
    bool  is_refined  () const { return ! _sub_nodes.empty(); }

    // give access to sub nodes
    auto  sub_nodes ()       -> decltype(_sub_nodes) { return _sub_nodes; };
    auto  sub_nodes () const -> decltype(_sub_nodes) { return _sub_nodes; };
    
    // split node into subnodes and update dependencies
    // if retval is empty, no refinement was done
    void  refine ();

    //
    // refine dependencies for sub nodes
    //
    void  refine_sub_deps ();
    
    //
    // refine local dependencies, e.g., if destinations were refined
    // - return true, if dependencies were refined
    //
    bool  refine_deps  ();

    //
    // mutex functions
    //

    void lock   () { _mutex.lock(); }
    void unlock () { _mutex.unlock(); }

    //
    // output and visualization
    //
    
    // print node with full edge information
    void  print () const;
    
    // return text version of node
    virtual std::string  to_string () const { return "Node"; }

    // (optional) color for DAG visualization (format: RRGGBB)
    virtual std::string  color     () const { return "FFFFFF"; }

private:

    //
    // private functions used by above public wrappers
    //
    
    virtual
    void
    run_ ( const HLIB::TTruncAcc & ) = 0;

    virtual
    const block_list_t
    in_blocks_ () const
    {
        return block_list_t();
    }
    
    virtual
    const block_list_t
    out_blocks_ () const
    {
        return block_list_t();
    }

    virtual
    LocalGraph
    refine_ () = 0;
};

//
// wrapper to simultaneously allocate node and put into list of subnodes
//
template < typename T,
           typename ... Args >
T *
alloc_node ( std::list< Node * > & subnodes,
             Args && ...           args )
{
    auto  node = new T( std::forward< Args >( args ) ... );

    subnodes.push_back( node );

    return node;
}

template < typename T,
           typename ... Args >
T *
alloc_node ( LocalGraph &  g,
             Args && ...   args )
{
    auto  node = new T( std::forward< Args >( args ) ... );

    g.push_back( node );

    return node;
}

//
// special class for an empty node without anything to
// compute and without refinement
//
class EmptyNode : public Node
{
    // return text version of node
    virtual std::string  to_string () const { return "Empty"; }

    // (optional) color for DAG visualization (format: RRGGBB)
    virtual std::string  color     () const { return "888A85"; }

private:

    virtual void        run_    ( const HLIB::TTruncAcc & ) {}
    virtual LocalGraph  refine_ ()                          { return {}; }
};

}// namespace DAG

}// namespace HLR

#endif // __HLR_DAG_NODE_HH
