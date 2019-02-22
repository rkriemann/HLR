#ifndef __HLR_DAG_GRAPH_HH
#define __HLR_DAG_GRAPH_HH
//
// Project     : HLib
// File        : Graph.hh
// Description : Graph representing compute DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "Node.hh"

namespace HLR
{

namespace DAG
{

//
// directed acyclic graph (DAG)
// - only holds list of nodes, start and end nodes
//
class Graph
{
private:
    node_list_t  _nodes;
    node_list_t  _start;
    node_list_t  _end;

public:
    // ctor
    Graph ( node_list_t &  nodes,
            node_list_t &  start,
            node_list_t &  end );

    // return number of nodes
    size_t  nnodes () const { return _nodes.size(); }

    // return number of (out) edges
    size_t  nedges () const;

    // return list of all/start/end nodes
    node_list_t &        nodes ()       { return _nodes; }
    const node_list_t &  nodes () const { return _nodes; }
    node_list_t &        start ()       { return _start; }
    const node_list_t &  start () const { return _start; }
    node_list_t &        end   ()       { return _end; }
    const node_list_t &  end   () const { return _end; }

    // add given set of nodes to DAG
    // (assumption: dependency counter already set)
    void    add_nodes ( node_list_t &  nodes );
    
    // simulate execution of DAG and look if all nodes are handled and
    // all end nodes are reached
    void    test     ();

    // output DAG
    void    print () const;
    
    // output DAG in DOT format
    void    print_dot ( const std::string &  filename ) const;
};

//
// construct DAG based on refinement of given node
//
Graph
refine  ( Node *  node );

}// namespace DAG

}// namespace HLR

#endif // __HLR_DAG_GRAPH_HH
