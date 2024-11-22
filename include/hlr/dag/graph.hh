#ifndef __HLR_DAG_GRAPH_HH
#define __HLR_DAG_GRAPH_HH
//
// Project     : HLR
// Module      : graph.hh
// Description : graph representing compute DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <map>
#include <functional>

#include <hlr/dag/node.hh>

namespace hlr
{

namespace dag
{

// signals multiple or single end node(s)
enum end_nodes_mode_t
{
    use_multiple_end_nodes = false,
    use_single_end_node    = true
};

//
// directed acyclic graph (DAG)
// - only holds list of nodes, start and end nodes
//
class graph
{
private:
    node_list_t  _nodes;
    node_list_t  _start;
    node_list_t  _end;

public:
    // ctor
    graph ()
    {}
    
    graph ( node_list_t &           nodes,
            node_list_t &           start,
            node_list_t &           end,
            const end_nodes_mode_t  end_mode = use_single_end_node );

    graph ( node_list_t &&          nodes,
            node_list_t &&          start,
            node_list_t &&          end,
            const end_nodes_mode_t  end_mode = use_single_end_node );

    graph ( graph &&                g )
    {
        _nodes = std::move( g._nodes );
        _start = std::move( g._start );
        _end   = std::move( g._end );
    }

    graph &  operator = ( graph &&  g )
    {
        clear();
        
        _nodes = std::move( g._nodes );
        _start = std::move( g._start );
        _end   = std::move( g._end );

        return *this;
    }

    // dtor
    ~graph ()
    {
        for ( auto  node : _nodes )
            delete node;
    }
    
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

    // ensure graph has single end node
    void    make_single_end ();

    // remove all nodes in graph
    void    clear ()
    {
        for ( auto  n : _nodes )
            delete n;

        _nodes.clear();
        _start.clear();
        _end.clear();
    }

    // remove direct edges between nodes if path of length <max_path_len> exists
    void    sparsify ( const uint  max_path_len = def_path_len );
    
    // simulate execution of DAG and look if all nodes are handled and
    // all end nodes are reached
    void    test     ();

    // output DAG
    void    print () const;
    
    // output DAG in DOT format
    void    print_dot  ( const std::string &  filename ) const; 
    
    // output DAG in DOT format with additional node grouping
    void    print_dot  ( const std::string &                           filename,
                         const std::map< std::string, node_list_t > &  groups ) const; 
    
    // output DAG in GEXF format
    void    print_gexf ( const std::string &  filename ) const;

    // return memory usage of graph (with all nodes and edges)
    size_t  mem_size   () const;
};

//
// function type for graph generation through node refinement
//
using  refine_func_t = std::function< graph ( node *,
                                              const size_t,
                                              const end_nodes_mode_t ) >;

//
// function type for graph execution
//
using  exec_func_t   = std::function< void ( graph &,
                                             const HLIB::TTruncAcc & ) >;

//
// concatenate g1 and g2
// - end nodes of g1 are dependencies for start nodes of g2
//
graph
concat  ( graph &  g1,
          graph &  g2 );

//
// merge g1 and g2
//
graph
merge   ( graph &  g1,
          graph &  g2 );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_GRAPH_HH
