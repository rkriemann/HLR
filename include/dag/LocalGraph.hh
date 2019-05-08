#ifndef __HLR_DAG_LOCALGRAPH_HH
#define __HLR_DAG_LOCALGRAPH_HH
//
// Project     : HLib
// File        : LocalGraph.hh
// Description : represents a local graph during refinment
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <string>

namespace HLR
{

namespace DAG
{

// forward decl. for Node
class Node;

// list of nodes
using  node_list_t  = std::list< Node * >;

//!
//! class for local sub graph during refinement
//!
class LocalGraph : public node_list_t
{
public:
    // add node and apply dependencies based on existing nodes
    void add_node_and_dependencies ( Node *  node );

    // only add node to graph
    void add_node                  ( Node *  node )
    {
        push_back( node );
    }

    // set dependencies between all nodes in graph based on
    // in/out data blocks of nodes
    void set_dependencies ();

    // output graph
    void    print () const;
    
    // output graph in DOT format
    void    print_dot ( const std::string &  filename ) const;
};

}// namespace DAG

}// namespace HLR

#endif // __HLR_DAG_NODE_HH