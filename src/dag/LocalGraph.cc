//
// Project     : HLib
// File        : LocalGraph.cc
// Description : represents a local graph during refinment
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>

#include "dag/LocalGraph.hh"
#include "dag/Node.hh"

namespace HLR
{

namespace DAG
{

//
// add node and apply dependencies based on existing nodes
//
void
LocalGraph::add_node_and_dependencies ( Node *  node )
{
    for ( auto  n : *this )
    {
        if ( is_intersecting( node->in_blocks(), n->out_blocks() ) )
            node->after( n );

        if ( is_intersecting( node->out_blocks(), n->in_blocks() ) )
            node->before( n );
    }// for

    push_back( node );
}

//
// set dependencies between all nodes in graph based on
// in/out data blocks of nodes
//
void
LocalGraph::set_dependencies ()
{
    //
    // check all nodes against all other
    // - only one direction to prevent double edges
    //
    
    for ( auto  n1 : *this )
    {
        for ( auto  n2 : *this )
        {
            if ( n1 != n2 )
            {
                if ( is_intersecting( n1->in_blocks(), n2->out_blocks() ) )
                    n1->after( n2 );
            }// if
        }// for
    }// for
}

//
// output DAG
//
void
LocalGraph::print () const
{
    for ( auto  node : *this )
        node->print();
}

//
// output DAG in DOT format
//
void
LocalGraph::print_dot ( const std::string &  filename ) const
{
    std::ofstream  out( filename );

    out << "digraph G {" << std::endl
        << "  size  = \"16,16\";" << std::endl
        << "  ratio = \"1.5\";" << std::endl
        << "  node [ shape = box, style = \"filled,rounded\", fontsize = 20, fontname = \"Noto Sans\", height = 1.5, width = 4, fixedsize = true ];" << std::endl
        << "  edge [ arrowhead = open, color = \"#babdb6\" ];" << std::endl;

    for ( auto node : *this )
    {
        out << size_t(node)
            << "[ label = \"" << node->to_string() << "\", "
            << "color = \"#" << node->color()
            << "\" ];" << std::endl;
    }// for

    for ( auto node : *this )
    {
        auto  succ = node->successors().begin();

        if ( succ != node->successors().end() )
        {
            out << size_t(node) << " -> {";

            out << size_t(*succ);
        
            while ( ++succ != node->successors().end() )
                out << ";" << size_t(*succ);
            
            out << "};" << std::endl;
        }// if
    }// for

    out << "}" << std::endl;
}// if

}// namespace DAG

}// namespace HLR
