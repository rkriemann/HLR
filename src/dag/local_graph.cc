//
// Project     : HLib
// File        : local_graph.cc
// Description : represents a local graph during refinment
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>

#include "hlr/dag/local_graph.hh"
#include "hlr/dag/node.hh"

namespace hlr
{

namespace dag
{

//
// add node and apply dependencies based on existing nodes
//
void
local_graph::add_node_and_dependencies ( node *  node )
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
local_graph::set_dependencies ()
{
    // no edge construction if alread finished
    if ( _finished )
        return;
    
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
                {
                    HLR_LOG( 4, "    " + n2->to_string() + " → " + n1->to_string() );
                    n1->after( n2 );
                }// if
            }// if
        }// for
    }// for
}

//
// output DAG
//
void
local_graph::print () const
{
    for ( auto  node : *this )
        node->print();
}

//
// output DAG in DOT format
//
void
local_graph::print_dot ( const std::string &  filename ) const
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

}// namespace dag

}// namespace hlr
