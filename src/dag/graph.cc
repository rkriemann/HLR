//
// Project     : HLib
// File        : graph.cc
// Description : represents a graph in a DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>
#include <deque>
#include <unordered_set>
#include <cassert>

#include "hlr/utils/log.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/graph.hh"

namespace hlr
{

namespace dag
{

using namespace HLIB;

//////////////////////////////////////////////
//
// graph
//
//////////////////////////////////////////////

//
// ctor
//
graph::graph ( node_list_t &           nodes,
               node_list_t &           start,
               node_list_t &           end,
               const end_nodes_mode_t  end_mode )
        : _nodes( nodes )
        , _start( start )
        , _end(   end )
{
    if ( end_mode == use_single_end_node )
        make_single_end();
}

graph::graph ( node_list_t &&          nodes,
               node_list_t &&          start,
               node_list_t &&          end,
               const end_nodes_mode_t  end_mode )
        : _nodes( std::move( nodes ) )
        , _start( std::move( start ) )
        , _end(   std::move( end ) )
{
    if ( end_mode == use_single_end_node )
        make_single_end();
}

//
// return number of (out) edges
//
size_t
graph::nedges () const
{
    size_t  n = 0;
    
    for ( auto  node : _nodes )
        n += node->successors().size();
    
    return n;
}

//
// add given set of nodes to DAG
//
void
graph::add_nodes ( node_list_t &  nodes )
{
    for ( auto  node : nodes )
    {
        _nodes.push_back( node );

        // adjust dependency counters
        for ( auto  succ : node->successors() )
            succ->inc_dep_cnt();
    }// for

    _start.clear();
    _end.clear();
    
    for ( auto  node : _nodes )
    {
        if ( node->dep_cnt() == 0 )
            _start.push_back( node );
    
        if ( node->successors().empty() )
            _end.push_back( node );
    }// for
}

//
// ensure graph has single end node
//
void
graph::make_single_end ()
{
    if ( _end.size() > 1 )
    {
        auto  new_end = new empty_node();

        for ( auto  node : _end )
            new_end->after( node );

        new_end->set_dep_cnt( _end.size() );

        _nodes.push_back( new_end );
        _end.clear();
        _end.push_back( new_end );
    }// if
}
    
//
// output DAG
//
void
graph::print () const
{
    for ( auto  node : _nodes )
        node->print();
}

//
// output DAG in DOT format
//
void
graph::print_dot ( const std::string &  filename ) const
{
    std::ofstream  out( filename );

    out << "digraph G {" << std::endl
        << "  size  = \"16,16\";" << std::endl
        << "  ratio = \"1.5\";" << std::endl
        << "  node [ shape = box, style = \"filled,rounded\", fontsize = 20, fontname = \"Noto Sans\", height = 1.5, width = 4, fixedsize = true ];" << std::endl
        << "  edge [ arrowhead = open, color = \"#babdb6\" ];" << std::endl;

    for ( auto node : _nodes )
    {
        out << size_t(node) << "[ label = \"" << node->to_string() << "\", ";
        
        if ( node->successors().empty()  )
            out << "shape = parallelogram, ";

        if ( node->dep_cnt() == 0 )
            out << "penwidth = 5, fillcolor = \"#" << node->color();
        else 
            out << "color = \"#" << node->color();
        
        out << "\" ];" << std::endl;
    }// for

    for ( auto node : _nodes )
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

//
// simulate execution of DAG and
// look if all nodes are handled and
// all ende nodes are reached
//
void
graph::test ()
{
    std::unordered_set< node * >  executed;
    std::list< node * >           scheduled;
        
    for ( auto  t : _start )
        scheduled.push_back( t );
    
    while ( ! scheduled.empty() )
    {
        auto  t = behead( scheduled );

        // log( 6, t->to_string() );

        executed.insert( t );

        for ( auto  succ : t->successors() )
        {
            auto  nsuccs = succ->dec_dep_cnt();

            if ( nsuccs == 0 )
                scheduled.push_front( succ );

            assert( nsuccs >= 0 );
        }// for
    }// while

    //
    // look, if all nodes are handled
    //

    for ( auto  node : _nodes )
    {
        if ( executed.find( node ) == executed.end() )
            std::cout << "  not executed : " + node->to_string() << std::endl;
    }// for
    
    //
    // reset dependency counters of all nodes
    //
    
    for ( auto  node : _nodes )
        for ( auto  succ : node->successors() )
            succ->inc_dep_cnt();
}

//
// return memory usage of graph (with all nodes and edges)
//
size_t
graph::mem_size  () const
{
    size_t  size = sizeof(_nodes) + sizeof(_start) + sizeof(_end);

    for ( auto  n : _nodes )
    {
        size += sizeof(node) + sizeof(node*);
        size += sizeof(node*) * n->successors().size();
    }// for

    size += sizeof(node*) * _start.size();
    size += sizeof(node*) * _end.size();
    
    return size;
}

}// namespace dag

}// namespace hlr
