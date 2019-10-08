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
#include <map>

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
        {
            new_end->after( node );
            new_end->inc_dep_cnt();
        }// for

        _nodes.push_back( new_end );
        _end.clear();
        _end.push_back( new_end );
    }// if
}

//
// remove direct edges between nodes if path of length <max_path_len> exists
//
void
graph::sparsify ( const uint  max_path_len )
{
    for ( auto  node : nodes() )
    {
        node->sparsify( max_path_len );

        node->set_dep_cnt( 0 );
    }// for

    for ( auto  node : nodes() )
    {
        for ( auto  succ : node->successors() )
            succ->inc_dep_cnt();
    }// for

    _start.clear();
    _end.clear();
    
    for ( auto  node : nodes() )
    {
        node->finalize();

        if ( node->dep_cnt() == 0 )
            _start.push_back( node );

        if ( node->successors().empty() )
            _end.push_back( node );
    }// for
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
        << "  edge [ arrowhead = open, color = \"#000000\" ];" << std::endl; // #babdb6

    std::map< node *, HLIB::id_t >  node_ids;
    HLIB::id_t                      id = 0;
    
    for ( auto node : _nodes )
    {
        node_ids[ node ] = id;
        
        out << "  " << id << " [ label = \"" << node->to_string() << "\", ";

        ++id;
        
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
            out << "  " << node_ids[node] << " -> {";

            out << node_ids[*succ];
        
            while ( ++succ != node->successors().end() )
                out << ";" << node_ids[*succ];
            
            out << "};" << std::endl;
        }// if
    }// for

    out << "}" << std::endl;
}// if

//
// output DAG in GEXF format
//
void
graph::print_gexf ( const std::string &  filename ) const
{
    std::ofstream  out( filename );

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl
        << "<gexf xmlns=\"http://www.gexf.net/1.2draft\" version=\"1.2\">" << std::endl
        << "  <graph mode=\"static\" defaultedgetype=\"directed\">" << std::endl;

    std::map< node *, HLIB::id_t >  node_ids;
    HLIB::id_t                      nid = 0;
    HLIB::id_t                      eid = 0;
    
    out << "    <nodes>" << std::endl;
    for ( auto node : _nodes )
    {
        out << "      <node id=\"" << nid << "\" label=\"" << node->to_string() << "\" />" << std::endl;
        node_ids[ node ] = nid++;
    }// for
    out << "    </nodes>" << std::endl;

    out << "    <edges>" << std::endl;
    for ( auto node : _nodes )
        for ( auto  succ : node->successors() )
            out << "      <edge id=\"" << eid++ << "\" source=\"" << node_ids[node] << "\" target=\"" << node_ids[succ] << "\" />" << std::endl;
    out << "    </edges>" << std::endl;

    out << "  </graph>" << std::endl
        << "</gexf>" << std::endl;
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
        size += n->mem_size() + sizeof(node*);

    size += sizeof(node*) * _start.size();
    size += sizeof(node*) * _end.size();
    
    return size;
}

//
// concatenate dag1 and dag2
// - end nodes of dag1 are dependencies for start nodes of dag2
// - input DAGs are destroyed
//
graph
concat ( graph &  g1,
         graph &  g2 )
{
    std::list< node * >  nodes, start, end;

    // collect all nodes
    for ( auto  n : g1.nodes() ) nodes.push_back( n );
    for ( auto  n : g2.nodes() ) nodes.push_back( n );

    for ( auto  n : g1.start() ) start.push_back( n );
    for ( auto  n : g2.end()   ) end.push_back( n );

    for ( auto  start2 : g2.start() )
    {
        for ( auto  end1 : g1.end() )
        {
            start2->after( end1 );
            start2->inc_dep_cnt();
        }// for
    }// for

    return graph( std::move( nodes ), std::move( start ), std::move( end ), use_multiple_end_nodes );
}

//
// merge g1 and g2
//
graph
merge ( graph &  g1,
        graph &  g2 )
{
    std::list< node * >  nodes, start, end;

    // collect all nodes
    for ( auto  n : g1.nodes() ) nodes.push_back( n );
    for ( auto  n : g2.nodes() ) nodes.push_back( n );

    for ( auto  n : g1.start() ) start.push_back( n );
    for ( auto  n : g2.start() ) start.push_back( n );

    for ( auto  n : g1.end()   ) end.push_back( n );
    for ( auto  n : g2.end()   ) end.push_back( n );

    return graph( std::move( nodes ), std::move( start ), std::move( end ), use_multiple_end_nodes );
}

}// namespace dag

}// namespace hlr
