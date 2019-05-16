//
// Project     : HLib
// File        : Graph.cc
// Description : represents a Graph in a DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>
#include <deque>
#include <unordered_set>
#include <cassert>
#include <mutex>

#include <tbb/parallel_for_each.h>

#include "utils/log.hh"
#include "utils/tools.hh"
#include "dag/Graph.hh"

namespace HLR
{

namespace DAG
{

using namespace HLIB;

//////////////////////////////////////////////
//
// Graph
//
//////////////////////////////////////////////

//
// ctor
//
Graph::Graph ( node_list_t &  nodes,
               node_list_t &  start,
               node_list_t &  end )
        : _nodes( nodes )
        , _start( start )
        , _end(   end )
{}

//
// return number of (out) edges
//
size_t
Graph::nedges () const
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
Graph::add_nodes ( node_list_t &  nodes )
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
// output DAG
//
void
Graph::print () const
{
    for ( auto  node : _nodes )
        node->print();
}

//
// output DAG in DOT format
//
void
Graph::print_dot ( const std::string &  filename ) const
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
Graph::test ()
{
    std::unordered_set< Node * >  executed;
    std::list< Node * >           scheduled;
        
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
// construct DAG using refinement of given node
//
Graph
refine ( Node *  root )
{
    assert( root != nullptr );
    
    std::deque< Node * >  nodes;
    std::list< Node * >   tasks, start, end;
    std::mutex            mtx;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  subnodes, del_nodes;

        auto  node_dep_refine = [&] ( Node * node )
        {
            const bool  node_changed = node->refine_deps();

            if ( node->is_refined() )       // node was refined; collect all sub nodes
            {
                std::scoped_lock  lock( mtx );
                    
                for ( auto  sub : node->sub_nodes() )
                    subnodes.push_back( sub );
                    
                del_nodes.push_back( node );
            }// if
            else if ( node_changed )        // node was not refined but dependencies were
            {
                std::scoped_lock  lock( mtx );
                    
                subnodes.push_back( node );
            }// if
            else                            // neither node nor dependencies changed: reached final state
            {
                {
                    std::scoped_lock  lock( mtx );
                    
                    tasks.push_back( node );
                }

                // adjust dependency counter of successors (which were NOT refined!)
                for ( auto  succ : node->successors() )
                    succ->inc_dep_cnt();
            }// else
        };

        // first refine nodes
        tbb::parallel_for_each( nodes.begin(), nodes.end(),
                                [] ( Node * node ) { node->refine(); } );

        // then refine dependencies and collect new nodes
        tbb::parallel_for_each( nodes.begin(), nodes.end(),
                                node_dep_refine );

        // delete all refined nodes (only after "dep_refine" since accessed in "refine_deps")
        tbb::parallel_for_each( del_nodes.begin(), del_nodes.end(),
                                [] ( Node * node ) { delete node; } );
        
        nodes = std::move( subnodes );
    }// while

    //
    // adjust dependency counter
    //
    
    // for ( auto  t : tasks )
    // {
    //     for ( auto  succ : t->successors() )
    //         succ->inc_dep_cnt();
    // }// for

    //
    // collect start and end nodes
    //
    
    // for ( auto  t : tasks )
    tbb::parallel_do( tasks,
                      [&] ( Node * node )
                      {
                          if ( node->dep_cnt() == 0 )
                          {
                              std::scoped_lock  lock( mtx );
                              
                              start.push_back( node );
                          }// if
                          
                          if ( node->successors().empty() )
                          {
                              std::scoped_lock  lock( mtx );
                              
                              end.push_back( node );
                          }// if
                      } );

    return Graph( tasks, start, end );
}

}// namespace DAG

}// namespace HLR
