//
// Project     : HLib
// File        : dag.cc
// Description : classes and functions for compute DAGs
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>
#include <deque>
#include <unordered_set>
#include <cassert>

#include <tbb/mutex.h>
#include <tbb/parallel_do.h>

#include "../tools.hh"
#include "Graph.hh"

namespace DAG
{

using namespace HLIB;

// enables some debug output
#define  LOG( lvl, msg )  if ( HLIB::verbose( lvl ) ) DBG::print( msg )

// abbrv. for locking
#define  LOCK( mtx )  scoped_lock_t  lock( mtx )

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

        // LOG::print( t->to_string() );

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
            LOG::print( "  not executed : " + node->to_string() );
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
    using  mutex_t       = tbb::mutex;
    using  scoped_lock_t = mutex_t::scoped_lock;
    
    assert( root != nullptr );
    
    std::deque< Node * >  nodes;
    std::list< Node * >   tasks, start, end;
    mutex_t               mtx_tasks, mtx_sub;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  subnodes;

        auto  node_refine      = []  ( Node * node ) { node->refine(); };
        auto  node_refine_deps = []  ( Node * node ) { node->refine_sub_deps(); };
        auto  node_delete      = []  ( Node * node ) { if ( node->is_refined() ) { LOG( 5, "delete : " + node->to_string() ); delete node; } };
        auto  node_collect     = [&] ( Node * node )
        {
            if ( node->is_refined() )       // node was refined; collect all subs
            {
                LOCK( mtx_sub );
                for ( auto  sub : node->sub_nodes() )
                    subnodes.push_back( sub );
            }// if
            else if ( node->refine_deps() ) // node was not refined but dependencies were
            {
                LOCK( mtx_sub );
                subnodes.push_back( node );
            }// if
            else                            // neither node nore dependencies have changed: will not be touched
            { 
                LOCK( mtx_tasks );
                tasks.push_back( node );
            }// else
        };

        if ( true )
        {
            for ( auto  node : nodes ) node_refine( node );      // first refine nodes
            for ( auto  node : nodes ) node_refine_deps( node ); // then refine dependencies between sub nodes
            for ( auto  node : nodes ) node_collect( node );     // collect new (and delete refined) nodes
            for ( auto  node : nodes ) node_delete( node );      // delete all refined nodes
                                                                 // (only after "collect" since accessed in "collect>refine_deps")
        }// if
        else
        {
            // same as above put in parallel
            tbb::parallel_do( nodes, node_refine );
            tbb::parallel_do( nodes, node_refine_deps );
            tbb::parallel_do( nodes, node_collect );
            tbb::parallel_do( nodes, node_delete );
        }// else
        
        nodes = std::move( subnodes );
    }// while

    //
    // adjust dependency counter
    //
    
    for ( auto  t : tasks )
    {
        for ( auto  succ : t->successors() )
            succ->inc_dep_cnt();
    }// for
    
    for ( auto  t : tasks )
    {
        // t->dep_cnt = t->in.size();

        if ( t->dep_cnt() == 0 )
            start.push_back( t );

        if ( t->successors().empty() )
            end.push_back( t );
    }// for

    return Graph( tasks, start, end );
}

}// namespace DAG
