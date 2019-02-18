//
// Project     : HLib
// File        : dag.cc
// Description : classes and functions for compute DAGs
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <fstream>
#include <deque>
#include <unordered_set>

#include <tbb/task.h>
#include <tbb/mutex.h>
#include <tbb/parallel_do.h>

#include "tools.hh"
#include "dag.hh"

namespace DAG
{

using namespace HLIB;

// enables some debug output
#define  LOG( msg )  // DBG::print( msg )

// abbrv. for locking
#define  LOCK( mtx )  scoped_lock_t  lock( mtx )

// controls edge sparsification
const bool  sparsify = true;

//
// return true if any of the indexsets in vis0 intersects
// with and of the indexsets in vis1
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

//////////////////////////////////////////////
//
// node for collecting dependencies
// without any computation
//
//////////////////////////////////////////////

struct EmptyNode : public Node
{
    // return text version of node
    virtual std::string  to_string () const { return "Empty"; }

    // (optional) color for DAG visualization (format: RRGGBB)
    virtual std::string  color     () const { return "888A85"; }

private:

    virtual void  run_    ( const TTruncAcc & ) {}
    virtual void  refine_ ( node_list_t & )     {}
};
    
//////////////////////////////////////////////
//
// TBB based DAG execution
//
//////////////////////////////////////////////

//
// helper class for executing Node via TBB
//
class RuntimeTask : public tbb::task
{
private:
    Node *             _node;
    const TTruncAcc &  _acc;
    
public:
    RuntimeTask ( Node *             anode,
                  const TTruncAcc &  aacc )
            : _node( anode )
            , _acc( aacc )
    {
        set_ref_count( _node->dep_cnt() );
    }

    tbb::task *  execute ()
    {
        _node->run( _acc );

        for ( auto  succ : _node->successors() )
        {
            if ( succ->task()->decrement_ref_count() == 0 )
                spawn( * succ->task() );
        }// for

        return nullptr;
    }
};


//
// convert DAG to TBB tasks and execute
//
void
tbb_run ( Graph &            dag,
          const TTruncAcc &  acc )
{
    //
    // TBB needs single end node
    //
    
    Node *  final        = nullptr;
    bool    multiple_end = false;

    if ( dag.end().size() > 1 )
    {
        LOG( "INFO: multiple end nodes" );

        multiple_end = true;
        
        //
        // create single special end node
        //

        final = new EmptyNode();

        for ( auto  node : dag.end() )
            final->add_dep( node );

        final->set_dep_cnt( dag.end().size() ); // final->in.size();
        final->alloc_task( acc );
    }// if

    // create tbb tasks for all nodes
    for ( auto  node : dag.nodes() )
        node->alloc_task( acc );
    
    // if DAG has single end node, get pointer to it
    if ( final == nullptr )
        final = dag.end().front();
    
    tbb::task_list  work_queue;
    
    for ( auto  node : dag.start() )
    {
        if ( node != final )
            work_queue.push_back( * node->task() );
    }// for
    
    final->task()->increment_ref_count();                // for "tbb::wait" to actually wait for final node
    final->task()->spawn_and_wait_for_all( work_queue ); // execute all nodes except final node
    final->task()->execute();                            // and the final node explicitly
    tbb::task::destroy( * final->task() );               // not done by TBB since executed manually

    if ( multiple_end )
    {
        //
        // remove node from DAG
        //
        
        for ( auto  node : dag.end() )
            node->successors().remove( final );
        
        delete final;
    }// if
}

//////////////////////////////////////////////
//
// Node
//
//////////////////////////////////////////////

//
// ctor
//
Node::Node ()
        : _dep_cnt( 0 )
        , _task( nullptr )
{}

//
// per node initialization
//
void
Node::init ()
{
    _in_blk_deps  = in_blocks_();
    _out_blk_deps = out_blocks_();

    LOG( to_string() );
}
    
//
// handles execution of node code and spawning of successor nodes
//
void
Node::run ( const TTruncAcc & acc )
{
    LOG( Term::on_green( "run( " + this->to_string() + " )" ) );
    
    run_( acc );
}

//
// allocate runtime task
//
void
Node::alloc_task ( const TTruncAcc &  acc )
{
    if ( _task != nullptr )
        HERROR( ERR_CONSISTENCY, "(Node) alloc_task", "already have runtime task" );

    _task = new ( tbb::task::allocate_root() ) RuntimeTask( this, acc );
}

//
// split node into subnodes and update dependencies
// if retval is empty, no refinement was done
//
void
Node::refine ()
{
    LOG( Term::on_blue( "refine( " + to_string() + " )" ) );

    //
    // create subnodes
    //

    std::list< Node * >  subnodes;
        
    refine_( subnodes );

    //
    // copy nodes to local array
    //

    size_t  pos = 0;
        
    _sub_nodes.resize( subnodes.size() );

    for ( auto  node : subnodes )
    {
        _sub_nodes[pos++] = node;
    }// for
}

namespace
{

//
// return set of nodes reachable by <steps> steps
// - if <neighbourhood> is non-empty, search is restricted to nodes
//   in <neighbourhood>
//
std::unordered_set< Node * >
reachable_indirect ( Node *                                root,
                     const std::unordered_set< Node * > &  neighbourhood = {},
                     const uint                            steps         = 2 )
{
    const bool                    no_neigh = ( neighbourhood.size() == 0 );
    std::deque< Node * >          nodes;
    std::unordered_set< Node * >  descendants;
    uint                          step  = 1;

    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  sons;

        for ( auto  node : nodes )
        {
            for ( auto  out : node->successors() )
            {
                if ( no_neigh || ( neighbourhood.find( out ) != neighbourhood.end() ))
                {
                    sons.push_back( out );

                    if ( step > 1 )
                        descendants.insert( out );
                }// if
            }// for
        }// for

        nodes = std::move( sons );

        if ( ++step > steps )
            break;
    }// while

    return descendants;
}
                     
}// namespace anonymous

//
// refine dependencies for sub nodes
//
void
Node::refine_sub_deps ()
{
    LOG( "refine_sub_deps( " + to_string() + " )" );
    
    if ( _sub_nodes.size() == 0 )
        return;

    //
    // first lock all nodes
    //

    node_list_t  locked;
    
    locked.push_back( this );

    for ( auto  node : _sub_nodes )
        locked.push_back( node );
    
    for ( auto  succ : successors() )
    {
        if ( succ->is_refined() )
        {
            for ( auto  succ_sub : succ->sub_nodes() )
                locked.push_back( succ_sub );
        }// if
        else
            locked.push_back( succ );
    }// for

    // remove duplicates
    locked.sort();
    locked.unique();
    
    for ( auto  node : locked )
    {
        node->lock();
        LOG( Term::on_yellow( "locked: " + node->to_string() + " by " + this->to_string() ) );
    }// for
                
    //
    // add dependencies for all sub nodes to old or refined successors
    //
    
    for ( auto  succ : successors() )
    {
        if ( succ->is_refined() )
        {
            for ( auto  succ_sub : succ->sub_nodes() )
            {
                for ( auto  node : _sub_nodes )
                {
                    if ( is_intersecting( node->out_blocks(), succ_sub->in_blocks() ) )
                    {
                        LOG( node->to_string() + " ⟶ " + succ_sub->to_string() );
                        node->successors().push_back( succ_sub );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            for ( auto  node : _sub_nodes )
            {
                if ( is_intersecting( node->out_blocks(), succ->in_blocks() ) )
                {
                    LOG( node->to_string() + " ⟶ " + succ->to_string() );
                    node->successors().push_back( succ );
                }// if
            }// for
        }// else
    }// for

    //
    // remove direct outgoing edges if reachable via path
    //
    
    if ( sparsify )
    {
        std::unordered_set< Node * >  neighbourhood;

        // restrict search to local nodes and neighbouring nodes
        for ( auto  node : _sub_nodes )
            neighbourhood.insert( node );
        
        for ( auto  node : _successors )
        {
            if ( node->is_refined() )
            {
                for ( auto  node_sub : node->sub_nodes() )
                    neighbourhood.insert( node_sub );
            }// if
            else
                neighbourhood.insert( node );
        }// for
        
        // now test reachability
        for ( auto  node : _sub_nodes )
        {
            auto  descendants = reachable_indirect( node, neighbourhood );

            // if a direct edge to otherwise reachable node exists, remove it
            for ( auto  succ_iter = node->successors().begin(); succ_iter != node->successors().end(); )
            {
                if ( descendants.find( *succ_iter ) != descendants.end() )
                {
                    LOG( "  removing " + node->to_string() + " ⟶ " + (*succ_iter)->to_string() + " from " + node->to_string() );
                    succ_iter = node->successors().erase( succ_iter );
                    // node->print();
                }// if
                else
                    ++succ_iter;
            }// for
        }// for
    }// if

    //
    // remove duplicates
    //

    for ( auto  node : _sub_nodes )
    {
        node->successors().sort();
        node->successors().unique();
    }// for

    //
    // finally unlock nodes
    //
    
    for ( auto  node : locked )
    {
        node->unlock();
        LOG( Term::on_yellow( "unlocked: " + node->to_string() + " by " + this->to_string() ) );
    }// for
}
    
//
// check local dependencies for refinement
//
bool
Node::refine_deps ()
{
    LOG( "refine_deps( " + to_string() + " )" );

    //
    // first lock all nodes
    //

    node_list_t  locked;
    
    locked.push_back( this );

    for ( auto  node : _sub_nodes )
        locked.push_back( node );
    
    for ( auto  succ : successors() )
    {
        if ( succ->is_refined() )
        {
            for ( auto  succ_sub : succ->sub_nodes() )
                locked.push_back( succ_sub );
        }// if
        else
            locked.push_back( succ );
    }// for

    // remove duplicates
    locked.sort();
    locked.unique();
    
    for ( auto  node : locked )
    {
        node->lock();
        LOG( Term::on_yellow( "locked: " + node->to_string() + " by " + this->to_string() ) );
    }// for

    //
    // replace successor by refined successors if available
    //
    
    bool         changed = false;
    node_list_t  new_out;
    auto         succ = successors().begin();
            
    while ( succ != successors().end() )
    {
        if ( (*succ)->is_refined() )
        {
            changed = true;
                
            // insert succendencies for subnodes (intersection test neccessary???)
            for ( auto  succ_sub : (*succ)->sub_nodes() )
            {
                LOG( to_string() + " ⟶ " + succ_sub->to_string() );
                new_out.push_back( succ_sub );
            }// for

            // remove previous succendency
            succ = successors().erase( succ );
        }// if
        else
            ++succ;
    }// for

    successors().splice( successors().end(), new_out );

    // remove duplicates
    successors().sort();
    successors().unique();
    
    //
    // remove direct outgoing edges if reachable otherwise
    //
    
    if ( changed && sparsify )
    {
        // restrict search to local nodes and neighbouring nodes
        std::unordered_set< Node * >  neighbourhood{ this };
        
        for ( auto  node : successors() )
            neighbourhood.insert( node );
        
        // now test reachability
        auto  descendants = reachable_indirect( this, neighbourhood );

        // if a direct edge to otherwise reachable node exists, remove it
        for ( auto  succ_iter = successors().begin(); succ_iter != successors().end(); )
        {
            if ( descendants.find( *succ_iter ) != descendants.end() )
            {
                LOG( "  removing " + to_string() + " ⟶ " + (*succ_iter)->to_string() + " from " + to_string() );
                succ_iter = successors().erase( succ_iter );
                // node->print();
            }// if
            else
                ++succ_iter;
        }// for
    }// if
    
    //
    // finally unlock nodes
    //
    
    for ( auto  node : locked )
    {
        node->unlock();
        LOG( Term::on_yellow( "unlocked: " + node->to_string() + " by " + this->to_string() ) );
    }// for
    
    return changed;
}

//
// print node with full edge information
//
void
Node::print () const
{
    std::cout << to_string() << std::endl;
    std::cout << "   #deps : " << _dep_cnt << std::endl;

    std::cout << "   succ  : " << successors().size() << std::endl;
    for ( auto  succ : successors() )
        std::cout << "      " << succ->to_string() << std::endl;
}
    

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
// construct DAG using refinement of given node
//
Graph
refine ( Node *  root )
{
    using  mutex_t       = tbb::mutex;
    using  scoped_lock_t = mutex_t::scoped_lock;
    
    if ( root == nullptr )
        HERROR( ERR_ARG, "refine", "root node is NULL" );
    
    std::deque< Node * >  nodes;
    std::list< Node * >   tasks, start, end;
    mutex_t               mtx_tasks, mtx_sub;
    
    nodes.push_back( root );

    while ( ! nodes.empty() )
    {
        std::deque< Node * >  subnodes;

        auto  node_refine      = []  ( Node * node ) { node->refine(); };
        auto  node_refine_deps = []  ( Node * node ) { node->refine_sub_deps(); };
        auto  node_delete      = []  ( Node * node ) { if ( node->is_refined() ) { LOG( Term::on_red( "delete : " + node->to_string() ) ); delete node; } };
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

//
// execute given DAG
//
void
Graph::run ( const TTruncAcc &  acc )
{
    //
    // run dag
    //

    tbb_run( *this, acc );
}

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
            
            if ( nsuccs < 0 )
                HERROR( ERR_CONSISTENCY, "test_dag", "negative deps in " + succ->to_string() );
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

}// namespace DAG
