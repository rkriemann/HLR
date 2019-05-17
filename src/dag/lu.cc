//
// Project     : HLib
// File        : lu.cc
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include <matrix/structure.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "utils/tensor.hh"
#include "utils/checks.hh"
#include "utils/tools.hh"
#include "dag/lu.hh"

using namespace HLIB;

namespace HLR
{

namespace DAG
{

namespace
{

// map for ApplyUpdatesNode nodes
using  apply_map_t = std::unordered_map< HLIB::id_t, Node * >;

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_U = 'U';

struct LUNode : public Node
{
    TMatrix *      A;
    apply_map_t &  apply_nodes;
    
    LUNode ( TMatrix *      aA,
             apply_map_t &  aapply_nodes )
            : A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual LocalGraph          refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct SolveUNode : public Node
{
    const TMatrix *  U;
    TMatrix *        A;
    apply_map_t &    apply_nodes;
    
    SolveUNode ( const TMatrix *  aU,
                 TMatrix *        aA,
                 apply_map_t &    aapply_nodes )
            : U( aU )
            , A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d, %d )",
                                                                      U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual LocalGraph          refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct SolveLNode : public Node
{
    const TMatrix *  L;
    TMatrix *        A;
    apply_map_t &    apply_nodes;

    SolveLNode ( const TMatrix *  aL,
                 TMatrix *        aA,
                 apply_map_t &    aapply_nodes )
            : L( aL )
            , A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d, %d )",
                                                                      L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual LocalGraph          refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};
    
struct UpdateNode : public Node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;
    apply_map_t &    apply_nodes;

    UpdateNode ( const TMatrix *  aA,
                 const TMatrix *  aB,
                 TMatrix *        aC,
                 apply_map_t &    aapply_nodes )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %d, %d, %d )",
                                                                      A->id(), B->id(), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual LocalGraph          refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_A, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { id_U, C->block_is() } };
        else                        return { { id_A, C->block_is() } };
    }
};

struct ApplyUpdatesNode : public Node
{
    TMatrix *  A;
    
    ApplyUpdatesNode ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "apply( %d )", A->id() ); }
    virtual std::string  color     () const { return "edd400"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual LocalGraph          refine_      () { return {}; } // not needed because of direct DAG generation
    virtual const block_list_t  in_blocks_   () const { return { { id_U, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_leaf( A ) ) return { { id_A, A->block_is() } };
        else                return { };
    }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// LUNode
//
///////////////////////////////////////////////////////////////////////////////////////

LocalGraph
LUNode::refine_ ()
{
    LocalGraph  g;

    if ( is_blocked( A ) && ! is_small( A ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto        B   = ptrcast( A, TBlockMatrix );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();

        //
        // then create factorise/solve nodes for all blocks
        //
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = B->block( i, i );

            assert( A_ii != nullptr );

            HLR::DAG::alloc_node< LUNode >( g, A_ii, apply_nodes );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                    HLR::DAG::alloc_node< SolveUNode >( g, A_ii, B->block( j, i ), apply_nodes );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                    HLR::DAG::alloc_node< SolveLNode >( g, A_ii, B->block( i, j ), apply_nodes );
        }// for

        //
        // now create update nodes with dependencies
        //
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                        HLR::DAG::alloc_node< UpdateNode >( g, B->block( j, i ), B->block( i, l ), B->block( j, l ), apply_nodes );
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

void
LUNode::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveLNode
//
///////////////////////////////////////////////////////////////////////////////////////

LocalGraph
SolveLNode::refine_ ()
{
    LocalGraph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( A, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        //
        // first create all solve nodes
        //
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            //
            // solve in current block row
            //

            if ( ! is_null( L_ii ) )
            {
                for ( uint j = 0; j < nbc; ++j )
                    if ( ! is_null( BA->block( i, j ) ) )
                        HLR::DAG::alloc_node< SolveLNode >( g, L_ii, BA->block( i, j ), apply_nodes );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint i = 0; i < nbr; ++i )
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        HLR::DAG::alloc_node< UpdateNode >( g, BL->block( k, i ), BA->block( i, j ), BA->block( k, j ), apply_nodes );
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

void
SolveLNode::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveUNode
//
///////////////////////////////////////////////////////////////////////////////////////

LocalGraph
SolveUNode::refine_ ()
{
    LocalGraph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( A, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        //
        // first create all solve nodes
        //
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            if ( ! is_null( U_jj ) )
            {
                for ( uint i = 0; i < nbr; ++i )
                    if ( ! is_null( BA->block(i,j) ) )
                        HLR::DAG::alloc_node< SolveUNode >( g, U_jj, BA->block( i, j ), apply_nodes );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint j = 0; j < nbc; ++j )
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        HLR::DAG::alloc_node< UpdateNode >( g, BA->block( i, j ), BU->block( j, k ), BA->block( i, k ), apply_nodes );
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

void
SolveUNode::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// UpdateNode
//
///////////////////////////////////////////////////////////////////////////////////////

LocalGraph
UpdateNode::refine_ ()
{
    LocalGraph  g;

    if ( is_blocked_all( A, B, C ) && ! is_small_any( A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix );
        auto  BB = cptrcast( B, TBlockMatrix );
        auto  BC = ptrcast(  C, TBlockMatrix );

        for ( uint  i = 0; i < BC->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->block_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->block_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        HLR::DAG::alloc_node< UpdateNode >( g,
                                                            BA->block( i, k ),
                                                            BB->block( k, j ),
                                                            BC->block( i, j ),
                                                            apply_nodes );
                }// for
            }// for
        }// for
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ C->id() ];
        
        assert( apply != nullptr );

        apply->after( this );
    }// if

    return g;
}

void
UpdateNode::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        add_product( real(-1),
                     apply_normal, A,
                     apply_normal, B,
                     C, acc );
    else
        multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// ApplyUpdatesNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
ApplyUpdatesNode::run_ ( const TTruncAcc &  acc )
{
    if ( is_blocked( A ) && ! is_small( A ) )
        A->apply_updates( acc, nonrecursive );
    else
        A->apply_updates( acc, recursive );
}

//
// construct DAG for applying updates
//
void
build_apply_dag ( TMatrix *              A,
                  Node *                 parent,
                  apply_map_t &          apply_map,
                  std::list< Node * > &  apply_nodes )
{
    if ( is_null( A ) )
        return;
    
    // DBG::printf( "apply( %d )", A->id() );
    
    auto  apply = DAG::alloc_node< ApplyUpdatesNode >( apply_nodes, A );

    apply_map[ A->id() ] = apply;

    if ( parent != nullptr )
        apply->after( parent );
    
    if ( is_blocked( A ) && ! is_small( A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );

        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                    build_apply_dag( BA->block( i, j ), apply, apply_map, apply_nodes );
            }// for
        }// for
    }// if
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

Graph
gen_LU_dag ( TMatrix *  A )
{
    //
    // generate DAG for shifting and applying updates
    //
    
    apply_map_t       apply_map;
    DAG::node_list_t  apply_nodes;

    if ( CFG::Arith::use_accu )
        build_apply_dag( A, nullptr, apply_map, apply_nodes );

    //
    // construct DAG for LU
    //
    
    auto  dag = HLR::DAG::refine( new LUNode( A, apply_map ) );

    if ( ! CFG::Arith::use_accu )
        return dag;
    
    dag.add_nodes( apply_nodes );

    //
    // remove apply update nodes without updates
    //

    using  node_set_t = std::unordered_set< Node * >;

    DAG::node_list_t  work;
    node_set_t        deleted;

    for ( auto  node : dag.start() )
        work.push_back( node );

    while ( ! work.empty() )
    {
        DAG::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( dynamic_cast< ApplyUpdatesNode * >( node ) != nullptr )
            {
                if ( node->dep_cnt() == 0 )
                {
                    for ( auto  out : node->successors() )
                    {
                        out->dec_dep_cnt();
                        succ.push_back( out );
                    }// for
                    
                    deleted.insert( node );
                }// if
            }// if
        }// while

        work = std::move( succ );
    }// while

    DAG::node_list_t  nodes, start, end;

    for ( auto  node : dag.nodes() )
    {
        if ( deleted.find( node ) != deleted.end() )
        {
            // DBG::print( Term::on_red( "deleting " + node->to_string() ) );
            delete node;
        }// if
        else
        {
            nodes.push_back( node );
            
            if ( node->dep_cnt() == 0 )
                start.push_back( node );

            if ( node->successors().empty() )
                end.push_back( node );
        }// else
    }// for
    
    return  DAG::Graph( nodes, start, end );
}

}// namespace LU

}// namespace DAG
