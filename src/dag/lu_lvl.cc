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
#include <map>

#include <hpro/matrix/structure.hh>
#include <hpro/algebra/solve_tri.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_fac.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

using namespace HLIB;

namespace
{

struct lu_node : public node
{
    TMatrix *  A;
    
    lu_node ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

struct solve_upper_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    
    solve_upper_node ( const TMatrix *  aU,
                       TMatrix *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d, %d )", U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

struct solve_lower_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;

    solve_lower_node ( const TMatrix *  aL,
                       TMatrix *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d, %d )", L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};
    
struct update_node : public node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;

    update_node ( const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %d, %d, %d )", A->id(), B->id(), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

struct apply_node : public node
{
    TMatrix *  A;
    bool       is_recursive;
    
    apply_node ( TMatrix *  aA )
            : A( aA )
            , is_recursive( false )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "apply( %d )", A->id() ); }
    virtual std::string  color     () const { return "edd400"; }

    void set_recursive () { is_recursive = true; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; } // not needed because of direct DAG generation
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
lu_node::run_ ( const TTruncAcc &  acc )
{
    if ( is_small( A ) || is_dense( A ) )
    {
        HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
    }// if
}

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
solve_lower_node::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_upper_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
solve_upper_node::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
update_node::run_ ( const TTruncAcc &  acc )
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
// apply_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
apply_node::run_ ( const TTruncAcc &  acc )
{
    if ( ! is_recursive )
        A->apply_updates( acc, nonrecursive );
    
    // if ( is_blocked( A ) && ! is_small( A ) )
    //     A->apply_updates( acc, nonrecursive );
    // else
    //     A->apply_updates( acc, recursive );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// level-wise generation of DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

using node_map_t      = std::vector< node * >;
using nodelist_map_t  = std::vector< node_list_t >;

//
// add dependency from all sub blocks of A to "node"
//
void
add_dep_from_all_sub ( node *           node,
                       const TMatrix *  A,
                       node_map_t &     nodes )
{
    auto  node_A = nodes[ A->id() ];
    
    if (( node_A != nullptr ) && ( node_A != node ))
    {
        node->after( node_A );
        node->inc_dep_cnt();
    }// if
    else if ( is_blocked( A ) )
    {
        auto  B = cptrcast( A, TBlockMatrix );
        
        for ( uint j = 0; j < B->block_cols(); ++j )
        {
            for ( uint i = 0; i < B->block_rows(); ++i )
            {
                const TMatrix *  B_ij = B->block( i, j );
                
                if ( B_ij != nullptr )
                    add_dep_from_all_sub( node, B_ij, nodes );
            }// for
        }// for
    }// if
}

//
// add all dependencies of A and of all subblocks of A to "node"
//
void
add_dep_from_all_sub ( node *            node,
                       const TMatrix *   A,
                       node_map_t &      final_map,
                       nodelist_map_t &  updates )
{
    // add dependencies of A
    for ( auto  dep : updates[ A->id() ] )
    {
        node->after( dep );
        node->inc_dep_cnt();
    }// for
    
    // recurse
    if ( is_blocked( A ) )
    {
        auto  B = cptrcast( A, TBlockMatrix );
        
        for ( uint j = 0; j < B->block_cols(); ++j )
        {
            for ( uint i = 0; i < B->block_rows(); ++i )
            {
                const TMatrix *  B_ij = B->block( i, j );
                
                if ( B_ij == nullptr )
                    continue;

                add_dep_from_all_sub( node, B_ij, final_map, updates );
            }// for
        }// for
    }// if
}

//
// generate nodes for level-wise LU
//
node *
dag_lu_lvl ( TMatrix *         A,
             node_list_t &     nodes,
             node_map_t &      final_map,
             nodelist_map_t &  updates,
             std::mutex &      mtx_nodes )
{
    local_graph  g;

    ///////////////////////////////////////////////////////////////
    //
    // recurse to visit all diagonal blocks
    //
    ///////////////////////////////////////////////////////////////

    const bool  A_is_leaf = ( is_leaf( A ) || is_small( A ) );
    node *      node_A    = nullptr;

    {
        std::scoped_lock  lock( mtx_nodes );
        
        node_A = hlr::dag::alloc_node< lu_node >( nodes, A );
    }

    final_map[ A->id() ] = node_A;
    
    if ( ! A_is_leaf )
    {
        auto        B      = ptrcast( A, TBlockMatrix );
        const uint  nbrows = B->nblock_rows();
        const uint  nbcols = B->nblock_cols();
        std::mutex  mtx;

        for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
        // ::tbb::parallel_for< uint >(
        //     0, std::min( nbrows, nbcols ),
        //     [&,node_A,B] ( const uint  i )
            {
                auto  A_ii = B->block( i, i );
                
                assert( ! is_null( A_ii ) );
                
                auto  node_A_ii = dag_lu_lvl( A_ii, nodes, final_map, updates, mtx_nodes );

                node_A->after( node_A_ii );
                node_A->inc_dep_cnt();
            }
    }// if

    ///////////////////////////////////////////////////////////////
    //
    // actual factorisation of A, solving in block row/column of A
    // and update of trailing sub matrix with respect to A (all on L)
    //
    ///////////////////////////////////////////////////////////////

    //
    // off-diagonal solves in current block row/column
    //
    
    for ( auto  L_ij = A->next_in_block_row(); L_ij != nullptr; L_ij = L_ij->next_in_block_row() )
    {
        if ( ! is_null( L_ij ) && ( is_leaf( L_ij ) || A_is_leaf ))
        {
            node *  solve_node = hlr::dag::alloc_node< solve_lower_node >( nodes, A, L_ij );

            solve_node->after( node_A );
            solve_node->inc_dep_cnt();
            final_map[ L_ij->id() ] = solve_node;
        }// if
    }// for
        
    for ( auto  L_ji = A->next_in_block_col(); L_ji != nullptr; L_ji = L_ji->next_in_block_col() )
    {
        if ( ! is_null( L_ji ) && ( is_leaf( L_ji ) || A_is_leaf ))
        {
            node *  solve_node = hlr::dag::alloc_node< solve_upper_node >( nodes, A, L_ji );

            solve_node->after( node_A );
            solve_node->inc_dep_cnt();
            final_map[ L_ji->id() ] = solve_node;
        }// if
    }// for

    //
    // update of trailing sub matrix
    //

    for ( auto  L_ji = A->next_in_block_col(); L_ji != nullptr; L_ji = L_ji->next_in_block_col() )
    {
        for ( auto  L_il = A->next_in_block_row(); L_il != nullptr; L_il = L_il->next_in_block_row() )
        {
            auto  L_jl = product_block( L_ji, L_il );
            
            if ( ! is_null_any( L_ji, L_il, L_jl ) && ( is_leaf_any( L_ji, L_il, L_jl ) || A_is_leaf ))
            {
                node *  upd_node = hlr::dag::alloc_node< update_node >( nodes, L_ji, L_il, L_jl );
                
                updates[ L_jl->id() ].push_back( upd_node );

                add_dep_from_all_sub( upd_node, L_ji, final_map );
                add_dep_from_all_sub( upd_node, L_il, final_map );
            }// if
        }// for
    }// for
    
    return node_A;
}

//
// assign collected dependencies
//
void
assign_dependencies ( TMatrix *            A,
                      node_map_t &         final_map,
                      nodelist_map_t &     updates,
                      const node_list_t &  parent_deps = {} )
{
    if ( A == nullptr )
        return;
    
    auto        node_A       = final_map[ A->id() ];
    const bool  is_final     = ( node_A != nullptr );
    // inner factorisation nodes are only dummies, hence exclude them
    const bool  is_inner_fac = ( is_on_diag( A ) && is_blocked( A ) && ! is_small( A ) );

    if ( is_final && ! is_inner_fac )
    {
        // add dependencies of A and of all subblocks
        add_dep_from_all_sub( node_A, A, final_map, updates );

        // and dependencies from parent nodes
        for ( auto  node : parent_deps )
        {
            node_A->after( node );
            node_A->inc_dep_cnt();
        }// for
    }// if
    else
    {
        // move dependencies to subblocks
        if ( is_blocked( A ) )
        {
            // collect dependencies from parent and from A
            node_list_t  deps_A;
        
            for ( auto  dep : parent_deps )
                deps_A.push_back( dep );

            for ( auto  dep : updates[ A->id() ] )
                deps_A.push_back( dep );
            
            auto  B = ptrcast( A, TBlockMatrix );
        
            for ( uint j = 0; j < B->block_cols(); ++j )
            {
                for ( uint i = 0; i < B->block_rows(); ++i )
                {
                    auto  B_ij = B->block( i, j );
                
                    if ( B_ij != nullptr )
                        assign_dependencies( B_ij, final_map, updates, deps_A );
                }// for
            }// for
        }// if
        else
        {
            assert( A->row_is() != A->col_is() );
        }// else
    }// else
}

//
// add dependencies from all updates applied to blocks below M_root
// to node "root_apply"
//
apply_node *
add_dep_from_all_updates_below ( const TMatrix *   M,
                                 TMatrix *         M_root,
                                 apply_node *      root_apply,
                                 node_list_t &     nodes,
                                 nodelist_map_t &  updates )
{
    if ( M == nullptr )
        return root_apply;
    
    if ( ! updates[ M->id() ].empty() && ( M != M_root ))
    {
        // create apply node for root block if updates exist
        if ( root_apply == nullptr )
            root_apply = hlr::dag::alloc_node< apply_node >( nodes, M_root );
                
        for ( auto  upd : updates[ M->id() ] )
        {
            root_apply->after( upd );
            root_apply->inc_dep_cnt();
        }// for
    }// if

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( M, TBlockMatrix );
        
        for ( uint  i = 0; i < B->block_rows(); ++i )
            for ( uint  j = 0; j < B->block_cols(); ++j )
                root_apply = add_dep_from_all_updates_below( B->block( i, j ), M_root, root_apply, nodes, updates );
    }// if

    return root_apply;
}

//
// create apply nodes for all blocks below M (including) with updates
// and set dependencies to the final node of M
//
void
build_apply_dep ( TMatrix *         M,
                  node *            parent_apply,
                  node_list_t &     nodes,
                  node_map_t &      final_map,
                  nodelist_map_t &  updates )
{
    if ( M == nullptr )
        return;

    auto          M_final      = final_map[ M->id() ];
    const bool    is_inner_fac = ( is_on_diag( M ) && is_blocked( M ) && ! is_small( M ) );
    apply_node *  M_apply      = nullptr;

    //
    // if node has updates, it needs an apply node
    //

    if ( ! updates[ M->id() ].empty() )
    {
        M_apply = hlr::dag::alloc_node< apply_node >( nodes, M );

        // all updates form dependency
        for ( auto  upd : updates[ M->id() ] )
        {
            M_apply->after( upd );
            M_apply->inc_dep_cnt();
        }// for
    }// if

    //
    // also, if parent has apply node, so needs son to ensure
    // that updates are applied through hierarchy
    //

    if (( parent_apply != nullptr ) && ( M_apply == nullptr ))
    {
        M_apply = hlr::dag::alloc_node< apply_node >( nodes, M );
    }// if
    
    //
    // apply nodes depend on previous apply nodes
    // (parent_apply != nullptr  =>  M_apply != nullptr)
    //
    
    if ( parent_apply != nullptr )
    {
        M_apply->after( parent_apply );
        M_apply->inc_dep_cnt();
    }// if

    //
    // either finish with a final node or recurse
    //
    
    if ( ! is_inner_fac && ( M_final != nullptr ))
    {
        // collect updates from below and create apply node if not yet existing
        M_apply = add_dep_from_all_updates_below( M, M, M_apply, nodes, updates );

        if ( M_apply != nullptr )
        {
            //
            // apply node synchronises all updates to blocks below
            //
            
            M_final->after( M_apply );
            M_final->inc_dep_cnt();
            M_apply->set_recursive();
        }// if
    }// if
    else
    {
        if ( is_blocked( M ) )
        {
            auto  B = ptrcast( M, TBlockMatrix );

            for ( uint  i = 0; i < B->block_rows(); ++i )
                for ( uint  j = 0; j < B->block_cols(); ++j )
                    build_apply_dep( B->block( i, j ), M_apply, nodes, final_map, updates );
        }// if
        else
            assert( false );
    }// else
}

}// namespace anonymous

graph
gen_dag_lu_lvl ( TMatrix &  A )
{
    //
    // construct DAG for LU
    //

    node_list_t     nodes;
    node_map_t      final_map;
    nodelist_map_t  updates;
    std::mutex      mtx_nodes;
    const auto      nid = max_id( & A ) + 1;

    final_map.resize( nid );
    updates.resize( nid );

    auto  final = dag_lu_lvl( & A, nodes, final_map, updates, mtx_nodes );

    if ( CFG::Arith::use_accu )
        build_apply_dep( & A, nullptr, nodes, final_map, updates );
    else
        assign_dependencies( & A, final_map, updates );
    
    dag::node_list_t  start, end{ final };

    for ( auto  node : nodes )
    {
        node->finalize();
        
        if ( node->dep_cnt() == 0 )
            start.push_back( node );
    }// for

    return dag::graph( nodes, start, end );
}

}}// namespace hlr::dag
