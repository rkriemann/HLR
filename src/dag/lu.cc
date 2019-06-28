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

#include <matrix/structure.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/dag/lu.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

using hlr::matrix::level_matrix;

// map for apply_node nodes
using  apply_map_t = std::unordered_map< HLIB::id_t, node * >;

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_U = 'U';

struct lu_node : public node
{
    TMatrix *      A;
    apply_map_t &  apply_nodes;
    
    lu_node ( TMatrix *      aA,
              apply_map_t &  aapply_nodes )
            : A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct lvllu_node : public node
{
    TMatrix *       A;
    apply_map_t &   apply_nodes;
    
    lvllu_node ( TMatrix *       aA,
                 apply_map_t &   aapply_nodes )
            : A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lvllu( %d )", A->id() ); }
    virtual std::string  color     () const { return "a40000"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct solve_upper_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    apply_map_t &    apply_nodes;
    
    solve_upper_node ( const TMatrix *  aU,
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
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct solve_lower_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;
    apply_map_t &    apply_nodes;

    solve_lower_node ( const TMatrix *  aL,
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
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};
    
struct update_node : public node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;
    apply_map_t &    apply_nodes;

    update_node ( const TMatrix *  aA,
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
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_A, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { id_U, C->block_is() } };
        else                        return { { id_A, C->block_is() } };
    }
};

struct apply_node : public node
{
    TMatrix *  A;
    
    apply_node ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "apply( %d )", A->id() ); }
    virtual std::string  color     () const { return "edd400"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      () { return {}; } // not needed because of direct DAG generation
    virtual const block_list_t  in_blocks_   () const { return { { id_U, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_leaf( A ) ) return { { id_A, A->block_is() } };
        else                return { };
    }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
lu_node::refine_ ()
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( A ) )
    {
        auto        B   = ptrcast( A, TBlockMatrix );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();

        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = B->block( i, i );

            assert( A_ii != nullptr );

            hlr::dag::alloc_node< lu_node >( g, A_ii, apply_nodes );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                    hlr::dag::alloc_node< solve_upper_node >( g, A_ii, B->block( j, i ), apply_nodes );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                    hlr::dag::alloc_node< solve_lower_node >( g, A_ii, B->block( i, j ), apply_nodes );

            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                        hlr::dag::alloc_node< update_node >( g, B->block( j, i ), B->block( i, l ), B->block( j, l ), apply_nodes );
        }// for
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
lu_node::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// lvllu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
lvllu_node::refine_ ()
{
    return {};
}

void
lvllu_node::run_ ( const TTruncAcc &  acc )
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

local_graph
solve_lower_node::refine_ ()
{
    local_graph  g;

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
                        hlr::dag::alloc_node< solve_lower_node >( g, L_ii, BA->block( i, j ), apply_nodes );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint i = 0; i < nbr; ++i )
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        hlr::dag::alloc_node< update_node >( g, BL->block( k, i ), BA->block( i, j ), BA->block( k, j ), apply_nodes );
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

local_graph
solve_upper_node::refine_ ()
{
    local_graph  g;

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
                        hlr::dag::alloc_node< solve_upper_node >( g, U_jj, BA->block( i, j ), apply_nodes );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint j = 0; j < nbc; ++j )
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        hlr::dag::alloc_node< update_node >( g, BA->block( i, j ), BU->block( j, k ), BA->block( i, k ), apply_nodes );
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

local_graph
update_node::refine_ ()
{
    local_graph  g;

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
                        hlr::dag::alloc_node< update_node >( g,
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
    if ( is_blocked( A ) && ! is_small( A ) )
        A->apply_updates( acc, nonrecursive );
    else
        A->apply_updates( acc, recursive );
}

//
// construct DAG for applying updates
//
void
build_apply_dag ( TMatrix *           A,
                  node *              parent,
                  apply_map_t &       apply_map,
                  dag::node_list_t &  apply_nodes )
{
    if ( is_null( A ) )
        return;
    
    // DBG::printf( "apply( %d )", A->id() );
    
    auto  apply = dag::alloc_node< apply_node >( apply_nodes, A );

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

graph
gen_dag_lu_rec ( TMatrix *                          A,
                 std::function< graph ( node * ) >  refine )
{
    //
    // generate DAG for shifting and applying updates
    //
    
    apply_map_t       apply_map;
    dag::node_list_t  apply_nodes;

    if ( CFG::Arith::use_accu )
        build_apply_dag( A, nullptr, apply_map, apply_nodes );

    //
    // construct DAG for LU
    //
    
    auto  dag = refine( new lu_node( A, apply_map ) );

    if ( ! CFG::Arith::use_accu )
        return dag;
    
    dag.add_nodes( apply_nodes );

    //
    // remove apply update nodes without updates
    //
    // TEST: loop over apply_nodes (top to bottom) instead of performing
    //       BFS in full DAG
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;

    // deleted.reserve( dag.nnodes() );
    
    for ( auto  node : dag.start() )
        work.push_back( node );

    while ( ! work.empty() )
    {
        dag::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( dynamic_cast< apply_node * >( node ) != nullptr )
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

    dag::node_list_t  nodes, start, end;

    for ( auto  node : dag.nodes() )
    {
        if ( contains( deleted, node ) )
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
    
    return  dag::graph( nodes, start, end );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// level-wise generation of DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

namespace
{

using node_map_t      = std::vector< node * >;
using nodelist_map_t  = std::vector< node_list_t >;
// using node_map_t      = std::unordered_map< HLIB::id_t, node * >;
// using nodelist_map_t  = std::unordered_map< HLIB::id_t, node_list_t >;

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
//              level_matrix *    L,
             node_list_t &     nodes,
             node_map_t &      final_map,
             nodelist_map_t &  updates,
             apply_map_t &     apply_nodes )
{
    local_graph  g;

    ///////////////////////////////////////////////////////////////
    //
    // recurse to visit all diagonal blocks
    //
    ///////////////////////////////////////////////////////////////

    const bool  A_is_leaf = ( is_leaf( A ) || is_small( A ) );
    auto        node_A    = hlr::dag::alloc_node< lvllu_node >( nodes, A, apply_nodes );

    final_map[ A->id() ] = node_A;
    
    if ( ! A_is_leaf )
    {
        auto        B      = ptrcast( A, TBlockMatrix );
        const uint  nbrows = B->nblock_rows();
        const uint  nbcols = B->nblock_cols();

        for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
        {
            auto  A_ii = B->block( i, i );
            
            assert( ! is_null( A_ii ) );
            
            auto  node_A_ii = dag_lu_lvl( A_ii,
                                          // L->below(),
                                          nodes, final_map, updates, apply_nodes );

            node_A->after( node_A_ii );
            node_A->inc_dep_cnt();
        }// for
    }// if

    ///////////////////////////////////////////////////////////////
    //
    // actual factorisation of A, solving in block row/column of A
    // and update of trailing sub matrix with respect to A (all on L)
    //
    ///////////////////////////////////////////////////////////////

    #if 0

    //
    // get block row/column of A in L
    //

    const auto  [ bi, bj ] = L->get_index( A );

    // check if found in L
    assert(( bi != L->nblock_rows() ) && ( bj != L->nblock_cols() ));
    
    // should be on diagonal
    assert( bi == bj );
    
    //
    // off-diagonal solves in current block row/column
    //
    
    auto  end_i = L->col_end( bi );
    auto  end_j = L->row_end( bj );
    
    for ( auto  row_iter = L->row_iter( bi, bj+1 ); row_iter != end_j; ++row_iter )
    {
        auto  L_ij = row_iter->second;
            
        if ( ! is_null( L_ij ) && ( is_leaf( L_ij ) || A_is_leaf ))
        {
            // DBG::printf( "solve_lower_left( %d, %d )", A->id(), L_ij->id() );

            auto  solve_node = hlr::dag::alloc_node< solve_lower_node >( nodes, A, L_ij, apply_nodes );

            solve_node->after( node_A );
            solve_node->inc_dep_cnt();
            final_map[ L_ij->id() ] = solve_node;
        }// if
    }// for
        
    for ( auto  col_iter = L->col_iter( bj+1, bi ); col_iter != end_i; ++col_iter )
    {
        auto  L_ji = col_iter->second;
            
        if ( ! is_null( L_ji ) && ( is_leaf( L_ji ) || A_is_leaf ))
        {
            // DBG::printf( "solve_upper_right( %d, %d )", A->id(), L_ji->id() );

            auto  solve_node = hlr::dag::alloc_node< solve_upper_node >( nodes, A, L_ji, apply_nodes );

            solve_node->after( node_A );
            solve_node->inc_dep_cnt();
            final_map[ L_ji->id() ] = solve_node;
        }// if
    }// for

    //
    // update of trailing sub matrix
    //

    for ( auto  col_iter = L->col_iter( bj+1, bi ); col_iter != end_i; ++col_iter )
    {
        auto  L_ji = col_iter->second;
            
        for ( auto  row_iter = L->row_iter( bi, bj+1 ); row_iter != end_j; ++row_iter )
        {
            auto  L_il = row_iter->second;
            auto  L_jl = L->block( row_iter->first, col_iter->first );
            
            if ( ! is_null_any( L_ji, L_il, L_jl ) && ( is_leaf_any( L_ji, L_il, L_jl ) || A_is_leaf ))
            {
                // DBG::printf( "update( %d, %d, %d )", L_ji->id(), L_il->id(), L_jl->id() );
                
                auto  upd_node = hlr::dag::alloc_node< update_node >( nodes, L_ji, L_il, L_jl, apply_nodes );

                updates[ L_jl->id() ].push_back( upd_node );

                add_dep_from_all_sub( upd_node, L_ji, final_map );
                add_dep_from_all_sub( upd_node, L_il, final_map );
            }// if
        }// for
    }// for

    #else

    //
    // off-diagonal solves in current block row/column
    //
    
    for ( auto  L_ij = A->next_in_block_row(); L_ij != nullptr; L_ij = L_ij->next_in_block_row() )
    {
        if ( ! is_null( L_ij ) && ( is_leaf( L_ij ) || A_is_leaf ))
        {
            auto  solve_node = hlr::dag::alloc_node< solve_lower_node >( nodes, A, L_ij, apply_nodes );

            solve_node->after( node_A );
            solve_node->inc_dep_cnt();
            final_map[ L_ij->id() ] = solve_node;
        }// if
    }// for
        
    for ( auto  L_ji = A->next_in_block_col(); L_ji != nullptr; L_ji = L_ji->next_in_block_col() )
    {
        if ( ! is_null( L_ji ) && ( is_leaf( L_ji ) || A_is_leaf ))
        {
            auto  solve_node = hlr::dag::alloc_node< solve_upper_node >( nodes, A, L_ji, apply_nodes );

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
                auto  upd_node = hlr::dag::alloc_node< update_node >( nodes, L_ji, L_il, L_jl, apply_nodes );

                updates[ L_jl->id() ].push_back( upd_node );

                add_dep_from_all_sub( upd_node, L_ji, final_map );
                add_dep_from_all_sub( upd_node, L_il, final_map );
            }// if
        }// for
    }// for

    #endif
    
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
    apply_map_t     apply_nodes;
    const auto      nid = max_id( & A ) + 1;

    final_map.resize( nid );
    updates.resize( nid );

    auto  final = dag_lu_lvl( & A,
                              // & L,
                              nodes, final_map, updates, apply_nodes );
    
    if ( CFG::Arith::use_accu )
    {
        // build_apply_dep( A, nullptr, node_store.final, node_store.updates, acc, start );
    }// if
    else
    {
        assign_dependencies( & A, final_map, updates );
    }// else
    
    dag::node_list_t  start, end{ final };

    for ( auto  node : nodes )
    {
        if ( node->dep_cnt() == 0 )
            start.push_back( node );
    }// for

    return  dag::graph( nodes, start, end );
}

//
// return graph representing compute DAG for solving L X = A
//
graph
gen_dag_solve_lower  ( const HLIB::TMatrix *                        L,
                       HLIB::TMatrix *                              A,
                       std::function< dag::graph ( dag::node * ) >  refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new solve_lower_node( L, A, apply_map ) );

    return dag;
}

//
// return graph representing compute DAG for solving X U = A
//
graph
gen_dag_solve_upper  ( const HLIB::TMatrix *                        U,
                       HLIB::TMatrix *                              A,
                       std::function< dag::graph ( dag::node * ) >  refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new solve_upper_node( U, A, apply_map ) );

    return dag;
}

//
// return graph representing compute DAG for C = A B + C
//
graph
gen_dag_update       ( const HLIB::TMatrix *                        A,
                       const HLIB::TMatrix *                        B,
                       HLIB::TMatrix *                              C,
                       std::function< dag::graph ( dag::node * ) >  refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new update_node( A, B, C, apply_map ) );

    return dag;
}

}// namespace dag

}// namespace hlr
