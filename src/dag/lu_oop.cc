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

// #include <tbb/parallel_for.h>

#include <matrix/structure.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

using namespace HLIB;

namespace
{

using HLIB::id_t;

// map for apply_node nodes
using  apply_map_t = std::unordered_map< id_t, node * >;

// identifiers for memory blocks
const id_t  ID_A  = 'A';
const id_t  ID_L  = 'L';
const id_t  ID_U  = 'U';
const id_t  ID_UP = 'X';

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
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
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
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
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
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
struct update_node : public node
{
    const TMatrix *  A;
    const id_t       id_A;
    const TMatrix *  B;
    const id_t       id_B;
    TMatrix *        C;
    const id_t       id_C;
    apply_map_t &    apply_nodes;

    update_node ( const TMatrix *  aA,
                  const id_t       aid_A,
                  const TMatrix *  aB,
                  const id_t       aid_B,
                  TMatrix *        aC,
                  const id_t       aid_C,
                  apply_map_t &    aapply_nodes )
            : A(    aA )
            , id_A( aid_A )
            , B(    aB )
            , id_B( aid_B )
            , C(    aC )
            , id_C( aid_C )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %c%d, %c%d, %c%d )",
                                                                      char(id_A), A->id(), char(id_B), B->id(), char(id_C), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_B, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { ID_UP, C->block_is() } };
        else                        return { { id_C, C->block_is() } };
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
    virtual local_graph         refine_      ( const size_t ) { return {}; } // not needed because of direct DAG generation
    virtual const block_list_t  in_blocks_   () const { return { { ID_UP, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_leaf( A ) ) return { { ID_A, A->block_is() } };
        else                return { };
    }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
lu_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! hlr::is_small( min_size, A ) )
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
                        hlr::dag::alloc_node< update_node >( g, B->block( j, i ), ID_L, B->block( i, l ), ID_U, B->block( j, l ), ID_A, apply_nodes );
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
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
solve_lower_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! hlr::is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    hlr::dag::alloc_node< solve_lower_node >( g, L_ii, BA->block( i, j ), apply_nodes );

            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        hlr::dag::alloc_node< update_node >( g, BL->block( k, i ), ID_L, BA->block( i, j ), ID_U, BA->block( k, j ), ID_A, apply_nodes );
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
solve_upper_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! hlr::is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                    hlr::dag::alloc_node< solve_upper_node >( g, U_jj, BA->block( i, j ), apply_nodes );

            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        hlr::dag::alloc_node< update_node >( g, BA->block( i, j ), ID_L, BU->block( j, k ), ID_U, BA->block( i, k ), ID_A, apply_nodes );
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
update_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, B, C ) && ! hlr::is_small_any( min_size, A, B, C ) )
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
                                                             BA->block( i, k ), id_A,
                                                             BB->block( k, j ), id_B,
                                                             BC->block( i, j ), id_C,
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
gen_dag_lu_oop ( TMatrix *      A,
                 refine_func_t  refine )
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
    
    auto  dag = refine( new lu_node( A, apply_map ), HLIB::CFG::Arith::max_seq_size );

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

}// namespace dag

}// namespace hlr
