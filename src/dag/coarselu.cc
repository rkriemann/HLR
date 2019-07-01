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
#include "hlr/seq/dag.hh"
#include "hlr/tbb/dag.hh"
#include "hlr/dag/lu.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

//
// return true if matrix is still large enough
//
template < typename T >
bool
is_large ( const T *  A )
{
    assert( ! is_null( A ) );
    
    return ( std::min( A->nrows(), A->ncols() ) >= 5000 );
}

template < typename T >
bool is_large_all  ( T *  A )               noexcept { return is_large( A ); }

template < typename T1, typename... T2 >
bool is_large_all  ( T1 *  A, T2...  mtrs ) noexcept { return is_large( A ) && is_large_all( mtrs... ); }



// map for apply_node nodes
using  apply_map_t = std::unordered_map< HLIB::id_t, node * >;

// refine and rnu functions for fine graphs
using  refine_func_t = std::function< dag::graph ( dag::node * ) >;
using  run_func_t    = std::function< void ( hlr::dag::graph &, const HLIB::TTruncAcc & ) >;

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_U = 'U';

struct lu_node : public node
{
    TMatrix *      A;
    apply_map_t &  apply_nodes;
    refine_func_t  refine_func;
    run_func_t     run_func;
    
    lu_node ( TMatrix *      aA,
              apply_map_t &  aapply_nodes,
              refine_func_t  arefine_func,
              run_func_t     arun_func )
            : A( aA )
            , apply_nodes( aapply_nodes )
            , refine_func( arefine_func )
            , run_func(    arun_func )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
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
    refine_func_t    refine_func;
    run_func_t       run_func;
    
    solve_upper_node ( const TMatrix *  aU,
                       TMatrix *        aA,
                       apply_map_t &    aapply_nodes,
                       refine_func_t    arefine_func,
                       run_func_t       arun_func )
            : U( aU )
            , A( aA )
            , apply_nodes( aapply_nodes )
            , refine_func( arefine_func )
            , run_func(    arun_func )
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
    refine_func_t    refine_func;
    run_func_t       run_func;

    solve_lower_node ( const TMatrix *  aL,
                       TMatrix *        aA,
                       apply_map_t &    aapply_nodes,
                       refine_func_t    arefine_func,
                       run_func_t       arun_func )
            : L( aL )
            , A( aA )
            , apply_nodes( aapply_nodes )
            , refine_func( arefine_func )
            , run_func(    arun_func )
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
    refine_func_t    refine_func;
    run_func_t       run_func;

    update_node ( const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC,
                  apply_map_t &    aapply_nodes,
                  refine_func_t    arefine_func,
                  run_func_t       arun_func )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_nodes( aapply_nodes )
            , refine_func( arefine_func )
            , run_func(    arun_func )
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

    if ( is_blocked( A ) && is_large( A ) )
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

            hlr::dag::alloc_node< lu_node >( g, A_ii,
                                             apply_nodes, refine_func, run_func );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                    hlr::dag::alloc_node< solve_upper_node >( g, A_ii, B->block( j, i ),
                                                              apply_nodes, refine_func, run_func );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                    hlr::dag::alloc_node< solve_lower_node >( g, A_ii, B->block( i, j ),
                                                              apply_nodes, refine_func, run_func );

            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                        hlr::dag::alloc_node< update_node >( g, B->block( j, i ), B->block( i, l ), B->block( j, l ),
                                                             apply_nodes, refine_func, run_func );
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
    auto  dag = gen_dag_lu_rec( A, refine_func );

    run_func( dag, acc );
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

    if ( is_blocked_all( A, L ) && is_large_all( A, L ) )
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
                        hlr::dag::alloc_node< solve_lower_node >( g, L_ii, BA->block( i, j ),
                                                                  apply_nodes, refine_func, run_func );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint i = 0; i < nbr; ++i )
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        hlr::dag::alloc_node< update_node >( g, BL->block( k, i ), BA->block( i, j ), BA->block( k, j ),
                                                             apply_nodes, refine_func, run_func );
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
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    auto  dag = gen_dag_solve_lower( L, A, refine_func );

    run_func( dag, acc );
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

    if ( is_blocked_all( A, U ) && is_large_all( A, U ) )
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
                        hlr::dag::alloc_node< solve_upper_node >( g, U_jj, BA->block( i, j ),
                                                                  apply_nodes, refine_func, run_func );
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint j = 0; j < nbc; ++j )
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        hlr::dag::alloc_node< update_node >( g, BA->block( i, j ), BU->block( j, k ), BA->block( i, k ),
                                                             apply_nodes, refine_func, run_func );
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
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    auto  dag = gen_dag_solve_upper( U, A, refine_func );

    run_func( dag, acc );
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

    if ( is_blocked_all( A, B, C ) && is_large_all( A, B, C ) )
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
                                                             apply_nodes,
                                                             refine_func,
                                                             run_func );
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
    // multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
    auto  dag = gen_dag_update( A, B, C, refine_func );

    run_func( dag, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// apply_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
apply_node::run_ ( const TTruncAcc &  acc )
{
    if ( is_blocked( A ) && is_large( A ) )
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
    
    if ( is_blocked( A ) && is_large( A ) )
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
gen_dag_coarselu ( TMatrix *                                                  A,
                   const std::function< graph ( node * ) > &                  coarse_refine,
                   const std::function< dag::graph ( dag::node * ) > &        fine_refine,
                   const std::function< void ( hlr::dag::graph &,
                                               const HLIB::TTruncAcc & ) > &  fine_run )
{
    apply_map_t  apply_map;
    auto         dag = coarse_refine( new lu_node( A, apply_map, fine_refine, fine_run ) );

    return dag;
}

}// namespace dag

}// namespace hlr
