//
// Project     : HLib
// File        : lu_coarse.cc
// Description : generate coarse DAG for LU factorization
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
#include <algebra/mat_fac.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/seq/dag.hh"
#include "hlr/dag/lu.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

// map for apply_node nodes
using  apply_map_t = std::unordered_map< HLIB::id_t, node * >;

using HLIB::id_t;

// identifiers for memory blocks
constexpr id_t  ID_A    = 'A';
constexpr id_t  ID_L    = 'L';
constexpr id_t  ID_U    = 'U';
constexpr id_t  ID_ACCU = 'X';

struct lu_node : public node
{
    TMatrix *      A;
    apply_map_t &  apply_nodes;
    exec_func_t    run_func;
    
    lu_node ( TMatrix *      aA,
              apply_map_t &  aapply_nodes,
              exec_func_t    arun_func )
            : A( aA )
            , apply_nodes( aapply_nodes )
            , run_func(    arun_func )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

struct trsmu_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    apply_map_t &    apply_nodes;
    exec_func_t      run_func;
    
    trsmu_node ( const TMatrix *  aU,
                       TMatrix *        aA,
                       apply_map_t &    aapply_nodes,
                       exec_func_t      arun_func )
            : U( aU )
            , A( aA )
            , apply_nodes( aapply_nodes )
            , run_func(    arun_func )
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

struct trsml_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;
    apply_map_t &    apply_nodes;
    exec_func_t      run_func;

    trsml_node ( const TMatrix *  aL,
                       TMatrix *        aA,
                       apply_map_t &    aapply_nodes,
                       exec_func_t      arun_func )
            : L( aL )
            , A( aA )
            , apply_nodes( aapply_nodes )
            , run_func(    arun_func )
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
    const TMatrix *  B;
    TMatrix *        C;
    apply_map_t &    apply_nodes;
    exec_func_t      run_func;

    update_node ( const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC,
                  apply_map_t &    aapply_nodes,
                  exec_func_t      arun_func )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_nodes( aapply_nodes )
            , run_func(    arun_func )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %d, %d, %d )",
                                                                      A->id(), B->id(), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { ID_ACCU, C->block_is() } };
        else                        return { { ID_A,    C->block_is() } };
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, A->block_is() } }; }
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

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        B   = ptrcast( A, TBlockMatrix );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();

        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            auto  A_ii  = B->block( i, i );

            assert( A_ii != nullptr );

            g.alloc_node< lu_node >( A_ii, apply_nodes, run_func );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                    g.alloc_node< trsmu_node >( A_ii, B->block( j, i ), apply_nodes, run_func );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                    g.alloc_node< trsml_node >( A_ii, B->block( i, j ), apply_nodes, run_func );

            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                        g.alloc_node< update_node >( B->block( j, i ), B->block( i, l ), B->block( j, l ),
                                                     apply_nodes, run_func );
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
    const auto  min_size = HLIB::CFG::Arith::max_seq_size;
        
    if ( is_small( min_size, A ) || is_leaf( A ) )
    {
        HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
    }// if
    else
    {
        apply_map_t  apply_map;
        auto         dag = seq::dag::refine( new lu_node( A, apply_map, run_func ), min_size );

        run_func( dag, acc );
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
trsml_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( min_size, A, L ) )
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
                    g.alloc_node< trsml_node >( L_ii, BA->block( i, j ), apply_nodes, run_func );

            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        g.alloc_node< update_node >( BL->block( k, i ), BA->block( i, j ), BA->block( k, j ),
                                                     apply_nodes, run_func );
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
trsml_node::run_ ( const TTruncAcc &  acc )
{
    const auto  min_size = HLIB::CFG::Arith::max_seq_size;
        
    if ( is_small_any( min_size, A, L ) || is_leaf_any( A, L ) )
    {
        solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    }// if
    else
    {
        apply_map_t  apply_map;
        auto         dag = seq::dag::refine( new trsml_node( L, A, apply_map, run_func ), min_size );

        run_func( dag, acc );
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
trsmu_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( min_size, A, U ) )
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
                    g.alloc_node< trsmu_node >( U_jj, BA->block( i, j ), apply_nodes, run_func );

            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        g.alloc_node< update_node >( BA->block( i, j ), BU->block( j, k ), BA->block( i, k ),
                                                     apply_nodes, run_func );
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
trsmu_node::run_ ( const TTruncAcc &  acc )
{
    const auto  min_size = HLIB::CFG::Arith::max_seq_size;

    if ( is_small_any( min_size, U, A ) || is_leaf_any( U, A ) )
    {
        solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    }// if
    else
    {
        apply_map_t  apply_map;
        auto         dag = seq::dag::refine( new trsmu_node( U, A, apply_map, run_func ), min_size );

        run_func( dag, acc );
    }// else
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

    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
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
                        g.alloc_node< update_node >( BA->block( i, k ), BB->block( k, j ), BC->block( i, j ),
                                                     apply_nodes, run_func );
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
    const auto  min_size = HLIB::CFG::Arith::max_seq_size;

    if ( is_small_any( min_size, A, B, C ) || is_leaf_any( A, B, C ) )
    {
        multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
    }// if
    else
    {
        apply_map_t  apply_map;
        auto         dag = seq::dag::refine( new update_node( A, B, C, apply_map, run_func ), min_size );

        run_func( dag, acc );
    }// else
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
    
    auto  apply = dag::alloc_node< apply_node >( apply_nodes, A );

    apply_map[ A->id() ] = apply;

    if ( parent != nullptr )
        apply->after( parent );
    
    if ( is_blocked( A ) && is_small( A ) )
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
gen_dag_lu_oop_coarse ( TMatrix &            A,
                        const refine_func_t  refine,
                        const exec_func_t    fine_run,
                        const size_t         ncoarse )
{
    // if coarse size if too small, generate standard LU DAG
    if ( ncoarse <= HLIB::CFG::Arith::max_seq_size )
        return gen_dag_lu_oop_auto( A, refine );
            
    apply_map_t  apply_map;
    auto         dag = refine( new lu_node( & A, apply_map, fine_run ), ncoarse, use_single_end_node );

    return std::move( dag );
}

}// namespace dag

}// namespace hlr
