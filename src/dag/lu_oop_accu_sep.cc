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

#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/matrix.hh"

namespace hlr { namespace dag {

using namespace HLIB;

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
    apply_map_t &  apply_map;
    
    lu_node ( TMatrix *      aA,
              apply_map_t &  aapply_map )
            : A( aA )
            , apply_map( aapply_map )
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
    apply_map_t &    apply_map;
    
    trsmu_node ( const TMatrix *  aU,
                 TMatrix *        aA,
                 apply_map_t &    aapply_map )
            : U( aU )
            , A( aA )
            , apply_map( aapply_map )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "L%d = trsmu( U%d, A%d )", A->id(), U->id(), A->id() ); }
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
    apply_map_t &    apply_map;

    trsml_node ( const TMatrix *  aL,
                 TMatrix *        aA,
                 apply_map_t &    aapply_map )
            : L( aL )
            , A( aA )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "U%d = trsml( L%d, A%d )", A->id(), L->id(), A->id() ); }
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
    apply_map_t &    apply_map;

    update_node ( const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC,
                  apply_map_t &    aapply_map )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
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
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_ACCU, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_leaf( A ) ) return { { ID_A,    A->block_is() } };
        else                return { { ID_ACCU, A->block_is() } };
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
        auto        BA  = ptrcast( A, TBlockMatrix );
        auto        BL  = BA;
        auto        BU  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = BA->block( i, i );
            auto  L_ii  = A_ii;
            auto  U_ii  = A_ii;

            assert( ! is_null_any( A_ii, L_ii, U_ii ) );

            g.alloc_node< lu_node >( A_ii, apply_map );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( BA->block( j, i ) ) )
                    g.alloc_node< trsmu_node >( U_ii, BA->block( j, i ), apply_map );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                    g.alloc_node< trsml_node >( L_ii, BA->block( i, j ), apply_map );

            for ( uint j = i+1; j < nbr; j++ )
            {
                for ( uint l = i+1; l < nbc; l++ )
                {
                    if ( ! is_null_any( BL->block( j, i ), BU->block( i, l ), BA->block( j, l ) ) )
                        g.alloc_node< update_node >( BL->block( j, i ),
                                                     BU->block( i, l ),
                                                     BA->block( j, l ),
                                                     apply_map );
                }// for
            }// for
        }// for
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_map[ A->id() ];
        
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

    HLR_ERROR( "todo" );
    // HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
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
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                    g.alloc_node< trsmu_node >(  U_jj, BA->block( i, j ), apply_map );

            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        g.alloc_node< update_node >( BX->block( i, j ),
                                                     BU->block( j, k ),
                                                     BA->block( i, k ),
                                                     apply_map );
        }// for
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_map[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if
    
    return g;
}

void
trsmu_node::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    HLR_ERROR( "todo" );
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
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
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    g.alloc_node< trsml_node >(  L_ii, BA->block( i, j ), apply_map );
        
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        g.alloc_node< update_node >( BL->block( k, i ),
                                                     BX->block( i, j ),
                                                     BA->block( k, j ),
                                                     apply_map );
        }// for
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_map[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if
    
    return g;
}

void
trsml_node::run_ ( const TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, recursive );
    
    HLR_ERROR( "todo" );
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
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

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node >( BA->block( i, k ),
                                                     BB->block( k, j ),
                                                     BC->block( i, j ),
                                                     apply_map );
                }// for
            }// for
        }// for
    }// if
    else if ( CFG::Arith::use_accu )
    {
        auto  apply = apply_map[ C->id() ];
        
        assert( apply != nullptr );

        apply->after( this );
    }// if

    g.finalize();
    
    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    
    // if ( CFG::Arith::use_accu )
    //     add_product( real(-1),
    //                  apply_normal, A,
    //                  apply_normal, B,
    //                  C, acc );
    // else
    //     multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// apply_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
apply_node::run_ ( const TTruncAcc &  acc )
{
    if ( is_blocked( A ) && ! hpro::is_small( A ) )
        A->apply_updates( acc, nonrecursive );
    else
        A->apply_updates( acc, recursive );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// apply DAG
//
///////////////////////////////////////////////////////////////////////////////////////

//
// construct DAG for applying updates
//
void
build_apply_dag ( TMatrix *           A,
                  node *              parent,
                  apply_map_t &       apply_map,
                  dag::node_list_t &  apply_nodes,
                  const size_t        min_size )
{
    if ( is_null( A ) )
        return;

    auto  apply = dag::alloc_node< apply_node >( apply_nodes, A );

    apply_map[ A->id() ] = apply;

    if ( parent != nullptr )
        apply->after( parent );
    
    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );

        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                    build_apply_dag( BA->block( i, j ), apply, apply_map, apply_nodes, min_size );
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
gen_dag_lu_oop_accu_sep ( TMatrix &      A,
                          const size_t   min_size,
                          refine_func_t  refine )
{
    //
    // generate DAG for shifting and applying updates
    //
    
    apply_map_t       apply_map;
    dag::node_list_t  apply_nodes;

    build_apply_dag( & A, nullptr, apply_map, apply_nodes, min_size );
    
    auto  dag = refine( new lu_node( & A, apply_map ), min_size, use_single_end_node );

    //
    // add apply/shift nodes with shift(A) as new start
    //
        
    for ( auto  node : apply_nodes )
    {
        dag.nodes().push_back( node );

        // adjust dependency counters
        for ( auto  succ : node->successors() )
            succ->inc_dep_cnt();
    }// for

    dag.start().clear();
    dag.start().push_back( apply_map[ A.id() ] );
        
    return dag;
}

}// namespace dag

}// namespace hlr
