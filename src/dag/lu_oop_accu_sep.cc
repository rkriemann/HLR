//
// Project     : HLR
// Module      : lu.cc
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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
using  apply_map_t = std::unordered_map< Hpro::id_t, node * >;

using Hpro::id_t;

// identifiers for memory blocks
constexpr id_t  ID_A    = 'A';
constexpr id_t  ID_L    = 'L';
constexpr id_t  ID_U    = 'U';
constexpr id_t  ID_ACCU = 'X';

template < typename value_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *      A;
    apply_map_t &  apply_map;
    
    lu_node ( Hpro::TMatrix< value_t > *      aA,
              apply_map_t &  aapply_map )
            : A( aA )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

template < typename value_t >
struct trsmu_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    apply_map_t &    apply_map;
    
    trsmu_node ( const Hpro::TMatrix< value_t > *  aU,
                 Hpro::TMatrix< value_t > *        aA,
                 apply_map_t &    aapply_map )
            : U( aU )
            , A( aA )
            , apply_map( aapply_map )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "L%d = trsmu( U%d, A%d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

template < typename value_t >
struct trsml_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;
    apply_map_t &    apply_map;

    trsml_node ( const Hpro::TMatrix< value_t > *  aL,
                 Hpro::TMatrix< value_t > *        aA,
                 apply_map_t &    aapply_map )
            : L( aL )
            , A( aA )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "U%d = trsml( L%d, A%d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
template < typename value_t >
struct update_node : public node
{
    const Hpro::TMatrix< value_t > *  A;
    const Hpro::TMatrix< value_t > *  B;
    Hpro::TMatrix< value_t > *        C;
    apply_map_t &    apply_map;

    update_node ( const Hpro::TMatrix< value_t > *  aA,
                  const Hpro::TMatrix< value_t > *  aB,
                  Hpro::TMatrix< value_t > *        aC,
                  apply_map_t &    aapply_map )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { ID_ACCU, C->block_is() } };
        else                        return { { ID_A,    C->block_is() } };
    }
};

template < typename value_t >
struct apply_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    apply_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "apply( %d )", A->id() ); }
    virtual std::string  color     () const { return "edd400"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
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

template < typename value_t >
local_graph
lu_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
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

            g.alloc_node< lu_node< value_t > >( A_ii, apply_map );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( BA->block( j, i ) ) )
                    g.alloc_node< trsmu_node< value_t > >( U_ii, BA->block( j, i ), apply_map );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                    g.alloc_node< trsml_node< value_t > >( L_ii, BA->block( i, j ), apply_map );

            for ( uint j = i+1; j < nbr; j++ )
            {
                for ( uint l = i+1; l < nbc; l++ )
                {
                    if ( ! is_null_any( BL->block( j, i ), BU->block( i, l ), BA->block( j, l ) ) )
                        g.alloc_node< update_node< value_t > >( BL->block( j, i ),
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

template < typename value_t >
void
lu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );

    HLR_ERROR( "todo" );
    // Hpro::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
trsmu_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                    g.alloc_node< trsmu_node< value_t > >(  U_jj, BA->block( i, j ), apply_map );

            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        g.alloc_node< update_node< value_t > >( BX->block( i, j ),
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

template < typename value_t >
void
trsmu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );
    
    HLR_ERROR( "todo" );
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
trsml_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    g.alloc_node< trsml_node< value_t > >(  L_ii, BA->block( i, j ), apply_map );
        
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        g.alloc_node< update_node< value_t > >( BL->block( k, i ),
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

template < typename value_t >
void
trsml_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );
    
    HLR_ERROR( "todo" );
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
update_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  C, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node< value_t > >( BA->block( i, k ),
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

template < typename value_t >
void
update_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
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

template < typename value_t >
void
apply_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( is_blocked( A ) && ! hpro::is_small( A ) )
        A->apply_updates( acc, Hpro::nonrecursive );
    else
        A->apply_updates( acc, Hpro::recursive );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// apply DAG
//
///////////////////////////////////////////////////////////////////////////////////////

//
// construct DAG for applying updates
//
template < typename value_t >
void
build_apply_dag ( Hpro::TMatrix< value_t > * A,
                  node *                     parent,
                  apply_map_t &              apply_map,
                  dag::node_list_t &         apply_nodes,
                  const size_t               min_size )
{
    if ( is_null( A ) )
        return;

    auto  apply = dag::alloc_node< apply_node< value_t > >( apply_nodes, A );

    apply_map[ A->id() ] = apply;

    if ( parent != nullptr )
        apply->after( parent );
    
    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA = ptrcast( A, Hpro::TBlockMatrix< value_t > );

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

template < typename value_t >
graph
gen_dag_lu_oop_accu_sep ( Hpro::TMatrix< value_t > & A,
                          const size_t               min_size,
                          refine_func_t              refine )
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

#define INST_ALL( type )                    \
    template graph gen_dag_lu_oop_accu_sep< type > ( Hpro::TMatrix< type > &, \
                                                     const size_t           , \
                                                     refine_func_t          );

INST_ALL( float )
INST_ALL( double )
INST_ALL( std::complex< float > )
INST_ALL( std::complex< double > )

}}// namespace hlr::dag
