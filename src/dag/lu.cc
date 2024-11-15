//
// Project     : HLR
// Module      : lu.cc
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <hpro/matrix/structure.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"
#include "hlr/approx/svd.hh"

namespace hlr { namespace dag {

namespace
{

using Hpro::id_t;

// map for apply_node nodes
using  apply_map_t = std::unordered_map< Hpro::id_t, node * >;

// identifiers for memory blocks
const id_t  ID_A    = 'A';
const id_t  ID_ACCU = 'X';

template < typename value_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    apply_map_t &               apply_nodes;
    
    lu_node ( Hpro::TMatrix< value_t > *  aA,
              apply_map_t &               aapply_nodes )
            : A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct trsmu_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    apply_map_t &                     apply_nodes;
    
    trsmu_node ( const Hpro::TMatrix< value_t > *  aU,
                 Hpro::TMatrix< value_t > *        aA,
                 apply_map_t &                     aapply_nodes )
            : U( aU )
            , A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsmu( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct trsml_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;
    apply_map_t &                     apply_nodes;

    trsml_node ( const Hpro::TMatrix< value_t > *  aL,
                 Hpro::TMatrix< value_t > *        aA,
                 apply_map_t &                     aapply_nodes )
            : L( aL )
            , A( aA )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsml( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};
    
template < typename value_t >
struct update_node : public node
{
    const Hpro::TMatrix< value_t > *  A;
    const Hpro::TMatrix< value_t > *  B;
    Hpro::TMatrix< value_t > *        C;
    apply_map_t &                     apply_nodes;

    update_node ( const Hpro::TMatrix< value_t > *  aA,
                  const Hpro::TMatrix< value_t > *  aB,
                  Hpro::TMatrix< value_t > *        aC,
                  apply_map_t &    aapply_nodes )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_nodes( aapply_nodes )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%d = mul( %d, %d )",
                                                                      C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { ID_A, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( Hpro::CFG::Arith::use_accu ) return { { ID_ACCU, C->block_is() } };
        else                        return { { ID_A, C->block_is() } };
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
        auto        B   = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();
        
        if ( is_nd( A ) )
        {
            for ( uint i = 0; i < std::min( nbr, nbc )-1; ++i )
            {
                //
                // factorise diagonal block
                //
            
                auto  A_ii  = B->block( i, i );

                assert( A_ii != nullptr );

                g.alloc_node< lu_node< value_t > >( A_ii, apply_nodes );

                if ( ! is_null( B->block( nbr-1, i ) ) )
                    g.alloc_node< trsmu_node< value_t > >( A_ii, B->block( nbr-1, i ), apply_nodes );

                if ( ! is_null( B->block( i, nbc-1 ) ) )
                    g.alloc_node< trsml_node< value_t > >( A_ii, B->block( i, nbc-1 ), apply_nodes );

                if ( ! is_null_any( B->block( nbr-1, i ), B->block( i, nbc-1 ), B->block( nbr-1, nbc-1 ) ) )
                    g.alloc_node< update_node< value_t > >( B->block( nbr-1, i ),
                                                 B->block( i, nbc-1 ),
                                                 B->block( nbr-1, nbc-1 ),
                                                 apply_nodes );
            }// for

            g.alloc_node< lu_node< value_t > >( B->block( nbr-1, nbc-1 ), apply_nodes );
        }// if
        else
        {
            for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
            {
                //
                // factorise diagonal block
                //
            
                auto  A_ii  = B->block( i, i );

                assert( A_ii != nullptr );

                g.alloc_node< lu_node< value_t > >( A_ii, apply_nodes );

                for ( uint j = i+1; j < nbr; j++ )
                    if ( ! is_null( B->block( j, i ) ) )
                        g.alloc_node< trsmu_node< value_t > >( A_ii, B->block( j, i ), apply_nodes );

                for ( uint j = i+1; j < nbc; j++ )
                    if ( ! is_null( B->block( i, j ) ) )
                        g.alloc_node< trsml_node< value_t > >( A_ii, B->block( i, j ), apply_nodes );

                for ( uint j = i+1; j < nbr; j++ )
                    for ( uint l = i+1; l < nbc; l++ )
                        if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                            g.alloc_node< update_node< value_t > >( B->block( j, i ),
                                                         B->block( i, l ),
                                                         B->block( j, l ),
                                                         apply_nodes );
            }// for
        }// else
    }// if
    else if ( Hpro::CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

template < typename value_t >
void
lu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( Hpro::CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );

    // Hpro::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );

    hlr::approx::SVD< value_t >  apx;
    
    hlr::lu< value_t >( *A, acc, apx );
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
        auto        BA  =  ptrcast( A, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        // if ( is_nd( U ) )
        // {
        //     for ( uint j = 0; j < nbc-1; ++j )
        //     {
        //         const auto  U_jj = BU->block( j, j );
                
        //         HLR_ASSERT( ! is_null( U_jj ) );
                
        //         for ( uint i = 0; i < nbr; ++i )
        //         {
        //             auto  A_ij = BA->block( i, j );
                    
        //             if ( ! is_null( A_ij ) )
        //             {
        //                 g.alloc_node< trsmu_node< value_t > >( U_jj, A_ij, apply_nodes );

        //                 if ( ! is_null_any( BU->block( j, nbc-1 ), BA->block( i, nbc-1 ) ) )
        //                     g.alloc_node< update_node< value_t > >( A_ij,
        //                                                  BU->block( j, nbc-1 ),
        //                                                  BA->block( i, nbc-1 ),
        //                                                  apply_nodes );
        //             }// if
        //         }// for
        //     }// for
                
        //     HLR_ASSERT( ! is_null( BU->block( nbc-1, nbc-1 ) ) );
            
        //     for ( uint  i = 0; i < nbr; ++i )
        //     {
        //         auto  A_ij = BA->block( i, nbc-1 );
                
        //         if ( ! is_null( A_ij ) )
        //             g.alloc_node< trsmu_node< value_t > >( BU->block( nbc-1, nbc-1 ), A_ij, apply_nodes );
        //     }// for
        // }// if
        // else
        {
            for ( uint j = 0; j < nbc; ++j )
            {
                const auto  U_jj = BU->block( j, j );
                
                if ( ! is_null( U_jj ) )
                {
                    for ( uint i = 0; i < nbr; ++i )
                        if ( ! is_null( BA->block(i,j) ) )
                            g.alloc_node< trsmu_node< value_t > >( U_jj, BA->block( i, j ), apply_nodes );
                }// if
                
                for ( uint  k = j+1; k < nbc; ++k )
                    for ( uint  i = 0; i < nbr; ++i )
                        if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                            g.alloc_node< update_node< value_t > >( BA->block( i, j ),
                                                         BU->block( j, k ),
                                                         BA->block( i, k ),
                                                         apply_nodes );
            }// for
        }// else
    }// if
    else if ( Hpro::CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];

        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

template < typename value_t >
void
trsmu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( Hpro::CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );
    
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    
    hlr::approx::SVD< value_t >  apx;
    
    hlr::solve_upper_tri< value_t >( from_right, general_diag, *U, *A, acc, apx );
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
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        if ( is_nd( L ) )
        {
        }// if
        else
        {
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
                            g.alloc_node< trsml_node< value_t > >( L_ii, BA->block( i, j ), apply_nodes );
                }// if

                for ( uint  k = i+1; k < nbr; ++k )
                    for ( uint  j = 0; j < nbc; ++j )
                        if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                            g.alloc_node< update_node< value_t > >( BL->block( k, i ),
                                                         BA->block( i, j ),
                                                         BA->block( k, j ),
                                                         apply_nodes );
            }// for
        }// else
    }// if
    else if ( Hpro::CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ A->id() ];
        
        assert( apply != nullptr );

        apply->before( this );
    }// if

    return g;
}

template < typename value_t >
void
trsml_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( Hpro::CFG::Arith::use_accu )
        A->apply_updates( acc, Hpro::recursive );
    
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );

    hlr::approx::SVD< value_t >  apx;
    
    hlr::solve_lower_tri< value_t >( from_left, unit_diag, *L, *A, acc, apx );
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

        for ( uint  i = 0; i < BC->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->block_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->block_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node< value_t > >( BA->block( i, k ),
                                                     BB->block( k, j ),
                                                     BC->block( i, j ),
                                                     apply_nodes );
                }// for
            }// for
        }// for
    }// if
    else if ( Hpro::CFG::Arith::use_accu )
    {
        auto  apply = apply_nodes[ C->id() ];
        
        assert( apply != nullptr );

        apply->after( this );
    }// if

    // no dependencies between updates
    g.finalize();
    
    return g;
}

template < typename value_t >
void
update_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    if ( Hpro::CFG::Arith::use_accu )
    {
        HLR_ERROR( "todo" );
    
        // add_product( real(-1),
        //              apply_normal, A,
        //              apply_normal, B,
        //              C, acc );
    }// if
    else
    {
        // multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );

        hlr::approx::SVD< value_t >  apx;
    
        hlr::multiply( value_t(-1), apply_normal, *A, apply_normal, *B, *C, acc, apx );
    }// else
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
    if ( is_blocked( A ) && ! Hpro::is_small( A ) )
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
build_apply_dag ( Hpro::TMatrix< value_t > *  A,
                  node *                      parent,
                  apply_map_t &               apply_map,
                  dag::node_list_t &          apply_nodes,
                  const size_t                min_size )
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
gen_dag_lu_ip ( Hpro::TMatrix< value_t > &  A,
                const size_t                min_size,
                refine_func_t               refine )
{
    //
    // generate DAG for shifting and applying updates
    //
    
    apply_map_t       apply_map;
    dag::node_list_t  apply_nodes;

    if ( Hpro::CFG::Arith::use_accu )
        build_apply_dag( & A, nullptr, apply_map, apply_nodes, min_size );
    
    //
    // construct DAG for LU
    //

    auto  dag = refine( new lu_node< value_t >( & A, apply_map ), min_size, use_single_end_node );

    if ( ! Hpro::CFG::Arith::use_accu )
        return dag;
    else
    {
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

        // update old nodes as well
        for ( auto  node : dag.nodes() )
            node->finalize();
        
        dag.start().clear();
        dag.start().push_back( apply_map[ A.id() ] );
        
        return dag;
    }// else

    //
    // loop over apply nodes from top to bottom and remove nodes without updates
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;
    auto              is_apply_node = [] ( node * node ) { return ( dynamic_cast< apply_node< value_t > * >( node ) != nullptr ); };
    
    work.push_back( apply_map[ A.id() ] );

    while ( ! work.empty() )
    {
        dag::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( is_apply_node( node ) )
            {
                if ( node->dep_cnt() == 0 )
                {
                    for ( auto  out : node->successors() )
                    {
                        out->dec_dep_cnt();
                        
                        if ( is_apply_node( out ) )
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
    
    return  dag::graph( std::move( nodes ), std::move( start ), std::move( end ) );
}

//
// return graph representing compute DAG for solving L X = A
//
template < typename value_t >
graph
gen_dag_solve_lower  ( const Hpro::TMatrix< value_t > &  L,
                       Hpro::TMatrix< value_t > &        A,
                       const size_t                      min_size,
                       refine_func_t                     refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new trsml_node< value_t >( &L, &A, apply_map ), min_size, use_single_end_node );

    return dag;
}

//
// return graph representing compute DAG for solving X U = A
//
template < typename value_t >
graph
gen_dag_solve_upper  ( const Hpro::TMatrix< value_t > &  U,
                       Hpro::TMatrix< value_t > &        A,
                       const size_t                      min_size,
                       refine_func_t                     refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new trsmu_node< value_t >( &U, &A, apply_map ), min_size, use_single_end_node );

    return dag;
}

//
// return graph representing compute DAG for C = A B + C
//
template < typename value_t >
graph
gen_dag_update       ( const Hpro::TMatrix< value_t > &  A,
                       const Hpro::TMatrix< value_t > &  B,
                       Hpro::TMatrix< value_t > &        C,
                       const size_t                      min_size,
                       refine_func_t                     refine )
{
    apply_map_t  apply_map;
    auto         dag = refine( new update_node< value_t >( &A, &B, &C, apply_map ), min_size, use_single_end_node );

    return dag;
}

#define INST_ALL( type )                                                \
    template graph gen_dag_lu_ip< type >        ( Hpro::TMatrix< type > &      , \
                                                  const size_t                 , \
                                                  refine_func_t                ); \
    template graph gen_dag_solve_lower< type >  ( const Hpro::TMatrix< type > &, \
                                                  Hpro::TMatrix< type > &      , \
                                                  const size_t                 , \
                                                  refine_func_t                ); \
    template graph gen_dag_solve_upper< type >  ( const Hpro::TMatrix< type > &, \
                                                  Hpro::TMatrix< type > &      , \
                                                  const size_t                 , \
                                                  refine_func_t                ); \
    template graph gen_dag_update< type >       ( const Hpro::TMatrix< type > &, \
                                                  const Hpro::TMatrix< type > &, \
                                                  Hpro::TMatrix< type > &      , \
                                                  const size_t                 , \
                                                  refine_func_t                );

INST_ALL( float )
INST_ALL( double )
INST_ALL( std::complex< float > )
INST_ALL( std::complex< double > )
    
}}// namespace hlr::dag
