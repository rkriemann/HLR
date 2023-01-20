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

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

namespace
{

using Hpro::id_t;

// identifiers for memory blocks
const id_t  ID_A    = 'A';
const id_t  ID_L    = 'L';
const id_t  ID_U    = 'U';
const id_t  ID_ACCU = 'X';

template < typename value_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    lu_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

template < typename value_t >
struct lu_leaf_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    lu_leaf_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

template < typename value_t >
struct trsmu_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    
    trsmu_node ( const Hpro::TMatrix< value_t > *  aU,
                 Hpro::TMatrix< value_t > *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsmu( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

template < typename value_t >
struct trsmu_leaf_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    
    trsmu_leaf_node ( const Hpro::TMatrix< value_t > *  aU,
                      Hpro::TMatrix< value_t > *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsmu( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

template < typename value_t >
struct trsml_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;

    trsml_node ( const Hpro::TMatrix< value_t > *  aL,
                 Hpro::TMatrix< value_t > *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsml( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
template < typename value_t >
struct trsml_leaf_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;

    trsml_leaf_node ( const Hpro::TMatrix< value_t > *  aL,
                      Hpro::TMatrix< value_t > *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%d = trsml( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
template < typename value_t >
struct add_prod_node : public node
{
    const Hpro::TMatrix< value_t > *  A;
    const Hpro::TMatrix< value_t > *  B;
    Hpro::TMatrix< value_t > *        C;

    add_prod_node ( const Hpro::TMatrix< value_t > *  aA,
                    const Hpro::TMatrix< value_t > *  aB,
                    Hpro::TMatrix< value_t > *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%d = add_prod( %d, %d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, C->block_is() }, { id_t(C), C->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const
    {
        if ( A->parent() != nullptr ) return { { id_t(A->parent()), A->block_is() }, { id_t(A), A->block_is() } };
        else                          return { { id_t(A), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct shift_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    shift_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "shift( %d )", A->id() ); }
    virtual std::string  color     () const { return "c4a000"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const
    {
        if ( A->parent() != nullptr ) return { { id_t(A->parent()), A->block_is() }, { id_t(A), A->block_is() } };
        else                          return { { id_t(A), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(A), A->block_is() } }; }
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

        auto  shift_A = g.alloc_node< shift_node< value_t > >( A );

        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            auto  A_ii = B->block( i, i );

            assert( A_ii != nullptr );

            finished(i,i) = g.alloc_node< lu_node< value_t > >( A_ii );
            finished(i,i)->after( shift_A );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                {
                    finished(j,i) = g.alloc_node< trsmu_node< value_t > >( A_ii, B->block( j, i ) );
                    finished(j,i)->after( finished(i,i) );
                    finished(j,i)->after( shift_A );
                }// if

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                {
                    finished(i,j) = g.alloc_node< trsml_node< value_t > >( A_ii, B->block( i, j ) );
                    finished(i,j)->after( finished(i,i) );
                    finished(i,j)->after( shift_A );
                }// if
        }// for

        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                    {
                        auto  update = g.alloc_node< add_prod_node< value_t > >( B->block( j, i ),
                                                                      B->block( i, l ),
                                                                      B->block( j, l ) );

                        update->after( finished(j,i) );
                        update->after( finished(i,l) );
                        finished(j,l)->after( update );
                    }// if
        }// for
    }// if
    else
    {
        g.alloc_node< apply_node< value_t > >( A )->before( g.alloc_node< lu_leaf_node< value_t > >( A ) );
    }// else

    g.finalize();
    
    return g;
}

template < typename value_t >
void
lu_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    assert( false );
}

template < typename value_t >
void
lu_leaf_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
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
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        auto  shift_A = g.alloc_node< shift_node< value_t > >( A );
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                {
                    finished(i,j) = g.alloc_node< trsmu_node< value_t > >( U_jj, BA->block( i, j ) );
                    finished(i,j)->after( shift_A );
                }// if
        }// for

        for ( uint j = 0; j < nbc; ++j )
        {
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                    {
                        auto  update = g.alloc_node< add_prod_node< value_t > >( BA->block( i, j ),
                                                                      BU->block( j, k ),
                                                                      BA->block( i, k ) );

                        update->after( finished(i,j) );
                        finished(i,k)->after( update );
                    }// if
        }// for
    }// if
    else
    {
        g.alloc_node< apply_node< value_t > >( A )->before( g.alloc_node< trsmu_leaf_node< value_t > >( U, A ) );
    }// else

    g.finalize();

    return g;
}

template < typename value_t >
void
trsmu_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    assert( false );
}

template < typename value_t >
void
trsmu_leaf_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
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
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        auto  shift_A = g.alloc_node< shift_node< value_t > >( A );
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished(i,j) = g.alloc_node< trsml_node< value_t > >( L_ii, BA->block( i, j ) );
                    finished(i,j)->after( shift_A );
                }// if
        }// for

        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                    {
                        auto  update = g.alloc_node< add_prod_node< value_t > >( BL->block( k, i ),
                                                                      BA->block( i, j ),
                                                                      BA->block( k, j ) );

                        update->after( finished(i,j) );
                        finished(k,j)->after( update );
                    }// if
        }// for
    }// if
    else
    {
        g.alloc_node< apply_node< value_t > >( A )->before( g.alloc_node< trsml_leaf_node< value_t > >( L, A ) );
    }// else

    g.finalize();
    
    return g;
}

template < typename value_t >
void
trsml_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    assert( false );
}

template < typename value_t >
void
trsml_leaf_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// add_prod_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
add_prod_node< value_t >::refine_ ( const size_t  min_size )
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
                        g.alloc_node< add_prod_node< value_t > >( BA->block( i, k ),
                                                                  BB->block( k, j ),
                                                                  BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    // no dependendies here
    g.finalize();
    
    return g;
}

template < typename value_t >
void
add_prod_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // add_product( real(-1),
    //              apply_normal, A,
    //              apply_normal, B,
    //              C, acc );
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
    A->apply_updates( acc, Hpro::recursive );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// shift_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
shift_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    A->apply_updates( acc, Hpro::nonrecursive );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
graph
gen_dag_lu_oop_accu ( Hpro::TMatrix< value_t > &  A,
                      const size_t                min_size,
                      refine_func_t               refine )
{
    if ( hlr::dag::sparsify_mode != hlr::dag::sparsify_none )
        hlr::log( 0, term::red( term::bold( "SPARSIFICATION NOT WORKING WITH ACCUMULATOR ARITHMETIC" ) ) );
    
    //
    // construct DAG for LU
    //
    
    auto  dag = refine( new lu_node( & A ), min_size, use_single_end_node );

    return dag;
    
    //
    // loop over accumulator nodes from top to bottom and remove nodes without updates
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;
    auto              is_apply_node = [] ( node * node )
                                      {
                                          return ( ! is_null_all( dynamic_cast< apply_node< value_t > * >( node ),
                                                                  dynamic_cast< shift_node< value_t > * >( node ) ) );
                                      };

    for ( auto  node : dag.start() )
        work.push_back( node );

    while ( ! work.empty() )
    {
        dag::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( is_apply_node( node ) )
            {
                if (( node->dep_cnt() == 0 ) && ( deleted.find( node ) == deleted.end() ))
                {
                    HLR_LOG( 6, "delete " + node->to_string() );

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

#define INST_ALL( type )                    \
    template graph gen_dag_lu_oop_accu< type > ( Hpro::TMatrix< type > &, \
                                                 const size_t           , \
                                                 refine_func_t          );

INST_ALL( float )
INST_ALL( double )
INST_ALL( std::complex< float > )
INST_ALL( std::complex< double > )

}}// namespace hlr::dag
