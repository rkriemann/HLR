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
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

using namespace HLIB;

namespace
{

using HLIB::id_t;

// identifiers for memory blocks
const id_t  ID_A    = 'A';
const id_t  ID_L    = 'L';
const id_t  ID_U    = 'U';
const id_t  ID_ACCU = 'X';

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
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( A->parent() ) ) return { { ID_A, A->block_is() } };
        else                          return { { ID_A, A->block_is() }, { id_t(A->parent()), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

struct lu_leaf_node : public node
{
    TMatrix *  A;
    
    lu_leaf_node ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

struct trsmu_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    
    trsmu_node ( const TMatrix *  aU,
                 TMatrix *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = trsmu( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( A->parent() ) ) return { { ID_U, U->block_is() }, { ID_A, A->block_is() } };
        else                          return { { ID_U, U->block_is() }, { ID_A, A->block_is() }, { id_t(A->parent()), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

struct trsmu_leaf_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    
    trsmu_leaf_node ( const TMatrix *  aU,
                      TMatrix *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = trsmu( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

struct trsml_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;

    trsml_node ( const TMatrix *  aL,
                 TMatrix *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = trsml( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( A->parent() ) ) return { { ID_L, L->block_is() }, { ID_A, A->block_is() } };
        else                          return { { ID_L, L->block_is() }, { ID_A, A->block_is() }, { id_t(A->parent()), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
struct trsml_leaf_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;

    trsml_leaf_node ( const TMatrix *  aL,
                      TMatrix *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = trsml( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
struct add_prod_node : public node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;

    add_prod_node ( const TMatrix *  aA,
                    const TMatrix *  aB,
                    TMatrix *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = add_prod( %d, %d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, C->block_is() }, { id_t(C), C->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const
    {
        if ( A->parent() != nullptr ) return { { id_t(A->parent()), A->block_is() }, { id_t(A), A->block_is() } };
        else                          return { { id_t(A), A->block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

struct shift_node : public node
{
    TMatrix *  A;
    
    shift_node ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "shift( %d )", A->id() ); }
    virtual std::string  color     () const { return "c4a000"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
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

local_graph
lu_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! hlr::is_small( min_size, A ) )
    {
        auto        B   = ptrcast( A, TBlockMatrix );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();

        auto  shift_A = g.alloc_node< shift_node >( A );

        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            auto  A_ii = B->block( i, i );

            assert( A_ii != nullptr );

            finished(i,i) = g.alloc_node< lu_node >( A_ii );
            finished(i,i)->after( shift_A );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                {
                    finished(j,i) = g.alloc_node< trsmu_node >( A_ii, B->block( j, i ) );
                    finished(j,i)->after( finished(i,i) );
                    finished(j,i)->after( shift_A );
                }// if

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                {
                    finished(i,j) = g.alloc_node< trsml_node >( A_ii, B->block( i, j ) );
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
                        auto  update = g.alloc_node< add_prod_node >( B->block( j, i ),
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
        g.alloc_node< apply_node >( A )->before( g.alloc_node< lu_leaf_node >( A ) );
    }// else

    g.finalize();
    
    return g;
}

void
lu_node::run_ ( const TTruncAcc & )
{
    assert( false );
}

void
lu_leaf_node::run_ ( const TTruncAcc &  acc )
{
    HLIB::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
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

    if ( is_blocked_all( A, U ) && ! hlr::is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        auto  shift_A = g.alloc_node< shift_node >( A );
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                {
                    finished(i,j) = g.alloc_node< trsmu_node >( U_jj, BA->block( i, j ) );
                    finished(i,j)->after( shift_A );
                }// if
        }// for

        for ( uint j = 0; j < nbc; ++j )
        {
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                    {
                        auto  update = g.alloc_node< add_prod_node >( BA->block( i, j ),
                                                                      BU->block( j, k ),
                                                                      BA->block( i, k ) );

                        update->after( finished(i,j) );
                        finished(i,k)->after( update );
                    }// if
        }// for
    }// if
    else
    {
        g.alloc_node< apply_node >( A )->before( g.alloc_node< trsmu_leaf_node >( U, A ) );
    }// else

    g.finalize();

    return g;
}

void
trsmu_node::run_ ( const TTruncAcc & )
{
    assert( false );
}

void
trsmu_leaf_node::run_ ( const TTruncAcc &  acc )
{
    solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
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

    if ( is_blocked_all( A, L ) && ! hlr::is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        auto  shift_A = g.alloc_node< shift_node >( A );
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished(i,j) = g.alloc_node< trsml_node >( L_ii, BA->block( i, j ) );
                    finished(i,j)->after( shift_A );
                }// if
        }// for

        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                    {
                        auto  update = g.alloc_node< add_prod_node >( BL->block( k, i ),
                                                                      BA->block( i, j ),
                                                                      BA->block( k, j ) );

                        update->after( finished(i,j) );
                        finished(k,j)->after( update );
                    }// if
        }// for
    }// if
    else
    {
        g.alloc_node< apply_node >( A )->before( g.alloc_node< trsml_leaf_node >( L, A ) );
    }// else

    g.finalize();
    
    return g;
}

void
trsml_node::run_ ( const TTruncAcc & )
{
    assert( false );
}

void
trsml_leaf_node::run_ ( const TTruncAcc &  acc )
{
    solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// add_prod_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
add_prod_node::refine_ ( const size_t  min_size )
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
                        g.alloc_node< add_prod_node >( BA->block( i, k ),
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

void
add_prod_node::run_ ( const TTruncAcc &  acc )
{
    add_product( real(-1),
                 apply_normal, A,
                 apply_normal, B,
                 C, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// apply_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
apply_node::run_ ( const TTruncAcc &  acc )
{
    A->apply_updates( acc, recursive );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// shift_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
shift_node::run_ ( const TTruncAcc &  acc )
{
    A->apply_updates( acc, nonrecursive );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_lu_accu ( TMatrix *      A,
                  refine_func_t  refine )
{
    //
    // construct DAG for LU
    //
    
    auto  dag = refine( new lu_node( A ), HLIB::CFG::Arith::max_seq_size );

    return std::move( dag );
    
    //
    // loop over accumulator nodes from top to bottom and remove nodes without updates
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;
    auto              is_apply_node = [] ( node * node )
                                      {
                                          return ( ! is_null_all( dynamic_cast< apply_node * >( node ),
                                                                  dynamic_cast< shift_node * >( node ) ) );
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
    
    return  std::move( dag::graph( std::move( nodes ), std::move( start ), std::move( end ) ) );
}

}// namespace dag

}// namespace hlr
