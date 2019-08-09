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
using  apply_map_t = std::unordered_map< HLIB::id_t, node * >;

// identifiers for memory blocks
const id_t  ID_A    = 'A';
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, U->block_is() }, { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, U->block_is() }, { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, L->block_is() }, { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, L->block_is() }, { ID_A, A->block_is() }, { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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

    virtual std::string  to_string () const { return HLIB::to_string( "%d = mul( %d, %d )",
                                                                      C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { ID_A, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(C), C->block_is() } }; }
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
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_ACCU, A->block_is() }, { id_t(A), A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A,    A->block_is() } }; }
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
    virtual const block_list_t  in_blocks_   () const         { return { { ID_ACCU, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_blocked( A ) )
        {
            auto          B = ptrcast( A, TBlockMatrix );
            block_list_t  mblocks;

            for ( uint  i = 0; i < B->nblock_rows(); ++i )
                for ( uint  j = 0; j < B->nblock_rows(); ++j )
                    mblocks.push_back( { ID_ACCU, B->block(i,j)->block_is() } );

            return mblocks;
        }// if
        else
            return {};
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

        hlr::dag::alloc_node< shift_node >( g, A );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            auto  A_ii = B->block( i, i );

            assert( A_ii != nullptr );

            hlr::dag::alloc_node< lu_node >( g, A_ii );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( B->block( j, i ) ) )
                    hlr::dag::alloc_node< trsmu_node >( g, A_ii, B->block( j, i ) );

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( B->block( i, j ) ) )
                    hlr::dag::alloc_node< trsml_node >( g, A_ii, B->block( i, j ) );

            for ( uint j = i+1; j < nbr; j++ )
                for ( uint l = i+1; l < nbc; l++ )
                    if ( ! is_null_any( B->block( j, i ), B->block( i, l ), B->block( j, l ) ) )
                        hlr::dag::alloc_node< update_node >( g,
                                                             B->block( j, i ),
                                                             B->block( i, l ),
                                                             B->block( j, l ) );
        }// for
    }// if
    else
    {
        hlr::dag::alloc_node< apply_node >( g, A );
        hlr::dag::alloc_node< lu_leaf_node >( g, A );
    }// else

    return g;
}

void
lu_node::run_ ( const TTruncAcc & )
{
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

        hlr::dag::alloc_node< shift_node >( g, A );
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                    hlr::dag::alloc_node< trsmu_node >( g, U_jj, BA->block( i, j ) );

            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        hlr::dag::alloc_node< update_node >( g,
                                                             BA->block( i, j ),
                                                             BU->block( j, k ),
                                                             BA->block( i, k ) );
        }// for
    }// if
    else
    {
        hlr::dag::alloc_node< apply_node >( g, A );
        hlr::dag::alloc_node< trsmu_leaf_node >( g, U, A );
    }// else

    return g;
}

void
trsmu_node::run_ ( const TTruncAcc & )
{
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

        hlr::dag::alloc_node< shift_node >( g, A );
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    hlr::dag::alloc_node< trsml_node >( g, L_ii, BA->block( i, j ) );

            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        hlr::dag::alloc_node< update_node >( g,
                                                             BL->block( k, i ),
                                                             BA->block( i, j ),
                                                             BA->block( k, j ) );
        }// for
    }// if
    else
    {
        hlr::dag::alloc_node< apply_node >( g, A );
        hlr::dag::alloc_node< trsml_leaf_node >( g, L, A );
    }// else

    return g;
}

void
trsml_node::run_ ( const TTruncAcc & )
{
}

void
trsml_leaf_node::run_ ( const TTruncAcc &  acc )
{
    solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
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
                                                             BA->block( i, k ),
                                                             BB->block( k, j ),
                                                             BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
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

local_graph
apply_node::refine_ ( const size_t )
{
    local_graph  g;

    // if ( is_blocked( A ) && ! hlr::is_small( min_size, A ) )
    // {
    //     auto  B = ptrcast( A, TBlockMatrix );
        
    //     hlr::dag::alloc_node< shift_node >( g, A );

    //     for ( uint  i = 0; i < B->nblock_rows(); ++i )
    //         for ( uint  j = 0; j < B->nblock_cols(); ++j )
    //             hlr::dag::alloc_node< apply_node >( g, B->block( i, j ) );
    // }// if

    return g;
}

void
apply_node::run_ ( const TTruncAcc &  acc )
{
    // if ( is_blocked( A ) && ! is_small( A ) )
    //     A->apply_updates( acc, nonrecursive );
    // else
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
    
    return  refine( new lu_node( A ), HLIB::CFG::Arith::max_seq_size );
}

}// namespace dag

}// namespace hlr
