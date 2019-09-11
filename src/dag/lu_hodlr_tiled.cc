//
// Project     : HLib
// File        : lu_hodlr_tiled.cc
// Description : generate DAG for tiled LU factorization of HODLR matrices
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
#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"

namespace hlr { namespace dag {

using namespace HLIB;

namespace
{

////////////////////////////////////////////////////////////////////////////////
//
// auxiliary functions
//
////////////////////////////////////////////////////////////////////////////////

//
// access to matrices U/V of lowrank matrices
//
template < typename value_t >
BLAS::Matrix< value_t > &
mat_U ( TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return blas_mat_A< value_t >( A );
}

template < typename value_t >
BLAS::Matrix< value_t > &
mat_V ( TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return blas_mat_B< value_t >( A );
}

template < typename value_t >
const BLAS::Matrix< value_t > &
mat_U ( const TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return blas_mat_A< value_t >( A );
}

template < typename value_t >
const BLAS::Matrix< value_t > &
mat_V ( const TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return blas_mat_B< value_t >( A );
}

template < typename value_t >
BLAS::Matrix< value_t > &
mat_U ( TRkMatrix &  A )
{
    return blas_mat_A< value_t >( & A );
}

template < typename value_t >
BLAS::Matrix< value_t > &
mat_V ( TRkMatrix &  A )
{
    return blas_mat_B< value_t >( & A );
}

template < typename value_t >
const BLAS::Matrix< value_t > &
mat_U ( const TRkMatrix &  A )
{
    return blas_mat_A< value_t >( & A );
}

template < typename value_t >
const BLAS::Matrix< value_t > &
mat_V ( const TRkMatrix &  A )
{
    return blas_mat_B< value_t >( & A );
}

//
// split given range into <n> subsets
//
inline
std::vector< BLAS::Range >
split ( const BLAS::Range &  r,
        const size_t         n )
{
    if ( n == 2 )
    {
        const BLAS::Range  r0( r.first(), (r.first() + r.last()) / 2 - 1 );
        const BLAS::Range  r1( r0.last() + 1, r.last() );

        return { std::move(r0), std::move(r1) };
    }// if
    else
        assert( false );

    return {};
}

////////////////////////////////////////////////////////////////////////////////
//
// tasks
//
////////////////////////////////////////////////////////////////////////////////

using HLIB::id_t;

// identifiers for memory blocks
constexpr id_t  ID_A = 'A';
constexpr id_t  ID_L = 'L';
constexpr id_t  ID_U = 'U';

struct lu_node : public node
{
    TMatrix *     A;
    const size_t  ntile;
    
    lu_node ( TMatrix *     aA,
              const size_t  antile )
            : A( aA )
            , ntile( antile )
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
    const TMatrix *         U;
    const TBlockIndexSet    is_A;
    BLAS::Matrix< real > &  X;
    const size_t            ntile;
    
    trsmu_node ( const TMatrix *         aU,
                 const TBlockIndexSet    ais_X,
                 BLAS::Matrix< real > &  aX,
                 const size_t            antile )
            : U( aU )
            , is_X( ais_X )
            , X( aX )
            , ntile( antile )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "L = trsmu( U%d, A )", U->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, is_X } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, is_X } }; }
};

struct trsml_node : public node
{
    const TMatrix *         L;
    const TBlockIndexSet    is_X;
    BLAS::Matrix< real > &  X;
    const size_t            ntile;

    trsml_node ( const TMatrix *         aL,
                 const TBlockIndexSet    ais_X,
                 BLAS::Matrix< real > &  aX,
                 const size_t            antile )
            : L( aL )
            , is_X( ais_X )
            , X( aX )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "U = trsml( L%d, A )", L->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, is_X } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, is_X } }; }
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
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  BL  = BA;
        auto  BU  = BA;
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );

        assert(( BA->nblock_rows() == 2 ) && ( BA->nblock_cols() == 2 ));
        assert( ! is_null( A10 ) && is_lowrank( A10 ));
        assert( ! is_null( A01 ) && is_lowrank( A01 ));
            
        auto  lu_00    = g.alloc_node< lu_node >( BA->block( 0, 0 ), ntile );
        auto  solve_10 = g.alloc_node< trsmu_node >( BU->block( 0, 0 ), bis( A10->col_is(), is( A10->rank() ) ), mat_V< real >( A10 ), ntile );
        auto  solve_01 = g.alloc_node< trsml_node >( BL->block( 0, 0 ), bis( A01->row_is(), is( A01->rank() ) ), mat_U< real >( A01 ), ntile );
        auto  T        = make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< tsmul_node >( bis( A10->col_is(), is( A10->rank() ) ), mat_V< value_t >( A10 ),
                                                     bis( A01->row_is(), is( A01->rank() ) ), mat_U< value_t >( A01 ),
                                                     T,
                                                     ntile );
        auto  addlr    = g.alloc_node< addlr_node >( bis( A10->row_is(), is( A10->rank() ) ), mat_U< value_t >( A10 ),
                                                     T,
                                                     bis( A01->col_is(), is( A01->rank() ) ), mat_V< value_t >( A01 ),
                                                     BA->block( 1, 1 ),
                                                     ntile );
        auto  lu_11    = g.alloc_node< lu_node >( BA->block( 1, 1 ), ntile );

        solve_10->after( lu_00 );
        solve_01->after( lu_00 );
        tsmul->after( solve_10 );
        tsmul->after( solve_01 );
        addlr->after( tsmul );
        lu_11->after( addlr );
    }// if

    g.finalize();
    
    return g;
}

void
lu_node::run_ ( const TTruncAcc &  acc )
{
    hlr::seq::tile::hodlr::lu( A, ntile );
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

    if ( is_blocked( U ) && ! hlr::is_small( min_size, U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        const auto               is0 = U00->col_is();
        const auto               is1 = U11->col_is();
        BLAS::Matrix< value_t >  X0( X, is0 - U->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, is1 - U->col_ofs(), BLAS::Range::all );

        auto  solve_00 = g.alloc_node< trsmu_node >( U00, is0, X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< tsmul_node >(           mat_U< value_t >( U01 ), T, is0, X0, ntile );
        auto  tsadd    = g.alloc_node< tsadd_node >( real(-1), mat_V< value_t >( U01 ), T, is1, X1, ntile );
        auto  solve_11 = g.alloc_node< trsmu_node >( U00, is0, X0, ntile );

        tsmul->after( solve_00 );
        tsadd->after( tsmul );
        solve_11->after( tsadd );
    }// if

    g.finalize();
    
    return g;
}

void
trsmu_node::run_ ( const TTruncAcc &  acc )
{
    hlr::seq::tile::hodlr::trsmuh( U, X, ntile );
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

    if ( is_blocked( L ) && ! hlr::is_small( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        const auto               is0 = L00->row_is();
        const auto               is1 = L11->row_is();
        BLAS::Matrix< value_t >  X0( X, is0 - L->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, is0 - L->row_ofs(), BLAS::Range::all );
            
        auto  solve_00 = g.alloc_node< trsml_node >( L00, is0, X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< tsmul_node >(           mat_V< real >( L10 ), T, is0, X0, ntile );
        auto  tsadd    = g.alloc_node< tsadd_node >( real(-1), mat_U< real >( L10 ), T, is1, X1, ntile );
        auto  solve_11 = g.alloc_node< trsml_node >( L11, is1, X1, ntile );

        tsmul->after( solve_00 );
        tsadd->after( tsmul );
        solve_11->after( tsadd );
    }// if

    g.finalize();
    
    return g;
}

void
trsml_node::run_ ( const TTruncAcc &  acc )
{
    hlr::seq::tile::hodlr::trsmuh( L, X, ntile );
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
                                                     BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
{
    multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_lu_oop ( TMatrix &      A,
                 refine_func_t  refine )
{
    return std::move( refine( new lu_node( & A ), HLIB::CFG::Arith::max_seq_size ) );
}

}// namespace dag

}// namespace hlr
