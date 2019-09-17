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
// split given range/is into <n> subsets
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

inline
std::vector< TIndexSet >
split ( const TIndexSet &  is,
        const size_t       n )
{
    if ( n == 2 )
    {
        const TIndexSet  is0( is.first(), (is.first() + is.last()) / 2 - 1 );
        const TIndexSet  is1( is0.last() + 1, is.last() );

        return { std::move(is0), std::move(is1) };
    }// if
    else
        assert( false );

    return {};
}

//
// return a block indexset for a BLAS::Matrix
//
template < typename value_t >
TBlockIndexSet
bis ( const BLAS::Matrix< value_t > &  M )
{
    return TBlockIndexSet( is( 0, M.nrows()-1 ), is( 0, M.ncols()-1 ) );
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

//
// compute A = LU
//
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

//
// solve X U = M with upper triangular U and M given
//
struct trsmu_node : public node
{
    const TMatrix *         U;
    const TBlockIndexSet    is_X;
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

//
// solve L X = M with lower triangular L and M given
//
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
    
//
// compute T := A^H · B
//
struct tsmul_node : public node
{
    const id_t                               id_A;
    const TBlockIndexSet                     is_A;
    const BLAS::Matrix< real > &             A;
    const id_t                               id_B;
    const TBlockIndexSet                     is_B;
    const BLAS::Matrix< real > &             B;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const size_t                             ntile;

    tsmul_node ( const id_t              aid_A,
                 const TBlockIndexSet    ais_A,
                 BLAS::Matrix< real > &  aA,
                 const id_t              aid_B,
                 const TBlockIndexSet    ais_B,
                 BLAS::Matrix< real > &  aB,
                 std::shared_ptr< BLAS::Matrix< real > >  aT,
                 const size_t            antile )
            : id_A( aid_A ), is_A( ais_A ), A( aA )
            , id_B( aid_B ), is_B( ais_B ), B( aB )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "tsmul( %c-[%d,%d], %c-[%d,%d] )",
                                                                      char(id_A), is_A.row_is().first(), is_A.row_is().last(),
                                                                      char(id_B), is_B.row_is().first(), is_B.row_is().last() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, is_A }, { id_B, is_B } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(T.get()), bis( *T ) } }; } // empty indexset???
};

//
// compute B := B + α·A·T
//
struct tsadd_node : public node
{
    const real                    alpha;
    const id_t                    id_A;
    const TBlockIndexSet          is_A;
    const BLAS::Matrix< real > &  A;
    const id_t                    id_T;
    const TBlockIndexSet          is_T;
    const BLAS::Matrix< real > &  T;
    const id_t                    id_B;
    const TBlockIndexSet          is_B;
    BLAS::Matrix< real > &        B;
    const size_t                  ntile;

    tsadd_node ( const real              aalpha,
                 const id_t              aid_A,
                 const TBlockIndexSet    ais_A,
                 BLAS::Matrix< real > &  aA,
                 const id_t              aid_T,
                 const TBlockIndexSet    ais_T,
                 BLAS::Matrix< real > &  aT,
                 const id_t              aid_B,
                 const TBlockIndexSet    ais_B,
                 BLAS::Matrix< real > &  aB,
                 const size_t            antile )
            : alpha( aalpha )
            , id_A( aid_A ), is_A( ais_A ), A( aA )
            , id_T( aid_T ), is_T( ais_T ), T( aT )
            , id_B( aid_B ), is_B( ais_B ), B( aB )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "tsadd( %c-[%d,%d], %c-[%d,%d] )",
                                                                      char(id_A), is_A.row_is().first(), is_A.row_is().last(),
                                                                      char(id_B), is_B.row_is().first(), is_B.row_is().last() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, is_A }, { id_B, is_B }, { id_T, is_T } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_B, is_B } }; }
};

//
// compute A := A - U·T·V^H
//
struct addlr_node : public node
{
    const id_t                    id_U;
    const TBlockIndexSet          is_U;
    const BLAS::Matrix< real > &  U;
    const id_t                    id_T;
    const TBlockIndexSet          is_T;
    const BLAS::Matrix< real > &  T;
    const id_t                    id_V;
    const TBlockIndexSet          is_V;
    BLAS::Matrix< real > &        V;
    TMatrix *                     A;
    const size_t                  ntile;

    addlr_node ( const id_t              aid_U,
                 const TBlockIndexSet    ais_U,
                 BLAS::Matrix< real > &  aU,
                 const id_t              aid_T,
                 const TBlockIndexSet    ais_T,
                 BLAS::Matrix< real > &  aT,
                 const id_t              aid_V,
                 const TBlockIndexSet    ais_V,
                 BLAS::Matrix< real > &  aV,
                 TMatrix *               aA,
                 const size_t            antile )
            : id_U( aid_U ), is_U( ais_U ), U( aU )
            , id_T( aid_T ), is_T( ais_T ), T( aT )
            , id_V( aid_V ), is_V( ais_V ), V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "addlr( %c-[%d,%d], %c-[%d,%d], %d )",
                                                                      char(id_U), is_U.row_is().first(), is_U.row_is().last(),
                                                                      char(id_V), is_V.row_is().first(), is_V.row_is().last(),
                                                                      A->id() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_U, is_U }, { id_V, is_V }, { id_T, is_T } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

//
// compute T := T_0 + T_1
//
struct tadd_node : public node
{
    std::shared_ptr< BLAS::Matrix< real > >  T0;
    std::shared_ptr< BLAS::Matrix< real > >  T1;
    std::shared_ptr< BLAS::Matrix< real > >  T;

    tadd_node ( std::shared_ptr< BLAS::Matrix< real > >  aT0,
                std::shared_ptr< BLAS::Matrix< real > >  aT1,
                std::shared_ptr< BLAS::Matrix< real > >  aT )
            : T0( aT0 )
            , T1( aT1 )
            , T( aT )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = tadd( %d, %d )",
                                                                      id_t(T.get()), id_t(T0.get()), id_t(T1.get()) ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_t(T0.get()), bis( *T0 ) }, { id_t(T1.get()), bis( *T1 ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(T.get()), bis( *T ) } }; }
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
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tsmul_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( A.nrows() == B.nrows() );

    if ( A.nrows() > ntile )
    {
        const auto                     R     = split( BLAS::Range( 0, A.nrows()-1 ), 2 );
        const auto                     sis_A = split( is_A, 2 );
        const auto                     sis_B = split( is_B, 2 );
        const BLAS::Matrix< value_t >  A0( A, R[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  A1( A, R[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  B0( B, R[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  B1( B, R[1], BLAS::Range::all );

        auto  T0     = std::make_shared< BLAS::Matrix< real > >();
        auto  T1     = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul0 = g.alloc_node< tsmul_node >( id_A, sis_A[0], A0, id_B, sis_B[0], B0, T0, ntile );
        auto  tsmul1 = g.alloc_node< tsmul_node >( id_A, sis_A[1], A1, id_B, sis_B[1], B1, T1, ntile );
        auto  add    = g.alloc_node< tadd_node >( T0, T1, T );

        add->after( tsmul0 );
        add->after( tsmul1 );
    }// if

    g.finalize();

    return g;
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tsadd_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( A.nrows() == B.nrows() );

    if ( A.nrows() > ntile )
    {
        const auto                  R     = split( BLAS::Range( 0, A.nrows()-1 ), 2 );
        const auto                  sis_A = split( is_A, 2 );
        const auto                  sis_B = split( is_B, 2 );
        const BLAS::Matrix< real >  A0( A, R[0], BLAS::Range::all );
        const BLAS::Matrix< real >  A1( A, R[1], BLAS::Range::all );
        BLAS::Matrix< real >        B0( B, R[0], BLAS::Range::all );
        BLAS::Matrix< real >        B1( B, R[1], BLAS::Range::all );

        auto  tsadd0 = g.alloc_node< tsadd_node >( id_A, sis_A[0], A0, id_T, is_T, T, id_B, sis_B[0], B0, ntile );
        auto  tsadd1 = g.alloc_node< tsadd_node >( id_A, sis_A[1], A1, id_T, is_T, T, id_B, sis_B[1], B1, ntile );
    }// if

    g.finalize();

    return g;
}

///////////////////////////////////////////////////////////////////////////////////////
//
// addlr_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
addlr_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );
        
        const BLAS::Matrix< real >  U0( U, A00->row_is() - A->row_ofs(), BLAS::Range::all );
        const BLAS::Matrix< real >  U1( U, A11->row_is() - A->row_ofs(), BLAS::Range::all );
        const BLAS::Matrix< real >  V0( V, A00->col_is() - A->col_ofs(), BLAS::Range::all );
        const BLAS::Matrix< real >  V1( V, A11->col_is() - A->col_ofs(), BLAS::Range::all );

        auto  task00 = g.alloc_node< addlr_node >( id_U, A00->row_is(), U0,
                                                   id_T, is_T, T,
                                                   id_V, A00->col_is(), V0,
                                                   A00,
                                                   ntile );

        // {
        //     auto  [ U01, V01 ] = truncate( value_t(-1), U0, T, V1, mat_U< value_t >( A01 ), mat_V< value_t >( A01 ), acc, ntile );

        //     A01->set_lrmat( U01, V01 );
        // }

        // {
        //     auto  [ U10, V10 ] = truncate( value_t(-1), U1, T, V0, mat_U< value_t >( A10 ), mat_V< value_t >( A10 ), acc, ntile );
            
        //     A10->set_lrmat( U10, V10 );
        // }

        auto  task11 = g.alloc_node< addlr_node >( id_U, A11->row_is(), U1,
                                                   id_T, is_T, T,
                                                   id_V, A11->col_is(), V1,
                                                   A11,
                                                   ntile );
    }// if

    g.finalize();

    return g;
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tadd_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
tadd_node::run_ ( const TTruncAcc &  acc )
{
    assert(( T0->nrows() == T1->nrows() ) && ( T0->ncols() == T1->ncols() ));
    
    if (( T->nrows() != T0->nrows() ) || ( T->ncols() != T0->ncols() ))
        *T = BLAS::Matrix< real >( T0->nrows(), T0->ncols() );
    
    BLAS::add( value_t(1), *T0, *T );
    BLAS::add( value_t(1), *T1, *T );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_lu_hodlr_tiled ( TMatrix &      A,
                         refine_func_t  refine )
{
    BLAS::Matrix< real >  A( 512, 16 );
    BLAS::Matrix< real >  B( 512, 16 );
    auto                  T = std::make_shared< BLAS::Matrix< real > >( 16, 16 );
    
    return refine( new tsmul_node( id_t('A'), bis( is( 0, 511 ), is( 0, 15 ) ), A,
                                   id_t('B'), bis( is( 0, 511 ), is( 0, 15 ) ), B,
                                   T, 128 ) );
    
    // return std::move( refine( new lu_node( & A ), HLIB::CFG::Arith::max_seq_size ) );
}

}// namespace dag

}// namespace hlr
