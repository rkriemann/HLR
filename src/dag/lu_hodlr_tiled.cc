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
        const BLAS::Range  r0( r.first(), r.first() + r.size() / 2 - 1 );
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
        const TIndexSet  is0( is.first(), is.first() + is.size() / 2 - 1 );
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

// dummy indexset for T operations (rank/size unknown during DAG and only object is of interest)
const auto  BIS_ONE = TBlockIndexSet( TIndexSet( 0, 0 ), TIndexSet( 0, 0 ) );

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
    BLAS::Matrix< real >    X;
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
    BLAS::Matrix< real >    X;
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
// compute A := A - U·T·V^H
//
struct addlr_node : public node
{
    const id_t                    id_U;
    const TBlockIndexSet          is_U;
    const BLAS::Matrix< real >    U;
    const id_t                    id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const id_t                    id_V;
    const TBlockIndexSet          is_V;
    const BLAS::Matrix< real >    V;
    TMatrix *                     A;
    const size_t                  ntile;

    addlr_node ( const id_t              aid_U,
                 const TBlockIndexSet    ais_U,
                 const BLAS::Matrix< real > &  aU,
                 const id_t              aid_T,
                 std::shared_ptr< BLAS::Matrix< real > >  aT,
                 const id_t              aid_V,
                 const TBlockIndexSet    ais_V,
                 const BLAS::Matrix< real > &  aV,
                 TMatrix *               aA,
                 const size_t            antile )
            : id_U( aid_U ), is_U( ais_U ), U( aU )
            , id_T( aid_T ), T( aT )
            , id_V( aid_V ), is_V( ais_V ), V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "addlr( %c[%d,%d], %c[%d,%d], %d )",
                                                                      char(id_U), is_U.row_is().first(), is_U.row_is().last(),
                                                                      char(id_V), is_V.row_is().first(), is_V.row_is().last(),
                                                                      A->id() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_U, is_U }, { id_V, is_V }, { id_T, BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

//
// compute T := A^H · B
//
struct dot_node : public node
{
    const id_t                               id_A;
    const TBlockIndexSet                     is_A;
    const BLAS::Matrix< real >               A;
    const id_t                               id_B;
    const TBlockIndexSet                     is_B;
    const BLAS::Matrix< real >               B;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const size_t                             ntile;

    dot_node ( const id_t                   aid_A,
                 const TBlockIndexSet         ais_A,
                 const BLAS::Matrix< real > & aA,
                 const id_t                   aid_B,
                 const TBlockIndexSet         ais_B,
                 const BLAS::Matrix< real > & aB,
                 std::shared_ptr< BLAS::Matrix< real > >  aT,
                 const size_t                 antile )
            : id_A( aid_A ), is_A( ais_A ), A( aA )
            , id_B( aid_B ), is_B( ais_B ), B( aB )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "dot( %c[%d,%d], %c[%d,%d] )",
                                                                      char(id_A), is_A.row_is().first(), is_A.row_is().last(),
                                                                      char(id_B), is_B.row_is().first(), is_B.row_is().last() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, is_A }, { id_B, is_B } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(T.get()), BIS_ONE } }; }
};

//
// compute Y := β·Y + α·X·T
//
struct tprod_node : public node
{
    const real                    alpha;
    const id_t                    id_X;
    const TBlockIndexSet          is_X;
    const BLAS::Matrix< real >    X;
    const id_t                    id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const real                    beta;
    const id_t                    id_Y;
    const TBlockIndexSet          is_Y;
    BLAS::Matrix< real >          Y;
    const size_t                  ntile;

    tprod_node ( const real                   aalpha,
                 const id_t                   aid_X,
                 const TBlockIndexSet         ais_X,
                 const BLAS::Matrix< real > & aX,
                 const id_t                   aid_T,
                 std::shared_ptr< BLAS::Matrix< real > >  aT,
                 const real                   abeta,
                 const id_t                   aid_Y,
                 const TBlockIndexSet         ais_Y,
                 BLAS::Matrix< real > &       aY,
                 const size_t                 antile )
            : alpha( aalpha )
            , id_X( aid_X ), is_X( ais_X ), X( aX )
            , id_T( aid_T ), T( aT )
            , beta( abeta )
            , id_Y( aid_Y ), is_Y( ais_Y ), Y( aY )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "tprod( %c[%d,%d], %c[%d,%d] )",
                                                                      char(id_X), is_X.row_is().first(), is_X.row_is().last(),
                                                                      char(id_Y), is_Y.row_is().first(), is_Y.row_is().last() ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( beta == real(0) ) return { { id_X, is_X }, { id_T, BIS_ONE } };
        else                   return { { id_X, is_X }, { id_T, BIS_ONE }, { id_Y, is_Y } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { id_Y, is_Y } }; }
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
    virtual const block_list_t  in_blocks_   () const { return { { id_t(T0.get()), BIS_ONE }, { id_t(T1.get()), BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_t(T.get()), BIS_ONE } }; }
};

//
// truncate α X T Y^H + U(A) V(A)^H
//
// struct truncate_node : public node
// {
//     const id_t                               id_X;
//     const TBlockIndexSet                     is_X;
//     const BLAS::Matrix< real >               X;
//     const id_t                               id_T;
//     std::shared_ptr< BLAS::Matrix< real > >  T;
//     const id_t                               id_Y;
//     const TBlockIndexSet                     is_Y;
//     const BLAS::Matrix< real >               Y;
//     TRkMatrix *                              A;
//     const size_t                             ntile;

//     truncate_node ( const id_t                               aid_X,
//                     const TBlockIndexSet                     ais_X,
//                     const BLAS::Matrix< real > &             aX,
//                     const id_t                               aid_T,
//                     std::shared_ptr< BLAS::Matrix< real > >  aT,
//                     const id_t                               aid_Y,
//                     const TBlockIndexSet                     ais_Y,
//                     const BLAS::Matrix< real > &             aY,
//                     TRkMatrix *                              aA,
//                     const size_t                             antile )
//             : T0( aT0 )
//             , T1( aT1 )
//             , T( aT )
//     { init(); }

//     virtual std::string  to_string () const { return HLIB::to_string( "truncate( %d )", A->id() ); }
//     virtual std::string  color     () const { return "e9b96e"; }

// private:
//     virtual void                run_         ( const TTruncAcc &  acc );
//     virtual local_graph         refine_      ( const size_t ) { return {}; }
//     virtual const block_list_t  in_blocks_   () const { return { { id_X, is_X }, { id_T, BIS_ONE }, { id_Y, is_Y } }; }
//     virtual const block_list_t  out_blocks_  () const { return { { A->block_is(), ID_A } }; }
// };

//
// QR factorization of [αX·T,U]
//
struct tsqr_node : public node
{
    const real                               alpha;
    const id_t                               id_X;
    const TBlockIndexSet                     is_X;
    const BLAS::Matrix< real >               X;
    const id_t                               id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const id_t                               id_U;
    const TBlockIndexSet                     is_U;
    const BLAS::Matrix< real >               U;
    const id_t                               id_Q;
    const TBlockIndexSet                     is_Q;
    BLAS::Matrix< real >                     Q;
    const id_t                               id_R;
    std::shared_ptr< BLAS::Matrix< real > >  R;
    const size_t                             ntile;

    tsqr_node ( const real                               aalpha,
                const id_t                               aid_X,
                const TBlockIndexSet                     ais_X,
                const BLAS::Matrix< real > &             aX,
                const id_t                               aid_T,
                std::shared_ptr< BLAS::Matrix< real > >  aT,
                const id_t                               aid_U,
                const TBlockIndexSet                     ais_U,
                const BLAS::Matrix< real > &             aU,
                const id_t                               aid_Q,
                const TBlockIndexSet                     ais_Q,
                BLAS::Matrix< real >                     aQ,
                const id_t                               aid_R,
                std::shared_ptr< BLAS::Matrix< real > >  aR,
                const size_t                             antile )
            : alpha( aalpha )
            , id_X( aid_X ), is_X( ais_X ), X( aX )
            , id_T( aid_T ), T( aT )
            , id_U( aid_U ), is_U( ais_U ), U( aU )
            , id_Q( aid_Q ), is_Q( ais_Q ), Q( aQ )
            , id_R( aid_R ), R( aR )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "tsqr( %d )", X.nrows() ); }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_X, is_X }, { id_T, BIS_ONE }, { id_U, is_U } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_Q, is_Q }, { id_R, BIS_ONE } }; }
};

//
// QR factorization of [R0;R1] with Q written to [R0;R1] 
//
struct qr_node : public node
{
    const id_t                               id_R0;
    std::shared_ptr< BLAS::Matrix< real > >  R0;
    const id_t                               id_R1;
    std::shared_ptr< BLAS::Matrix< real > >  R1;
    const id_t                               id_R;
    std::shared_ptr< BLAS::Matrix< real > >  R;

    qr_node ( const id_t                               aid_R0,
              std::shared_ptr< BLAS::Matrix< real > >  aR0,
              const id_t                               aid_R1,
              std::shared_ptr< BLAS::Matrix< real > >  aR1,
              const id_t                               aid_R,
              std::shared_ptr< BLAS::Matrix< real > >  aR )
            : id_R0( aid_R0 ), R0( aR0 )
            , id_R1( aid_R1 ), R1( aR1 )
            , id_R(  aid_R  ), R(  aR  )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = qr( %d, %d )", id_t(R.get()), id_t(R0.get()), id_t(R1.get()) ); }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_R0, BIS_ONE }, { id_R1, BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_R0, BIS_ONE }, { id_R1, BIS_ONE }, { id_R, BIS_ONE } }; }
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
        auto  solve_10 = g.alloc_node< trsmu_node >( BU->block( 0, 0 ), bis( A10->col_is(), is( 0, A10->rank()-1 ) ), mat_V< real >( A10 ), ntile );
        auto  solve_01 = g.alloc_node< trsml_node >( BL->block( 0, 0 ), bis( A01->row_is(), is( 0, A01->rank()-1 ) ), mat_U< real >( A01 ), ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( ID_L, bis( A10->col_is(), is( 0, A10->rank()-1 ) ), mat_V< real >( A10 ),
                                                     ID_U, bis( A01->row_is(), is( 0, A01->rank()-1 ) ), mat_U< real >( A01 ),
                                                     T,
                                                     ntile );
        auto  addlr    = g.alloc_node< addlr_node >( ID_L, bis( A10->row_is(), is( 0, A10->rank()-1 ) ), mat_U< real >( A10 ),
                                                     id_t( T.get() ), T,
                                                     ID_U, bis( A01->col_is(), is( 0, A01->rank()-1 ) ), mat_V< real >( A01 ),
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
    hlr::seq::tile::hodlr::lu< real >( A, acc, ntile );
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
        //
        //  ⎡ R_00^T │        ⎤ ⎡X_0⎤   ⎡ R_00^T            │        ⎤ ⎡X_0⎤   ⎡M_0⎤
        //  ⎢────────┼────────⎥ ⎢───⎥ = ⎢───────────────────┼────────⎥ ⎢───⎥ = ⎢───⎥
        //  ⎣ R_01^T │ R_11^T ⎦ ⎣X_1⎦   ⎣ V(R_01) U(R_01)^T │ R_11^T ⎦ ⎣X_1⎦   ⎣M_1⎦
        //
        
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        const auto            sis_X = split( is_X.row_is(), 2 );
        const auto            is1   = U11->col_is();
        const auto            is0   = U00->col_is();
        BLAS::Matrix< real >  X0( X, is0 - U->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< real >  X1( X, is1 - U->col_ofs(), BLAS::Range::all );

        auto  solve_00 = g.alloc_node< trsmu_node >( U00, bis( is0, is_X.col_is() ), X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( ID_U, bis( U01->row_is(), is( 0, U01->rank()-1 ) ), mat_U< real >( U01 ),
                                                     ID_L, bis( sis_X[0], is_X.col_is() ), X0,
                                                     T, ntile );
        auto  tprod    = g.alloc_node< tprod_node >( real(-1),
                                                     ID_U, bis( U01->col_is(), is( 0, U01->rank()-1 ) ), mat_V< real >( U01 ),
                                                     id_t(T.get()), T, // dummy index set
                                                     real(1),
                                                     ID_L, bis( sis_X[1], is_X.col_is() ), X1,
                                                     ntile );
        auto  solve_11 = g.alloc_node< trsmu_node >( U11, bis( is1, is_X.col_is() ), X1, ntile );

        tsmul->after( solve_00 );
        tprod->after( tsmul );
        solve_11->after( tprod );
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
        //
        //  ⎡ L_00 │      ⎤ ⎡X_0⎤   ⎡ L_00              │      ⎤ ⎡X_0⎤   ⎡M_0⎤
        //  ⎢──────┼──────⎥ ⎢───⎥ = ⎢───────────────────┼──────⎥ ⎢───⎥ = ⎢───⎥
        //  ⎣ L_10 │ L_11 ⎦ ⎣X_1⎦   ⎣ U(L_01) V(L_01)^T │ L_11 ⎦ ⎣X_1⎦   ⎣M_1⎦
        //
        
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        const auto            sis_X = split( is_X.row_is(), 2 );
        const auto            is0   = L00->row_is();
        const auto            is1   = L11->row_is();
        BLAS::Matrix< real >  X0( X, is0 - L->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< real >  X1( X, is0 - L->row_ofs(), BLAS::Range::all );
            
        auto  solve_00 = g.alloc_node< trsml_node >( L00, bis( is0, is_X.col_is() ), X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( ID_L, bis( L10->col_is(), is( 0, L10->rank()-1 ) ), mat_V< real >( L10 ),
                                                     ID_U, bis( sis_X[0], is_X.col_is() ), X0,
                                                     T, ntile );
        auto  tprod    = g.alloc_node< tprod_node >( real(-1),
                                                     ID_L, bis( L10->row_is(), is( 0, L10->rank()-1 ) ), mat_U< real >( L10 ),
                                                     id_t(T.get()), T, // dummy index set
                                                     real(1),
                                                     ID_U, bis( sis_X[1], is_X.col_is() ), X1,
                                                     ntile );
        auto  solve_11 = g.alloc_node< trsml_node >( L11, bis( is1, is_X.col_is() ), X1, ntile );

        tsmul->after( solve_00 );
        tprod->after( tsmul );
        solve_11->after( tprod );
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
dot_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( A.nrows() == B.nrows() );

    if ( A.nrows() > ntile )
    {
        const auto                  sis_A = split( is_A.row_is(), 2 );
        const auto                  sis_B = split( is_B.row_is(), 2 );
        const BLAS::Matrix< real >  A0( A, sis_A[0] - is_A.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  A1( A, sis_A[1] - is_A.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  B0( B, sis_B[0] - is_A.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  B1( B, sis_B[1] - is_A.row_is().first(), BLAS::Range::all );

        auto  T0     = std::make_shared< BLAS::Matrix< real > >();
        auto  T1     = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul0 = g.alloc_node< dot_node >( id_A, bis( sis_A[0], is_A.col_is() ), A0,
                                                   id_B, bis( sis_B[0], is_B.col_is() ), B0,
                                                   T0, ntile );
        auto  tsmul1 = g.alloc_node< dot_node >( id_A, bis( sis_A[1], is_A.col_is() ), A1,
                                                   id_B, bis( sis_B[1], is_B.col_is() ), B1,
                                                   T1, ntile );
        auto  add    = g.alloc_node< tadd_node >( T0, T1, T );

        add->after( tsmul0 );
        add->after( tsmul1 );
    }// if

    g.finalize();

    return g;
}

void
dot_node::run_ ( const TTruncAcc &  acc )
{
    *T = std::move( BLAS::prod( real(1), BLAS::adjoint( A ), B ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tprod_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( X.nrows() == Y.nrows() );

    if ( X.nrows() > ntile )
    {
        const auto                  sis_X = split( is_X.row_is(), 2 );
        const auto                  sis_Y = split( is_Y.row_is(), 2 );
        const BLAS::Matrix< real >  X0( X, sis_X[0] - is_X.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  X1( X, sis_X[1] - is_X.row_is().first(), BLAS::Range::all );
        BLAS::Matrix< real >        Y0( Y, sis_Y[0] - is_Y.row_is().first(), BLAS::Range::all );
        BLAS::Matrix< real >        Y1( Y, sis_Y[1] - is_Y.row_is().first(), BLAS::Range::all );

        g.alloc_node< tprod_node >( alpha,
                                    id_X, bis( sis_X[0], is_X.col_is() ), X0,
                                    id_T, T,
                                    beta,
                                    id_Y, bis( sis_Y[0], is_Y.col_is() ), Y0,
                                    ntile );
        g.alloc_node< tprod_node >( alpha,
                                    id_X, bis( sis_X[1], is_X.col_is() ), X1,
                                    id_T, T,
                                    beta,
                                    id_Y, bis( sis_Y[1], is_Y.col_is() ), Y1,
                                    ntile );
    }// if

    g.finalize();

    return g;
}

void
tprod_node::run_ ( const TTruncAcc &  acc )
{
    BLAS::prod( alpha, X, *T, beta, Y );
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

        g.alloc_node< addlr_node    >( id_U, bis( A00->row_is(), is_U.col_is() ), U0,
                                       id_T, T,
                                       id_V, bis( A00->col_is(), is_V.col_is() ), V0,
                                       A00,
                                       ntile );
        
        // g.alloc_node< truncate_node >( id_U, bis( A00->row_is(), is_U.col_is() ), U0,
        //                                id_T, T,
        //                                id_V, bis( A11->col_is(), is_V.col_is() ), V1,
        //                                A01, ntile );

        // g.alloc_node< truncate_node >( id_U, bis( A11->row_is(), is_U.col_is() ), U1,
        //                                id_T, T,
        //                                id_V, bis( A00->col_is(), is_V.col_is() ), V0,
        //                                A10, ntile );
        
        g.alloc_node< addlr_node    >( id_U, bis( A11->row_is(), is_U.col_is() ), U1,
                                       id_T, T,
                                       id_V, bis( A11->col_is(), is_V.col_is() ), V1,
                                       A11,
                                       ntile );
    }// if

    g.finalize();

    return g;
}

void
addlr_node::run_ ( const TTruncAcc & )
{
    assert( is_dense( A ) );
    
    auto        D = ptrcast( A, TDenseMatrix );
    const auto  W = BLAS::prod( real(1), U, *T );

    BLAS::prod( real(-1), W, BLAS::adjoint( V ), real(1), blas_mat< real >( D ) );
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
    
    BLAS::add( real(1), *T0, *T );
    BLAS::add( real(1), *T1, *T );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// truncate_node
//
///////////////////////////////////////////////////////////////////////////////////////

// local_graph
// truncate_node::refine_ ( const size_t  min_size )
// {
//     local_graph  g;

//     // assert( X.nrows() == A->nrows() );
//     // assert( Y.nrows() == A->ncols() );
//     // assert( X.ncols() == T.nrows() );
//     // assert( T.ncols() == Y.ncols() );
    
//     // if ( Y.ncols() + A->rank() > A->nrows() / 2 )
//     // {
//     //     // M = α X T Y^H + U(A) V(A)^H
//     //     auto  W = BLAS::prod( value_t(1), X, T );
//     //     auto  M = BLAS::prod( value_t(1), U, BLAS::adjoint( V ) );

//     //     BLAS::prod( alpha, W, BLAS::adjoint( Y ), value_t(1), M );
            
//     //     // truncate to rank-k
//     //     return std::move( hlr::approx_svd( M, acc ) );
//     // }// if
//     // else
//     // {
//     //     auto  qr_U   = g.alloc_node< tsqr_node >( alpha,      X, T, U, Q0, R0, ntile );
//     //     auto  qr_V   = g.alloc_node< tsqr_node >( value_t(1), Y,    V, Q1, R1, ntile );

//     //     auto  svd    = g.alloc_node< svd_node  >();

//     //     auto  mul_U  = g.alloc_node< tmul_node >();
//     //     auto  mul_V  = g.alloc_node< tmul_node >();

//     //     auto  assign = g.alloc_node< assign_node >();

//     //     svd->after( qr_U );
//     //     svd->after( qr_V );

//     //     mul_U->after( svd );
//     //     mul_V->after( svd );

//     //     assign->after( mul_U );
//     //     assign->after( mul_V );

        
//     //     auto                     R  = BLAS::prod( value_t(1), R0, BLAS::adjoint( R1 ) );
//     //     auto                     Us = std::move( R );
//     //     BLAS::Matrix< value_t >  Vs;
//     //     BLAS::Vector< value_t >  Ss;
        
//     //     BLAS::svd( Us, Ss, Vs );
        
//     //     auto  k  = acc.trunc_rank( Ss );

//     //     BLAS::Matrix< value_t >  Usk( Us, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
//     //     BLAS::Matrix< value_t >  Vsk( Vs, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
        
//     //     BLAS::prod_diag( Usk, Ss, k );

//     //     BLAS::Matrix< value_t >  Uk( U.nrows(), k );
//     //     BLAS::Matrix< value_t >  Vk( V.nrows(), k );



//     //     ::tbb::parallel_invoke( [&,ntile] { tmul( Q0, Usk, Uk, ntile ); },
//     //                             [&,ntile] { tmul( Q1, Vsk, Vk, ntile ); } );

//     //     return { std::move( Uk ), std::move( Vk ) };
//     // }// else

//     // if ( is_blocked( A ) && ! is_small( min_size, A ) )
//     // {
//     //     auto  BA  = ptrcast( A, TBlockMatrix );
//     //     auto  A00 = BA->block( 0, 0 );
//     //     auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
//     //     auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
//     //     auto  A11 = BA->block( 1, 1 );
        
//     //     const BLAS::Matrix< real >  U0( U, A00->row_is() - A->row_ofs(), BLAS::Range::all );
//     //     const BLAS::Matrix< real >  U1( U, A11->row_is() - A->row_ofs(), BLAS::Range::all );
//     //     const BLAS::Matrix< real >  V0( V, A00->col_is() - A->col_ofs(), BLAS::Range::all );
//     //     const BLAS::Matrix< real >  V1( V, A11->col_is() - A->col_ofs(), BLAS::Range::all );

//     //     g.alloc_node< addlr_node    >( id_U, bis( A00->row_is(), is_U.col_is() ), U0,
//     //                                    id_T, T,
//     //                                    id_V, bis( A00->col_is(), is_V.col_is() ), V0,
//     //                                    A00,
//     //                                    ntile );
        
//     //     g.alloc_node< truncate_node >( id_U, bis( A00->row_is(), is_U.col_is() ), U0,
//     //                                    id_T, T,
//     //                                    id_V, bis( A11->col_is(), is_V.col_is() ), V1,
//     //                                    A01, ntile );

//     //     g.alloc_node< truncate_node >( id_U, bis( A11->row_is(), is_U.col_is() ), U1,
//     //                                    id_T, T,
//     //                                    id_V, bis( A00->col_is(), is_V.col_is() ), V0,
//     //                                    A10, ntile );
        
//     //     g.alloc_node< addlr_node    >( id_U, bis( A11->row_is(), is_U.col_is() ), U1,
//     //                                    id_T, T,
//     //                                    id_V, bis( A11->col_is(), is_V.col_is() ), V1,
//     //                                    A11,
//     //                                    ntile );
//     // }// if

//     g.finalize();

//     return g;
// }

///////////////////////////////////////////////////////////////////////////////////////
//
// tsqr_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tsqr_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( X.nrows() == U.nrows() );
    
    if ( X.nrows() > ntile )
    {
        //
        // A = | Q0 R0 | = | Q0   | | R0 | = | Q0   | Q01 R
        //     | Q1 R1 |   |   Q1 | | R1 |   |   Q1 | 
        //
        
        const auto                  sis_X = split( is_X.row_is(), 2 );
        const BLAS::Matrix< real >  X0( X, sis_X[0] - is_X.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  X1( X, sis_X[1] - is_X.row_is().first(), BLAS::Range::all );
        const auto                  sis_U = split( is_U.row_is(), 2 );
        const BLAS::Matrix< real >  U0( U, sis_U[0] - is_U.row_is().first(), BLAS::Range::all );
        const BLAS::Matrix< real >  U1( U, sis_U[1] - is_U.row_is().first(), BLAS::Range::all );

        auto  Q0    = std::make_shared< BLAS::Matrix< real > >();
        auto  R0    = std::make_shared< BLAS::Matrix< real > >();
        auto  tsqr0 = g.alloc_node< tsqr_node >( alpha,
                                                 id_X, bis( sis_X[0], is_X.col_is() ), X0,
                                                 id_T, T,
                                                 id_U, bis( sis_U[0], is_U.col_is() ), U0,
                                                 id_t(Q0.get()), bis( sis_X[0], is_X.col_is() ), Q0,
                                                 id_t(R0.get()), R0,
                                                 ntile );
        
        auto  Q1    = std::make_shared< BLAS::Matrix< real > >();
        auto  R1    = std::make_shared< BLAS::Matrix< real > >();
        auto  tsqr1 = g.alloc_node< tsqr_node >( alpha,
                                                 id_X, bis( sis_X[1], is_X.col_is() ), X1,
                                                 id_T, T,
                                                 id_U, bis( sis_U[1], is_U.col_is() ), U1,
                                                 id_t(Q1.get()), bis( sis_X[1], is_X.col_is() ), Q1,
                                                 id_t(R1.get()), R1,
                                                 ntile );

        auto  Q01   = std::make_shared< BLAS::Matrix< real > >();
        auto  qr01  = g.alloc_node< qr_node >( id_t(R0.get()), R0, id_t(R0.get()), R1, id_R, R );

        qr01->after( tsqr0 );
        qr01->after( tsqr1 );
        
        auto  mul0 = g.alloc_node< tprod_node >( real(1),
                                                 id_t(Q0.get()), bis( sis_X[0], is_X.col_is() ), Q0,
                                                 id_t(R0.get()), R0,
                                                 real(0),
                                                 id_Q, is_Q, Q,  // TODO: is_Q oder is_Q0 ???
                                                 ntile );
        auto  mul1 = g.alloc_node< tprod_node >( real(1),
                                                 id_t(Q1.get()), bis( sis_X[1], is_X.col_is() ), Q1,
                                                 id_t(R1.get()), R1,
                                                 real(0),
                                                 id_Q, is_Q, Q,
                                                 ntile );

        mul0->after( qr01 );
        mul1->after( qr01 );
    }// if

    g.finalize();

    return g;
}

void
tsqr_node::run_ ( const TTruncAcc & )
{
    // TODO: asserts
    
    auto                  W = BLAS::prod( alpha, X, *T );
    BLAS::Matrix< real >  WU( W.nrows(), W.ncols() + U.ncols () );
    BLAS::Matrix< real >  WU_W( WU, BLAS::Range::all, BLAS::Range( 0, W.ncols()-1 ) );
    BLAS::Matrix< real >  WU_U( WU, BLAS::Range::all, BLAS::Range( W.ncols(), WU.ncols()-1 ) );

    BLAS::copy( W, WU_W );
    BLAS::copy( U, WU_U );

    BLAS::qr( WU, *R );

    Q = std::move( WU );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// qr_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
qr_node::run_ ( const TTruncAcc & )
{
    // Q = | R0 |
    //     | R1 |
    BLAS::Matrix< real >  Q(   R0->nrows() + R1->nrows(), R0->ncols() );
    BLAS::Matrix< real >  Q_0( Q, BLAS::Range(           0, R0->nrows()-1 ), BLAS::Range::all );
    BLAS::Matrix< real >  Q_1( Q, BLAS::Range( R0->nrows(), Q.nrows()-1   ), BLAS::Range::all );
        
    BLAS::copy( *R0, Q_0 );
    BLAS::copy( *R1, Q_1 );

    BLAS::qr( Q, *R );

    BLAS::copy( Q_0, *R0 );
    BLAS::copy( Q_1, *R1 );
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
    BLAS::Matrix< real >  X( A.nrows(), 16 );
    BLAS::Matrix< real >  U( A.nrows(), 16 );
    auto                  T = std::make_shared< BLAS::Matrix< real > >();
    BLAS::Matrix< real >  Q( A.nrows(), 16 );
    auto                  R = std::make_shared< BLAS::Matrix< real > >();
    
    return refine( new tsqr_node( -1,
                                  id_t('X'), bis( is( 0, X.nrows()-1 ), is( 0, X.ncols()-1 ) ), X,
                                  id_t('T'), T,
                                  id_t('U'), bis( is( 0, U.nrows()-1 ), is( 0, U.ncols()-1 ) ), U,
                                  id_t('Q'), bis( is( 0, X.nrows()-1 ), is( 0, X.ncols()-1 ) ), Q,
                                  id_t('R'), R,
                                  128 ),
                   HLIB::CFG::Arith::max_seq_size );
    
    // return std::move( refine( new lu_node( & A, 128 ), HLIB::CFG::Arith::max_seq_size ) );
}

}// namespace dag

}// namespace hlr
