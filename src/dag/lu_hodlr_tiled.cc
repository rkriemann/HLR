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

using HLIB::id_t;

// dummy indexset for T operations (rank/size unknown during DAG and only object is of interest)
const auto  IS_ONE  = TIndexSet( 0, 0 );
const auto  BIS_ONE = TBlockIndexSet( TIndexSet( 0, 0 ), TIndexSet( 0, 0 ) );

//
// structure to address BLAS::Matrix
//
template < typename matrix_t >
struct matrix
{
    const id_t       id;       // id of matrix
    const TIndexSet  is;       // index set of associated data
    const TIndexSet  base_is;  // index set of data this matrix is part of
    matrix_t         data;     // matrix data

    matrix ( const id_t       aid,
             const TIndexSet  ais,
             const TIndexSet  abase_is,
             matrix_t         adata )
            : id( aid )
            , is( ais )
            , base_is( abase_is )
            , data( adata )
    {}

    matrix ( const id_t       aid,
             const TIndexSet  ais,
             matrix_t         adata )
            : id( aid )
            , is( ais )
            , base_is( ais )
            , data( adata )
    {}

    matrix ( const id_t       aid,
             matrix_t         adata )
            : id( aid )
            , is( IS_ONE )
            , base_is( IS_ONE )
            , data( adata )
    {}

    matrix ( const TIndexSet  ais,
             matrix &         amat )
            : id( amat.id )
            , is( ais )
            , base_is( amat.base_is )
            , data( amat.data )
    {}

    const TBlockIndexSet block_is () const { return TBlockIndexSet( is, IS_ONE ); }
};

using shared_matrix = matrix< std::shared_ptr< BLAS::Matrix< real > > >;

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

inline
TBlockIndexSet
bis_row ( const TIndexSet &  row_is )
{
    return TBlockIndexSet( row_is, is( 0, 0 ) );
}

inline
TBlockIndexSet
bis_col ( const TIndexSet &  col_is )
{
    return TBlockIndexSet( is( 0, 0 ), col_is );
}

////////////////////////////////////////////////////////////////////////////////
//
// tasks
//
////////////////////////////////////////////////////////////////////////////////

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
template < typename matrixX_t, typename matrixY_t >
struct tprod_node : public node
{
    const real           alpha;
    matrix< matrixX_t >  X;
    shared_matrix        T;
    const real           beta;
    matrix< matrixY_t >  Y;
    const size_t         ntile;

    tprod_node ( const real           aalpha,
                 matrix< matrixX_t >  aX,
                 shared_matrix        aT,
                 const real           abeta,
                 matrix< matrixY_t >  aY,
                 const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , beta( abeta )
            , Y( aY )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "tprod( %d, %d )",
                                                                      X.is.first() / ntile,
                                                                      Y.is.first() / ntile ); }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( beta == real(0) ) return { { X.id, X.block_is() }, { T.id, T.block_is() } };
        else                   return { { X.id, X.block_is() }, { T.id, T.block_is() }, { Y.id, Y.block_is() } };
    }
    virtual const block_list_t  out_blocks_  () const { return { { Y.id, Y.block_is() } }; }
};

template < typename matrix_t >
struct tprod_ip_node : public node
{
    const real                    alpha;
    const id_t                    id_X;
    const TBlockIndexSet          is_X;
    matrix_t                      X;
    const id_t                    id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const size_t                  ntile;

    tprod_ip_node ( const real                   aalpha,
                    const id_t                   aid_X,
                    const TBlockIndexSet         ais_X,
                    matrix_t                     aX,
                    const id_t                   aid_T,
                    std::shared_ptr< BLAS::Matrix< real > >  aT,
                    const size_t                 antile )
            : alpha( aalpha )
            , id_X( aid_X ), is_X( ais_X ), X( aX )
            , id_T( aid_T ), T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( is_X.row_is().size() > ntile )
            return HLIB::to_string( "tprod_ip( %d:%d )",
                                    is_X.row_is().first() / ntile,
                                    is_X.row_is().last() / ntile );
        else
            return HLIB::to_string( "tprod_ip( %d )", is_X.row_is().first() / ntile );
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_X, is_X }, { id_T, BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_X, is_X } }; }
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
struct truncate_node : public node
{
    const real                               alpha;
    const id_t                               id_X;
    const TBlockIndexSet                     is_X;
    const BLAS::Matrix< real >               X;
    const id_t                               id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const id_t                               id_Y;
    const TBlockIndexSet                     is_Y;
    const BLAS::Matrix< real >               Y;
    TRkMatrix *                              A;
    const size_t                             ntile;

    truncate_node ( const real                               aalpha,
                    const id_t                               aid_X,
                    const TBlockIndexSet                     ais_X,
                    const BLAS::Matrix< real > &             aX,
                    const id_t                               aid_T,
                    std::shared_ptr< BLAS::Matrix< real > >  aT,
                    const id_t                               aid_Y,
                    const TBlockIndexSet                     ais_Y,
                    const BLAS::Matrix< real > &             aY,
                    TRkMatrix *                              aA,
                    const size_t                             antile )
            : alpha( aalpha )
            , id_X( aid_X ), is_X( ais_X ), X( aX )
            , id_T( aid_T ), T( aT )
            , id_Y( aid_Y ), is_Y( ais_Y ), Y( aY )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "truncate( %d )", A->id() ); }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_X, is_X }, { id_T, BIS_ONE }, { id_Y, is_Y } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

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
    std::shared_ptr< BLAS::Matrix< real > >  Q;
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
                std::shared_ptr< BLAS::Matrix< real > >  aQ,
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

    virtual std::string  to_string () const
    {
        if ( is_X.row_is().size() > ntile )
            return HLIB::to_string( "tsqr( %d:%d )",
                                    is_X.row_is().first() / ntile,
                                    is_X.row_is().last() / ntile );
        else
            return HLIB::to_string( "tsqr( %d )", is_X.row_is().first() / ntile );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( T ) ) return { { id_X, is_X }, { id_U, is_U }, { id_Q, is_Q } };
        else                return { { id_X, is_X }, { id_U, is_U }, { id_Q, is_Q }, { id_T, BIS_ONE } };
    }
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

    virtual std::string  to_string () const { return HLIB::to_string( "qr( %d )", id_t(R.get()) ); }
    virtual std::string  color     () const { return "c17d11"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_R0, BIS_ONE }, { id_R1, BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_R0, BIS_ONE }, { id_R1, BIS_ONE }, { id_R, BIS_ONE } }; }
};

//
// allocate Q matrices for QR during truncation
//
struct alloc_node : public node
{
    const id_t                               id_X;
    const TBlockIndexSet                     is_X;
    const BLAS::Matrix< real >               X;
    const id_t                               id_T;
    std::shared_ptr< BLAS::Matrix< real > >  T;
    const id_t                               id_Y;
    const TBlockIndexSet                     is_Y;
    const BLAS::Matrix< real >               Y;
    const id_t                               id_Q0;
    const TBlockIndexSet                     is_Q0;
    std::shared_ptr< BLAS::Matrix< real > >  Q0;
    const id_t                               id_Q1;
    const TBlockIndexSet                     is_Q1;
    std::shared_ptr< BLAS::Matrix< real > >  Q1;
    const TRkMatrix *                        A;

    alloc_node ( const id_t                               aid_X,
                 const TBlockIndexSet                     ais_X,
                 const BLAS::Matrix< real >               aX,
                 const id_t                               aid_T,
                 std::shared_ptr< BLAS::Matrix< real > >  aT,
                 const id_t                               aid_Y,
                 const TBlockIndexSet                     ais_Y,
                 const BLAS::Matrix< real >               aY,
                 const id_t                               aid_Q0,
                 const TBlockIndexSet                     ais_Q0,
                 std::shared_ptr< BLAS::Matrix< real > >  aQ0,
                 const id_t                               aid_Q1,
                 const TBlockIndexSet                     ais_Q1,
                 std::shared_ptr< BLAS::Matrix< real > >  aQ1,
                 const TRkMatrix *                        aA )
            : id_X( aid_X ), is_X( ais_X ), X( aX )
            , id_T( aid_T ), T( aT )
            , id_Y( aid_Y ), is_Y( ais_Y ), Y( aY )
            , id_Q0( aid_Q0 ), is_Q0( ais_Q0 ), Q0( aQ0 )
            , id_Q1( aid_Q1 ), is_Q1( ais_Q1 ), Q1( aQ1 )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "alloc( %d )", A->id() ); }
    virtual std::string  color     () const { return "aaaaaa"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_X, is_X }, { id_T, BIS_ONE }, { id_Y, is_Y }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_Q0, is_Q0 }, { id_Q1, is_Q1 } }; }
};

//
// compute truncation rank and set up matrices for truncation
//
struct svd_node : public node
{
    const id_t                               id_R0;
    std::shared_ptr< BLAS::Matrix< real > >  R0;
    const id_t                               id_R1;
    std::shared_ptr< BLAS::Matrix< real > >  R1;
    const id_t                               id_Uk;
    std::shared_ptr< BLAS::Matrix< real > >  Uk;
    const id_t                               id_Vk;
    std::shared_ptr< BLAS::Matrix< real > >  Vk;
    const id_t                               id_U;
    const TBlockIndexSet                     is_U;
    std::shared_ptr< BLAS::Matrix< real > >  U;
    const id_t                               id_V;
    const TBlockIndexSet                     is_V;
    std::shared_ptr< BLAS::Matrix< real > >  V;

    svd_node ( const id_t                               aid_R0,
               std::shared_ptr< BLAS::Matrix< real > >  aR0,
               const id_t                               aid_R1,
               std::shared_ptr< BLAS::Matrix< real > >  aR1,
               const id_t                               aid_Uk,
               std::shared_ptr< BLAS::Matrix< real > >  aUk,
               const id_t                               aid_Vk,
               std::shared_ptr< BLAS::Matrix< real > >  aVk,
               const id_t                               aid_U,
               const TBlockIndexSet                     ais_U,
               std::shared_ptr< BLAS::Matrix< real > >  aU,
               const id_t                               aid_V,
               const TBlockIndexSet                     ais_V,
               std::shared_ptr< BLAS::Matrix< real > >  aV )
            : id_R0( aid_R0 ), R0( aR0 )
            , id_R1( aid_R1 ), R1( aR1 )
            , id_Uk( aid_Uk ), Uk( aUk )
            , id_Vk( aid_Vk ), Vk( aVk )
            , id_U( aid_U ), is_U( ais_U ), U( aU )
            , id_V( aid_V ), is_V( ais_V ), V( aV )
    { init(); }

    virtual std::string  to_string () const { return "svd"; }
    virtual std::string  color     () const { return "ad7fa8"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_R0, BIS_ONE }, { id_R1, BIS_ONE } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_Uk, BIS_ONE }, { id_Vk, BIS_ONE }, { id_U, is_U }, { id_V, is_V } }; }
};

//
// assign result of truncation to low-rank matrix
//
struct assign_node : public node
{
    const id_t                  id_U;
    const TBlockIndexSet        is_U;
    std::shared_ptr< BLAS::Matrix< real > >  U;
    const id_t                  id_V;
    const TBlockIndexSet        is_V;
    std::shared_ptr< BLAS::Matrix< real > >  V;
    TRkMatrix *                 A;

    assign_node ( const id_t                               aid_U,
                  const TBlockIndexSet                     ais_U,
                  std::shared_ptr< BLAS::Matrix< real > >  aU,
                  const id_t                               aid_V,
                  const TBlockIndexSet                     ais_V,
                  std::shared_ptr< BLAS::Matrix< real > >  aV,
                  TRkMatrix *                              aA )
            : id_U( aid_U ), is_U( ais_U ), U( aU )
            , id_V( aid_V ), is_V( ais_V ), V( aV )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "assign( %d )", A->id() ); }
    virtual std::string  color     () const { return "aaaaaa"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { { id_U, is_U }, { id_V, is_V } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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

        auto  solve_00 = g.alloc_node< trsmu_node >( U00, bis_row( is0 ), X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( ID_U, bis_row( U01->row_is() ), mat_U< real >( U01 ),
                                                   ID_L, bis_row( sis_X[0] ), X0,
                                                   T, ntile );
        auto  tprod    = new tprod_node( real(-1),
                                         matrix( ID_U, U01->col_is(), mat_V< real >( U01 ) ),
                                         shared_matrix( id_t(T.get()), T ),
                                         real(1),
                                         matrix( ID_L, sis_X[1], X1 ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsmu_node >( U11, bis_row( is1 ), X1, ntile );

        g.add_node( tprod );
        
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
            
        auto  solve_00 = g.alloc_node< trsml_node >( L00, bis_row( is0 ), X0, ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( ID_L, bis_row( L10->col_is() ), mat_V< real >( L10 ),
                                                   ID_U, bis_row( sis_X[0] ), X0,
                                                   T, ntile );
        auto  tprod    = new tprod_node( real(-1),
                                         matrix( ID_L, L10->row_is(), mat_U< real >( L10 ) ),
                                         shared_matrix( id_t(T.get()), T ),
                                         real(1),
                                         matrix( ID_U, sis_X[1], X1 ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsml_node >( L11, bis_row( is1 ), X1, ntile );

        g.add_node( tprod );
        
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
        // const BLAS::Matrix< real >  A0( A, sis_A[0] - is_A.row_is().first(), BLAS::Range::all );
        // const BLAS::Matrix< real >  A1( A, sis_A[1] - is_A.row_is().first(), BLAS::Range::all );
        // const BLAS::Matrix< real >  B0( B, sis_B[0] - is_A.row_is().first(), BLAS::Range::all );
        // const BLAS::Matrix< real >  B1( B, sis_B[1] - is_A.row_is().first(), BLAS::Range::all );

        auto  T0     = std::make_shared< BLAS::Matrix< real > >();
        auto  T1     = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul0 = g.alloc_node< dot_node  >( id_A, bis( sis_A[0], is_A.col_is() ), A,
                                                  id_B, bis( sis_B[0], is_B.col_is() ), B,
                                                  T0, ntile );
        auto  tsmul1 = g.alloc_node< dot_node  >( id_A, bis( sis_A[1], is_A.col_is() ), A,
                                                  id_B, bis( sis_B[1], is_B.col_is() ), B,
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
    const BLAS::Matrix< real >  Ai( A, is_A.row_is(), BLAS::Range::all );
    const BLAS::Matrix< real >  Bi( B, is_B.row_is(), BLAS::Range::all );
    
    *T = std::move( BLAS::prod( real(1), BLAS::adjoint( Ai ), Bi ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tprod_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename matrixX_t, typename matrixY_t >
local_graph
tprod_node< matrixX_t, matrixY_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( X.is.size() == Y.is.size() );

    if ( X.is.size() > ntile )
    {
        const auto  sis_X = split( X.is, 2 );
        const auto  sis_Y = split( Y.is, 2 );

        g.alloc_node< tprod_node >( alpha, matrix( sis_X[0], X ), T, beta, matrix( sis_Y[0], Y ), ntile );
        g.alloc_node< tprod_node >( alpha, matrix( sis_X[1], X ), T, beta, matrix( sis_Y[1], Y ), ntile );
    }// if

    g.finalize();

    return g;
}

template <>
void
tprod_node< BLAS::Matrix< real >, BLAS::Matrix< real > >::run_ ( const TTruncAcc &  acc )
{
    const BLAS::Matrix< real >  Xi( X.data, X.is - X.base_is.first(), BLAS::Range::all );
    BLAS::Matrix< real >        Yi( Y.data, Y.is - Y.base_is.first(), BLAS::Range::all );
    
    BLAS::prod( alpha, Xi, *(T.data), beta, Yi );
}

template <>
void
tprod_node< BLAS::Matrix< real >, std::shared_ptr< BLAS::Matrix< real > > >::run_ ( const TTruncAcc &  acc )
{
    const BLAS::Matrix< real >  Xi(         X, X.is - X.base_is.first(), BLAS::Range::all );
    BLAS::Matrix< real >        Yi( *(Y.data), Y.is - Y.base_is.first(), BLAS::Range::all );
    
    BLAS::prod( alpha, Xi, *(T.data), beta, Yi );
}

template <>
void
tprod_node< std::shared_ptr< BLAS::Matrix< real > >, BLAS::Matrix< real > >::run_ ( const TTruncAcc &  acc )
{
    const BLAS::Matrix< real >  Xi( *(X.data), X.is - X.base_is.first(), BLAS::Range::all );
    BLAS::Matrix< real >        Yi(         Y, Y.is - Y.base_is.first(), BLAS::Range::all );
    
    BLAS::prod( alpha, Xi, *(T.data), beta, Yi );
}

template <>
void
tprod_node< std::shared_ptr< BLAS::Matrix< real > >, std::shared_ptr< BLAS::Matrix< real > > >::run_ ( const TTruncAcc &  acc )
{
    const BLAS::Matrix< real >  Xi( *(X.data), X.is - X.base_is.first(), BLAS::Range::all );
    BLAS::Matrix< real >        Yi( *(Y.data), Y.is - Y.base_is.first(), BLAS::Range::all );
    
    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( Yi, "Y1.mat", "Y1" );
    // DBG::write( *T, "T1.mat", "T1" );
    
    BLAS::prod( alpha, Xi, *(T.data), beta, Yi );

    // DBG::write( Yi,  "Z1.mat", "Z1" );
}

template < typename matrix_t >
local_graph
tprod_ip_node< matrix_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_X.row_is().size() > ntile )
    {
        const auto  sis_X = split( is_X.row_is(), 2 );

        g.alloc_node< tprod_ip_node >( alpha,
                                       id_X, bis_row( sis_X[0] ), X,
                                       id_T, T,
                                       ntile );
        g.alloc_node< tprod_ip_node >( alpha,
                                       id_X, bis_row( sis_X[1] ), X,
                                       id_T, T,
                                       ntile );
    }// if

    g.finalize();

    return g;
}

template <>
void
tprod_ip_node< BLAS::Matrix< real > >::run_ ( const TTruncAcc &  acc )
{
    BLAS::Matrix< real >        Xi( X, is_X.row_is(), BLAS::Range::all );
    const BLAS::Matrix< real >  Xc( Xi, HLIB::copy_value );
    
    BLAS::prod( alpha, Xc, *T, 0.0, Xi );
}

template <>
void
tprod_ip_node< std::shared_ptr< BLAS::Matrix< real > > >::run_ ( const TTruncAcc &  acc )
{
    BLAS::Matrix< real >        Xi( *X, is_X.row_is(), BLAS::Range::all );
    const BLAS::Matrix< real >  Xc( Xi, HLIB::copy_value );
    
    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( *T, "T1.mat", "T1" );

    BLAS::prod( alpha, Xc, *T, 0.0, Xi );

    // DBG::write( Xi,  "Z1.mat", "Z1" );
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

local_graph
truncate_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    assert( is_X.row_is() == A->row_is() );
    assert( is_Y.row_is() == A->col_is() );
    
    // assert( X.nrows() == A->nrows() );
    // assert( Y.nrows() == A->ncols() );
    // assert( X.ncols() == T.nrows() );
    // assert( T.ncols() == Y.ncols() );
    
    auto  Q0     = std::make_shared< BLAS::Matrix< real > >();
    auto  R0     = std::make_shared< BLAS::Matrix< real > >();
    auto  Q1     = std::make_shared< BLAS::Matrix< real > >();
    auto  R1     = std::make_shared< BLAS::Matrix< real > >();
    auto  bis_X  = bis_row( is_X.row_is() );
    auto  bis_Y  = bis_row( is_Y.row_is() );

    // ensure that Q0/Q1 are allocated with correct size before QR
    auto  alloc  = g.alloc_node< alloc_node >( id_X, bis_row( is_X.row_is() ), X,
                                               id_T, T,
                                               id_Y, bis_row( is_Y.row_is() ), Y,
                                               id_t(Q0.get()), bis_X, Q0,
                                               id_t(Q1.get()), bis_Y, Q1,
                                               A );
    
    // perform QR for U/V
    auto  qr_U   = g.alloc_node< tsqr_node >( alpha,
                                              id_X, bis_row( is_X.row_is() ), X,
                                              id_T, T,
                                              id_t( A->blas_rmat_A().data() ), bis_row( A->row_is() ), A->blas_rmat_A(),
                                              id_t(Q0.get()), bis_X, Q0,
                                              id_t(R0.get()), R0,
                                              ntile );

    auto  qr_V   = g.alloc_node< tsqr_node >( real(1),
                                              id_Y, bis_row( is_Y.row_is() ), Y,
                                              id_T, std::shared_ptr< BLAS::Matrix< real > >(), // T = 0
                                              id_t( A->blas_rmat_B().data() ), bis_row( A->col_is() ), A->blas_rmat_B(),
                                              id_t(Q1.get()), bis_Y, Q1,
                                              id_t(R1.get()), R1,
                                              ntile );

    qr_U->after( alloc );
    qr_V->after( alloc );

    // determine truncated rank and allocate destination matrices
    auto  Uk     = std::make_shared< BLAS::Matrix< real > >();
    auto  Vk     = std::make_shared< BLAS::Matrix< real > >();
    auto  U      = std::make_shared< BLAS::Matrix< real > >();
    auto  V      = std::make_shared< BLAS::Matrix< real > >();
    auto  svd    = g.alloc_node< svd_node >( id_t(R0.get()), R0,
                                             id_t(R1.get()), R1,
                                             id_t(Uk.get()), Uk,
                                             id_t(Vk.get()), Vk,
                                             id_t(U.get()),  bis_X, U,
                                             id_t(V.get()),  bis_Y, V );

    svd->after( qr_U );
    svd->after( qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( real(1),
                                   matrix( id_t(Q0.get()), is_X.row_is(), Q0 ),
                                   shared_matrix( id_t(Uk.get()), Uk ),
                                   real(0),
                                   matrix( id_t(U.get()),  is_X.row_is(), U ),
                                   ntile );
    auto  mul_V  = new tprod_node( real(1),
                                   matrix( id_t(Q1.get()), is_Y.row_is(), Q1 ),
                                   shared_matrix( id_t(Vk.get()), Vk ),
                                   real(0),
                                   matrix( id_t(V.get()),  is_Y.row_is(), V ),
                                   ntile );

    g.add_node( mul_U );
    g.add_node( mul_V );
    
    mul_U->after( qr_U );
    mul_U->after( svd );
    mul_V->after( qr_V );
    mul_V->after( svd );

    // assign destination to actual matrix
    auto  assign = g.alloc_node< assign_node >( id_t(U.get()), bis_X, U,
                                                id_t(V.get()), bis_Y, V,
                                                A );

    assign->after( mul_U );
    assign->after( mul_V );
    
    g.finalize();

    return g;
}

void
truncate_node::run_ ( const TTruncAcc &  acc )
{
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tsqr_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tsqr_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_X.row_is().size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis_X = split( is_X.row_is(), 2 );
        const auto  sis_U = split( is_U.row_is(), 2 );
        const auto  sis_Q = split( is_Q.row_is(), 2 );
        auto        R0    = std::make_shared< BLAS::Matrix< real > >();
        auto        R1    = std::make_shared< BLAS::Matrix< real > >();

        auto  tsqr0 = g.alloc_node< tsqr_node >( alpha,
                                                 id_X, bis_row( sis_X[0] ), X,
                                                 id_T, T,
                                                 id_U, bis_row( sis_U[0] ), U,
                                                 id_Q, bis_row( sis_Q[0] ), Q,
                                                 id_t(R0.get()), R0,
                                                 ntile );
        
        auto  tsqr1 = g.alloc_node< tsqr_node >( alpha,
                                                 id_X, bis_row( sis_X[1] ), X,
                                                 id_T, T,
                                                 id_U, bis_row( sis_U[1] ), U,
                                                 id_Q, bis_row( sis_Q[1] ), Q,
                                                 id_t(R1.get()), R1,
                                                 ntile );

        auto  qr01  = g.alloc_node< qr_node >( id_t(R0.get()), R0,
                                               id_t(R1.get()), R1,
                                               id_R, R );

        qr01->after( tsqr0 );
        qr01->after( tsqr1 );
        
        auto  mul0 = new tprod_ip_node( real(1),
                                        id_Q, bis_row( sis_Q[0] ), Q,
                                        id_t(R0.get()), R0,
                                        ntile );
        auto  mul1 = new tprod_ip_node( real(1),
                                        id_Q, bis_row( sis_Q[1] ), Q,
                                        id_t(R1.get()), R1,
                                        ntile );

        g.add_node( mul0 );
        g.add_node( mul1 );
        
        mul0->after( tsqr0 );
        mul0->after( qr01 );
        mul1->after( tsqr1 );
        mul1->after( qr01 );
    }// if

    g.finalize();

    return g;
}

void
tsqr_node::run_ ( const TTruncAcc & )
{
    // TODO: asserts

    std::cout << is_X.row_is() << ", " << is_U.row_is() << std::endl;
    
    const BLAS::Matrix< real >  Xi( X, is_X.row_is(), BLAS::Range::all );
    const BLAS::Matrix< real >  Ui( U, is_U.row_is(), BLAS::Range::all );

    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( *T, "T1.mat", "T1" );
    // DBG::write( Ui, "U1.mat", "U1" );
    
    if ( is_null( T ) )
    {
        BLAS::Matrix< real >  XU( Xi.nrows(), Xi.ncols() + Ui.ncols () );
        BLAS::Matrix< real >  XU_X( XU, BLAS::Range::all, BLAS::Range( 0, Xi.ncols()-1 ) );
        BLAS::Matrix< real >  XU_U( XU, BLAS::Range::all, BLAS::Range( Xi.ncols(), XU.ncols()-1 ) );

        BLAS::copy( Xi, XU_X );
        BLAS::copy( Ui, XU_U );
        
        BLAS::qr( XU, *R );

        // DBG::write( WU, "Q1.mat", "Q1" );
        // DBG::write( *R, "R1.mat", "R1" );
        
        BLAS::Matrix< real >  Qi( *Q, is_Q.row_is(), BLAS::Range::all );
        
        BLAS::copy( XU, Qi );
    }// if
    else
    {
        auto                  XT = BLAS::prod( alpha, Xi, *T );

        // DBG::write( XT, "XT1.mat", "XT1" );
    
        BLAS::Matrix< real >  XU( Xi.nrows(), XT.ncols() + Ui.ncols () );
        BLAS::Matrix< real >  XU_X( XU, BLAS::Range::all, BLAS::Range( 0, XT.ncols()-1 ) );
        BLAS::Matrix< real >  XU_U( XU, BLAS::Range::all, BLAS::Range( XT.ncols(), XU.ncols()-1 ) );

        BLAS::copy( XT, XU_X );
        BLAS::copy( Ui, XU_U );
        
        BLAS::qr( XU, *R );

        // DBG::write( XU, "Q1.mat", "Q1" );
        // DBG::write( *R, "R1.mat", "R1" );
        
        BLAS::Matrix< real >  Qi( *Q, is_Q.row_is(), BLAS::Range::all );
        
        BLAS::copy( XU, Qi );
    }// else
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

    // DBG::write( Q, "R1.mat", "R1" );
    
    BLAS::qr( Q, *R );

    // DBG::write( Q, "Q1.mat", "Q1" );

    BLAS::copy( Q_0, *R0 );
    BLAS::copy( Q_1, *R1 );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// alloc_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
alloc_node::run_ ( const TTruncAcc & )
{
    *Q0 = std::move( BLAS::Matrix< real >( X.nrows(), T->ncols() + A->rank() ) );
    *Q1 = std::move( BLAS::Matrix< real >( Y.nrows(), Y.ncols()  + A->rank() ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// svd_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
svd_node::run_ ( const TTruncAcc &  acc )
{
    auto                  Us = BLAS::prod( real(1), *R0, BLAS::adjoint( *R1 ) );
    BLAS::Matrix< real >  Vs;
    BLAS::Vector< real >  Ss;
        
    BLAS::svd( Us, Ss, Vs );
        
    const auto            k  = acc.trunc_rank( Ss );

    for ( uint i = 0; i < k; ++i )
        std::cout << Ss(i) << std::endl;
    for ( uint i = k; i < Ss.length(); ++i )
        std::cout << Ss(i) << std::endl;
    
    BLAS::Matrix< real >  Usk( Us, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
    BLAS::Matrix< real >  Vsk( Vs, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
        
    BLAS::prod_diag( Usk, Ss, k );

    *Uk = std::move( BLAS::Matrix< real >( Usk.nrows(), Usk.ncols() ) );
    *Vk = std::move( BLAS::Matrix< real >( Vsk.nrows(), Vsk.ncols() ) );

    BLAS::copy( Usk, *Uk );
    BLAS::copy( Vsk, *Vk );
    
    *U  = std::move( BLAS::Matrix< real >( is_U.row_is().size(), k ) );
    *V  = std::move( BLAS::Matrix< real >( is_V.row_is().size(), k ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// alloc_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
assign_node::run_ ( const TTruncAcc & )
{
    A->set_lrmat( *U, *V );
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
    auto                  Q = std::make_shared< BLAS::Matrix< real > >();
    auto                  R = std::make_shared< BLAS::Matrix< real > >();
    
    return refine( new tsqr_node( -1,
                                  id_t('X'), bis_row( is( 0, X.nrows()-1 ) ), X,
                                  id_t('T'), T,
                                  id_t('U'), bis_row( is( 0, U.nrows()-1 ) ), U,
                                  id_t('Q'), bis_row( is( 0, X.nrows()-1 ) ), Q,
                                  id_t('R'), R,
                                  128 ),
                   HLIB::CFG::Arith::max_seq_size );
    
    // return std::move( refine( new lu_node( & A, 128 ), HLIB::CFG::Arith::max_seq_size ) );
}

//
// compute DAG for TSQR( X·T, U )
//
graph
gen_dag_tsqr ( HLIB::BLAS::Matrix< HLIB::real > &                     X,
               std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  T,
               HLIB::BLAS::Matrix< HLIB::real > &                     U,
               std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  Q,
               std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  R,
               refine_func_t                                          refine )
{
    return refine( new tsqr_node( 1,
                                  id_t('X'), bis_row( is( 0, X.nrows()-1 ) ), X,
                                  id_t('T'), T,
                                  id_t('U'), bis_row( is( 0, U.nrows()-1 ) ), U,
                                  id_t('Q'), bis_row( is( 0, X.nrows()-1 ) ), Q,
                                  id_t('R'), R,
                                  128 ),
                   HLIB::CFG::Arith::max_seq_size );
}

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_truncate ( HLIB::BLAS::Matrix< HLIB::real > &                     X,
                   std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  T,
                   HLIB::BLAS::Matrix< HLIB::real > &                     Y,
                   TRkMatrix *                                            A,
                   refine_func_t                                          refine )
{
    return refine( new truncate_node( real(1),
                                      id_t('X'), bis_row( A->row_is() ), X,
                                      id_t('T'), T,
                                      id_t('U'), bis_row( A->col_is() ), Y,
                                      A,
                                      128 ),
                   HLIB::CFG::Arith::max_seq_size );
}

}// namespace dag

}// namespace hlr
