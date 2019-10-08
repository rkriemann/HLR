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

    matrix ( matrix< matrix_t > &  aM )
            : id( aM.id )
            , is( aM.is )
            , base_is( aM.base_is )
            , data( aM.data )
    {}

    matrix ( matrix< matrix_t > &&  aM )
            : id( aM.id )
            , is( aM.is )
            , base_is( aM.base_is )
            , data( std::move( aM.data ) )
    {}

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
    
    matrix ( const TIndexSet  is,
             matrix_t         adata );
    
    matrix ( matrix_t         adata );
    
    matrix ( const TIndexSet  ais,
             matrix &         amat )
            : id( amat.id )
            , is( ais )
            , base_is( amat.base_is )
            , data( amat.data )
    {}

    virtual ~matrix ();

    operator matrix_t () { return data; }
    
    const BLAS::Range    range     () const { return is - base_is.first(); }
    const TBlockIndexSet block_is  () const { return TBlockIndexSet( is, IS_ONE ); }
    const mem_block_t    mem_block () const { return { id, block_is() }; }
};

template <>
matrix< BLAS::Matrix< real > >::~matrix ()
{}

template <>
matrix< std::shared_ptr< BLAS::Matrix< real > > >::~matrix ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix< BLAS::Matrix< real > >::matrix ( const TIndexSet       is,
                                         BLAS::Matrix< real >  adata )
        : id( id_t(adata.data()) )
        , is( is )
        , base_is( is )
        , data( adata )
{}

template <>
matrix< std::shared_ptr< BLAS::Matrix< real > > >::matrix ( const TIndexSet                          is,
                                                            std::shared_ptr< BLAS::Matrix< real > >  adata )
        : id( id_t(adata.get()) )
        , is( is )
        , base_is( is )
        , data( adata )
{}

// template <>
// matrix< BLAS::Matrix< real > >::matrix ( BLAS::Matrix< real >  adata )
//         : id( id_t(adata.data()) )
//         , is( IS_ONE )
//         , base_is( IS_ONE )
//         , data( adata )
// {}

template <>
matrix< std::shared_ptr< BLAS::Matrix< real > > >::matrix ( std::shared_ptr< BLAS::Matrix< real > >  adata )
        : id( id_t(adata.get()) )
        , is( IS_ONE )
        , base_is( IS_ONE )
        , data( adata )
{}


using blas_matrix   = matrix< BLAS::Matrix< real > >;
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
constexpr id_t  ID_A    = 'A';
constexpr id_t  ID_L    = 'L';
constexpr id_t  ID_U    = 'U';
constexpr id_t  ID_NONE = '0';

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
    virtual size_t              mem_size_    () const { return sizeof(lu_node); }
};

//
// solve X U = M with upper triangular U and M given
//
struct trsmu_node : public node
{
    const TMatrix *  U;
    blas_matrix      X;
    const size_t     ntile;
    
    trsmu_node ( const TMatrix *  aU,
                 blas_matrix      aX,
                 const size_t     antile )
            : U( aU )
            , X( aX )
            , ntile( antile )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "L = trsmu( U%d, A )", U->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, X.block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, X.block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(trsmu_node); }
};

//
// solve L X = M with lower triangular L and M given
//
struct trsml_node : public node
{
    const TMatrix *  L;
    blas_matrix      X;
    const size_t     ntile;

    trsml_node ( const TMatrix *  aL,
                 blas_matrix      aX,
                 const size_t     antile )
            : L( aL )
            , X( aX )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "U = trsml( L%d, A )", L->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, X.block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, X.block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(trsml_node); }
};
    
//
// compute A := A - U·T·V^H
//
struct addlr_node : public node
{
    blas_matrix    U;
    shared_matrix  T;
    blas_matrix    V;
    TMatrix *      A;
    const size_t   ntile;

    addlr_node ( blas_matrix    aU,
                 shared_matrix  aT,
                 blas_matrix    aV,
                 TMatrix *      aA,
                 const size_t   antile )
            : U( aU )
            , T( aT )
            , V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( U.is.size() > ntile )
            return HLIB::to_string( "addlr( %d:%d, %d:%d, %d )",
                                    U.is.first() / ntile, U.is.last()/ntile,
                                    V.is.first() / ntile, V.is.last()/ntile,
                                    A->id() );
        else
            return HLIB::to_string( "addlr( %d, %d, %d )",
                                    U.is.first() / ntile,
                                    V.is.first() / ntile,
                                    A->id() );
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { U.mem_block(), V.mem_block(), T.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(addlr_node); }
};

//
// compute T := A^H · B
//
struct dot_node : public node
{
    blas_matrix    A;
    blas_matrix    B;
    shared_matrix  T;
    const size_t   ntile;

    dot_node ( blas_matrix    aA,
               blas_matrix    aB,
               shared_matrix  aT,
               const size_t   antile )
            : A( aA )
            , B( aB )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( A.is.size() > ntile )
            return HLIB::to_string( "dot( %c[%d:%d], %c[%d:%d] )",
                                    char(A.id), A.is.first() / ntile, A.is.last() / ntile,
                                    char(B.id), B.is.first() / ntile, B.is.last() / ntile );
        else
            return HLIB::to_string( "dot( %c[%d], %c[%d] )",
                                    char(A.id), A.is.first() / ntile,
                                    char(B.id), B.is.first() / ntile );
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { A.mem_block(), B.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { T.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(dot_node); }
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
        if ( beta == real(0) ) return { X.mem_block(), T.mem_block() };
        else                   return { X.mem_block(), T.mem_block(), Y.mem_block() };
    }
    virtual const block_list_t  out_blocks_  () const { return { Y.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tprod_node); }
};

template < typename matrix_t >
struct tprod_ip_node : public node
{
    const real          alpha;
    matrix< matrix_t >  X;
    shared_matrix       T;
    const size_t        ntile;

    tprod_ip_node ( const real          aalpha,
                    matrix< matrix_t >  aX,
                    shared_matrix       aT,
                    const size_t        antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( X.is.size() > ntile )
            return HLIB::to_string( "tprod_ip( %d:%d )", X.is.first() / ntile, X.is.last() / ntile );
        else
            return HLIB::to_string( "tprod_ip( %d )", X.is.first() / ntile );
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { X.mem_block(), T.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { X.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tprod_ip_node); }
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
    virtual size_t              mem_size_    () const { return sizeof(tadd_node); }
};

//
// truncate α X T Y^H + U(A) V(A)^H
//
struct truncate_node : public node
{
    const real     alpha;
    blas_matrix    X;
    shared_matrix  T;
    blas_matrix    Y;
    TRkMatrix *    A;
    const size_t   ntile;

    truncate_node ( const real     aalpha,
                    blas_matrix    aX,
                    shared_matrix  aT,
                    blas_matrix    aY,
                    TRkMatrix *    aA,
                    const size_t   antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , Y( aY )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "truncate( %d )", A->id() ); }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { X.mem_block(), T.mem_block(), Y.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate_node); }
};

//
// QR factorization of [αX·T,U]
//
struct tsqr_node : public node
{
    const real     alpha;
    blas_matrix    X;
    shared_matrix  T;
    blas_matrix    U;
    shared_matrix  Q;
    shared_matrix  R;
    const size_t   ntile;

    tsqr_node ( const real     aalpha,
                blas_matrix    aX,
                shared_matrix  aT,
                blas_matrix    aU,
                shared_matrix  aQ,
                shared_matrix  aR,
                const size_t   antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , U( aU )
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    tsqr_node ( const real     aalpha,
                blas_matrix    aX,
                blas_matrix    aU,
                shared_matrix  aQ,
                shared_matrix  aR,
                const size_t   antile )
            : alpha( aalpha )
            , X( aX )
            , T( shared_matrix( ID_NONE, std::shared_ptr< BLAS::Matrix< real > >() ) ) // T = 0
            , U( aU )
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( X.is.size() > ntile )
            return HLIB::to_string( "tsqr( %d:%d )", X.is.first() / ntile, X.is.last() / ntile );
        else
            return HLIB::to_string( "tsqr( %d )", X.is.first() / ntile );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( T.data ) ) return { X.mem_block(), U.mem_block(), Q.mem_block() };
        else                     return { X.mem_block(), U.mem_block(), Q.mem_block(), T.mem_block() };
    }
    virtual const block_list_t  out_blocks_  () const { return { Q.mem_block(), R.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tsqr_node); }
};

//
// QR factorization of [R0;R1] with Q written to [R0;R1] 
//
struct qr_node : public node
{
    shared_matrix  R0;
    shared_matrix  R1;
    shared_matrix  R;

    qr_node ( shared_matrix  aR0,
              shared_matrix  aR1,
              shared_matrix  aR )
            : R0( aR0 )
            , R1( aR1 )
            , R(  aR  )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "qr( %d )", R.id ); }
    virtual std::string  color     () const { return "c17d11"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { R0.mem_block(), R1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { R0.mem_block(), R1.mem_block(), R.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(qr_node); }
};

//
// allocate Q matrices for QR during truncation
//
struct alloc_node : public node
{
    blas_matrix        X;
    shared_matrix      T;
    blas_matrix        Y;
    shared_matrix      Q0;
    shared_matrix      Q1;
    const TRkMatrix *  A;

    alloc_node ( blas_matrix        aX,
                 shared_matrix      aT,
                 blas_matrix        aY,
                 shared_matrix      aQ0,
                 shared_matrix      aQ1,
                 const TRkMatrix *  aA )
            : X( aX )
            , T( aT )
            , Y( aY )
            , Q0( aQ0 )
            , Q1( aQ1 )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "alloc( %d )", A->id() ); }
    virtual std::string  color     () const { return "aaaaaa"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { X.mem_block(), T.mem_block(), Y.mem_block(), { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { Q0.mem_block(), Q1.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(alloc_node); }
};

//
// compute truncation rank and set up matrices for truncation
//
struct svd_node : public node
{
    shared_matrix  R0;
    shared_matrix  R1;
    shared_matrix  Uk;
    shared_matrix  Vk;
    shared_matrix  U;
    shared_matrix  V;

    svd_node ( shared_matrix  aR0,
               shared_matrix  aR1,
               shared_matrix  aUk,
               shared_matrix  aVk,
               shared_matrix  aU,
               shared_matrix  aV )
            : R0( aR0 )
            , R1( aR1 )
            , Uk( aUk )
            , Vk( aVk )
            , U( aU )
            , V( aV )
    { init(); }

    virtual std::string  to_string () const { return "svd"; }
    virtual std::string  color     () const { return "ad7fa8"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { R0.mem_block(), R1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { Uk.mem_block(), Vk.mem_block(), U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(svd_node); }
};

//
// assign result of truncation to low-rank matrix
//
struct assign_node : public node
{
    shared_matrix  U;
    shared_matrix  V;
    TRkMatrix *    A;

    assign_node ( shared_matrix  aU,
                  shared_matrix  aV,
                  TRkMatrix *    aA )
            : U( aU )
            , V( aV )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "assign( %d )", A->id() ); }
    virtual std::string  color     () const { return "aaaaaa"; }

private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { U.mem_block(), V.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(assign_node); }
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
        auto  solve_10 = g.alloc_node< trsmu_node >( BU->block( 0, 0 ), matrix( ID_A, A10->col_is(), mat_V< real >( A10 ) ), ntile );
        auto  solve_01 = g.alloc_node< trsml_node >( BL->block( 0, 0 ), matrix( ID_A, A01->row_is(), mat_U< real >( A01 ) ), ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( matrix( ID_L, A10->col_is(), mat_V< real >( A10 ) ),
                                                   matrix( ID_U, A01->row_is(), mat_U< real >( A01 ) ),
                                                   T,
                                                   ntile );
        auto  addlr    = g.alloc_node< addlr_node >( matrix( ID_L, A10->row_is(), mat_U< real >( A10 ) ),
                                                     matrix( T ),
                                                     matrix( ID_U, A01->col_is(), mat_V< real >( A01 ) ),
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

        const auto  sis_X = split( X.is, 2 );
        const auto  is1   = U11->col_is();
        const auto  is0   = U00->col_is();

        auto  solve_00 = g.alloc_node< trsmu_node >( U00, matrix( is0, X ), ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( matrix( ID_U, U01->row_is(), mat_U< real >( U01 ) ),
                                                   matrix( sis_X[0], X ),
                                                   matrix( T ),
                                                   ntile );
        auto  tprod    = new tprod_node( real(-1),
                                         matrix( ID_U, U01->col_is(), mat_V< real >( U01 ) ),
                                         matrix( T ),
                                         real(1),
                                         matrix( sis_X[1], X ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsmu_node >( U11, matrix( is1, X ), ntile );

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
    BLAS::Matrix< real >  Xi( X.data, X.range(), BLAS::Range::all );
    
    hlr::seq::tile::hodlr::trsmuh( U, Xi, ntile );
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

        const auto  sis_X = split( X.is, 2 );
        const auto  is0   = L00->row_is();
        const auto  is1   = L11->row_is();
            
        auto  solve_00 = g.alloc_node< trsml_node >( L00, matrix( is0, X ), ntile );
        auto  T        = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul    = g.alloc_node< dot_node >( matrix( ID_L, L10->col_is(), mat_V< real >( L10 ) ),
                                                   matrix( sis_X[0], X ),
                                                   matrix( T ),
                                                   ntile );
        auto  tprod    = new tprod_node( real(-1),
                                         matrix( ID_L, L10->row_is(), mat_U< real >( L10 ) ),
                                         matrix( T ),
                                         real(1),
                                         matrix( sis_X[1], X ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsml_node >( L11, matrix( is1, X ), ntile );

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
    BLAS::Matrix< real >  Xi( X.data, X.range(), BLAS::Range::all );

    hlr::seq::tile::hodlr::trsml( L, Xi, ntile );
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

    assert( A.is.size() == B.is.size() );

    if ( A.is.size() > ntile )
    {
        const auto  sis_A = split( A.is, 2 );
        const auto  sis_B = split( B.is, 2 );

        auto  T0     = std::make_shared< BLAS::Matrix< real > >();
        auto  T1     = std::make_shared< BLAS::Matrix< real > >();
        auto  tsmul0 = g.alloc_node< dot_node  >( matrix( sis_A[0], A ), matrix( sis_B[0], B ), matrix( T0 ), ntile );
        auto  tsmul1 = g.alloc_node< dot_node  >( matrix( sis_A[1], A ), matrix( sis_B[1], B ), matrix( T1 ), ntile );
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
    const BLAS::Matrix< real >  Ai( A.data, A.range(), BLAS::Range::all );
    const BLAS::Matrix< real >  Bi( B.data, B.range(), BLAS::Range::all );
    
    *(T.data) = std::move( BLAS::prod( real(1), BLAS::adjoint( Ai ), Bi ) );

    std::cout << "dot : " << BLAS::norm_F( *(T.data) ) << std::endl;
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

        g.alloc_node< tprod_node >( alpha, matrix( sis_X[0], X.data ), T, beta, matrix( sis_Y[0], Y.data ), ntile );
        g.alloc_node< tprod_node >( alpha, matrix( sis_X[1], X.data ), T, beta, matrix( sis_Y[1], Y.data ), ntile );
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

// template <>
// void
// tprod_node< BLAS::Matrix< real >, std::shared_ptr< BLAS::Matrix< real > > >::run_ ( const TTruncAcc &  acc )
// {
//     const BLAS::Matrix< real >  Xi(         X, X.is - X.base_is.first(), BLAS::Range::all );
//     BLAS::Matrix< real >        Yi( *(Y.data), Y.is - Y.base_is.first(), BLAS::Range::all );
    
//     BLAS::prod( alpha, Xi, *(T.data), beta, Yi );
// }

// template <>
// void
// tprod_node< std::shared_ptr< BLAS::Matrix< real > >, BLAS::Matrix< real > >::run_ ( const TTruncAcc &  acc )
// {
//     const BLAS::Matrix< real >  Xi( *(X.data), X.is - X.base_is.first(), BLAS::Range::all );
//     BLAS::Matrix< real >        Yi(         Y, Y.is - Y.base_is.first(), BLAS::Range::all );
    
//     BLAS::prod( alpha, Xi, *(T.data), beta, Yi );
// }

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

    if ( X.is.size() > ntile )
    {
        const auto  sis_X = split( X.is, 2 );

        g.alloc_node< tprod_ip_node >( alpha, matrix( sis_X[0], X ), T, ntile );
        g.alloc_node< tprod_ip_node >( alpha, matrix( sis_X[1], X ), T, ntile );
    }// if

    g.finalize();

    return g;
}

// template <>
// void
// tprod_ip_node< BLAS::Matrix< real > >::run_ ( const TTruncAcc &  acc )
// {
//     BLAS::Matrix< real >        Xi( X.data, X.is - X.base_is.first(), BLAS::Range::all );
//     const BLAS::Matrix< real >  Xc( Xi, HLIB::copy_value );
    
//     BLAS::prod( alpha, Xc, *(T.data), 0.0, Xi );
// }

template <>
void
tprod_ip_node< std::shared_ptr< BLAS::Matrix< real > > >::run_ ( const TTruncAcc &  acc )
{
    BLAS::Matrix< real >        Xi( *(X.data), X.is - X.base_is.first(), BLAS::Range::all );
    const BLAS::Matrix< real >  Xc( Xi, HLIB::copy_value );
    
    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( *T, "T1.mat", "T1" );

    BLAS::prod( alpha, Xc, *(T.data), 0.0, Xi );

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
        
        g.alloc_node< addlr_node    >(           matrix( A00->row_is(), U ), T, matrix( A00->col_is(), V ), A00, ntile );
        g.alloc_node< truncate_node >( real(-1), matrix( A01->row_is(), U ), T, matrix( A01->col_is(), V ), A01, ntile );
        g.alloc_node< truncate_node >( real(-1), matrix( A10->row_is(), U ), T, matrix( A10->col_is(), V ), A10, ntile );
        g.alloc_node< addlr_node    >(           matrix( A11->row_is(), U ), T, matrix( A11->col_is(), V ), A11, ntile );
    }// if

    g.finalize();

    return g;
}

void
addlr_node::run_ ( const TTruncAcc & )
{
    assert( is_dense( A ) );
    
    const BLAS::Matrix< real >  Ui( U.data, U.range(), BLAS::Range::all );
    const BLAS::Matrix< real >  Vi( V.data, V.range(), BLAS::Range::all );
    
    auto        D = ptrcast( A, TDenseMatrix );
    const auto  W = BLAS::prod( real(1), Ui, *(T.data) );

    BLAS::prod( real(-1), W, BLAS::adjoint( Vi ), real(1), blas_mat< real >( D ) );
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

    assert( X.is == A->row_is() );
    assert( Y.is == A->col_is() );

    auto  Q0 = matrix( X.is, std::make_shared< BLAS::Matrix< real > >() );
    auto  R0 = matrix(       std::make_shared< BLAS::Matrix< real > >() );
    auto  Q1 = matrix( Y.is, std::make_shared< BLAS::Matrix< real > >() );
    auto  R1 = matrix(       std::make_shared< BLAS::Matrix< real > >() );
    auto  AU = matrix( A->row_is(), A->blas_rmat_A() );
    auto  AV = matrix( A->col_is(), A->blas_rmat_B() );

    // ensure that Q0/Q1 are allocated with correct size before QR
    auto  alloc  = g.alloc_node< alloc_node >( X, T, Y, Q0, Q1, A );
    
    // perform QR for U/V
    auto  qr_U   = g.alloc_node< tsqr_node >( alpha,   X, T, AU, Q0, R0, ntile );
    auto  qr_V   = g.alloc_node< tsqr_node >( real(1), Y,    AV, Q1, R1, ntile );

    qr_U->after( alloc );
    qr_V->after( alloc );

    // determine truncated rank and allocate destination matrices
    auto  Uk     = matrix(       std::make_shared< BLAS::Matrix< real > >() );
    auto  Vk     = matrix(       std::make_shared< BLAS::Matrix< real > >() );
    auto  U      = matrix( X.is, std::make_shared< BLAS::Matrix< real > >() );
    auto  V      = matrix( Y.is, std::make_shared< BLAS::Matrix< real > >() );
    auto  svd    = g.alloc_node< svd_node >( R0, R1, Uk, Vk, U, V );

    svd->after( qr_U );
    svd->after( qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( real(1), Q0, Uk, real(0), U, ntile );
    auto  mul_V  = new tprod_node( real(1), Q1, Vk, real(0), V, ntile );

    g.add_node( mul_U );
    g.add_node( mul_V );
    
    mul_U->after( qr_U );
    mul_U->after( svd );
    mul_V->after( qr_V );
    mul_V->after( svd );

    // assign destination to actual matrix
    auto  assign = g.alloc_node< assign_node >( U, V, A );

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

    if ( X.is.size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis_X = split( X.is, 2 );
        const auto  sis_U = split( U.is, 2 );
        const auto  sis_Q = split( Q.is, 2 );
        auto        Q0    = matrix( sis_Q[0], Q );
        auto        Q1    = matrix( sis_Q[1], Q );
        auto        R0    = shared_matrix( std::make_shared< BLAS::Matrix< real > >() );
        auto        R1    = shared_matrix( std::make_shared< BLAS::Matrix< real > >() );

        // std::cout << "R0 = " << R0.data.get() << std::endl;
        // std::cout << "R1 = " << R1.data.get() << std::endl;
        
        auto  tsqr0 = g.alloc_node< tsqr_node >( alpha, matrix( sis_X[0], X ), T, matrix( sis_U[0], U ), Q0, R0, ntile );
        auto  tsqr1 = g.alloc_node< tsqr_node >( alpha, matrix( sis_X[1], X ), T, matrix( sis_U[1], U ), Q1, R1, ntile );
        auto  qr01  = g.alloc_node< qr_node >( R0, R1, R );

        qr01->after( tsqr0 );
        qr01->after( tsqr1 );
        
        auto  mul0 = new tprod_ip_node( real(1), matrix( sis_Q[0], Q ), R0, ntile );
        auto  mul1 = new tprod_ip_node( real(1), matrix( sis_Q[1], Q ), R1, ntile );

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

    const BLAS::Matrix< real >  Xi( X.data, X.range(), BLAS::Range::all );
    const BLAS::Matrix< real >  Ui( U.data, U.range(), BLAS::Range::all );

    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( *T, "T1.mat", "T1" );
    // DBG::write( Ui, "U1.mat", "U1" );
    
    if ( is_null( T.data ) )
    {
        BLAS::Matrix< real >  XU( Xi.nrows(), Xi.ncols() + Ui.ncols () );
        BLAS::Matrix< real >  XU_X( XU, BLAS::Range::all, BLAS::Range( 0, Xi.ncols()-1 ) );
        BLAS::Matrix< real >  XU_U( XU, BLAS::Range::all, BLAS::Range( Xi.ncols(), XU.ncols()-1 ) );

        BLAS::copy( Xi, XU_X );
        BLAS::copy( Ui, XU_U );
        
        BLAS::qr( XU, *(R.data) );

        // DBG::write( WU, "Q1.mat", "Q1" );
        // DBG::write( *R, "R1.mat", "R1" );
        
        BLAS::Matrix< real >  Qi( *(Q.data), Q.range(), BLAS::Range::all );
        
        BLAS::copy( XU, Qi );
    }// if
    else
    {
        auto                  XT = BLAS::prod( alpha, Xi, *(T.data) );

        // DBG::write( XT, "XT1.mat", "XT1" );
    
        BLAS::Matrix< real >  XU( Xi.nrows(), XT.ncols() + Ui.ncols () );
        BLAS::Matrix< real >  XU_X( XU, BLAS::Range::all, BLAS::Range( 0, XT.ncols()-1 ) );
        BLAS::Matrix< real >  XU_U( XU, BLAS::Range::all, BLAS::Range( XT.ncols(), XU.ncols()-1 ) );

        BLAS::copy( XT, XU_X );
        BLAS::copy( Ui, XU_U );
        
        BLAS::qr( XU, *(R.data) );

        // DBG::write( XU, "Q1.mat", "Q1" );
        // DBG::write( *R, "R1.mat", "R1" );
        
        BLAS::Matrix< real >  Qi( *(Q.data), Q.range(), BLAS::Range::all );
        
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
    BLAS::Matrix< real >  Q(   R0.data->nrows() + R1.data->nrows(), R0.data->ncols() );
    BLAS::Matrix< real >  Q_0( Q, BLAS::Range(                0, R0.data->nrows()-1 ), BLAS::Range::all );
    BLAS::Matrix< real >  Q_1( Q, BLAS::Range( R0.data->nrows(), Q.nrows()-1        ), BLAS::Range::all );
        
    BLAS::copy( *(R0.data), Q_0 );
    BLAS::copy( *(R1.data), Q_1 );

    // DBG::write( Q, "R1.mat", "R1" );
    
    BLAS::qr( Q, *(R.data) );

    // DBG::write( Q, "Q1.mat", "Q1" );

    BLAS::copy( Q_0, *(R0.data) );
    BLAS::copy( Q_1, *(R1.data) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// alloc_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
alloc_node::run_ ( const TTruncAcc & )
{
    const BLAS::Matrix< real >  Xi( X.data, X.range(), BLAS::Range::all );
    const BLAS::Matrix< real >  Yi( Y.data, Y.range(), BLAS::Range::all );
    
    // std::cout << "alloc " << Xi.nrows() << " x " << T.data->ncols() + A->rank() << std::endl;
    // std::cout << "alloc " << Yi.nrows() << " x " << Yi.ncols() + A->rank() << std::endl;
    
    *(Q0.data) = std::move( BLAS::Matrix< real >( Xi.nrows(), T.data->ncols() + A->rank() ) );
    *(Q1.data) = std::move( BLAS::Matrix< real >( Yi.nrows(), Yi.ncols()      + A->rank() ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// svd_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
svd_node::run_ ( const TTruncAcc &  acc )
{
    auto                  Us = BLAS::prod( real(1), *(R0.data), BLAS::adjoint( *(R1.data) ) );
    BLAS::Matrix< real >  Vs;
    BLAS::Vector< real >  Ss;
        
    BLAS::svd( Us, Ss, Vs );
        
    const auto            k  = acc.trunc_rank( Ss );

    BLAS::Matrix< real >  Usk( Us, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
    BLAS::Matrix< real >  Vsk( Vs, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
        
    BLAS::prod_diag( Usk, Ss, k );

    *(Uk.data) = std::move( BLAS::Matrix< real >( Usk.nrows(), Usk.ncols() ) );
    *(Vk.data) = std::move( BLAS::Matrix< real >( Vsk.nrows(), Vsk.ncols() ) );

    BLAS::copy( Usk, *(Uk.data) );
    BLAS::copy( Vsk, *(Vk.data) );
    
    *(U.data) = std::move( BLAS::Matrix< real >( U.is.size(), k ) );
    *(V.data) = std::move( BLAS::Matrix< real >( V.is.size(), k ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// assign_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
assign_node::run_ ( const TTruncAcc & )
{
    A->set_lrmat( *(U.data), *(V.data) );
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
    
    return std::move( refine( new lu_node( & A, 128 ), HLIB::CFG::Arith::max_seq_size, use_single_end_node ) );
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
                                  matrix( id_t('X'), is( 0, X.nrows()-1 ), X ),
                                  matrix( id_t('T'), T ),
                                  matrix( id_t('U'), is( 0, U.nrows()-1 ), U ),
                                  matrix( id_t('Q'), is( 0, X.nrows()-1 ), Q ),
                                  matrix( id_t('R'), R ),
                                  128 ),
                   HLIB::CFG::Arith::max_seq_size,
                   use_single_end_node );
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
                                      matrix( id_t('X'), A->row_is(), X ),
                                      matrix( id_t('T'), T ),
                                      matrix( id_t('U'), A->col_is(), Y ),
                                      A,
                                      128 ),
                   HLIB::CFG::Arith::max_seq_size,
                   use_single_end_node );
}

graph
gen_dag_addlr ( HLIB::BLAS::Matrix< HLIB::real > &                     X,
                std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  T,
                HLIB::BLAS::Matrix< HLIB::real > &                     Y,
                TMatrix *                                              A,
                refine_func_t                                          refine )
{
    return refine( new addlr_node( matrix( id_t('X'), A->row_is(), X ),
                                   matrix( id_t('T'), T ),
                                   matrix( id_t('U'), A->col_is(), Y ),
                                   A,
                                   128 ),
                   HLIB::CFG::Arith::max_seq_size,
                   use_single_end_node );
}

}// namespace dag

}// namespace hlr
