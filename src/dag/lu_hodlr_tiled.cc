//
// Project     : HLR
// Module      : lu_hodlr_tiled.cc
// Description : generate DAG for tiled LU factorization of HODLR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <hpro/matrix/structure.hh>

#include "hlr/arith/blas.hh"
#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/utils/text.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_tiled_v2.hh"
// #include "hlr/tbb/arith_tiled_v2.hh"

namespace hlr { namespace dag {

using Hpro::id_t;

// dense matrix
template < typename value_t >
using  matrix   = blas::Matrix< value_t >;

// dense vector
template < typename value_t >
using  vector   = blas::Vector< value_t >;

// import matrix types
using hlr::matrix::indexset;
using hlr::matrix::tile;
using hlr::matrix::tile_storage;
using hlr::matrix::tiled_lrmatrix;

namespace
{

// dummy indexset for T operations (rank/size unknown during DAG and only object is of interest)
const auto  IS_ONE  = indexset( -1, -1 );
const auto  BIS_ONE = Hpro::TBlockIndexSet( IS_ONE, IS_ONE );

//
// structure to address matrix
//
template < typename matrix_t >
struct matrix_info
{
    const id_t      name;     // id of matrix
    const id_t      id;       // id of matrix
    const indexset  is;       // index set of associated data
    matrix_t        data;     // matrix data

    matrix_info ( matrix_info< matrix_t > &  aM )
            : name( aM.name )
            , id( aM.id )
            , is( aM.is )
            , data( aM.data )
    {}

    matrix_info ( matrix_info< matrix_t > &&  aM )
            : name( aM.name )
            , id( aM.id )
            , is( aM.is )
            , data( std::move( aM.data ) )
    {}

    matrix_info ( const id_t      aname,
                  const indexset  ais,
                  matrix_t        adata )
            : name( aname )
            , id( -1 )
            , is( ais )
            , data( adata )
    {}

    matrix_info ( const id_t      aname,
                  const id_t      aid,
                  const indexset  ais,
                  matrix_t        adata )
            : name( aname )
            , id( aid )
            , is( ais )
            , data( adata )
    {}

    matrix_info ( const id_t      aname,
                  matrix_t        adata )
            : name( aname )
            , id( -1 )
            , is( IS_ONE )
            , data( adata )
    {}
    
    matrix_info ( const indexset  is,
                  matrix_t        adata );
    
    matrix_info ( matrix_t        adata );
    
    matrix_info ( const indexset  ais,
                  matrix_info &   amat )
            : name( amat.name )
            , id( amat.id )
            , is( ais )
            , data( amat.data )
    {}

    virtual ~matrix_info ();

    operator matrix_t () { return data; }
    
    const Hpro::TBlockIndexSet  block_is  () const { return Hpro::TBlockIndexSet( is, IS_ONE ); }
    const mem_block_t           mem_block () const;

    std::string
    to_string ( const size_t  ntile = 0 ) const
    {
        std::ostringstream  os;
        
        if ( name < 100 ) os << char(name);
        else              os << ( name & 0xff );

        if ( id != id_t(-1) ) os << id;

        if (( is != IS_ONE ) && ( ntile != 0 ))
        {
            if ( is.size() <= ntile ) os << Hpro::to_string( "[%d]", is.first() / ntile );
            else                      os << Hpro::to_string( "[%d:%d]", is.first() / ntile, is.last() / ntile );
        }// if

        return os.str();
    }
};

template <>
matrix_info< matrix< double > >::~matrix_info ()
{}

template <>
matrix_info< tile_storage< double > * >::~matrix_info ()
{}

template <>
matrix_info< std::shared_ptr< matrix< double > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< std::shared_ptr< tile_storage< double > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< tile_storage< double > * >::matrix_info ( const indexset          ais,
                                                     tile_storage< double > *  adata )
        : name( id_t(adata) )
        , id( -1 )
        , is( ais )
        , data( adata )
{}

// template <>
// matrix_info< std::shared_ptr< tile_storage< double > > >::matrix_info ( const indexset                           is,
//                                                                       std::shared_ptr< tile_storage< double > >  adata )
//         : name( id_t(adata.get()) )
//         , id( -1 )
//         , is( is )
//         , data( adata )
// {}

template <>
matrix_info< std::shared_ptr< matrix< double > > >::matrix_info ( std::shared_ptr< matrix< double > >  adata )
        : name( id_t(adata.get()) )
        , id( -1  )
        , is( IS_ONE )
        , data( adata )
{}

//  { return { name, block_is() }; }
template <>
const mem_block_t
matrix_info< tile_storage< double > * >::mem_block () const
{
    return { id_t(data), this->block_is() };
}

template <>
const mem_block_t
matrix_info< std::shared_ptr< tile_storage< double > > >::mem_block () const
{
    return { id_t(data.get()), this->block_is() };
}

template <>
const mem_block_t
matrix_info< std::shared_ptr< matrix< double > > >::mem_block () const
{
    return { id_t(data.get()), this->block_is() };
}


using dense_matrix        = matrix_info< matrix< double > >;
using tiled_matrix        = matrix_info< tile_storage< double > * >;
using shared_matrix       = matrix_info< std::shared_ptr< matrix< double > > >;
using shared_tiled_matrix = matrix_info< std::shared_ptr< tile_storage< double > > >;

////////////////////////////////////////////////////////////////////////////////
//
// auxiliary functions
//
////////////////////////////////////////////////////////////////////////////////

//
// split given indexset into <n> subsets
//
inline
std::vector< indexset >
split ( const indexset &  is,
        const size_t       n )
{
    if ( n == 2 )
    {
        const indexset  is0( is.first(), is.first() + is.size() / 2 - 1 );
        const indexset  is1( is0.last() + 1, is.last() );

        return { std::move(is0), std::move(is1) };
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

// identifiers for memory blocks
constexpr id_t  NAME_A    = 'A';
constexpr id_t  NAME_L    = 'A';
constexpr id_t  NAME_U    = 'A';
constexpr id_t  NAME_NONE = '0';

//
// compute A = LU
//
struct lu_node : public node
{
    Hpro::TMatrix< double > *     A;
    const size_t  ntile;
    
    lu_node ( Hpro::TMatrix< double > *     aA,
              const size_t  antile )
            : A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { NAME_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { NAME_L, A->block_is() }, { NAME_U, A->block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(lu_node); }
};

//
// solve X U = M with upper triangular U and M given
//
struct trsmu_node : public node
{
    const Hpro::TMatrix< double > *  U;
    tiled_matrix     X;
    tiled_matrix     M;
    const size_t     ntile;
    
    trsmu_node ( const Hpro::TMatrix< double > *  aU,
                 tiled_matrix       aX,
                 tiled_matrix       aM,
                 const size_t     antile )
            : U( aU )
            , X( aX )
            , M( aM )
            , ntile( antile )
    { init(); }
    
    virtual std::string  to_string () const
    {
        return X.to_string( ntile ) + Hpro::to_string( " = trsmu( %c%d, ", char(NAME_U), U->id() ) + M.to_string( ntile ) + " )";
    }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { NAME_U, U->block_is() }, M.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { X.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(trsmu_node); }
};

//
// solve L X = M with lower triangular L and M given
//
struct trsml_node : public node
{
    const Hpro::TMatrix< double > *  L;
    tiled_matrix     X;
    tiled_matrix     M;
    const size_t     ntile;

    trsml_node ( const Hpro::TMatrix< double > *  aL,
                 tiled_matrix       aX,
                 tiled_matrix       aM,
                 const size_t     antile )
            : L( aL )
            , X( aX )
            , M( aM )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return X.to_string( ntile ) + Hpro::to_string( " = trsml( %c%d, ", char(NAME_L), L->id() ) + M.to_string( ntile ) + " )";
    }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { NAME_L, L->block_is() }, M.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { X.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(trsml_node); }
};
    
//
// compute A := A - U·T·V^H
//
struct addlr_node : public node
{
    tiled_matrix   U;
    shared_matrix  T;
    tiled_matrix   V;
    Hpro::TMatrix< double > *      A;
    const size_t   ntile;

    addlr_node ( tiled_matrix   aU,
                 shared_matrix  aT,
                 tiled_matrix   aV,
                 Hpro::TMatrix< double > *      aA,
                 const size_t   antile )
            : U( aU )
            , T( aT )
            , V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "addlr(" + U.to_string( ntile ) + "×" + T.to_string() + "×" + V.to_string( ntile ) + " + \n" +
                 Hpro::to_string( "A%d", A->id() ) + ")" );
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { U.mem_block(), V.mem_block(), T.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { { NAME_A, A->block_is() } }; }
    virtual size_t              mem_size_    () const { return sizeof(addlr_node); }
};

//
// compute T := A^H · B
//
struct dot_node : public node
{
    tiled_matrix   A;
    tiled_matrix   B;
    shared_matrix  T;
    const size_t   ntile;

    dot_node ( tiled_matrix   aA,
               tiled_matrix   aB,
               shared_matrix  aT,
               const size_t   antile )
            : A( aA )
            , B( aB )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return T.to_string() + " = dot(" + A.to_string( ntile ) + "×" + B.to_string( ntile ) + " )";
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
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
    const double     alpha;
    matrixX_t      X;
    shared_matrix  T;
    const double     beta;
    matrixY_t      Y;
    const size_t   ntile;

    tprod_node ( const double     aalpha,
                 matrixX_t      aX,
                 shared_matrix  aT,
                 const double     abeta,
                 matrixY_t      aY,
                 const size_t   antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , beta( abeta )
            , Y( aY )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const 
    {
        return "Tprod(" + X.to_string( ntile ) + "×" + T.to_string() + "+" + Y.to_string( ntile ) + ")";
    }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( beta == double(0) ) return { X.mem_block(), T.mem_block() };
        else                   return { X.mem_block(), T.mem_block(), Y.mem_block() };
    }
    virtual const block_list_t  out_blocks_  () const { return { Y.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tprod_node); }
};

template < typename matrix_t >
struct tprod_ip_node : public node
{
    const double     alpha;
    matrix_t       X;
    shared_matrix  T;
    const size_t   ntile;

    tprod_ip_node ( const double     aalpha,
                    matrix_t       aX,
                    shared_matrix  aT,
                    const size_t   antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return "Tprod_ip(" + X.to_string( ntile ) + "×" + T.to_string() + ")"; }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
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
    shared_matrix  T0;
    shared_matrix  T1;
    shared_matrix  T;

    tadd_node ( shared_matrix  aT0,
                shared_matrix  aT1,
                shared_matrix  aT )
            : T0( aT0 )
            , T1( aT1 )
            , T( aT )
    { init(); }

    virtual std::string  to_string () const { return T.to_string() + " = Tadd(" + T0.to_string() + "+" + T1.to_string() + ")"; }
    virtual std::string  color     () const { return "8ae234"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { T0.mem_block(), T1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { T.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tadd_node); }
};

//
// truncate α X T Y^H + U(A) V(A)^H
//
struct truncate_node : public node
{
    const double                alpha;
    tiled_matrix              X;
    shared_matrix             T;
    tiled_matrix              Y;
    tiled_matrix              U;
    tiled_matrix              V;
    tiled_lrmatrix< double > *  A;
    const size_t              ntile;

    truncate_node ( const double                aalpha,
                    tiled_matrix              aX,
                    shared_matrix             aT,
                    tiled_matrix              aY,
                    tiled_matrix              aU,
                    tiled_matrix              aV,
                    tiled_lrmatrix< double > *  aA,
                    const size_t              antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , Y( aY )
            , U( aU )
            , V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "trunc( " + X.to_string( ntile ) + "×" + T.to_string() + "×" + Y.to_string( ntile ) + ",\n " +
                 U.to_string( ntile ) + "×" + V.to_string( ntile ) + " )" );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { X.mem_block(), T.mem_block(), Y.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate_node); }
};

//
// QR factorization of [αX·T,U]
//
struct tsqr_node : public node
{
    const double           alpha;
    tiled_matrix         X;
    shared_matrix        T;
    tiled_matrix         U;
    shared_tiled_matrix  Q;
    shared_matrix        R;
    const size_t         ntile;

    tsqr_node ( const double           aalpha,
                tiled_matrix         aX,
                shared_matrix        aT,
                tiled_matrix         aU,
                shared_tiled_matrix  aQ,
                shared_matrix        aR,
                const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , U( aU )
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    tsqr_node ( const double           aalpha,
                tiled_matrix         aX,
                tiled_matrix         aU,
                shared_tiled_matrix  aQ,
                shared_matrix        aR,
                const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( shared_matrix( NAME_NONE, std::shared_ptr< matrix< double > >() ) ) // T = 0
            , U( aU )
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( is_null( T.data ) )
            return Q.to_string( ntile ) + ", " + R.to_string() + " = tsqr( " + X.to_string( ntile ) + ", " + U.to_string( ntile ) + " )";
        else
            return Q.to_string( ntile ) + ", " + R.to_string() + " = tsqr( " + X.to_string( ntile ) + "×" + T.to_string() + ", " + U.to_string( ntile ) + " )";
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
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

    virtual std::string  to_string () const
    {
        return R.to_string() + " = qr( " + R0.to_string() + ", " + R1.to_string() + " )";
    }
    virtual std::string  color     () const { return "c17d11"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { R0.mem_block(), R1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { R0.mem_block(), R1.mem_block(), R.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(qr_node); }
};

//
// compute truncation rank
//
struct svd_node : public node
{
    shared_matrix  R0;
    shared_matrix  R1;
    shared_matrix  Uk;
    shared_matrix  Vk;

    svd_node ( shared_matrix  aR0,
               shared_matrix  aR1,
               shared_matrix  aUk,
               shared_matrix  aVk )
            : R0( aR0 )
            , R1( aR1 )
            , Uk( aUk )
            , Vk( aVk )
    { init(); }

    virtual std::string  to_string () const { return Uk.to_string() + ", " + Vk.to_string() + " = svd( " + R0.to_string() + ", " + R1.to_string() + " )"; }
    virtual std::string  color     () const { return "ad7fa8"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return { R0.mem_block(), R1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { Uk.mem_block(), Vk.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(svd_node); }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
lu_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( tile_size, A ) )
    {
        auto  BA  = ptrcast( A, Hpro::TBlockMatrix< double > );
        auto  BL  = BA;
        auto  BU  = BA;
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< double > );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< double > );

        assert(( BA->nblock_rows() == 2 ) && ( BA->nblock_cols() == 2 ));
        assert( is_tiled_lowrank( A10 ));
        assert( is_tiled_lowrank( A01 ));
            
        auto  lu_00    = g.alloc_node< lu_node >( BA->block( 0, 0 ), ntile );
        auto  solve_10 = g.alloc_node< trsmu_node >( BU->block( 0, 0 ),
                                                     tiled_matrix( NAME_L, A10->id(), A10->col_is(), & A10->V() ),
                                                     tiled_matrix( NAME_A, A10->id(), A10->col_is(), & A10->V() ),
                                                     ntile );
        auto  solve_01 = g.alloc_node< trsml_node >( BL->block( 0, 0 ),
                                                     tiled_matrix( NAME_U, A01->id(), A01->row_is(), & A01->U() ),
                                                     tiled_matrix( NAME_A, A01->id(), A01->row_is(), & A01->U() ),
                                                     ntile );
        auto  T        = std::make_shared< matrix< double > >();
        auto  tsmul    = g.alloc_node< dot_node >( tiled_matrix( NAME_L, A10->id(), A10->col_is(), & A10->V() ),
                                                   tiled_matrix( NAME_U, A01->id(), A01->row_is(), & A01->U() ),
                                                   shared_matrix( T ),
                                                   ntile );
        auto  addlr    = g.alloc_node< addlr_node >( tiled_matrix( NAME_L, A10->id(), A10->row_is(), & A10->U() ),
                                                     shared_matrix( T ),
                                                     tiled_matrix( NAME_U, A01->id(), A01->col_is(), & A01->V() ),
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
lu_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // hlr::tbb::tiled2::hodlr::lu< double >( A, acc, ntile );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
trsmu_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( is_blocked( U ) && ! is_small( tile_size, U ) )
    {
        //
        //  ⎡ R_00^T │        ⎤ ⎡X_0⎤   ⎡ R_00^T            │        ⎤ ⎡X_0⎤   ⎡M_0⎤
        //  ⎢────────┼────────⎥ ⎢───⎥ = ⎢───────────────────┼────────⎥ ⎢───⎥ = ⎢───⎥
        //  ⎣ R_01^T │ R_11^T ⎦ ⎣X_1⎦   ⎣ V(R_01) U(R_01)^T │ R_11^T ⎦ ⎣X_1⎦   ⎣M_1⎦
        //
        
        auto  BU  = cptrcast( U, Hpro::TBlockMatrix< double > );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = ptrcast( const_cast< Hpro::TBlockMatrix< double > * >( BU )->block( 0, 1 ), tiled_lrmatrix< double > );
        auto  U11 = BU->block( 1, 1 );

        assert( is_tiled_lowrank( U01 ) );
        
        const auto  is0 = U00->col_is();
        const auto  is1 = U11->col_is();

        auto  solve_00 = g.alloc_node< trsmu_node >( U00,
                                                     tiled_matrix( is0, X ),
                                                     tiled_matrix( is0, M ),
                                                     ntile );
        auto  T        = std::make_shared< matrix< double > >();
        auto  tsmul    = g.alloc_node< dot_node >( tiled_matrix( NAME_U, U01->id(), U01->row_is(), & U01->U() ),
                                                   tiled_matrix( is0, X ),
                                                   shared_matrix( T ),
                                                   ntile );
        auto  tprod    = new tprod_node( double(-1),
                                         tiled_matrix( NAME_U, U01->id(), U01->col_is(), & U01->V() ),
                                         shared_matrix( T ),
                                         double(1),
                                         tiled_matrix( is1, M ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsmu_node >( U11,
                                                     tiled_matrix( is1, X ),
                                                     tiled_matrix( is1, M ),
                                                     ntile );

        g.add_node( tprod );
        
        tsmul->after( solve_00 );
        tprod->after( tsmul );
        solve_11->after( tprod );
    }// if

    g.finalize();
    
    return g;
}

void
trsmu_node::run_ ( const Hpro::TTruncAcc & )
{
    hlr::seq::tiled2::hodlr::trsmuh( U, *(X.data), ntile );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
trsml_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( is_blocked( L ) && ! is_small( tile_size, L ) )
    {
        //
        //  ⎡ L_00 │      ⎤ ⎡X_0⎤   ⎡ L_00              │      ⎤ ⎡X_0⎤   ⎡M_0⎤
        //  ⎢──────┼──────⎥ ⎢───⎥ = ⎢───────────────────┼──────⎥ ⎢───⎥ = ⎢───⎥
        //  ⎣ L_10 │ L_11 ⎦ ⎣X_1⎦   ⎣ U(L_01) V(L_01)^T │ L_11 ⎦ ⎣X_1⎦   ⎣M_1⎦
        //
        
        auto  BL  = cptrcast( L, Hpro::TBlockMatrix< double > );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = ptrcast( const_cast< Hpro::TBlockMatrix< double > * >( BL )->block( 1, 0 ), tiled_lrmatrix< double > );
        auto  L11 = BL->block( 1, 1 );

        assert( is_tiled_lowrank( L10 ) );
        
        const auto  is0 = L00->row_is();
        const auto  is1 = L11->row_is();
            
        auto  solve_00 = g.alloc_node< trsml_node >( L00,
                                                     tiled_matrix( is0, X ),
                                                     tiled_matrix( is0, M ),
                                                     ntile );
        auto  T        = std::make_shared< matrix< double > >();
        auto  tsmul    = g.alloc_node< dot_node >( tiled_matrix( NAME_L, L10->id(), L10->col_is(), & L10->V() ),
                                                   tiled_matrix( is0, X ),
                                                   shared_matrix( T ),
                                                   ntile );
        auto  tprod    = new tprod_node( double(-1),
                                         tiled_matrix( NAME_L, L10->id(), L10->row_is(), & L10->U() ),
                                         shared_matrix( T ),
                                         double(1),
                                         tiled_matrix( is1, M ),
                                         ntile );
        auto  solve_11 = g.alloc_node< trsml_node >( L11,
                                                     tiled_matrix( is1, X ),
                                                     tiled_matrix( is1, M ),
                                                     ntile );

        g.add_node( tprod );
        
        tsmul->after( solve_00 );
        tprod->after( tsmul );
        solve_11->after( tprod );
    }// if

    g.finalize();
    
    return g;
}

void
trsml_node::run_ ( const Hpro::TTruncAcc & )
{
    hlr::seq::tiled2::hodlr::trsml( L, *(X.data), ntile );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
dot_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    assert( A.is.size() == B.is.size() );

    if ( A.is.size() > tile_size )
    {
        const auto  sis_A = split( A.is, 2 );
        const auto  sis_B = split( B.is, 2 );

        auto  T0     = std::make_shared< matrix< double > >();
        auto  T1     = std::make_shared< matrix< double > >();
        auto  tsmul0 = g.alloc_node< dot_node  >( tiled_matrix( sis_A[0], A ),
                                                  tiled_matrix( sis_B[0], B ),
                                                  shared_matrix( T0 ),
                                                  ntile );
        auto  tsmul1 = g.alloc_node< dot_node  >( tiled_matrix( sis_A[1], A ),
                                                  tiled_matrix( sis_B[1], B ),
                                                  shared_matrix( T1 ),
                                                  ntile );
        auto  add    = g.alloc_node< tadd_node >( shared_matrix( T0 ),
                                                  shared_matrix( T1 ),
                                                  T );

        add->after( tsmul0 );
        add->after( tsmul1 );
    }// if

    g.finalize();

    return g;
}

void
dot_node::run_ ( const Hpro::TTruncAcc & )
{
    *(T.data) = hlr::seq::tiled2::dot( A.is, *(A.data), *(B.data), ntile );

    HLR_LOG( 5, "         dot :       " + hlr::seq::tiled2::isstr( A.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(T.data) ) ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tprod_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename matrixX_t, typename matrixY_t >
local_graph
tprod_node< matrixX_t, matrixY_t >::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    assert( X.is.size() == Y.is.size() );

    if ( X.is.size() > tile_size )
    {
        const auto  sis_X = split( X.is, 2 );
        const auto  sis_Y = split( Y.is, 2 );

        g.alloc_node< tprod_node >( alpha, matrix_info( sis_X[0], X ), T, beta, matrix_info( sis_Y[0], Y ), ntile );
        g.alloc_node< tprod_node >( alpha, matrix_info( sis_X[1], X ), T, beta, matrix_info( sis_Y[1], Y ), ntile );
    }// if

    g.finalize();

    return g;
}

template < typename matrixX_t, typename matrixY_t >
void
tprod_node< matrixX_t, matrixY_t >::run_ ( const Hpro::TTruncAcc & )
{
    hlr::seq::tiled2::tprod( X.is, alpha, *(X.data), *(T.data), beta, *(Y.data), ntile );
}

template < typename matrix_t >
local_graph
tprod_ip_node< matrix_t >::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( X.is.size() > tile_size )
    {
        const auto  sis_X = split( X.is, 2 );

        g.alloc_node< tprod_ip_node >( alpha, matrix_info( sis_X[0], X ), T, ntile );
        g.alloc_node< tprod_ip_node >( alpha, matrix_info( sis_X[1], X ), T, ntile );
    }// if

    g.finalize();

    return g;
}

template <>
void
tprod_ip_node< shared_tiled_matrix >::run_ ( const Hpro::TTruncAcc & )
{
    hlr::seq::tiled2::tprod( X.is, alpha, *(X.data), *(T.data), ntile );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// addlr_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
addlr_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( tile_size, A ) )
    {
        auto  BA  = ptrcast( A, Hpro::TBlockMatrix< double > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< double > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< double > );
        auto  A11 = BA->block( 1, 1 );
        
        assert( is_tiled_lowrank( A01 ) );
        assert( is_tiled_lowrank( A10 ) );
        
        g.alloc_node< addlr_node    >( tiled_matrix( A00->row_is(), U ),
                                       T,
                                       tiled_matrix( A00->col_is(), V ),
                                       A00,
                                       ntile );
        g.alloc_node< truncate_node >( double(-1),
                                       tiled_matrix( A01->row_is(), U ),
                                       T,
                                       tiled_matrix( A01->col_is(), V ),
                                       tiled_matrix( NAME_A, A01->id(), A01->row_is(), & A01->U() ),
                                       tiled_matrix( NAME_A, A01->id(), A01->col_is(), & A01->V() ),
                                       A01,
                                       ntile );
        g.alloc_node< truncate_node >( double(-1),
                                       tiled_matrix( A10->row_is(), U ),
                                       T,
                                       tiled_matrix( A10->col_is(), V ),
                                       tiled_matrix( NAME_A, A10->id(), A10->row_is(), & A10->U() ),
                                       tiled_matrix( NAME_A, A10->id(), A10->col_is(), & A10->V() ),
                                       A10,
                                       ntile );
        g.alloc_node< addlr_node    >( tiled_matrix( A11->row_is(), U ),
                                       T,
                                       tiled_matrix( A11->col_is(), V ),
                                       A11,
                                       ntile );
    }// if

    g.finalize();

    return g;
}

void
addlr_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    Hpro::TScopedLock  lock( *A );
    
    hlr::seq::tiled2::hodlr::addlr( *(U.data), *(T.data), *(V.data), A, acc, ntile );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tadd_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
tadd_node::run_ ( const Hpro::TTruncAcc & )
{
    assert(( T0.data->nrows() == T1.data->nrows() ) && ( T0.data->ncols() == T1.data->ncols() ));
    
    if (( T.data->nrows() != T0.data->nrows() ) || ( T.data->ncols() != T0.data->ncols() ))
        *(T.data) = matrix< double >( T0.data->nrows(), T0.data->ncols() );
    
    blas::add( double(1), *(T0.data), *(T.data) );
    blas::add( double(1), *(T1.data), *(T.data) );

    HLR_LOG( 5, "        tadd :             = " + hlr::seq::tiled2::normstr( blas::normF( *(T.data) ) ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// truncate_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
truncate_node::refine_ ( const size_t )
{
    local_graph  g;

    assert( X.is == U.is );
    assert( Y.is == V.is );

    // auto  Q0 = shared_tiled_matrix( X.is, std::make_shared< tile_storage< double > >() );
    // auto  R0 = shared_matrix(             std::make_shared< matrix< double > >() );
    // auto  Q1 = shared_tiled_matrix( Y.is, std::make_shared< tile_storage< double > >() );
    // auto  R1 = shared_matrix(             std::make_shared< matrix< double > >() );

    // // perform QR for U/V
    // auto  qr_U   = g.alloc_node< tsqr_node >( alpha,   X, T, U, Q0, R0, ntile );
    // auto  qr_V   = g.alloc_node< tsqr_node >( double(1), Y,    V, Q1, R1, ntile );

    // // determine truncated rank and allocate destination matrices
    // auto  Uk     = shared_matrix( std::make_shared< matrix< double > >() );
    // auto  Vk     = shared_matrix( std::make_shared< matrix< double > >() );
    // auto  svd    = g.alloc_node< svd_node >( R0, R1, Uk, Vk );

    // svd->after( qr_U );
    // svd->after( qr_V );

    // // compute final result
    // auto  mul_U  = new tprod_node( double(1), Q0, Uk, double(0), U, ntile );
    // auto  mul_V  = new tprod_node( double(1), Q1, Vk, double(0), V, ntile );

    // g.add_node( mul_U );
    // g.add_node( mul_V );
    
    // mul_U->after( qr_U );
    // mul_U->after( svd );
    // mul_V->after( qr_V );
    // mul_V->after( svd );

    g.finalize();

    return g;
}

void
truncate_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    Hpro::TScopedLock  lock( *A );
    
    auto [ U2, V2 ] = hlr::seq::tiled2::truncate( U.is, V.is, alpha, *(X.data), *(T.data), *(Y.data), *(U.data), *(V.data), acc, ntile );

    *(U.data) = std::move( U2 );
    *(V.data) = std::move( V2 );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tsqr_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
tsqr_node::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( X.is.size() > tile_size )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis_X = split( X.is, 2 );
        const auto  sis_U = split( U.is, 2 );
        const auto  sis_Q = split( Q.is, 2 );
        auto        Q0    = matrix_info( sis_Q[0], Q );
        auto        Q1    = matrix_info( sis_Q[1], Q );
        auto        R0    = matrix_info( std::make_shared< matrix< double > >() );
        auto        R1    = matrix_info( std::make_shared< matrix< double > >() );

        // std::cout << "R0 = " << R0.data.get() << std::endl;
        // std::cout << "R1 = " << R1.data.get() << std::endl;
        
        auto  tsqr0 = g.alloc_node< tsqr_node >( alpha, matrix_info( sis_X[0], X ), T, matrix_info( sis_U[0], U ), Q0, R0, ntile );
        auto  tsqr1 = g.alloc_node< tsqr_node >( alpha, matrix_info( sis_X[1], X ), T, matrix_info( sis_U[1], U ), Q1, R1, ntile );
        auto  qr01  = g.alloc_node< qr_node >( R0, R1, R );

        qr01->after( tsqr0 );
        qr01->after( tsqr1 );
        
        auto  mul0 = new tprod_ip_node( double(1), matrix_info( sis_Q[0], Q ), R0, ntile );
        auto  mul1 = new tprod_ip_node( double(1), matrix_info( sis_Q[1], Q ), R1, ntile );

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
tsqr_node::run_ ( const Hpro::TTruncAcc & )
{
    // DBG::write( Xi, "X1.mat", "X1" );
    // DBG::write( *T, "T1.mat", "T1" );
    // DBG::write( Ui, "U1.mat", "U1" );
    
    if ( is_null( T.data ) )
    {
        // {
        //     auto  DX = hlr::matrix::to_dense( *(X.data) );
        //     auto  DU = hlr::matrix::to_dense( *(U.data) );
            
        //     DBG::write( DX, "Y1.mat", "Y1" );
        //     DBG::write( DU, "V1.mat", "V1" );
        // }
    
        assert( X.data->contains( X.is ) && U.data->contains( U.is ) );

        const auto      X_is = X.data->at( X.is );
        const auto      U_is = U.data->at( U.is );
        matrix< double >  XU( X_is.nrows(), X_is.ncols() + U_is.ncols () );
        matrix< double >  XU_X( XU, blas::range::all, blas::range( 0, X_is.ncols()-1 ) );
        matrix< double >  XU_U( XU, blas::range::all, blas::range( X_is.ncols(), XU.ncols()-1 ) );

        blas::copy( X_is, XU_X );
        blas::copy( U_is, XU_U );

        blas::qr( XU, *(R.data) );

        HLR_LOG( 5, "tsqr  :          Q , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( XU ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(R.data) ) ) );
        
        (*(Q.data))[ X.is ] = std::move( XU );
    }// if
    else
    {
        // {
        //     auto  DX = hlr::matrix::to_dense( *(X.data) );
        //     auto  DU = hlr::matrix::to_dense( *(U.data) );
            
        //     DBG::write( DX,        "X1.mat", "X1" );
        //     DBG::write( *(T.data), "T1.mat", "T1" );
        //     DBG::write( DU,        "U1.mat", "U1" );
        // }
    
        assert( X.data->contains( X.is ) && U.data->contains( U.is ) );

        const auto      X_is = X.data->at( X.is );
        const auto      U_is = U.data->at( U.is );
        auto            W    = blas::prod( alpha, X_is, *(T.data) );
        matrix< double >  WU( W.nrows(), W.ncols() + U_is.ncols () );
        matrix< double >  WU_W( WU, blas::range::all, blas::range( 0, W.ncols()-1 ) );
        matrix< double >  WU_U( WU, blas::range::all, blas::range( W.ncols(), WU.ncols()-1 ) );

        blas::copy( W,    WU_W );
        blas::copy( U_is, WU_U );

        blas::qr( WU, *(R.data) );

        HLR_LOG( 5, "tsqr  :          Q , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( WU ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(R.data) ) ) );
        
        (*(Q.data))[ X.is ] = std::move( WU );
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// qr_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
qr_node::run_ ( const Hpro::TTruncAcc & )
{
    // Q = | R0 |
    //     | R1 |
    matrix< double >  Q(   R0.data->nrows() + R1.data->nrows(), R0.data->ncols() );
    matrix< double >  Q_0( Q, blas::range(                0, R0.data->nrows()-1 ), blas::range::all );
    matrix< double >  Q_1( Q, blas::range( R0.data->nrows(), Q.nrows()-1        ), blas::range::all );
        
    blas::copy( *(R0.data), Q_0 );
    blas::copy( *(R1.data), Q_1 );

    // DBG::write( Q, "R1.mat", "R1" );
    
    blas::qr( Q, *(R.data) );

    // DBG::write( Q, "Q1.mat", "Q1" );

    blas::copy( Q_0, *(R0.data) );
    blas::copy( Q_1, *(R1.data) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// svd_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
svd_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    auto            Us = blas::prod( double(1), *(R0.data), blas::adjoint( *(R1.data) ) );
    matrix< double >  Vs;
    vector< double >  Ss;
        
    blas::svd( Us, Ss, Vs );
        
    const auto      k  = acc.trunc_rank( Ss );

    // for ( size_t  i = 0; i < k; ++i )
    //     std::cout << Ss(i) << std::endl;
    
    matrix< double >  Usk( Us, blas::range::all, blas::range( 0, k-1 ) );
    matrix< double >  Vsk( Vs, blas::range::all, blas::range( 0, k-1 ) );
        
    blas::prod_diag( Usk, Ss, k );

    *(Uk.data) = std::move( matrix< double >( Usk.nrows(), Usk.ncols() ) );
    *(Vk.data) = std::move( matrix< double >( Vsk.nrows(), Vsk.ncols() ) );

    blas::copy( Usk, *(Uk.data) );
    blas::copy( Vsk, *(Vk.data) );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_lu_hodlr_tiled ( Hpro::TMatrix< double > &  A,
                         const size_t               ntile,
                         refine_func_t              refine )
{
    // matrix< double >  X( A.nrows(), 16 );
    // matrix< double >  U( A.nrows(), 16 );
    // auto                  T = std::make_shared< matrix< double > >();
    // auto                  Q = std::make_shared< matrix< double > >();
    // auto                  R = std::make_shared< matrix< double > >();
    
    return refine( new lu_node( & A, ntile ), ntile, use_single_end_node );
}

//
// compute DAG for TSQR( X·T, U )
//
graph
gen_dag_tsqr ( const size_t                                 n,
               tile_storage< double > &                     X,
               std::shared_ptr< matrix< double > > &        T,
               tile_storage< double > &                     U,
               std::shared_ptr< tile_storage< double > > &  Q,
               std::shared_ptr< matrix< double > > &        R,
               refine_func_t                                refine )
{
    return refine( new tsqr_node( 1,
                                  matrix_info( id_t('X'), Hpro::is( 0, n-1 ), & X ),
                                  matrix_info( id_t('T'), T ),
                                  matrix_info( id_t('U'), Hpro::is( 0, n-1 ), & U ),
                                  matrix_info( id_t('Q'), Hpro::is( 0, n-1 ), Q ),
                                  matrix_info( id_t('R'), R ),
                                  32 ),
                   32,
                   use_single_end_node );
}

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_truncate ( tile_storage< double > &               X,
                   std::shared_ptr< matrix< double > > &  T,
                   tile_storage< double > &               Y,
                   tiled_lrmatrix< double > *             A,
                   refine_func_t                          refine )
{
    return refine( new truncate_node( double(1),
                                      matrix_info( id_t('X'), A->row_is(), & X ),
                                      matrix_info( id_t('T'), T ),
                                      matrix_info( id_t('U'), A->col_is(), & Y ),
                                      matrix_info( A->row_is(), & A->U() ),
                                      matrix_info( A->col_is(), & A->V() ),
                                      A,
                                      32 ),
                   32,
                   use_single_end_node );
}

graph
gen_dag_addlr ( tile_storage< double > &               X,
                std::shared_ptr< matrix< double > > &  T,
                tile_storage< double > &               Y,
                Hpro::TMatrix< double > *              A,
                refine_func_t                          refine )
{
    return refine( new addlr_node( matrix_info( id_t('X'), A->row_is(), & X ),
                                   matrix_info( id_t('T'), T ),
                                   matrix_info( id_t('U'), A->col_is(), & Y ),
                                   A,
                                   32 ),
                   32,
                   use_single_end_node );
}

}// namespace dag

}// namespace hlr
