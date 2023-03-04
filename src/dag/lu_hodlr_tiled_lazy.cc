//
// Project     : HLR
// Module      : lu_hodlr_tiled_laz.cc
// Description : generate DAG for lazy, tiled LU factorization of HODLR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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
#include "hlr/utils/text.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_tiled_v2.hh"

namespace hlr { namespace dag {

using hpro::id_t;

// import matrix types
using hlr::matrix::indexset;
using hlr::matrix::block_indexset;
using hlr::matrix::tile;
using hlr::matrix::tile_storage;
using hlr::matrix::tiled_lrmatrix;

namespace
{

// dummy indexset for T operations (rank/size unknown during DAG and only object is of interest)
const auto  IS_ONE  = indexset( -1, -1 );
const auto  BIS_ONE = block_indexset( IS_ONE, IS_ONE );

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

    matrix_info ( const matrix_info< matrix_t > &  aM )
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
    
    const block_indexset  block_is  () const { return block_indexset( is, IS_ONE ); }
    const mem_block_t     mem_block () const;

    std::string
    to_string ( const size_t  ntile = 0 ) const
    {
        std::ostringstream  os;
        
        if ( name < 100 ) os << char(name);
        else              os << ( name & 0xff );

        if ( id != id_t(-1) ) os << id;

        if (( is != IS_ONE ) && ( ntile != 0 ))
        {
            if ( is.size() <= ntile ) os << superscript( is.first() / ntile );
            else                      os << superscript( is.first() / ntile ) << "ʼ" << superscript( is.last() / ntile );
        }// if

        return os.str();
    }
};

template <>
matrix_info< blas::matrix< double > >::~matrix_info ()
{}

template <>
matrix_info< tile_storage< double > * >::~matrix_info ()
{}

template <>
matrix_info< std::shared_ptr< blas::matrix< double > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< std::shared_ptr< tile_storage< double > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< std::shared_ptr< tile_storage< double > > >::matrix_info ( const indexset                           ais,
                                                                      std::shared_ptr< tile_storage< double > >  adata )
        : name( id_t(adata.get()) )
        , id( -1 )
        , is( ais )
        , data( adata )
{}

template <>
matrix_info< std::shared_ptr< blas::matrix< double > > >::matrix_info ( std::shared_ptr< blas::matrix< double > >  adata )
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
    return { id_t(data), block_is() };
}

template <>
const mem_block_t
matrix_info< std::shared_ptr< tile_storage< double > > >::mem_block () const
{
    return { id_t(data.get()), block_is() };
}

template <>
const mem_block_t
matrix_info< std::shared_ptr< blas::matrix< double > > >::mem_block () const
{
    return { id_t(data.get()), block_is() };
}


using dense_matrix        = matrix_info< blas::matrix< double > >;
using tiled_matrix        = matrix_info< tile_storage< double > * >;
using shared_matrix       = matrix_info< std::shared_ptr< blas::matrix< double > > >;
using shared_tiled_matrix = matrix_info< std::shared_ptr< tile_storage< double > > >;

//
// update U·T·V^H to low-rank matrix
//
struct lr_update_t
{
    tiled_matrix   U;
    shared_matrix  T;
    tiled_matrix   V;
};

//
// mapping of updates to matrix blocks
//
using  update_list_t = std::list< node * >;
using  update_map_t  = std::map< id_t, update_list_t >;

std::mutex  upd_mtx;

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
    Hpro::TMatrix< double > *       A;
    update_map_t &  update_map;
    const size_t    ntile;
    
    lu_node ( Hpro::TMatrix< double > *       aA,
              update_map_t &  aupdate_map,
              const size_t    antile )
            : A( aA )
            , update_map( aupdate_map )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return hpro::to_string( "lu( %d )", A->id() ); }
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
    update_map_t &   update_map;
    const size_t     ntile;
    
    trsmu_node ( const Hpro::TMatrix< double > *  aU,
                 tiled_matrix     aX,
                 tiled_matrix     aM,
                 update_map_t &   aupdate_map,
                 const size_t     antile )
            : U( aU )
            , X( aX )
            , M( aM )
            , update_map( aupdate_map )
            , ntile( antile )
    { init(); }
    
    virtual std::string  to_string () const
    {
        return X.to_string( ntile ) + hpro::to_string( " = trsmu( %c%d, ", char(NAME_U), U->id() ) + M.to_string( ntile ) + " )";
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
    update_map_t &   update_map;
    const size_t     ntile;

    trsml_node ( const Hpro::TMatrix< double > *  aL,
                 tiled_matrix     aX,
                 tiled_matrix     aM,
                 update_map_t &   aupdate_map,
                 const size_t     antile )
            : L( aL )
            , X( aX )
            , M( aM )
            , update_map( aupdate_map )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return X.to_string( ntile ) + hpro::to_string( " = trsml( %c%d, ", char(NAME_L), L->id() ) + M.to_string( ntile ) + " )";
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
    tiled_matrix    U;
    shared_matrix   T;
    tiled_matrix    V;
    Hpro::TMatrix< double > *       A;
    update_map_t &  update_map;
    const size_t    ntile;

    addlr_node ( tiled_matrix    aU,
                 shared_matrix   aT,
                 tiled_matrix    aV,
                 Hpro::TMatrix< double > *       aA,
                 update_map_t &  aupdate_map,
                 const size_t    antile )
            : U( aU )
            , T( aT )
            , V( aV )
            , A( aA )
            , update_map( aupdate_map )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "addlr(" + U.to_string( ntile ) + "×" + T.to_string() + "×" + V.to_string( ntile ) + " + " +
                 hpro::to_string( "A%d", A->id() ) + ")" );
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
        return ( "trunc( " + X.to_string( ntile ) + "×" + T.to_string() + "×" + Y.to_string( ntile ) + ", " +
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
// truncate U₁ V₁^H + U₂ V₂^H
//
struct truncate2_node : public node
{
    shared_tiled_matrix       U1;
    shared_tiled_matrix       V1;
    shared_tiled_matrix       U2;
    shared_tiled_matrix       V2;
    shared_tiled_matrix       U;
    shared_tiled_matrix       V;
    const size_t              ntile;

    truncate2_node ( shared_tiled_matrix   aU1,
                     shared_tiled_matrix   aV1,
                     shared_tiled_matrix   aU2,
                     shared_tiled_matrix   aV2,
                     shared_tiled_matrix   aU,
                     shared_tiled_matrix   aV,
                     const size_t          antile )
            : U1( aU1 )
            , V1( aV1 )
            , U2( aU2 )
            , V2( aV2 )
            , U( aU )
            , V( aV )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "trunc( " +
                 U1.to_string( ntile ) + "×" + V1.to_string( ntile ) + " + " +
                 U2.to_string( ntile ) + "×" + V2.to_string( ntile ) +
                 " )" );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc & ) { assert( false ); }
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { U1.mem_block(), V1.mem_block(), U2.mem_block(), V2.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate2_node); }
};

//
// truncate U₁ T₁ V₁^H + U₂ T₁ V₂^H
//
struct truncate3_node : public node
{
    tiled_matrix              U1;
    shared_matrix             T1;
    tiled_matrix              V1;
    tiled_matrix              U2;
    shared_matrix             T2;
    tiled_matrix              V2;
    shared_tiled_matrix       U;
    shared_tiled_matrix       V;
    const size_t              ntile;

    truncate3_node ( tiled_matrix         aU1,
                     shared_matrix        aT1,
                     tiled_matrix         aV1,
                     tiled_matrix         aU2,
                     shared_matrix        aT2,
                     tiled_matrix         aV2,
                     shared_tiled_matrix  aU,
                     shared_tiled_matrix  aV,
                     const size_t         antile )
            : U1( aU1 )
            , T1( aT1 )
            , V1( aV1 )
            , U2( aU2 )
            , T2( aT2 )
            , V2( aV2 )
            , U( aU )
            , V( aV )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "trunc( " +
                 U1.to_string( ntile ) + "×" + T1.to_string() + "×" + V1.to_string( ntile ) + " + " +
                 U2.to_string( ntile ) + "×" + T2.to_string() + "×" + V2.to_string( ntile ) +
                 " )" );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc & ) { assert( false ); }
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { U1.mem_block(), T1.mem_block(), V1.mem_block(),
                                                                 U2.mem_block(), T2.mem_block(), V2.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate3_node); }
};

//
// truncate U₁ T₁ V₁^H
//
struct truncate4_node : public node
{
    tiled_matrix              U1;
    shared_matrix             T1;
    tiled_matrix              V1;
    shared_tiled_matrix       U;
    shared_tiled_matrix       V;
    const size_t              ntile;

    truncate4_node ( tiled_matrix         aU1,
                     shared_matrix        aT1,
                     tiled_matrix         aV1,
                     shared_tiled_matrix  aU,
                     shared_tiled_matrix  aV,
                     const size_t         antile )
            : U1( aU1 )
            , T1( aT1 )
            , V1( aV1 )
            , U( aU )
            , V( aV )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "trunc( " +
                 U1.to_string( ntile ) + "×" + T1.to_string() + "×" + V1.to_string( ntile ) + 
                 " )" );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc & ) { assert( false ); }
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { U1.mem_block(), T1.mem_block(), V1.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate4_node); }
};

//
// truncate α X Y^H + U(A) V(A)^H
//
struct truncate5_node : public node
{
    const double                alpha;
    shared_tiled_matrix       X;
    shared_tiled_matrix       Y;
    tiled_matrix              U;
    tiled_matrix              V;
    tiled_lrmatrix< double > *  A;
    const size_t              ntile;

    truncate5_node ( const double                aalpha,
                     shared_tiled_matrix       aX,
                     shared_tiled_matrix       aY,
                     tiled_matrix              aU,
                     tiled_matrix              aV,
                     tiled_lrmatrix< double > *  aA,
                     const size_t              antile )
            : alpha( aalpha )
            , X( aX )
            , Y( aY )
            , U( aU )
            , V( aV )
            , A( aA )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        return ( "trunc( " + X.to_string( ntile ) + "×" + Y.to_string( ntile ) + ", " +
                 U.to_string( ntile ) + "×" + V.to_string( ntile ) + " )" );
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc & ) { assert( false ); }
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { X.mem_block(), Y.mem_block() }; }
    virtual const block_list_t  out_blocks_  () const { return { U.mem_block(), V.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(truncate_node); }
};

//
// QR factorization of [αX·T,U]
//
template < typename  matrixX_t,
           typename  matrixU_t >
struct tsqr_node : public node
{
    const double           alpha;
    matrixX_t            X;
    shared_matrix        T;
    matrixU_t            U;
    shared_tiled_matrix  Q;
    shared_matrix        R;
    const size_t         ntile;

    tsqr_node ( const double           aalpha,
                matrixX_t            aX,
                shared_matrix        aT,
                matrixU_t            aU,
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
                matrixX_t            aX,
                matrixU_t            aU,
                shared_tiled_matrix  aQ,
                shared_matrix        aR,
                const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( shared_matrix( NAME_NONE, std::shared_ptr< blas::matrix< double > >() ) ) // T = 0
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

template < typename  matrixX_t >
struct tsqr1_node : public node
{
    const double           alpha;
    matrixX_t            X;
    shared_matrix        T;
    shared_tiled_matrix  Q;
    shared_matrix        R;
    const size_t         ntile;

    tsqr1_node ( const double           aalpha,
                 matrixX_t            aX,
                 shared_matrix        aT,
                 shared_tiled_matrix  aQ,
                 shared_matrix        aR,
                 const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( aT )
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    tsqr1_node ( const double           aalpha,
                 matrixX_t            aX,
                 shared_tiled_matrix  aQ,
                 shared_matrix        aR,
                 const size_t         antile )
            : alpha( aalpha )
            , X( aX )
            , T( shared_matrix( NAME_NONE, std::shared_ptr< blas::matrix< double > >() ) ) // T = 0
            , Q( aQ )
            , R( aR )
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const
    {
        if ( is_null( T.data ) )
            return Q.to_string( ntile ) + ", " + R.to_string() + " = tsqr( " + X.to_string( ntile ) + " )";
        else 
            return Q.to_string( ntile ) + ", " + R.to_string() + " = tsqr( " + X.to_string( ntile ) + "×" + T.to_string() + " )";
    }
    virtual std::string  color     () const { return "e9b96e"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const
    {
        if ( is_null( T.data ) ) return { X.mem_block(), Q.mem_block() };
        else                     return { X.mem_block(), Q.mem_block(), T.mem_block() };
    }
    virtual const block_list_t  out_blocks_  () const { return { Q.mem_block(), R.mem_block() }; }
    virtual size_t              mem_size_    () const { return sizeof(tsqr1_node); }
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

//
// apply updates to low-rank block
//
struct apply_node : public node
{
    tiled_lrmatrix< double > *  M;
    lr_update_t               upd;
    size_t                    ntile;

    apply_node ( tiled_lrmatrix< double > *  aM,
                 tiled_matrix              aU,
                 shared_matrix             aT,
                 tiled_matrix              aV,
                 size_t                    antile )
            : M( aM )
            , upd{ aU, aT, aV }
            , ntile( antile )
    { init(); }

    virtual std::string  to_string () const { return hpro::to_string( "%d = apply( ", M->id() ) + upd.U.to_string() + "×" + upd.T.to_string() + "×" + upd.V.to_string() + " )"; }
    virtual std::string  color     () const { return "c4a000"; }

private:
    virtual void                run_         ( const Hpro::TTruncAcc & ) {}
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const
    {
        return { upd.U.mem_block(), upd.T.mem_block(), upd.V.mem_block() };
    }
    virtual const block_list_t  out_blocks_  () const
    {
        return { { NAME_A, M->block_is() },
                 { id_t(& M->U()), block_indexset( M->row_is(), IS_ONE ) },
                 { id_t(& M->V()), block_indexset( M->col_is(), IS_ONE ) } };
    }
    virtual size_t              mem_size_    () const { return sizeof(apply_node); }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

std::tuple< node *,
            shared_tiled_matrix,
            shared_tiled_matrix >
build_upd_nodes_pairwise_svd ( local_graph &    g,
                               Hpro::TMatrix< double > *        M,
                               update_list_t &  upds,
                               const size_t     ntile )
{
    const auto  nupds = upds.size();
    
    if ( nupds > 2 )
    {
        const auto     mid = nupds / 2;
        size_t         pos = 0;
        update_list_t  upds1, upds2;

        for ( auto upd : upds )
        {
            if ( pos++ < mid ) upds1.push_back( upd );
            else               upds2.push_back( upd );
        }// for

        auto  [ upd1, U1, V1 ] = build_upd_nodes_pairwise_svd( g, M, upds1, ntile );
        auto  [ upd2, U2, V2 ] = build_upd_nodes_pairwise_svd( g, M, upds2, ntile );

        auto  first = ptrcast( upds.front(), apply_node );
        auto  U     = shared_tiled_matrix( first->upd.U.is, std::make_shared< tile_storage< double > >() );
        auto  V     = shared_tiled_matrix( first->upd.V.is, std::make_shared< tile_storage< double > >() );
        auto  sum   = g.alloc_node< truncate2_node >( U1, V1, U2, V2,
                                                      U, V, ntile );

        sum->after( upd1, upd2 );

        return { sum, U, V };
    }// if
    else if ( nupds == 2 )
    {
        auto  first = ptrcast( upds.front(), apply_node );
        auto  last  = ptrcast( upds.back(),  apply_node );
        auto  U     = shared_tiled_matrix( first->upd.U.is, std::make_shared< tile_storage< double > >() );
        auto  V     = shared_tiled_matrix( first->upd.V.is, std::make_shared< tile_storage< double > >() );
        auto  trunc = g.alloc_node< truncate3_node >( first->upd.U, first->upd.T, first->upd.V,
                                                      last->upd.U,  last->upd.T,  last->upd.V,
                                                      U, V, ntile );

        trunc->after( first, last );
        
        return { trunc, U, V };
    }// if
    else if ( nupds == 1 )
    {
        auto  upd   = ptrcast( upds.front(), apply_node );
        auto  U     = shared_tiled_matrix( upd->upd.U.is, std::make_shared< tile_storage< double > >() );
        auto  V     = shared_tiled_matrix( upd->upd.V.is, std::make_shared< tile_storage< double > >() );
        auto  trunc = g.alloc_node< truncate4_node >( upd->upd.U, upd->upd.T, upd->upd.V,
                                                      U, V, ntile );

        trunc->after( upd );
        
        return { trunc, U, V };
    }// if
    else
    {
        throw std::runtime_error( hpro::to_string( "unsupported number of updates (%d)", nupds ) );
    }// else
}

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
            
        auto  lu_00    = g.alloc_node< lu_node >( BA->block( 0, 0 ), update_map, ntile );

        // std::cout << A10->id() << " " << update_map[ A10->id() ].size() << std::endl;
        // std::cout << A01->id() << " " << update_map[ A01->id() ].size() << std::endl;

        // auto  apply_10 = g.alloc_node< apply_node >( A10, update_map[ A10->id() ], ntile );
        // auto  apply_01 = g.alloc_node< apply_node >( A01, update_map[ A01->id() ], ntile );

        auto  solve_10 = g.alloc_node< trsmu_node >( BU->block( 0, 0 ),
                                                     tiled_matrix( NAME_L, A10->id(), A10->col_is(), & A10->V() ),
                                                     tiled_matrix( NAME_A, A10->id(), A10->col_is(), & A10->V() ),
                                                     update_map, ntile );
        auto  solve_01 = g.alloc_node< trsml_node >( BL->block( 0, 0 ),
                                                     tiled_matrix( NAME_U, A01->id(), A01->row_is(), & A01->U() ),
                                                     tiled_matrix( NAME_A, A01->id(), A01->row_is(), & A01->U() ),
                                                     update_map, ntile );

        lu_00->before( solve_10, solve_01 );
        
        if ( update_map[ A10->id() ].size() > 0 )
        {
            auto  [ sum_10, U10, V10 ] = build_upd_nodes_pairwise_svd( g, A10, update_map[ A10->id() ], ntile );
            auto  upd_10               = g.alloc_node< truncate5_node >( double(-1),
                                                                         U10, V10,
                                                                         tiled_matrix( NAME_A, A10->id(), A10->row_is(), & A10->U() ),
                                                                         tiled_matrix( NAME_A, A10->id(), A10->col_is(), & A10->V() ),
                                                                         A10,
                                                                         ntile );
            
            upd_10->after( sum_10 );
            solve_10->after( upd_10 );
        }// if

        if ( update_map[ A01->id() ].size() > 0 )
        {
            auto  [ sum_01, U01, V01 ] = build_upd_nodes_pairwise_svd( g, A01, update_map[ A01->id() ], ntile );
            auto  upd_01               = g.alloc_node< truncate5_node >( double(-1),
                                                                         U01, V01,
                                                                         tiled_matrix( NAME_A, A01->id(), A01->row_is(), & A01->U() ),
                                                                         tiled_matrix( NAME_A, A01->id(), A01->col_is(), & A01->V() ),
                                                                         A01,
                                                                         ntile );
            
            upd_01->after( sum_01 );
            solve_01->after( upd_01 );
        }// if

        auto  T        = std::make_shared< blas::matrix< double > >();
        auto  tsmul    = g.alloc_node< dot_node >( tiled_matrix( NAME_L, A10->id(), A10->col_is(), & A10->V() ),
                                                   tiled_matrix( NAME_U, A01->id(), A01->row_is(), & A01->U() ),
                                                   shared_matrix( T ),
                                                   ntile );

        tsmul->after( solve_10 );
        tsmul->after( solve_01 );

        auto  addlr    = g.alloc_node< addlr_node >( tiled_matrix( NAME_L, A10->id(), A10->row_is(), & A10->U() ),
                                                     shared_matrix( T ),
                                                     tiled_matrix( NAME_U, A01->id(), A01->col_is(), & A01->V() ),
                                                     BA->block( 1, 1 ),
                                                     update_map, ntile );

        addlr->after( tsmul );

        auto  lu_11    = g.alloc_node< lu_node >( BA->block( 1, 1 ), update_map, ntile );

        lu_11->after( addlr );
    }// if

    g.finalize();
    
    return g;
}

void
lu_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    hlr::seq::tiled2::hodlr::lu< double >( A, acc, ntile );
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
                                                     update_map, ntile );
        auto  T        = std::make_shared< blas::matrix< double > >();
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
                                                     update_map, ntile );

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
                                                     update_map, ntile );
        auto  T        = std::make_shared< blas::matrix< double > >();
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
                                                     update_map, ntile );

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

        auto  T0     = std::make_shared< blas::matrix< double > >();
        auto  T1     = std::make_shared< blas::matrix< double > >();
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
                                       update_map, ntile );

        // TO CHECK: is "push_back" before handling of updates (trsml,trsmu)???
        auto apply_01 = g.alloc_node< apply_node >( A01,
                                                    tiled_matrix( A01->row_is(), U ),
                                                    T,
                                                    tiled_matrix( A01->col_is(), V ),
                                                    ntile );

        {
            std::scoped_lock  lock( upd_mtx );
            
            update_map[ A01->id() ].push_back( apply_01 );
        }

        auto  apply_10 = g.alloc_node< apply_node >( A10,
                                                     tiled_matrix( A10->row_is(), U ),
                                                     T,
                                                     tiled_matrix( A10->col_is(), V ),
                                                     ntile );
        
        {
            std::scoped_lock  lock( upd_mtx );
            
            update_map[ A10->id() ].push_back( apply_10 );
        }
        
        // g.alloc_node< truncate_node >( double(-1),
        //                                tiled_matrix( A01->row_is(), U ),
        //                                T,
        //                                tiled_matrix( A01->col_is(), V ),
        //                                tiled_matrix( NAME_A, A01->id(), A01->row_is(), & A01->U() ),
        //                                tiled_matrix( NAME_A, A01->id(), A01->col_is(), & A01->V() ),
        //                                A01,
        //                                ntile );
        // g.alloc_node< truncate_node >( double(-1),
        //                                tiled_matrix( A10->row_is(), U ),
        //                                T,
        //                                tiled_matrix( A10->col_is(), V ),
        //                                tiled_matrix( NAME_A, A10->id(), A10->row_is(), & A10->U() ),
        //                                tiled_matrix( NAME_A, A10->id(), A10->col_is(), & A10->V() ),
        //                                A10,
        //                                ntile );
        
        g.alloc_node< addlr_node    >( tiled_matrix( A11->row_is(), U ),
                                       T,
                                       tiled_matrix( A11->col_is(), V ),
                                       A11,
                                       update_map, ntile );
    }// if

    g.finalize();

    return g;
}

void
addlr_node::run_ ( const Hpro::TTruncAcc &  acc )
{
    hpro::TScopedLock  lock( *A );
    
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
        *(T.data) = blas::matrix< double >( T0.data->nrows(), T0.data->ncols() );
    
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

    auto  Q0 = shared_tiled_matrix( X.is, std::make_shared< tile_storage< double > >() );
    auto  R0 = shared_matrix(             std::make_shared< blas::matrix< double > >() );
    auto  Q1 = shared_tiled_matrix( Y.is, std::make_shared< tile_storage< double > >() );
    auto  R1 = shared_matrix(             std::make_shared< blas::matrix< double > >() );

    // perform QR for U/V
    auto  qr_U   = new tsqr_node( alpha,   X, T, U, Q0, R0, ntile );
    auto  qr_V   = new tsqr_node( double(1), Y,    V, Q1, R1, ntile );

    g.add_node( qr_U, qr_V );
    
    // determine truncated rank and allocate destination matrices
    auto  Uk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  Vk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  svd    = g.alloc_node< svd_node >( R0, R1, Uk, Vk );

    svd->after( qr_U, qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( double(1), Q0, Uk, double(0), U, ntile );
    auto  mul_V  = new tprod_node( double(1), Q1, Vk, double(0), V, ntile );

    g.add_node( mul_U );
    g.add_node( mul_V );
    
    mul_U->after( qr_U, svd );
    mul_V->after( qr_V, svd );

    g.finalize();

    return g;
}

local_graph
truncate2_node::refine_ ( const size_t )
{
    local_graph  g;

    assert( U1.is == U2.is );
    assert( U1.is == U.is );
    assert( V1.is == V2.is );
    assert( V1.is == V.is );

    auto  Q1 = shared_tiled_matrix( U1.is, std::make_shared< tile_storage< double > >() );
    auto  R1 = shared_matrix(              std::make_shared< blas::matrix< double > >() );
    auto  Q2 = shared_tiled_matrix( V1.is, std::make_shared< tile_storage< double > >() );
    auto  R2 = shared_matrix(              std::make_shared< blas::matrix< double > >() );

    // perform QR for U/V
    auto  qr_U   = new tsqr_node( double(1), U1, U2, Q1, R1, ntile );
    auto  qr_V   = new tsqr_node( double(1), V1, V2, Q2, R2, ntile );

    g.add_node( qr_U, qr_V );
    
    // determine truncated rank and allocate destination matrices
    auto  Uk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  Vk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  svd    = g.alloc_node< svd_node >( R1, R2, Uk, Vk );

    svd->after( qr_U, qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( double(1), Q1, Uk, double(0), U, ntile );
    auto  mul_V  = new tprod_node( double(1), Q2, Vk, double(0), V, ntile );

    g.add_node( mul_U, mul_V );
    
    mul_U->after( qr_U, svd );
    mul_V->after( qr_V, svd );

    g.finalize();

    return g;
}

local_graph
truncate3_node::refine_ ( const size_t )
{
    local_graph  g;

    assert( U1.is == U2.is );
    assert( U1.is == U.is );
    assert( V1.is == V2.is );
    assert( V1.is == V.is );

    auto  Q1 = shared_tiled_matrix( U1.is, std::make_shared< tile_storage< double > >() );
    auto  R1 = shared_matrix(              std::make_shared< blas::matrix< double > >() );
    auto  Q2 = shared_tiled_matrix( V1.is, std::make_shared< tile_storage< double > >() );
    auto  R2 = shared_matrix(              std::make_shared< blas::matrix< double > >() );

    // perform QR for U/V
    auto  qr_U   = new tsqr_node( double(1), U1, T1, U2, Q1, R1, ntile ); // TODO: check if order is correct
    auto  qr_V   = new tsqr_node( double(1), V2, T2, V1, Q2, R2, ntile );

    g.add_node( qr_U, qr_V );
    
    // determine truncated rank and allocate destination matrices
    auto  Uk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  Vk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  svd    = g.alloc_node< svd_node >( R1, R2, Uk, Vk );

    svd->after( qr_U, qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( double(1), Q1, Uk, double(0), U, ntile );
    auto  mul_V  = new tprod_node( double(1), Q2, Vk, double(0), V, ntile );

    g.add_node( mul_U, mul_V );
    
    mul_U->after( qr_U, svd );
    mul_V->after( qr_V, svd );

    g.finalize();

    return g;
}

local_graph
truncate4_node::refine_ ( const size_t )
{
    local_graph  g;

    auto  Q0 = shared_tiled_matrix( U1.is, std::make_shared< tile_storage< double > >() );
    auto  R0 = shared_matrix(              std::make_shared< blas::matrix< double > >() );
    auto  Q1 = shared_tiled_matrix( V1.is, std::make_shared< tile_storage< double > >() );
    auto  R1 = shared_matrix(              std::make_shared< blas::matrix< double > >() );

    // perform QR for U/V
    auto  qr_U   = new tsqr1_node( double(1), U1, T1, Q0, R0, ntile );
    auto  qr_V   = new tsqr1_node( double(1), V1,     Q1, R1, ntile );

    g.add_node( qr_U, qr_V );
    
    // determine truncated rank and allocate destination matrices
    auto  Uk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  Vk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  svd    = g.alloc_node< svd_node >( R0, R1, Uk, Vk );

    svd->after( qr_U, qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( double(1), Q0, Uk, double(0), U, ntile );
    auto  mul_V  = new tprod_node( double(1), Q1, Vk, double(0), V, ntile );

    g.add_node( mul_U );
    g.add_node( mul_V );
    
    mul_U->after( qr_U, svd );
    mul_V->after( qr_V, svd );

    g.finalize();

    return g;
}

local_graph
truncate5_node::refine_ ( const size_t )
{
    local_graph  g;

    assert( X.is == U.is );
    assert( Y.is == V.is );

    auto  Q0 = shared_tiled_matrix( X.is, std::make_shared< tile_storage< double > >() );
    auto  R0 = shared_matrix(             std::make_shared< blas::matrix< double > >() );
    auto  Q1 = shared_tiled_matrix( Y.is, std::make_shared< tile_storage< double > >() );
    auto  R1 = shared_matrix(             std::make_shared< blas::matrix< double > >() );

    // perform QR for U/V
    auto  qr_U   = new tsqr_node( double(1), X, U, Q0, R0, ntile );
    auto  qr_V   = new tsqr_node( double(1), Y, V, Q1, R1, ntile );

    g.add_node( qr_U, qr_V );
    
    // determine truncated rank and allocate destination matrices
    auto  Uk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  Vk     = shared_matrix( std::make_shared< blas::matrix< double > >() );
    auto  svd    = g.alloc_node< svd_node >( R0, R1, Uk, Vk );

    svd->after( qr_U, qr_V );

    // compute final result
    auto  mul_U  = new tprod_node( double(1), Q0, Uk, double(0), U, ntile );
    auto  mul_V  = new tprod_node( double(1), Q1, Vk, double(0), V, ntile );

    g.add_node( mul_U );
    g.add_node( mul_V );
    
    mul_U->after( qr_U, svd );
    mul_V->after( qr_V, svd );

    g.finalize();

    return g;
}

void
truncate_node::run_ ( const Hpro::TTruncAcc & )
{
    // hpro::TScopedLock  lock( *A );
    
    // auto [ U2, V2 ] = hlr::seq::tiled2::truncate( U.is, V.is, alpha, *(X.data), *(T.data), *(Y.data), *(U.data), *(V.data), acc, ntile );

    // *(U.data) = std::move( U2 );
    // *(V.data) = std::move( V2 );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// tsqr_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename  matrixX_t,
           typename  matrixU_t >
local_graph
tsqr_node< matrixX_t, matrixU_t >::refine_ ( const size_t  tile_size )
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
        auto        R0    = matrix_info( std::make_shared< blas::matrix< double > >() );
        auto        R1    = matrix_info( std::make_shared< blas::matrix< double > >() );

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

template < typename  matrixX_t,
           typename  matrixU_t >
void
tsqr_node< matrixX_t, matrixU_t >::run_ ( const Hpro::TTruncAcc & )
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
        blas::matrix< double >  XU( X_is.nrows(), X_is.ncols() + U_is.ncols () );
        blas::matrix< double >  XU_X( XU, blas::range::all, blas::range( 0, X_is.ncols()-1 ) );
        blas::matrix< double >  XU_U( XU, blas::range::all, blas::range( X_is.ncols(), XU.ncols()-1 ) );

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
        blas::matrix< double >  WU( W.nrows(), W.ncols() + U_is.ncols () );
        blas::matrix< double >  WU_W( WU, blas::range::all, blas::range( 0, W.ncols()-1 ) );
        blas::matrix< double >  WU_U( WU, blas::range::all, blas::range( W.ncols(), WU.ncols()-1 ) );

        blas::copy( W,    WU_W );
        blas::copy( U_is, WU_U );

        blas::qr( WU, *(R.data) );

        HLR_LOG( 5, "tsqr  :          Q , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( WU ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(R.data) ) ) );
        
        (*(Q.data))[ X.is ] = std::move( WU );
    }// else
}

template < typename  matrixX_t >
local_graph
tsqr1_node< matrixX_t >::refine_ ( const size_t  tile_size )
{
    local_graph  g;

    if ( X.is.size() > tile_size )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis_X = split( X.is, 2 );
        const auto  sis_Q = split( Q.is, 2 );
        auto        Q0    = matrix_info( sis_Q[0], Q );
        auto        Q1    = matrix_info( sis_Q[1], Q );
        auto        R0    = matrix_info( std::make_shared< blas::matrix< double > >() );
        auto        R1    = matrix_info( std::make_shared< blas::matrix< double > >() );

        // std::cout << "R0 = " << R0.data.get() << std::endl;
        // std::cout << "R1 = " << R1.data.get() << std::endl;
        
        auto  tsqr0 = g.alloc_node< tsqr1_node >( alpha, matrix_info( sis_X[0], X ), T, Q0, R0, ntile );
        auto  tsqr1 = g.alloc_node< tsqr1_node >( alpha, matrix_info( sis_X[1], X ), T, Q1, R1, ntile );
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

template < typename  matrixX_t >
void
tsqr1_node< matrixX_t >::run_ ( const Hpro::TTruncAcc & )
{
    if ( is_null( T.data ) )
    {
        assert( X.data->contains( X.is ) );

        auto  X_is = X.data->at( X.is );

        blas::qr( X_is, *(R.data) );

        HLR_LOG( 5, "tsqr  :          Q , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( X_is ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(R.data) ) ) );
        
        (*(Q.data))[ X.is ] = std::move( X_is );
    }// if
    else
    {
        assert( X.data->contains( X.is ) );

        const auto  X_is = X.data->at( X.is );
        auto        W    = blas::prod( alpha, X_is, *(T.data) );

        blas::qr( W, *(R.data) );

        HLR_LOG( 5, "tsqr  :          Q , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( W ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + hlr::seq::tiled2::isstr( X.is, ntile ) + " = " + hlr::seq::tiled2::normstr( blas::normF( *(R.data) ) ) );
        
        (*(Q.data))[ X.is ] = std::move( W );
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
    blas::matrix< double >  Q(   R0.data->nrows() + R1.data->nrows(), R0.data->ncols() );
    blas::matrix< double >  Q_0( Q, blas::range(                0, R0.data->nrows()-1 ), blas::range::all );
    blas::matrix< double >  Q_1( Q, blas::range( R0.data->nrows(), Q.nrows()-1        ), blas::range::all );
        
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
    blas::matrix< double >  Vs;
    blas::vector< double >  Ss;
        
    blas::svd( Us, Ss, Vs );
        
    const auto      k  = acc.trunc_rank( Ss );

    // for ( size_t  i = 0; i < k; ++i )
    //     std::cout << Ss(i) << std::endl;
    
    blas::matrix< double >  Usk( Us, blas::range::all, blas::range( 0, k-1 ) );
    blas::matrix< double >  Vsk( Vs, blas::range::all, blas::range( 0, k-1 ) );
        
    blas::prod_diag( Usk, Ss, k );

    *(Uk.data) = std::move( blas::matrix< double >( Usk.nrows(), Usk.ncols() ) );
    *(Vk.data) = std::move( blas::matrix< double >( Vsk.nrows(), Vsk.ncols() ) );

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
gen_dag_lu_hodlr_tiled_lazy ( Hpro::TMatrix< double > &      A,
                              const size_t   ntile,
                              refine_func_t  refine )
{
    update_map_t  update_map;

    return refine( new lu_node( & A, update_map, ntile ), ntile, use_single_end_node );
}

}// namespace dag

}// namespace hlr
