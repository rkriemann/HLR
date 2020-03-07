#ifndef __HLR_SEQ_ARITH_TILED_V3_HH
#define __HLR_SEQ_ARITH_TILED_V3_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential tile-based arithmetic functions v3
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr { namespace seq { namespace tiled3 {

// map HLIB namespaces to HLR
namespace hpro = HLIB;

using HLIB::id_t;
using HLIB::real;

// import matrix types
using hlr::matrix::indexset;
using hlr::matrix::tile;
using hlr::matrix::tile_storage;
using hlr::matrix::tiled_lrmatrix;

// dummy indexset for T operations (rank/size unknown during DAG and only object is of interest)
const auto  IS_ONE  = indexset( -1, -1 );
const auto  BIS_ONE = TBlockIndexSet( IS_ONE, IS_ONE );

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
    
    matrix_info ( const indexset  is );
    
    matrix_info ( const indexset  ais,
                  matrix_info &   amat )
            : name( amat.name )
            , id( amat.id )
            , is( ais )
            , data( amat.data )
    {}

    virtual ~matrix_info ();

    operator matrix_t () { return data; }
    
    const TBlockIndexSet block_is  () const { return TBlockIndexSet( is, IS_ONE ); }

    std::string
    to_string ( const size_t  ntile = 0 ) const
    {
        std::ostringstream  os;
        
        if ( name < 100 ) os << char(name);
        else              os << ( name & 0xff );

        if ( id != id_t(-1) ) os << id;

        if (( is != IS_ONE ) && ( ntile != 0 ))
        {
            if ( is.size() <= ntile ) os << HLIB::to_string( "[%d]", is.first() / ntile );
            else                      os << HLIB::to_string( "[%d:%d]", is.first() / ntile, is.last() / ntile );
        }// if

        return os.str();
    }
};

template <>
matrix_info< blas::matrix< real > >::~matrix_info ()
{}

template <>
matrix_info< tile_storage< real > * >::~matrix_info ()
{}

template <>
matrix_info< std::shared_ptr< blas::matrix< real > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< std::shared_ptr< tile_storage< real > > >::~matrix_info ()
{
//    std::cout << "matrix : " << data.get() << ", #" << data.use_count() << std::endl;
}

template <>
matrix_info< tile_storage< real > * >::matrix_info ( const indexset          ais,
                                                     tile_storage< real > *  adata )
        : name( id_t(adata) )
        , id( -1 )
        , is( ais )
        , data( adata )
{}

template <>
matrix_info< std::shared_ptr< tile_storage< real > > >::matrix_info ( const indexset                           ais,
                                                                      std::shared_ptr< tile_storage< real > >  adata )
        : name( id_t(adata.get()) )
        , id( -1 )
        , is( ais )
        , data( adata )
{}

template <>
matrix_info< blas::matrix< real > >::matrix_info ( blas::matrix< real >  adata )
        : name( id_t(adata.data()) )
        , id( -1  )
        , is( IS_ONE )
        , data( adata )
{}

template <>
matrix_info< std::shared_ptr< blas::matrix< real > > >::matrix_info ( std::shared_ptr< blas::matrix< real > >  adata )
        : name( id_t(adata.get()) )
        , id( -1  )
        , is( IS_ONE )
        , data( adata )
{}

template <>
matrix_info< tile_storage< real > * >::matrix_info ( const indexset  ais )
        : name( 255 )
        , id( -1 )
        , is( ais )
{}

using dense_matrix        = matrix_info< blas::matrix< real > >;
using tiled_matrix        = matrix_info< tile_storage< real > * >;
using shared_matrix       = matrix_info< std::shared_ptr< blas::matrix< real > > >;
using shared_tiled_matrix = matrix_info< std::shared_ptr< tile_storage< real > > >;

inline
std::string
idstr ( hpro::id_t    id )
{
    return hpro::to_string( "%3d", id );
}

inline
std::string
isstr ( indexset      is,
        const size_t  ntile )
{
    if ( is.size() <= ntile ) return hpro::to_string( "[%d]  ",    is.first() / ntile );
    else                      return hpro::to_string( "[%d:%d]", is.first() / ntile, is.last() / ntile );
}

inline
std::string
normstr ( double   f )
{
    return hpro::to_string( "%.4e", f );
}

//
// split given indexset into <n> subsets
//
inline
std::vector< indexset >
split ( const indexset &  is,
        const size_t      n )
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

//
// compute T := A^H · B where A, B ∈ K^{is × k}
//
inline
shared_matrix
dot ( tiled_matrix  A,
      tiled_matrix  B,
      const size_t  ntile )
{
    HLR_LOG( 4, "T = dot(" + A.to_string( ntile ) + "×" + B.to_string( ntile ) + " )" );
    
    HLR_ASSERT( A.is == B.is );
    
    if ( A.is.size() > ntile )
    {
        const auto  sis = split( A.is, 2 );
        auto        T0  = dot( tiled_matrix( sis[0], A ),
                               tiled_matrix( sis[0], B ),
                               ntile );
        auto        T1  = dot( tiled_matrix( sis[1], A ),
                               tiled_matrix( sis[1], B ),
                               ntile );

        // HLR_LOG( 4, T1.to_string() + " = Tadd(" + T0.to_string() + "+" + T1.to_string() + ")" );
        
        blas::add( real(1), *(T0.data), *(T1.data) );

        return T1;
    }// if
    else
    {
        HLR_ASSERT( A.data->contains( A.is ) && B.data->contains( B.is ) );

        auto  T = blas::prod( real(1), blas::adjoint( A.data->at( A.is ) ), B.data->at( B.is ) );

        // HLR_LOG( 5, "         dot :       " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( T ) ) );

        return shared_matrix( std::make_shared< blas::matrix< real > >( std::move( T ) ) );
    }// else
}

//
// compute B := β·B + α·A·T
//
inline
void
tprod ( const real      alpha,
        tiled_matrix    A,
        shared_matrix   T,
        const real      beta,
        tiled_matrix    B,
        const size_t    ntile )
{
    HLR_ASSERT( A.is == B.is );
        
    HLR_LOG( 4, "Tprod(" + A.to_string( ntile ) + "×" + T.to_string() + "+" + B.to_string( ntile ) + ")" );
    
    if ( A.is.size() > ntile )
    {
        const auto  sis = split( A.is, 2 );

        tprod( alpha, tiled_matrix( sis[0], A ), T, beta, tiled_matrix( sis[0], B ), ntile );
        tprod( alpha, tiled_matrix( sis[1], A ), T, beta, tiled_matrix( sis[1], B ), ntile );
    }// if
    else
    {
        HLR_ASSERT( A.data->contains( A.is ) );
        HLR_ASSERT( ( beta == real(0) ) || B.data->contains( B.is ) );

        if ( B.data->contains( B.is ) )
        {
            HLR_LOG( 5, "tprod :          A , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( A.data->at( A.is ) ) ) );
            HLR_LOG( 5, "tprod :          T , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( *(T.data) ) ) );
            HLR_LOG( 5, "tprod :          B , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( B.data->at( A.is ) ) ) );
            
            blas::prod( alpha, A.data->at( A.is ), *(T.data), beta, B.data->at( B.is ) );
        }// if
        else
        {
            HLR_LOG( 5, "tprod :          A , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( A.data->at( A.is ) ) ) );
            HLR_LOG( 5, "tprod :          T , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( *(T.data) ) ) );
            
            (*(B.data))[ B.is ] = std::move( blas::prod( alpha, A.data->at( A.is ), *(T.data) ) );
        }// else

        HLR_LOG( 5, "tprod :          C , " + isstr( A.is, ntile ) + " = " + normstr( blas::normF( B.data->at( A.is ) ) ) );
    }// else
}

inline
void
tprod ( const real     alpha,
        tiled_matrix   A,
        shared_matrix  T,
        const size_t   ntile )
{
    HLR_LOG( 4, "Tprod_ip(" + A.to_string( ntile ) + "×" + T.to_string() + ")" );
    
    if ( A.is.size() > ntile )
    {
        const auto  sis = split( A.is, 2 );

        tprod( alpha, tiled_matrix( sis[0], A ), T, ntile );
        tprod( alpha, tiled_matrix( sis[1], A ), T, ntile );
    }// if
    else
    {
        HLR_ASSERT( A.data->contains( A.is ) );
        
        blas::matrix< real >  Ac( A.data->at( A.is ), copy_value );
        
        blas::prod( alpha, Ac, *(T.data), real(0), A.data->at( A.is ) );
    }// else
}

//
// compute QR factorization of [αX·T,U]
//
inline
std::pair< tiled_matrix, shared_matrix >
tsqr ( const real     alpha,
       tiled_matrix   X,
       shared_matrix  T,
       tiled_matrix   U,
       const size_t   ntile )
{
    HLR_LOG( 4, "Q, R = tsqr( " + X.to_string( ntile ) + "×" + T.to_string() + ", " + U.to_string( ntile ) + " )" );
    
    HLR_ASSERT( X.is == U.is );
    
    if ( X.is.size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis = split( X.is, 2 );

        auto [ Q0, R0 ] = tsqr( alpha, tiled_matrix( sis[0], X ), T, tiled_matrix( sis[0], U ), ntile );
        auto [ Q1, R1 ] = tsqr( alpha, tiled_matrix( sis[1], X ), T, tiled_matrix( sis[1], U ), ntile );

        // Q = | R0 |
        //     | R1 |
        blas::matrix< real >  Q01(   R0.data->nrows() + R1.data->nrows(), R0.data->ncols() );
        blas::matrix< real >  Q01_0( Q01, blas::range( 0, R0.data->nrows()-1 ), blas::range::all );
        blas::matrix< real >  Q01_1( Q01, blas::range( R0.data->nrows(), Q01.nrows()-1 ), blas::range::all );
        blas::matrix< real >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( *(R0.data), Q01_0 );
        blas::copy( *(R1.data), Q01_1 );

        blas::qr( Q01, R );

        tiled_matrix  Q( X.is, new tile_storage< real > );

        shared_matrix  dQ01_0( std::make_shared< blas::matrix< real > >( Q01_0, hpro::copy_value ) );
        shared_matrix  dQ01_1( std::make_shared< blas::matrix< real > >( Q01_1, hpro::copy_value ) );
        
        tprod( real(1), Q0, dQ01_0, real(0), tiled_matrix( sis[0], Q ), ntile );
        tprod( real(1), Q1, dQ01_1, real(0), tiled_matrix( sis[1], Q ), ntile );

        return { std::move( Q ), std::make_shared< blas::matrix< real > >( R, hpro::copy_value ) };
    }// if
    else
    {
        HLR_ASSERT( X.data->contains( X.is ) && U.data->contains( U.is ) );

        const auto      X_is = X.data->at( X.is );
        const auto      U_is = U.data->at( U.is );
        auto            W    = blas::prod( alpha, X_is, *(T.data) );
        blas::matrix< real >  WU( W.nrows(), W.ncols() + U_is.ncols () );
        blas::matrix< real >  WU_W( WU, blas::range::all, blas::range( 0, W.ncols()-1 ) );
        blas::matrix< real >  WU_U( WU, blas::range::all, blas::range( W.ncols(), WU.ncols()-1 ) );

        HLR_LOG( 5, "tsqr  :          X , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( X.data->at( X.is ) ) ) );
        HLR_LOG( 5, "tsqr  :          W , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( W ) ) );
        HLR_LOG( 5, "tsqr  :          U , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( U.data->at( X.is ) ) ) );
        
        blas::copy( W,    WU_W );
        blas::copy( U_is, WU_U );

        blas::matrix< real >  R;
        
        blas::qr( WU, R );

        tiled_matrix  Q( X.is, new tile_storage< real > );

        (*(Q.data))[ X.is ] = std::move( WU );
        
        HLR_LOG( 5, "tsqr  :          Q , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( Q.data->at( X.is ) ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( R ) ) );
        
        return { std::move( Q ), std::make_shared< blas::matrix< real > >( R, hpro::copy_value ) };
    }// else
}

//
// compute QR factorization of [αX,U]
//
inline
std::pair< tiled_matrix, shared_matrix >
tsqr ( const real    alpha,
       tiled_matrix  X,
       tiled_matrix  U,
       const size_t  ntile )
{
    HLR_LOG( 4, "Q, R = tsqr( " + X.to_string( ntile ) + ", " + U.to_string( ntile ) + " )" );
    
    HLR_ASSERT( X.is == U.is );

    if ( X.is.size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis = split( X.is, 2 );

        auto [ Q0, R0 ] = tsqr( alpha, tiled_matrix( sis[0], X ), tiled_matrix( sis[0], U ), ntile );
        auto [ Q1, R1 ] = tsqr( alpha, tiled_matrix( sis[1], X ), tiled_matrix( sis[1], U ), ntile );

        // Q = | R0 |
        //     | R1 |
        blas::matrix< real >  Q01(   R0.data->nrows() + R1.data->nrows(), R0.data->ncols() );
        blas::matrix< real >  Q01_0( Q01, blas::range( 0, R0.data->nrows()-1 ), blas::range::all );
        blas::matrix< real >  Q01_1( Q01, blas::range( R0.data->nrows(), Q01.nrows()-1 ), blas::range::all );
        blas::matrix< real >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( *(R0.data), Q01_0 );
        blas::copy( *(R1.data), Q01_1 );

        blas::qr( Q01, R );

        tiled_matrix  Q( X.is, new tile_storage< real > );
        
        shared_matrix  dQ01_0( std::make_shared< blas::matrix< real > >( Q01_0, hpro::copy_value ) );
        shared_matrix  dQ01_1( std::make_shared< blas::matrix< real > >( Q01_1, hpro::copy_value ) );
        
        tprod( real(1), Q0, dQ01_0, real(0), tiled_matrix( sis[0], Q ), ntile );
        tprod( real(1), Q1, dQ01_1, real(0), tiled_matrix( sis[1], Q ), ntile );

        return { std::move( Q ), std::make_shared< blas::matrix< real > >( R, hpro::copy_value ) };
    }// if
    else
    {
        HLR_ASSERT( X.data->contains( X.is ) && U.data->contains( U.is ) );

        const auto         X_is = X.data->at( X.is );
        const auto         U_is = U.data->at( U.is );
        blas::matrix< real >  XU( X_is.nrows(), X_is.ncols() + U_is.ncols () );
        blas::matrix< real >  XU_X( XU, blas::range::all, blas::range( 0, X_is.ncols()-1 ) );
        blas::matrix< real >  XU_U( XU, blas::range::all, blas::range( X_is.ncols(), XU.ncols()-1 ) );

        HLR_LOG( 5, "tsqr  :          X , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( X.data->at( X.is ) ) ) );
        HLR_LOG( 5, "tsqr  :          U , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( U.data->at( X.is ) ) ) );

        blas::copy( X_is, XU_X );
        blas::copy( U_is, XU_U );

        blas::matrix< real >  R;
        
        blas::qr( XU, R );

        tiled_matrix  Q( X.is, new tile_storage< real > );

        (*(Q.data))[ X.is ] = std::move( XU );
        
        HLR_LOG( 5, "tsqr  :          Q , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( Q.data->at( X.is ) ) ) );
        HLR_LOG( 5, "tsqr  :          R , " + isstr( X.is, ntile ) + " = " + normstr( blas::normF( R ) ) );
        
        return { std::move( Q ), std::make_shared< blas::matrix< real > >( R, hpro::copy_value ) };
    }// else
}

//
// truncate α X T Y^H + U V^H
//
inline
std::pair< tiled_matrix,
           tiled_matrix >
truncate ( const real         alpha,
           tiled_matrix       X,
           shared_matrix      T,
           tiled_matrix       Y,
           tiled_matrix       U,
           tiled_matrix       V,
           const TTruncAcc &  acc,
           const size_t       ntile )
{
    HLR_LOG( 4,
             "trunc( " + X.to_string( ntile ) + "×" + T.to_string() + "×" + Y.to_string( ntile ) + ", " +
             U.to_string( ntile ) + "×" + V.to_string( ntile ) + " )" );
    
    HLR_ASSERT( X.is == U.is );
    HLR_ASSERT( Y.is == V.is );
    
    // if ( Y.ncols() + V.ncols() > X.nrows() / 2 )
    // {
    //     // M = α X T Y^H + U V^H
    //     auto  W = blas::prod( real(1), X, T );
    //     auto  M = blas::prod( real(1), U, blas::adjoint( V ) );

    //     blas::prod( alpha, W, blas::adjoint( Y ), real(1), M );
            
    //     // truncate to rank-k
    //     return std::move( hlr::approx_svd( M, acc ) );
    // }// if
    // else
    {
        auto [ Q0, R0 ] = tsqr( alpha,      X, T, U, ntile );
        auto [ Q1, R1 ] = tsqr( real(1), Y,    V, ntile );

        // auto  dQ0 = to_dense( Q0 );
        // auto  dQ1 = to_dense( Q1 );

        // DBG::write( dQ0, "Q0.mat", "Q0" );
        // DBG::write( dQ1, "Q1.mat", "Q1" );
        // DBG::write(  R0, "R0.mat", "R0" );
        // DBG::write(  R1, "R1.mat", "R1" );
        
        auto                  R  = blas::prod( real(1), *(R0.data), blas::adjoint( *(R1.data) ) );
        auto                  Us = std::move( R );
        blas::matrix< real >  Vs;
        blas::vector< real >  Ss;
        
        blas::svd( Us, Ss, Vs );

        // DBG::write( Us, "Us.mat", "Us" );
        // DBG::write( Ss, "Ss.mat", "Ss" );
        // DBG::write( Vs, "Vs.mat", "Vs" );
        
        auto  k = acc.trunc_rank( Ss );

        for ( size_t  i = 0; i < k; ++i )
            std::cout << Ss(i) << std::endl;
        
        blas::matrix< real >  Usk( Us, blas::range::all, blas::range( 0, k-1 ) );
        blas::matrix< real >  Vsk( Vs, blas::range::all, blas::range( 0, k-1 ) );
        
        blas::prod_diag( Usk, Ss, k );

        tiled_matrix  Uk( U.is, new tile_storage< real > );
        tiled_matrix  Vk( V.is, new tile_storage< real > );

        shared_matrix  dUsk( std::make_unique< blas::matrix< real > >( Usk, hpro::copy_value ) );
        shared_matrix  dVsk( std::make_unique< blas::matrix< real > >( Vsk, hpro::copy_value ) );
        
        tprod( real(1), Q0, dUsk, real(0), Uk, ntile );
        tprod( real(1), Q1, dVsk, real(0), Vk, ntile );

        return { std::move( Uk ), std::move( Vk ) };
    }// else
}
    
namespace hodlr
{

///////////////////////////////////////////////////////////////////////
//
// tile based arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

//
// compute A := A - U·T·V^H
//
inline
void
addlr ( tiled_matrix       U,
        shared_matrix      T,
        tiled_matrix       V,
        TMatrix *          A,
        const TTruncAcc &  acc,
        const size_t       ntile )
{
    HLR_LOG( 4,
             "addlr(" + HLIB::to_string( "A%d, ", A->id() ) +
             U.to_string( ntile ) + "×" + T.to_string() + "×" + V.to_string( ntile ) + ")" );
    
    HLR_ASSERT( U.is == A->row_is() );
    HLR_ASSERT( V.is == A->col_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< real > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< real > );
        auto  A11 = BA->block( 1, 1 );

        addlr( tiled_matrix( A00->row_is(), U ),
               T,
               tiled_matrix( A00->col_is(), V ),
               A00, acc, ntile );
        
        auto  [ U01, V01 ] = truncate( real(-1),
                                       tiled_matrix( A01->row_is(), U ),
                                       T,
                                       tiled_matrix( A01->col_is(), V ),
                                       tiled_matrix( 'A', A01->id(), A01->row_is(), & A01->U() ),
                                       tiled_matrix( 'A', A01->id(), A01->col_is(), & A01->V() ),
                                       acc, ntile );

        A01->set_lrmat( std::move( *(U01.data) ), std::move( *(V01.data) ) );

        auto  [ U10, V10 ] = truncate( real(-1),
                                       tiled_matrix( A10->row_is(), U ),
                                       T,
                                       tiled_matrix( A10->col_is(), V ),
                                       tiled_matrix( 'A', A10->id(), A10->row_is(), & A10->U() ),
                                       tiled_matrix( 'A', A10->id(), A10->col_is(), & A10->V() ),
                                       acc, ntile );

        A10->set_lrmat( std::move( *(U10.data) ), std::move( *(V10.data) ) );
        
        addlr( tiled_matrix( A11->row_is(), U ),
               T,
               tiled_matrix( A11->col_is(), V ),
               A11, acc, ntile );
    }// if
    else
    {
        HLR_ASSERT( U.data->contains( A->row_is() ) && V.data->contains( A->col_is() ) );
        
        auto        D = ptrcast( A, TDenseMatrix );
        const auto  W = blas::prod( real(1), U.data->at( A->row_is() ), *(T.data) );

        HLR_LOG( 5, "addlr :         " + idstr( A->id() ) + ",     D = " + normstr( hpro::norm_F( D ) ) );
        HLR_LOG( 5, "addlr :         " + idstr( A->id() ) + ",     U = " + normstr( blas::norm_F( U.data->at( A->row_is() )) ) );
        HLR_LOG( 5, "addlr :         " + idstr( A->id() ) + ",     T = " + normstr( blas::norm_F( *(T.data) ) ) );
        HLR_LOG( 5, "addlr :         " + idstr( A->id() ) + ",     W = " + normstr( blas::norm_F( W ) ) );

        blas::prod( real(-1), W, blas::adjoint( V.data->at( A->col_is() ) ),
                    real(1), blas_mat< real >( D ) );

        HLR_LOG( 5, "addlr :         " + idstr( A->id() ) + ",     D = " + normstr( hpro::norm_F( D ) ) );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
inline
void
trsmuh ( TMatrix *     U,
         tiled_matrix  X,
         const size_t  ntile )
{
    HLR_LOG( 4, X.to_string( ntile ) + HLIB::to_string( " = trsmu( U%d, ", U->id() ) + X.to_string( ntile ) + " )" );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = ptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = ptrcast( BU->block( 0, 1 ), tiled_lrmatrix< real > );
        auto  U11 = BU->block( 1, 1 );

        const auto  is0 = U00->col_is();
        const auto  is1 = U11->col_is();

        trsmuh( U00, tiled_matrix( is0, X ), ntile );

        auto  T = dot( tiled_matrix( 'A', U01->id(), U01->row_is(), & U01->U() ),
                       tiled_matrix( is0, X ),
                       ntile );
        
        HLR_LOG( 5, "trsmu :  dot :  " + idstr( U->id() ) + ", " + isstr( U01->row_is(), ntile ) + " = " + normstr( blas::normF( *(T.data) ) ) );
        
        tprod( real(-1),
               tiled_matrix( 'A', U01->id(), U01->col_is(), & U01->V() ),
               T,
               real(1),
               tiled_matrix( is1, X ),
               ntile );

        trsmuh( U11, tiled_matrix( is1, X ), ntile );
    }// if
    else
    {
        HLR_ASSERT( X.data->contains( U->row_is() ) );
        
        auto  DU = ptrcast( U, TDenseMatrix );

        auto            X_is = X.data->at( U->row_is() );
        blas::matrix< real >  Y( X_is, copy_value );

        blas::prod( real(1), blas::adjoint( blas_mat< real >( DU ) ), Y, real(0), X_is );

        HLR_LOG( 5, "trsmu :         " + idstr( U->id() ) + "        = " + normstr( blas::normF( X.data->at( X.is ) ) ) );
    }// else
}

//
// solve L X = M
// - on input, X = M
//
inline
void
trsml ( TMatrix *     L,
        tiled_matrix  X,
        const size_t  ntile )
{
    HLR_LOG( 4, X.to_string( ntile ) + HLIB::to_string( " = trsml( A%d, ", L->id() ) + X.to_string( ntile ) + " )" );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = ptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = ptrcast( BL->block( 1, 0 ), tiled_lrmatrix< real > );
        auto  L11 = BL->block( 1, 1 );

        const auto  is0 = L00->row_is();
        const auto  is1 = L11->row_is();

        
        trsml( L00, tiled_matrix( is0, X ), ntile );

        auto  T = dot( tiled_matrix( 'A', L10->id(), L10->col_is(), & L10->V() ),
                       tiled_matrix( is0, X ),
                       ntile );

        HLR_LOG( 5, "trsml :  dot :  " + idstr( L->id() ) + ", " + isstr( L10->col_is(), ntile ) + " = " + normstr( blas::normF( *(T.data) ) ) );
        
        tprod( real(-1),
               tiled_matrix( 'A', L10->id(), L10->row_is(), & L10->U() ),
               T,
               real(1),
               tiled_matrix( is1, X ),
               ntile );

        trsml( L11, tiled_matrix( is1, X ), ntile );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //

        HLR_LOG( 5, "trsml :         " + idstr( L->id() ) + "        = " + normstr( blas::normF( X.data->at( L->row_is() ) ) ) );

        // DEBUG
        // {
        //     auto  DX = hlr::matrix::to_dense( X );

        //     DBG::write( X->at( L->row_is() ), "X.mat", "X" );
        //     std::exit( 0 );
        // }
    }// else
}

//
// compute A = LU
//
inline
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc,
     const size_t       ntile )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< real > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< real > );
        auto  A11 = BA->block( 1, 1 );

        lu( A00, acc, ntile );
        
        trsml(  A00, tiled_matrix( 'A', A01->id(), A01->row_is(), & A01->U() ), ntile );
        trsmuh( A00, tiled_matrix( 'A', A10->id(), A10->col_is(), & A10->V() ), ntile );

        // T = ( V(A_10)^H · U(A_01) )
        auto  T = dot( tiled_matrix( 'A', A10->id(), A10->col_is(), & A10->V() ),
                       tiled_matrix( 'A', A01->id(), A01->row_is(), & A01->U() ),
                       ntile ); 

        HLR_LOG( 5, "lu    :  dot :  " + idstr( A->id() ) + ", " + isstr( A10->col_is(), ntile ) + " = " + normstr( blas::normF( *(T.data) ) ) );
        
        addlr( tiled_matrix( 'A', A10->id(), A10->row_is(), & A10->U() ),
               T,
               tiled_matrix( 'A', A01->id(), A01->col_is(), & A01->V() ),
               A11, acc, ntile );
        
        lu( A11, acc, ntile );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        blas::invert( blas_mat< real >( DA ) );

        HLR_LOG( 5, "lu    :         " + idstr( A->id() ) + "        = " + normstr( norm_F( A ) ) );
    }// else
}

}// namespace hodlr

}}}// namespace hlr::seq::tile

#endif // __HLR_SEQ_ARITH_TILED_V3_HH
