#ifndef __HLR_SEQ_ARITH_TILED_V2_HH
#define __HLR_SEQ_ARITH_TILED_V2_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential tile-based arithmetic functions v2
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr { namespace seq { namespace tiled2 {

using namespace HLIB;

// map HLIB namespaces to HLR
namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

// dense matrix
template < typename value_t >
using  matrix   = HLIB::BLAS::Matrix< value_t >;

// dense vector
template < typename value_t >
using  vector   = HLIB::BLAS::Vector< value_t >;

// import matrix types
using hlr::matrix::indexset;
using hlr::matrix::range;
using hlr::matrix::tile;
using hlr::matrix::tile_storage;
using hlr::matrix::tiled_lrmatrix;

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
// compute T := A^H · B where A, B ∈ K^{is × k}
//
template < typename value_t >
matrix< value_t >
dot ( const indexset &                 is,
      const tile_storage< value_t > &  A,
      const tile_storage< value_t > &  B,
      const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "dot( %d )", is.size() ) );
    
    if ( is.size() > ntile )
    {
        const auto  sis = split( is, 2 );
        auto        T0  = dot( sis[0], A, B, ntile );
        auto        T1  = dot( sis[1], A, B, ntile );

        blas::add( value_t(1), T0, T1 );

        return T1;
    }// if
    else
    {
        assert( A.contains( is ) && B.contains( is ) );

        return blas::prod( value_t(1), blas::adjoint( A.at( is ) ), B.at( is ) );
    }// else
}

//
// compute B := β·B + α·A·T
//
template < typename value_t >
void
tprod ( const indexset &                 is,
        const value_t                    alpha,
        const tile_storage< value_t > &  A,
        const matrix< value_t > &        T,
        const value_t                    beta,
        tile_storage< value_t > &        B,
        const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "tprod( %d )", is.size() ) );
    
    if ( is.size() > ntile )
    {
        const auto  sis = split( is, 2 );

        tprod( sis[0], alpha, A, T, beta, B, ntile );
        tprod( sis[1], alpha, A, T, beta, B, ntile );
    }// if
    else
    {
        assert( A.contains( is ) );
        assert( ( beta == value_t(0) ) || B.contains( is ) );

        if ( B.contains( is ) )
        {
            hlr::log( 0, "tprod :          A , " + isstr( is, ntile ) + " = " + normstr( blas::normF( A.at( is ) ) ) );
            hlr::log( 0, "tprod :          T , " + isstr( is, ntile ) + " = " + normstr( blas::normF( T ) ) );
            hlr::log( 0, "tprod :          B , " + isstr( is, ntile ) + " = " + normstr( blas::normF( B.at( is ) ) ) );
            
            blas::prod( alpha, A.at( is ), T, beta, B.at( is ) );
        }// if
        else
        {
            hlr::log( 0, "tprod :          A , " + isstr( is, ntile ) + " = " + normstr( blas::normF( A.at( is ) ) ) );
            hlr::log( 0, "tprod :          T , " + isstr( is, ntile ) + " = " + normstr( blas::normF( T ) ) );
            
            B[ is ] = std::move( blas::prod( alpha, A.at( is ), T ) );
        }// else

        hlr::log( 0, "tprod :          C , " + isstr( is, ntile ) + " = " + normstr( blas::normF( B.at( is ) ) ) );
    }// else
}

template < typename value_t >
void
tprod ( const indexset &           is,
        const value_t              alpha,
        tile_storage< value_t > &  A,
        const matrix< value_t > &  T,
        const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "tprod( %d )", is.size() ) );
    
    if ( is.size() > ntile )
    {
        const auto  sis = split( is, 2 );

        tprod( sis[0], alpha, A, T, ntile );
        tprod( sis[1], alpha, A, T, ntile );
    }// if
    else
    {
        assert( A.contains( is ) );
        
        matrix< value_t >  Ac( A.at( is ), copy_value );
        
        blas::prod( alpha, Ac, T, value_t(0), A.at( is ) );
    }// else
}

//
// compute QR factorization of [αX·T,U]
//
template < typename value_t >
std::pair< tile_storage< value_t >,
           matrix< value_t > >
tsqr ( const indexset &                 is,
       const value_t                    alpha,
       const tile_storage< value_t > &  X,
       const matrix< value_t > &        T,
       const tile_storage< value_t > &  U,
       const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "tsqr( %d )", is.size() ) );
    
    if ( is.size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis = split( is, 2 );

        auto [ Q0, R0 ] = tsqr( sis[0], alpha, X, T, U, ntile );
        auto [ Q1, R1 ] = tsqr( sis[1], alpha, X, T, U, ntile );

        // Q = | R0 |
        //     | R1 |
        matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        matrix< value_t >  Q01_0( Q01, range(          0, R0.nrows()-1  ), range::all );
        matrix< value_t >  Q01_1( Q01, range( R0.nrows(), Q01.nrows()-1 ), range::all );
        matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( R0, Q01_0 );
        blas::copy( R1, Q01_1 );

        blas::qr( Q01, R );

        tile_storage< value_t >  Q;

        tprod( sis[0], value_t(1), Q0, Q01_0, value_t(0), Q, ntile );
        tprod( sis[1], value_t(1), Q1, Q01_1, value_t(0), Q, ntile );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        assert( X.contains( is ) && U.contains( is ) );

        const auto         X_is = X.at( is );
        const auto         U_is = U.at( is );
        auto               W    = blas::prod( alpha, X_is, T );
        matrix< value_t >  WU( W.nrows(), W.ncols() + U_is.ncols () );
        matrix< value_t >  WU_W( WU, range::all, range( 0, W.ncols()-1 ) );
        matrix< value_t >  WU_U( WU, range::all, range( W.ncols(), WU.ncols()-1 ) );

        blas::copy( W,    WU_W );
        blas::copy( U_is, WU_U );

        matrix< value_t >  R;
        
        blas::qr( WU, R );

        tile_storage< value_t >  Q;

        Q[ is ] = std::move( WU );
        
        hlr::log( 0, "tsqr  :          Q , " + isstr( is, ntile ) + " = " + normstr( blas::normF( Q.at( is ) ) ) );
        hlr::log( 0, "tsqr  :          R , " + isstr( is, ntile ) + " = " + normstr( blas::normF( R ) ) );
        
        return { std::move( Q ), std::move( R ) };
    }// else
}

//
// compute QR factorization of [αX,U]
//
template < typename value_t >
std::pair< tile_storage< value_t >,
           matrix< value_t > >
tsqr ( const indexset &                 is,
       const value_t                    alpha,
       const tile_storage< value_t > &  X,
       const tile_storage< value_t > &  U,
       const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "tsqr( %d )", is.size() ) );
    
    if ( is.size() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto  sis = split( is, 2 );

        auto [ Q0, R0 ] = tsqr( sis[0], alpha, X, U, ntile );
        auto [ Q1, R1 ] = tsqr( sis[1], alpha, X, U, ntile );

        // Q = | R0 |
        //     | R1 |
        matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        matrix< value_t >  Q01_0( Q01, range(          0, R0.nrows()-1  ), range::all );
        matrix< value_t >  Q01_1( Q01, range( R0.nrows(), Q01.nrows()-1 ), range::all );
        matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( R0, Q01_0 );
        blas::copy( R1, Q01_1 );

        blas::qr( Q01, R );

        tile_storage< value_t >  Q;

        tprod( sis[0], value_t(1), Q0, Q01_0, value_t(0), Q, ntile );
        tprod( sis[1], value_t(1), Q1, Q01_1, value_t(0), Q, ntile );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        assert( X.contains( is ) && U.contains( is ) );

        const auto         X_is = X.at( is );
        const auto         U_is = U.at( is );
        matrix< value_t >  XU( X_is.nrows(), X_is.ncols() + U_is.ncols () );
        matrix< value_t >  XU_X( XU, range::all, range( 0, X_is.ncols()-1 ) );
        matrix< value_t >  XU_U( XU, range::all, range( X_is.ncols(), XU.ncols()-1 ) );

        blas::copy( X_is, XU_X );
        blas::copy( U_is, XU_U );

        matrix< value_t >  R;
        
        blas::qr( XU, R );

        tile_storage< value_t >  Q;

        Q[ is ] = std::move( XU );
        
        hlr::log( 0, "tsqr  :          Q , " + isstr( is, ntile ) + " = " + normstr( blas::normF( Q.at( is ) ) ) );
        hlr::log( 0, "tsqr  :          R , " + isstr( is, ntile ) + " = " + normstr( blas::normF( R ) ) );
        
        return { std::move( Q ), std::move( R ) };
    }// else
}

//
// truncate α X T Y^H + U V^H
//
template < typename value_t >
std::pair< tile_storage< value_t >,
           tile_storage< value_t > >
truncate ( const indexset &                 row_is,
           const indexset &                 col_is,
           const value_t                    alpha,
           const tile_storage< value_t > &  X,
           const matrix< value_t > &        T,
           const tile_storage< value_t > &  Y,
           const tile_storage< value_t > &  U,
           const tile_storage< value_t > &  V,
           const TTruncAcc &                acc,
           const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "truncate( %d )", row_is.size() ) );
    
    // if ( Y.ncols() + V.ncols() > X.nrows() / 2 )
    // {
    //     // M = α X T Y^H + U V^H
    //     auto  W = blas::prod( value_t(1), X, T );
    //     auto  M = blas::prod( value_t(1), U, blas::adjoint( V ) );

    //     blas::prod( alpha, W, blas::adjoint( Y ), value_t(1), M );
            
    //     // truncate to rank-k
    //     return std::move( hlr::approx_svd( M, acc ) );
    // }// if
    // else
    {
        {
            auto  DX = hlr::matrix::to_dense( X );
            auto  DU = hlr::matrix::to_dense( U );
            auto  DY = hlr::matrix::to_dense( Y );
            auto  DV = hlr::matrix::to_dense( V );
            
            DBG::write( DX, "X2.mat", "X2" );
            DBG::write(  T, "T2.mat", "T2" );
            DBG::write( DU, "U2.mat", "U2" );
            DBG::write( DY, "Y2.mat", "Y2" );
            DBG::write( DV, "V2.mat", "V2" );
        }
        
        auto [ Q0, R0 ] = tsqr( row_is, alpha,      X, T, U, ntile );
        auto [ Q1, R1 ] = tsqr( col_is, value_t(1), Y,    V, ntile );

        // auto  dQ0 = to_dense( Q0 );
        // auto  dQ1 = to_dense( Q1 );

        // DBG::write( dQ0, "Q0.mat", "Q0" );
        // DBG::write( dQ1, "Q1.mat", "Q1" );
        // DBG::write(  R0, "R0.mat", "R0" );
        // DBG::write(  R1, "R1.mat", "R1" );
        
        auto               R  = blas::prod( value_t(1), R0, blas::adjoint( R1 ) );
        auto               Us = std::move( R );
        matrix< value_t >  Vs;
        vector< value_t >  Ss;
        
        blas::svd( Us, Ss, Vs );

        // DBG::write( Us, "Us.mat", "Us" );
        // DBG::write( Ss, "Ss.mat", "Ss" );
        // DBG::write( Vs, "Vs.mat", "Vs" );
        
        auto  k = acc.trunc_rank( Ss );

        for ( int  i = 0; i < k; ++i )
            std::cout << Ss(i) << std::endl;
        
        matrix< value_t >  Usk( Us, range::all, range( 0, k-1 ) );
        matrix< value_t >  Vsk( Vs, range::all, range( 0, k-1 ) );
        
        blas::prod_diag( Usk, Ss, k );

        tile_storage< value_t >  Uk, Vk;

        tprod( row_is, value_t(1), Q0, Usk, value_t(0), Uk, ntile );
        tprod( col_is, value_t(1), Q1, Vsk, value_t(0), Vk, ntile );

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
template < typename value_t >
void
addlr ( const tile_storage< value_t > &  U,
        const matrix< value_t > &        T,
        const tile_storage< value_t > &  V,
        TMatrix *                        A,
        const TTruncAcc &                acc,
        const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        addlr( U, T, V, A00, acc, ntile );
        
        auto  [ U01, V01 ] = truncate( A01->row_is(), A01->col_is(),
                                       value_t(-1),
                                       U, T, V,
                                       A01->U(), A01->V(),
                                       acc, ntile );

        A01->set_lrmat( std::move( U01 ), std::move( V01 ) );

        auto  [ U10, V10 ] = truncate( A10->row_is(), A10->col_is(),
                                       value_t(-1),
                                       U, T, V,
                                       A10->U(), A10->V(),
                                       acc, ntile );

        A10->set_lrmat( std::move( U10 ), std::move( V10 ) );
        
        addlr( U, T, V, A11, acc, ntile );
    }// if
    else
    {
        assert( U.contains( A->row_is() ) && V.contains( A->col_is() ) );
        
        auto        D = ptrcast( A, TDenseMatrix );
        const auto  W = blas::prod( value_t(1), U.at( A->row_is() ), T );

        blas::prod( value_t(-1), W, blas::adjoint( V.at( A->col_is() ) ),
                    value_t(1), blas_mat< value_t >( D ) );

        hlr::log( 0, "addlr :         " + idstr( A->id() ) + ",       = " + normstr( hpro::norm_F( D ) ) );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const TMatrix *            U,
         tile_storage< value_t > &  X,
         const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  U11 = BU->block( 1, 1 );

        trsmuh( U00, X, ntile );

        auto  T = dot( U01->row_is(), U01->U(), X, ntile );

        hlr::log( 0, "trsmu :  dot :  " + idstr( U->id() ) + ", " + isstr( U01->row_is(), ntile ) + " = " + normstr( blas::normF( T ) ) );
        
        tprod( U01->col_is(), value_t(-1), U01->V(), T, value_t(1), X, ntile );

        trsmuh( U11, X, ntile );
    }// if
    else
    {
        assert( X.contains( U->row_is() ) );
        
        auto  DU = cptrcast( U, TDenseMatrix );

        auto               X_is = X.at( U->row_is() );
        matrix< value_t >  Y( X_is, copy_value );

        blas::prod( value_t(1), blas::adjoint( blas_mat< value_t >( DU ) ), Y, value_t(0), X_is );

        hlr::log( 0, "trsmu :         " + idstr( U->id() ) + "        = " + normstr( blas::normF( X_is ) ) );
    }// else
}

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const TMatrix *            L,
        tile_storage< value_t > &  X,
        const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  L11 = BL->block( 1, 1 );

        trsml( L00, X, ntile );

        auto  T = dot( L10->col_is(), L10->V(), X, ntile );

        hlr::log( 0, "trsml :  dot :  " + idstr( L->id() ) + ", " + isstr( L10->col_is(), ntile ) + " = " + normstr( blas::normF( T ) ) );
        
        tprod( L10->row_is(), value_t(-1), L10->U(), T, value_t(1), X, ntile );

        trsml( L11, X, ntile );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //

        hlr::log( 0, "trsml :         " + idstr( L->id() ) + "        = " + normstr( blas::normF( X.at( L->row_is() ) ) ) );

        // DEBUG
        // {
        //     auto  DX = hlr::matrix::to_dense( X );

        //     DBG::write( X.at( L->row_is() ), "X.mat", "X" );
        //     std::exit( 0 );
        // }
    }// else
}

//
// compute A = LU
//
template < typename value_t >
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
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc, ntile );
        
        trsml(  A00, A01->U(), ntile );
        trsmuh( A00, A10->V(), ntile );

        // T = ( V(A_10)^H · U(A_01) )
        auto  T = dot( A10->col_is(), A10->V(), A01->U(), ntile ); 

        hlr::log( 0, "lu    :  dot :  " + idstr( A->id() ) + ", " + isstr( A10->col_is(), ntile ) + " = " + normstr( blas::normF( T ) ) );
        
        addlr< value_t >( A10->U(), T, A01->V(), A11, acc, ntile );
        
        lu< value_t >( A11, acc, ntile );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        blas::invert( blas_mat< value_t >( DA ) );

        hlr::log( 0, "lu    :         " + idstr( A->id() ) + "        = " + normstr( norm_F( A ) ) );
    }// else
}

}// namespace hodlr

}}}// namespace hlr::seq::tile

#endif // __HLR_SEQ_ARITH_TILE_HH
