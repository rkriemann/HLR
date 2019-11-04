#ifndef __HLR_SEQ_ARITH_TILED_HH
#define __HLR_SEQ_ARITH_TILED_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential tile-based arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/matrix.hh"

namespace hlr { namespace seq { namespace tiled {

using namespace HLIB;

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
        const BLAS::Range  r0( r.first(), r.first() + r.size() / 2 - 1 );
        const BLAS::Range  r1( r0.last() + 1, r.last() );

        return { std::move(r0), std::move(r1) };
    }// if
    else
        assert( false );

    return {};
}

//
// compute T := A^H · B
//
template < typename value_t >
BLAS::Matrix< value_t >
dot ( const BLAS::Matrix< value_t > &  A,
      const BLAS::Matrix< value_t > &  B,
      const size_t                     ntile )
{
    assert( A.nrows() == B.nrows() );

    HLR_LOG( 4, HLIB::to_string( "dot( %d )", B.nrows() ) );
    
    if ( A.nrows() > ntile )
    {
        const auto                     R = split( BLAS::Range( 0, A.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  A0( A, R[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  A1( A, R[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  B0( B, R[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  B1( B, R[1], BLAS::Range::all );
        
        auto  T0 = dot( A0, B0, ntile );
        auto  T1 = dot( A1, B1, ntile );

        BLAS::add( value_t(1), T0, T1 );

        return T1;
    }// if
    else
    {
        return std::move( BLAS::prod( value_t(1), BLAS::adjoint( A ), B ) );
    }// else
}

//
// compute B := β·B + α·A·T
//
template < typename value_t >
void
tprod ( const value_t                    alpha,
        const BLAS::Matrix< value_t > &  A,
        const BLAS::Matrix< value_t > &  T,
        const value_t                    beta,
        BLAS::Matrix< value_t > &        B,
        const size_t                     ntile )
{
    assert( A.nrows() == B.nrows() );
    assert( A.ncols() == T.nrows() );
    assert( T.ncols() == B.ncols() );

    HLR_LOG( 4, HLIB::to_string( "tprod( %d )", B.nrows() ) );
    
    if ( A.ncols() > ntile )
    {
        const auto                     R = split( BLAS::Range( 0, A.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  A0( A, R[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  A1( A, R[1], BLAS::Range::all );
        BLAS::Matrix< value_t >        B0( B, R[0], BLAS::Range::all );
        BLAS::Matrix< value_t >        B1( B, R[1], BLAS::Range::all );

        tprod( alpha, A0, T, beta, B0, ntile );
        tprod( alpha, A1, T, beta, B1, ntile );
    }// if
    else
    {
        BLAS::prod( alpha, A, T, beta, B );
    }// else
}

template < typename value_t >
void
tprod ( const value_t                    alpha,
        BLAS::Matrix< value_t > &        A,
        const BLAS::Matrix< value_t > &  T,
        const size_t                     ntile )
{
    assert( A.ncols() == T.nrows() );

    HLR_LOG( 4, HLIB::to_string( "tprod( %d )", A.nrows() ) );
    
    if ( A.ncols() > ntile )
    {
        const auto               R = split( BLAS::Range( 0, A.nrows()-1 ), 2 );
        BLAS::Matrix< value_t >  A0( A, R[0], BLAS::Range::all );
        BLAS::Matrix< value_t >  A1( A, R[1], BLAS::Range::all );

        tprod( alpha, A0, T, ntile );
        tprod( alpha, A1, T, ntile );
    }// if
    else
    {
        BLAS::Matrix< value_t >  Ac( A, HLIB::copy_value );
        
        BLAS::prod( alpha, Ac, T, value_t(0), A );
    }// else
}

//
// compute QR factorization of [αX·T,U]
//
template < typename value_t >
std::pair< BLAS::Matrix< value_t >,
           BLAS::Matrix< value_t > >
tsqr ( const value_t                 alpha,
       const BLAS::Matrix< real > &  X,
       const BLAS::Matrix< real > &  T,
       const BLAS::Matrix< real > &  U,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    assert( X.ncols() == T.nrows() );
    
    HLR_LOG( 4, HLIB::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto                     rows = split( BLAS::Range( 0, X.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  X0( X, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  X1( X, rows[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U0( U, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U1( U, rows[1], BLAS::Range::all );

        auto [ Q0, R0 ] = tsqr( alpha, X0, T, U0, ntile );
        auto [ Q1, R1 ] = tsqr( alpha, X1, T, U1, ntile );

        // Q = | R0 |
        //     | R1 |
        BLAS::Matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        BLAS::Matrix< value_t >  Q01_0( Q01, BLAS::Range(          0, R0.nrows()-1  ), BLAS::Range::all );
        BLAS::Matrix< value_t >  Q01_1( Q01, BLAS::Range( R0.nrows(), Q01.nrows()-1 ), BLAS::Range::all );
        BLAS::Matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        BLAS::copy( R0, Q01_0 );
        BLAS::copy( R1, Q01_1 );

        BLAS::qr( Q01, R );

        BLAS::Matrix< value_t >  Q( X.nrows(), Q01.ncols() );
        BLAS::Matrix< value_t >  Q_0( Q, rows[0], BLAS::Range::all );
        BLAS::Matrix< value_t >  Q_1( Q, rows[1], BLAS::Range::all );

        tprod( value_t(1), Q0, Q01_0, value_t(0), Q_0, ntile );
        tprod( value_t(1), Q1, Q01_1, value_t(0), Q_1, ntile );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        auto                     W = BLAS::prod( alpha, X, T );
        BLAS::Matrix< value_t >  WU( W.nrows(), W.ncols() + U.ncols () );
        BLAS::Matrix< value_t >  WU_W( WU, BLAS::Range::all, BLAS::Range( 0, W.ncols()-1 ) );
        BLAS::Matrix< value_t >  WU_U( WU, BLAS::Range::all, BLAS::Range( W.ncols(), WU.ncols()-1 ) );

        BLAS::copy( W, WU_W );
        BLAS::copy( U, WU_U );

        BLAS::Matrix< value_t >  R;
        
        BLAS::qr( WU, R );

        return { std::move( WU ), std::move( R ) };
    }// else
}

template < typename value_t >
void
tsqr ( const value_t                 alpha,
       const BLAS::Matrix< real > &  X,
       const BLAS::Matrix< real > &  T,
       const BLAS::Matrix< real > &  U,
       BLAS::Matrix< real > &        Q,
       BLAS::Matrix< real > &        R,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    assert( X.ncols() == T.nrows() );
    
    HLR_LOG( 4, HLIB::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto                     rows = split( BLAS::Range( 0, X.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  X0( X, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  X1( X, rows[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U0( U, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U1( U, rows[1], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q0( Q, rows[0], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q1( Q, rows[1], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q01( Q0.ncols() + Q1.ncols(), Q0.ncols() );
        BLAS::Matrix< value_t >        R0( Q01, BLAS::Range(          0, Q0.ncols()-1  ), BLAS::Range::all );
        BLAS::Matrix< value_t >        R1( Q01, BLAS::Range( R0.nrows(), Q01.nrows()-1 ), BLAS::Range::all );

        tsqr( alpha, X0, T, U0, Q0, R0, ntile );
        tsqr( alpha, X1, T, U1, Q1, R1, ntile );

        // Q = | R0 |
        //     | R1 |
        BLAS::qr( Q01, R );

        tprod( value_t(1), Q0, R0, ntile );
        tprod( value_t(1), Q1, R1, ntile );
    }// if
    else
    {
        auto                     W = BLAS::prod( alpha, X, T );
        BLAS::Matrix< value_t >  WU_W( Q, BLAS::Range::all, BLAS::Range( 0, W.ncols()-1 ) );
        BLAS::Matrix< value_t >  WU_U( Q, BLAS::Range::all, BLAS::Range( W.ncols(), Q.ncols()-1 ) );

        BLAS::copy( W, WU_W );
        BLAS::copy( U, WU_U );

        BLAS::qr( Q, R );
    }// else
}

//
// compute QR factorization of [αX,U]
//
template < typename value_t >
std::pair< BLAS::Matrix< value_t >,
           BLAS::Matrix< value_t > >
tsqr ( const value_t                 alpha,
       const BLAS::Matrix< real > &  X,
       const BLAS::Matrix< real > &  U,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    
    HLR_LOG( 4, HLIB::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto                     rows = split( BLAS::Range( 0, X.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  X0( X, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  X1( X, rows[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U0( U, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U1( U, rows[1], BLAS::Range::all );

        auto [ Q0, R0 ] = tsqr( alpha, X0, U0, ntile );
        auto [ Q1, R1 ] = tsqr( alpha, X1, U1, ntile );

        // Q = | R0 |
        //     | R1 |
        BLAS::Matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        BLAS::Matrix< value_t >  Q01_0( Q01, BLAS::Range(          0, R0.nrows()-1  ), BLAS::Range::all );
        BLAS::Matrix< value_t >  Q01_1( Q01, BLAS::Range( R0.nrows(), Q01.nrows()-1 ), BLAS::Range::all );
        BLAS::Matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        BLAS::copy( R0, Q01_0 );
        BLAS::copy( R1, Q01_1 );

        BLAS::qr( Q01, R );

        BLAS::Matrix< value_t >  Q( X.nrows(), Q01.ncols() );
        BLAS::Matrix< value_t >  Q_0( Q, rows[0], BLAS::Range::all );
        BLAS::Matrix< value_t >  Q_1( Q, rows[1], BLAS::Range::all );

        tprod( value_t(1), Q0, Q01_0, value_t(0), Q_0, ntile );
        tprod( value_t(1), Q1, Q01_1, value_t(0), Q_1, ntile );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        BLAS::Matrix< value_t >  XU( X.nrows(), X.ncols() + U.ncols () );
        BLAS::Matrix< value_t >  XU_X( XU, BLAS::Range::all, BLAS::Range( 0, X.ncols()-1 ) );
        BLAS::Matrix< value_t >  XU_U( XU, BLAS::Range::all, BLAS::Range( X.ncols(), XU.ncols()-1 ) );

        BLAS::copy( X, XU_X );
        BLAS::copy( U, XU_U );

        BLAS::Matrix< value_t >  R;
        
        BLAS::qr( XU, R );

        return { std::move( XU ), std::move( R ) };
    }// else
}

template < typename value_t >
void
tsqr ( const value_t                 alpha,
       const BLAS::Matrix< real > &  X,
       const BLAS::Matrix< real > &  U,
       BLAS::Matrix< real > &        Q,
       BLAS::Matrix< real > &        R,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    
    HLR_LOG( 4, HLIB::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        const auto                     rows = split( BLAS::Range( 0, X.nrows()-1 ), 2 );
        const BLAS::Matrix< value_t >  X0( X, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  X1( X, rows[1], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U0( U, rows[0], BLAS::Range::all );
        const BLAS::Matrix< value_t >  U1( U, rows[1], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q0( Q, rows[0], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q1( Q, rows[1], BLAS::Range::all );
        BLAS::Matrix< value_t >        Q01( Q0.ncols() + Q1.ncols(), Q0.ncols() );
        BLAS::Matrix< value_t >        R0( Q01, BLAS::Range(          0, Q0.ncols()-1  ), BLAS::Range::all );
        BLAS::Matrix< value_t >        R1( Q01, BLAS::Range( R0.nrows(), Q01.nrows()-1 ), BLAS::Range::all );

        tsqr( alpha, X0, U0, Q0, R0, ntile );
        tsqr( alpha, X1, U1, Q1, R1, ntile );
        
        // Q = | R0 |
        //     | R1 |
        BLAS::qr( Q01, R );

        tprod( value_t(1), Q0, R0, ntile );
        tprod( value_t(1), Q1, R1, ntile );
    }// if
    else
    {
        BLAS::Matrix< value_t >  XU_X( Q, BLAS::Range::all, BLAS::Range( 0, X.ncols()-1 ) );
        BLAS::Matrix< value_t >  XU_U( Q, BLAS::Range::all, BLAS::Range( X.ncols(), Q.ncols()-1 ) );

        BLAS::copy( X, XU_X );
        BLAS::copy( U, XU_U );

        BLAS::qr( Q, R );
    }// else
}

//
// truncate α X T Y^H + U V^H
//
template < typename value_t >
std::pair< BLAS::Matrix< value_t >,
           BLAS::Matrix< value_t > >
truncate ( const value_t                 alpha,
           const BLAS::Matrix< real > &  X,
           const BLAS::Matrix< real > &  T,
           const BLAS::Matrix< real > &  Y,
           const BLAS::Matrix< real > &  U,
           const BLAS::Matrix< real > &  V,
           const TTruncAcc &             acc,
           const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    assert( Y.nrows() == V.nrows() );
    assert( X.ncols() == T.nrows() );
    assert( T.ncols() == Y.ncols() );
    assert( U.ncols() == V.ncols() );
    
    HLR_LOG( 4, HLIB::to_string( "truncate( %d )", X.nrows() ) );
    
    if ( Y.ncols() + V.ncols() > X.nrows() / 2 )
    {
        // M = α X T Y^H + U V^H
        auto  W = BLAS::prod( value_t(1), X, T );
        auto  M = BLAS::prod( value_t(1), U, BLAS::adjoint( V ) );

        BLAS::prod( alpha, W, BLAS::adjoint( Y ), value_t(1), M );
            
        // truncate to rank-k
        return std::move( hlr::approx_svd( M, acc ) );
    }// if
    else
    {
        #if 0
        
        BLAS::Matrix< value_t >  Q0( X.nrows(), T.ncols() + U.ncols() );
        BLAS::Matrix< value_t >  Q1( Y.nrows(), Y.ncols() + V.ncols() );
        BLAS::Matrix< value_t >  R0( Q0.ncols(), Q0.ncols() );
        BLAS::Matrix< value_t >  R1( Q1.ncols(), Q1.ncols() );
        
        tsqr( alpha,      X, T, U, Q0, R0, ntile );
        tsqr( value_t(1), Y,    V, Q1, R1, ntile );
        
        #else
        
        auto [ Q0, R0 ] = tsqr( alpha,      X, T, U, ntile );
        auto [ Q1, R1 ] = tsqr( value_t(1), Y,    V, ntile );
        
        #endif
        
        auto R = BLAS::prod( value_t(1), R0, BLAS::adjoint( R1 ) );

        auto                     Us = std::move( R );
        BLAS::Matrix< value_t >  Vs;
        BLAS::Vector< value_t >  Ss;
        
        BLAS::svd( Us, Ss, Vs );
        
        auto  k  = acc.trunc_rank( Ss );

        BLAS::Matrix< value_t >  Usk( Us, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
        BLAS::Matrix< value_t >  Vsk( Vs, BLAS::Range::all, BLAS::Range( 0, k-1 ) );
        
        BLAS::prod_diag( Usk, Ss, k );

        BLAS::Matrix< value_t >  Uk( U.nrows(), k );
        BLAS::Matrix< value_t >  Vk( V.nrows(), k );

        tprod( value_t(1), Q0, Usk, value_t(0), Uk, ntile );
        tprod( value_t(1), Q1, Vsk, value_t(0), Vk, ntile );

        return { std::move( Uk ), std::move( Vk ) };
    }// else
}
    
namespace hodlr
{

template < typename value_t >       BLAS::Matrix< value_t > &  mat_U ( TRkMatrix *        A ) { assert( ! is_null( A ) ); return blas_mat_A< value_t >( A ); }
template < typename value_t >       BLAS::Matrix< value_t > &  mat_V ( TRkMatrix *        A ) { assert( ! is_null( A ) ); return blas_mat_B< value_t >( A ); }

template < typename value_t > const BLAS::Matrix< value_t > &  mat_U ( const TRkMatrix *  A ) { assert( ! is_null( A ) ); return blas_mat_A< value_t >( A ); }
template < typename value_t > const BLAS::Matrix< value_t > &  mat_V ( const TRkMatrix *  A ) { assert( ! is_null( A ) ); return blas_mat_B< value_t >( A ); }

template < typename value_t >       BLAS::Matrix< value_t > &  mat_U ( TRkMatrix &        A ) { return blas_mat_A< value_t >( & A ); }
template < typename value_t >       BLAS::Matrix< value_t > &  mat_V ( TRkMatrix &        A ) { return blas_mat_B< value_t >( & A ); }

template < typename value_t > const BLAS::Matrix< value_t > &  mat_U ( const TRkMatrix &  A ) { return blas_mat_A< value_t >( & A ); }
template < typename value_t > const BLAS::Matrix< value_t > &  mat_V ( const TRkMatrix &  A ) { return blas_mat_B< value_t >( & A ); }

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
addlr ( const BLAS::Matrix< value_t > &  U,
        const BLAS::Matrix< value_t > &  T,
        const BLAS::Matrix< value_t > &  V,
        TMatrix *                        A,
        const TTruncAcc &                acc,
        const size_t                     ntile )
{
    HLR_LOG( 4, HLIB::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        BLAS::Matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), BLAS::Range::all );

        addlr( U0, T, V0, A00, acc, ntile );
        
        auto  [ U01, V01 ] = truncate( value_t(-1), U0, T, V1, mat_U< value_t >( A01 ), mat_V< value_t >( A01 ), acc, ntile );

        A01->set_lrmat( U01, V01 );

        auto  [ U10, V10 ] = truncate( value_t(-1), U1, T, V0, mat_U< value_t >( A10 ), mat_V< value_t >( A10 ), acc, ntile );

        A10->set_lrmat( U10, V10 );
        
        addlr( U1, T, V1, A11, acc, ntile );
    }// if
    else
    {
        auto        D = ptrcast( A, TDenseMatrix );
        const auto  W = BLAS::prod( value_t(1), U, T );

        BLAS::prod( value_t(-1), W, BLAS::adjoint( V ), value_t(1), blas_mat< value_t >( D ) );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const TMatrix *            U,
         BLAS::Matrix< value_t > &  X,
         const size_t               ntile )
{
    HLR_LOG( 4, HLIB::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), BLAS::Range::all );
            
        trsmuh( U00, X0, ntile );

        auto  T = dot( mat_U< value_t >( U01 ), X0, ntile );
        
        tprod( value_t(-1), mat_V< value_t >( U01 ), T, value_t(1), X1, ntile );

        trsmuh( U11, X1, ntile );
    }// if
    else
    {
        auto  DU = cptrcast( U, TDenseMatrix );
        
        BLAS::Matrix< value_t >  Y( X, copy_value );

        BLAS::prod( value_t(1), BLAS::adjoint( blas_mat< value_t >( DU ) ), Y, value_t(0), X );

        std::cout << "trsmu " << X.nrows() << " : " << BLAS::norm_F( X ) << std::endl;
    }// else
}

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const TMatrix *            L,
        BLAS::Matrix< value_t > &  X,
        const size_t               ntile )
{
    HLR_LOG( 4, HLIB::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), BLAS::Range::all );
            
        trsml( L00, X0, ntile );

        auto  T = dot( mat_V< value_t >( L10 ), X0, ntile );

        tprod( value_t(-1), mat_U< value_t >( L10 ), T, value_t(1), X1, ntile );

        trsml( L11, X1, ntile );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        std::cout << "trsml : " << BLAS::norm_F( X ) << std::endl;
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
    HLR_LOG( 4, HLIB::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc, ntile );
        
        trsml(  A00, mat_U< value_t >( A01 ), ntile );
        trsmuh( A00, mat_V< value_t >( A10 ), ntile );

        // T = ( V(A_10)^H · U(A_01) )
        auto  T  = dot( mat_V< value_t >( A10 ), mat_U< value_t >( A01 ), ntile ); 

        std::cout << "dot : " << BLAS::norm_F( T ) << std::endl;
        
        addlr< value_t >( mat_U< value_t >( A10 ), T, mat_V< value_t >( A01 ), A11, acc, ntile );
        
        lu< value_t >( A11, acc, ntile );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        BLAS::invert( blas_mat< value_t >( DA ) );

        std::cout << "lu : " << norm_F( A ) << std::endl;
    }// else
}

}// namespace hodlr

}}}// namespace hlr::seq::tiled

#endif // __HLR_SEQ_ARITH_TILED_HH
