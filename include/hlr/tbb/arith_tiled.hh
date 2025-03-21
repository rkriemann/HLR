#ifndef __HLR_TBB_ARITH_TILED_HH
#define __HLR_TBB_ARITH_TILED_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : tile-based arithmetic functions using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/approx/svd.hh"
#include "hlr/tbb/matrix.hh"

namespace hlr { namespace tbb { namespace tiled {

namespace hpro = HLIB;

//
// split given range into <n> subsets
//
inline
std::vector< blas::range >
split ( const blas::range &  r,
        const size_t         n )
{
    if ( n == 2 )
    {
        const blas::range  r0( r.first(), r.first() + r.size() / 2 - 1 );
        const blas::range  r1( r0.last() + 1, r.last() );

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
blas::matrix< value_t >
dot ( const blas::matrix< value_t > &  A,
      const blas::matrix< value_t > &  B,
      const size_t                     ntile )
{
    assert( A.nrows() == B.nrows() );

    HLR_LOG( 4, hpro::to_string( "dot( %d )", B.nrows() ) );
    
    if ( A.nrows() > ntile )
    {
        const auto                     R = split( blas::range( 0, A.nrows()-1 ), 2 );
        const blas::matrix< value_t >  A0( A, R[0], blas::range::all );
        const blas::matrix< value_t >  A1( A, R[1], blas::range::all );
        const blas::matrix< value_t >  B0( B, R[0], blas::range::all );
        const blas::matrix< value_t >  B1( B, R[1], blas::range::all );

        blas::matrix< value_t >  T0, T1;

        ::tbb::parallel_invoke( [&,ntile] { T0 = std::move( dot( A0, B0, ntile ) ); },
                                [&,ntile] { T1 = std::move( dot( A1, B1, ntile ) ); } );

        blas::add( value_t(1), T0, T1 );

        return T1;
    }// if
    else
    {
        return std::move( blas::prod( value_t(1), blas::adjoint( A ), B ) );
    }// else
}

//
// compute B := β·B + α·A·T
//
template < typename value_t >
void
tprod ( const value_t                    alpha,
        const blas::matrix< value_t > &  A,
        const blas::matrix< value_t > &  T,
        const value_t                    beta,
        blas::matrix< value_t > &        B,
        const size_t                     ntile )
{
    assert( A.nrows() == B.nrows() );
    assert( A.ncols() == T.nrows() );
    assert( T.ncols() == B.ncols() );

    HLR_LOG( 4, hpro::to_string( "tprod( %d )", B.nrows() ) );
    
    if ( A.ncols() > ntile )
    {
        const auto                     R = split( blas::range( 0, A.nrows()-1 ), 2 );
        const blas::matrix< value_t >  A0( A, R[0], blas::range::all );
        const blas::matrix< value_t >  A1( A, R[1], blas::range::all );
        blas::matrix< value_t >        B0( B, R[0], blas::range::all );
        blas::matrix< value_t >        B1( B, R[1], blas::range::all );

        ::tbb::parallel_invoke( [&,ntile] { tprod( alpha, A0, T, beta, B0, ntile ); },
                                [&,ntile] { tprod( alpha, A1, T, beta, B1, ntile ); } );
    }// if
    else
    {
        blas::prod( alpha, A, T, beta, B );
    }// else
}

//
// compute QR factorization of [αX·T,U]
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
tsqr ( const value_t                 alpha,
       const blas::matrix< real > &  X,
       const blas::matrix< real > &  T,
       const blas::matrix< real > &  U,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    assert( X.ncols() == T.nrows() );
    
    HLR_LOG( 4, hpro::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // A = | Q0 R0 | = | Q0   | | R0 | = | Q0   | Q01 R
        //     | Q1 R1 |   |   Q1 | | R1 |   |   Q1 | 
        //
        
        const auto                     rows = split( blas::range( 0, X.nrows()-1 ), 2 );
        const blas::matrix< value_t >  X0( X, rows[0], blas::range::all );
        const blas::matrix< value_t >  X1( X, rows[1], blas::range::all );
        const blas::matrix< value_t >  U0( U, rows[0], blas::range::all );
        const blas::matrix< value_t >  U1( U, rows[1], blas::range::all );

        blas::matrix< value_t >  Q0, Q1;
        blas::matrix< value_t >  R0, R1;
        
        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = std::move( tsqr( alpha, X0, T, U0, ntile ) ); },
                                [&,ntile] { std::tie( Q1, R1 ) = std::move( tsqr( alpha, X1, T, U1, ntile ) ); } );

        // Q = | R0 |
        //     | R1 |
        blas::matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        blas::matrix< value_t >  Q01_0( Q01, blas::range(          0, R0.nrows()-1  ), blas::range::all );
        blas::matrix< value_t >  Q01_1( Q01, blas::range( R0.nrows(), Q01.nrows()-1 ), blas::range::all );
        blas::matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( R0, Q01_0 );
        blas::copy( R1, Q01_1 );

        blas::qr( Q01, R );

        blas::matrix< value_t >  Q( X.nrows(), Q01.ncols() );
        blas::matrix< value_t >  Q_0( Q, rows[0], blas::range::all );
        blas::matrix< value_t >  Q_1( Q, rows[1], blas::range::all );

        ::tbb::parallel_invoke( [&,ntile] { tprod( value_t(1), Q0, Q01_0, value_t(0), Q_0, ntile ); },
                                [&,ntile] { tprod( value_t(1), Q1, Q01_1, value_t(0), Q_1, ntile ); } );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        auto                     W = blas::prod( alpha, X, T );
        blas::matrix< value_t >  WU( W.nrows(), W.ncols() + U.ncols () );
        blas::matrix< value_t >  WU_W( WU, blas::range::all, blas::range( 0, W.ncols()-1 ) );
        blas::matrix< value_t >  WU_U( WU, blas::range::all, blas::range( W.ncols(), WU.ncols()-1 ) );

        blas::copy( W, WU_W );
        blas::copy( U, WU_U );

        blas::matrix< value_t >  R;
        
        blas::qr( WU, R );

        return { std::move( WU ), std::move( R ) };
    }// else
}

//
// compute QR factorization of [αX,U]
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
tsqr ( const value_t                 alpha,
       const blas::matrix< real > &  X,
       const blas::matrix< real > &  U,
       const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    
    HLR_LOG( 4, hpro::to_string( "tsqr( %d )", X.nrows() ) );
    
    if ( X.nrows() > ntile )
    {
        //
        // A = | Q0 R0 | = | Q0   | | R0 | = | Q0   | Q01 R
        //     | Q1 R1 |   |   Q1 | | R1 |   |   Q1 | 
        //
        
        const auto                     rows = split( blas::range( 0, X.nrows()-1 ), 2 );
        const blas::matrix< value_t >  X0( X, rows[0], blas::range::all );
        const blas::matrix< value_t >  X1( X, rows[1], blas::range::all );
        const blas::matrix< value_t >  U0( U, rows[0], blas::range::all );
        const blas::matrix< value_t >  U1( U, rows[1], blas::range::all );

        blas::matrix< value_t >  Q0, Q1;
        blas::matrix< value_t >  R0, R1;
        
        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = std::move( tsqr( alpha, X0, U0, ntile ) ); },
                                [&,ntile] { std::tie( Q1, R1 ) = std::move( tsqr( alpha, X1, U1, ntile ) ); } );

        // Q = | R0 |
        //     | R1 |
        blas::matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        blas::matrix< value_t >  Q01_0( Q01, blas::range(          0, R0.nrows()-1  ), blas::range::all );
        blas::matrix< value_t >  Q01_1( Q01, blas::range( R0.nrows(), Q01.nrows()-1 ), blas::range::all );
        blas::matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( R0, Q01_0 );
        blas::copy( R1, Q01_1 );

        blas::qr( Q01, R );

        blas::matrix< value_t >  Q( X.nrows(), Q01.ncols() );
        blas::matrix< value_t >  Q_0( Q, rows[0], blas::range::all );
        blas::matrix< value_t >  Q_1( Q, rows[1], blas::range::all );

        ::tbb::parallel_invoke( [&,ntile] { tprod( value_t(1), Q0, Q01_0, value_t(0), Q_0, ntile ); },
                                [&,ntile] { tprod( value_t(1), Q1, Q01_1, value_t(0), Q_1, ntile ); } );

        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        blas::matrix< value_t >  XU( X.nrows(), X.ncols() + U.ncols () );
        blas::matrix< value_t >  XU_X( XU, blas::range::all, blas::range( 0, X.ncols()-1 ) );
        blas::matrix< value_t >  XU_U( XU, blas::range::all, blas::range( X.ncols(), XU.ncols()-1 ) );

        blas::copy( X, XU_X );
        blas::copy( U, XU_U );

        blas::matrix< value_t >  R;
        
        blas::qr( XU, R );

        return { std::move( XU ), std::move( R ) };
    }// else
}

//
// truncate α X T Y^H + U V^H
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
truncate ( const value_t                 alpha,
           const blas::matrix< real > &  X,
           const blas::matrix< real > &  T,
           const blas::matrix< real > &  Y,
           const blas::matrix< real > &  U,
           const blas::matrix< real > &  V,
           const hpro::TTruncAcc &       acc,
           const size_t                  ntile )
{
    assert( X.nrows() == U.nrows() );
    assert( Y.nrows() == V.nrows() );
    assert( X.ncols() == T.nrows() );
    assert( T.ncols() == Y.ncols() );
    assert( U.ncols() == V.ncols() );
    
    HLR_LOG( 4, hpro::to_string( "truncate( %d )", X.nrows() ) );
    
    if ( Y.ncols() + V.ncols() > X.nrows() / 2 )
    {
        // M = α X T Y^H + U V^H
        auto  W = blas::prod( value_t(1), X, T );
        auto  M = blas::prod( value_t(1), U, blas::adjoint( V ) );

        blas::prod( alpha, W, blas::adjoint( Y ), value_t(1), M );
            
        // truncate to rank-k
        return std::move( hlr::approx::svd( M, acc ) );
    }// if
    else
    {
        blas::matrix< value_t >  Q0, Q1;
        blas::matrix< value_t >  R0, R1;
        
        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = std::move( tsqr( alpha,      X, T, U, ntile ) ); },
                                [&,ntile] { std::tie( Q1, R1 ) = std::move( tsqr( value_t(1), Y,    V, ntile ) ); } );

        auto                     R  = blas::prod( value_t(1), R0, blas::adjoint( R1 ) );
        auto                     Us = std::move( R );
        blas::matrix< value_t >  Vs;
        blas::vector< value_t >  Ss;
        
        blas::svd( Us, Ss, Vs );
        
        auto  k  = acc.trunc_rank( Ss );

        blas::matrix< value_t >  Usk( Us, blas::range::all, blas::range( 0, k-1 ) );
        blas::matrix< value_t >  Vsk( Vs, blas::range::all, blas::range( 0, k-1 ) );
        
        blas::prod_diag( Usk, Ss, k );

        blas::matrix< value_t >  Uk( U.nrows(), k );
        blas::matrix< value_t >  Vk( V.nrows(), k );

        ::tbb::parallel_invoke( [&,ntile] { tprod( value_t(1), Q0, Usk, value_t(0), Uk, ntile ); },
                                [&,ntile] { tprod( value_t(1), Q1, Vsk, value_t(0), Vk, ntile ); } );

        return { std::move( Uk ), std::move( Vk ) };
    }// else
}
    
namespace hodlr
{

template < typename value_t >       blas::matrix< value_t > &  mat_U ( hpro::TRkMatrix *        A ) { assert( ! is_null( A ) ); return hpro::blas_mat_A< value_t >( A ); }
template < typename value_t >       blas::matrix< value_t > &  mat_V ( hpro::TRkMatrix *        A ) { assert( ! is_null( A ) ); return hpro::blas_mat_B< value_t >( A ); }

template < typename value_t > const blas::matrix< value_t > &  mat_U ( const hpro::TRkMatrix *  A ) { assert( ! is_null( A ) ); return hpro::blas_mat_A< value_t >( A ); }
template < typename value_t > const blas::matrix< value_t > &  mat_V ( const hpro::TRkMatrix *  A ) { assert( ! is_null( A ) ); return hpro::blas_mat_B< value_t >( A ); }

template < typename value_t >       blas::matrix< value_t > &  mat_U ( hpro::TRkMatrix &        A ) { return hpro::blas_mat_A< value_t >( & A ); }
template < typename value_t >       blas::matrix< value_t > &  mat_V ( hpro::TRkMatrix &        A ) { return hpro::blas_mat_B< value_t >( & A ); }

template < typename value_t > const blas::matrix< value_t > &  mat_U ( const hpro::TRkMatrix &  A ) { return hpro::blas_mat_A< value_t >( & A ); }
template < typename value_t > const blas::matrix< value_t > &  mat_V ( const hpro::TRkMatrix &  A ) { return hpro::blas_mat_B< value_t >( & A ); }

///////////////////////////////////////////////////////////////////////
//
// tile based arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const hpro::TMatrix *      L,
        blas::matrix< value_t > &  X,
        const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, hpro::TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), hpro::TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        blas::matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), blas::range::all );
            
        hodlr::trsml( L00, X0, ntile );

        auto  T = dot( mat_V< value_t >( L10 ), X0, ntile );

        tprod( value_t(-1), mat_U< value_t >( L10 ), T, value_t(1), X1, ntile );

        trsml( L11, X1, ntile );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
    }// else
}

//
// compute A := A - U·T·V^H
//
template < typename value_t >
void
addlr ( const blas::matrix< value_t > &  U,
        const blas::matrix< value_t > &  T,
        const blas::matrix< value_t > &  V,
        hpro::TMatrix *                  A,
        const hpro::TTruncAcc &          acc,
        const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), blas::range::all );

        ::tbb::parallel_invoke(
            [&,ntile]
            {
                addlr( U0, T, V0, A00, acc, ntile );
            },
            
            [&,ntile]
            {
                auto  [ U01, V01 ] = truncate( value_t(-1), U0, T, V1, mat_U< value_t >( A01 ), mat_V< value_t >( A01 ), acc, ntile );

                A01->set_lrmat( U01, V01 );
            },

            [&,ntile]
            {
                auto  [ U10, V10 ] = truncate( value_t(-1), U1, T, V0, mat_U< value_t >( A10 ), mat_V< value_t >( A10 ), acc, ntile );

                A10->set_lrmat( U10, V10 );
            },

            [&,ntile]
            {
                addlr( U1, T, V1, A11, acc, ntile );
            } );
    }// if
    else
    {
        auto        D = ptrcast( A, hpro::TDenseMatrix );
        const auto  W = blas::prod( value_t(1), U, T );

        blas::prod( value_t(-1), W, blas::adjoint( V ), value_t(1), hpro::blas_mat< value_t >( D ) );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const hpro::TMatrix *      U,
         blas::matrix< value_t > &  X,
         const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, hpro::TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), hpro::TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        blas::matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), blas::range::all );
            
        trsmuh( U00, X0, ntile );

        auto  T = dot( mat_U< value_t >( U01 ), X0, ntile );
        
        tprod( value_t(-1), mat_V< value_t >( U01 ), T, value_t(1), X1, ntile );

        trsmuh( U11, X1, ntile );
    }// if
    else
    {
        auto  DU = cptrcast( U, hpro::TDenseMatrix );
        
        blas::matrix< value_t >  Y( X, hpro::copy_value );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// compute A = LU
//
template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc,
     const size_t             ntile )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc, ntile );

        ::tbb::parallel_invoke( [&,ntile] { trsml(  A00, mat_U< value_t >( A01 ), ntile ); },
                                [&,ntile] { trsmuh( A00, mat_V< value_t >( A10 ), ntile ); } );

        // T = ( V(A_10)^H · U(A_01) )
        auto  T  = dot( mat_V< value_t >( A10 ), mat_U< value_t >( A01 ), ntile ); 

        addlr< value_t >( mat_U< value_t >( A10 ), T, mat_V< value_t >( A01 ), A11, acc, ntile );
        
        lu< value_t >( A11, acc, ntile );
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        blas::invert( hpro::blas_mat< value_t >( DA ) );
    }// else
}

}// namespace hodlr

}}}// namespace hlr::tbb::tiled

#endif // __HLR_TBB_ARITH_TILED_HH
