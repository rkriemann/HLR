#ifndef __HLR_TBB_ARITH_TILED_V2_HH
#define __HLR_TBB_ARITH_TILED_V2_HH
//
// Project     : HLib
// File        : arith.hh
// Description : tile-based arithmetic functions v2
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/tbb/matrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/vector/tiled_scalarvector.hh"

namespace hlr { namespace tbb { namespace tiled2 {

// map HLIB namespaces to HLR
namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

// cluster
using  cluster  = hpro::TCluster;

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
using hlr::vector::tiled_scalarvector;

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
        const auto         sis = split( is, 2 );
        matrix< value_t >  T0, T1;
        
        ::tbb::parallel_invoke( [&,ntile] { T0 = dot( sis[0], A, B, ntile ); },
                                [&,ntile] { T1 = dot( sis[1], A, B, ntile ); } );

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

        ::tbb::parallel_invoke( [&,ntile] { tprod( sis[0], alpha, A, T, beta, B, ntile ); },
                                [&,ntile] { tprod( sis[1], alpha, A, T, beta, B, ntile ); } );
    }// if
    else
    {
        assert( A.contains( is ) );
        assert( ( beta == value_t(0) ) || B.contains( is ) );

        if ( B.contains( is ) )
            blas::prod( alpha, A.at( is ), T, beta, B.at( is ) );
        else
            B[ is ] = std::move( blas::prod( alpha, A.at( is ), T ) );
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
    assert( A.ncols() == T.nrows() );

    HLR_LOG( 4, hpro::to_string( "tprod( %d )", A.nrows() ) );
    
    if ( A.ncols() > ntile )
    {
        const auto  sis = split( is, 2 );

        ::tbb::parallel_invoke( [&,ntile] { tprod( sis[0], alpha, A, T, ntile ); },
                                [&,ntile] { tprod( sis[1], alpha, A, T, ntile ); } );
    }// if
    else
    {
        assert( A.contains( is ) );
        
        matrix< value_t >  Ac( A.at( is ), hpro::copy_value );
        
        blas::prod( alpha, Ac, T, value_t(0), A.at( is ) );
    }// else
}

//
// compute B := A·T
// - assumption: binary cluster tree
//
template < typename value_t >
void
tprod ( const cluster &                  cl,
        const tile_storage< value_t > &  A,
        const matrix< value_t > &        T,
        tile_storage< value_t > &        B )
{
    HLR_LOG( 4, hpro::to_string( "tprod( %d )", cl.size() ) );
    
    if ( cl.is_leaf() )
    {
        assert( A.contains( cl ) );

        B[ cl ] = std::move( blas::prod( value_t(1), A.at( cl ), T ) );
    }// if
    else
    {
        assert(( cl.nsons() == 2 ) && ! is_null_any( cl.son( 0 ), cl.son( 1 )) );
        
        ::tbb::parallel_invoke( [&] { tprod( *cl.son( 0 ), A, T, B ); },
                                [&] { tprod( *cl.son( 1 ), A, T, B ); } );
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
        
        const auto               sis = split( is, 2 );
        tile_storage< value_t >  Q0, Q1;
        matrix< value_t >        R0, R1;

        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = tsqr( sis[0], alpha, X, T, U, ntile ); },
                                [&,ntile] { std::tie( Q1, R1 ) = tsqr( sis[1], alpha, X, T, U, ntile ); } );

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

        ::tbb::parallel_invoke( [&,ntile] { tprod( sis[0], value_t(1), Q0, Q01_0, value_t(0), Q, ntile ); },
                                [&,ntile] { tprod( sis[1], value_t(1), Q1, Q01_1, value_t(0), Q, ntile ); } );

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
        
        const auto               sis = split( is, 2 );
        tile_storage< value_t >  Q0, Q1;
        matrix< value_t >        R0, R1;

        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = tsqr( sis[0], alpha, X, U, ntile ); },
                                [&,ntile] { std::tie( Q1, R1 ) = tsqr( sis[1], alpha, X, U, ntile ); } );

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

        ::tbb::parallel_invoke( [&,ntile] { tprod( sis[0], value_t(1), Q0, Q01_0, value_t(0), Q, ntile ); },
                                [&,ntile] { tprod( sis[1], value_t(1), Q1, Q01_1, value_t(0), Q, ntile ); } );

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
        
        return { std::move( Q ), std::move( R ) };
    }// else
}

//
// compute QR factorization of X
// - assumption: binary cluster tree
//
template < typename value_t >
std::pair< tile_storage< value_t >,
           matrix< value_t > >
tsqr ( const cluster &                  cl,
       const tile_storage< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "tsqr( %d )", cl.size() ) );
    
    if ( cl.is_leaf() )
    {
        assert( X.contains( cl ) );

        auto                     X_cl = blas::copy( X.at( cl ) );
        matrix< value_t >        R;
        tile_storage< value_t >  Q;

        if ( X_cl.nrows() < X_cl.ncols() )
        {
            HLR_ERROR( "nrows < ncols" );
        }// if
        else
        {
            blas::qr( X_cl, R );
            Q[ cl ] = std::move( X_cl );
        }// else
        
        return { std::move( Q ), std::move( R ) };
    }// if
    else
    {
        assert(( cl.nsons() == 2 ) && ! is_null_any( cl.son( 0 ), cl.son( 1 )) );
        
        //
        // qr(A) = ⎡Q0  ⎤ qr⎡R0⎤ = ⎡⎡Q0  ⎤ Q01⎤ R
        //         ⎣  Q1⎦   ⎣R1⎦   ⎣⎣  Q1⎦    ⎦ 
        //
        
        tile_storage< value_t >  Q0, Q1;
        matrix< value_t >        R0, R1;

        ::tbb::parallel_invoke( [&] { std::tie( Q0, R0 ) = std::move( tsqr( *cl.son(0), X ) ); },
                                [&] { std::tie( Q1, R1 ) = std::move( tsqr( *cl.son(1), X ) ); } );

        // Q = | R0 |
        //     | R1 |
        matrix< value_t >  Q01(   R0.nrows() + R1.nrows(), R0.ncols() );
        matrix< value_t >  Q01_0( Q01, range( 0,          R0.nrows()-1  ), range::all );
        matrix< value_t >  Q01_1( Q01, range( R0.nrows(), Q01.nrows()-1 ), range::all );
        matrix< value_t >  R(     Q01.ncols(), Q01.ncols() );
        
        blas::copy( R0, Q01_0 );
        blas::copy( R1, Q01_1 );

        blas::qr( Q01, R );

        tile_storage< value_t >  Q;

        ::tbb::parallel_invoke( [&] { tprod( *cl.son(0), Q0, Q01_0, Q ); },
                                [&] { tprod( *cl.son(1), Q1, Q01_1, Q ); } );

        return { std::move( Q ), std::move( R ) };
    }// else
}

//
// truncate U V^H
//
template < typename value_t >
std::pair< tile_storage< value_t >,
           tile_storage< value_t > >
truncate ( const cluster &                  row_cl,
           const cluster &                  col_cl,
           const tile_storage< value_t > &  U,
           const tile_storage< value_t > &  V,
           const hpro::TTruncAcc &          acc )
{
    HLR_LOG( 4, hpro::to_string( "truncate( %d )", row_cl.size() ) );
    
    tile_storage< value_t >  Q0, Q1;
    matrix< value_t >        R0, R1;

    ::tbb::parallel_invoke( [&] { std::tie( Q0, R0 ) = std::move( tsqr( row_cl, U ) ); },
                            [&] { std::tie( Q1, R1 ) = std::move( tsqr( col_cl, V ) ); } );

    auto               R  = blas::prod( value_t(1), R0, blas::adjoint( R1 ) );
    auto               Us = std::move( R );
    matrix< value_t >  Vs;
    vector< value_t >  Ss;
        
    blas::svd( Us, Ss, Vs );

    auto  k = acc.trunc_rank( Ss );

    matrix< value_t >  Usk( Us, range::all, range( 0, k-1 ) );
    matrix< value_t >  Vsk( Vs, range::all, range( 0, k-1 ) );
        
    blas::prod_diag( Usk, Ss, k );

    tile_storage< value_t >  Uk, Vk;

    ::tbb::parallel_invoke( [&] { tprod( row_cl, Q0, Usk, Uk ); },
                            [&] { tprod( col_cl, Q1, Vsk, Vk ); } );

    return { std::move( Uk ), std::move( Vk ) };
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
           const hpro::TTruncAcc &          acc,
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
        tile_storage< value_t >  Q0, Q1;
        matrix< value_t >        R0, R1;

        ::tbb::parallel_invoke( [&,ntile] { std::tie( Q0, R0 ) = tsqr( row_is, alpha,      X, T, U, ntile ); },
                                [&,ntile] { std::tie( Q1, R1 ) = tsqr( col_is, value_t(1), Y,    V, ntile ); } );
        
        auto               R  = blas::prod( value_t(1), R0, blas::adjoint( R1 ) );
        auto               Us = std::move( R );
        matrix< value_t >  Vs;
        vector< value_t >  Ss;
        
        blas::svd( Us, Ss, Vs );
        
        auto  k = acc.trunc_rank( Ss );

        matrix< value_t >  Usk( Us, range::all, range( 0, k-1 ) );
        matrix< value_t >  Vsk( Vs, range::all, range( 0, k-1 ) );
        
        blas::prod_diag( Usk, Ss, k );

        tile_storage< value_t >  Uk, Vk;

        ::tbb::parallel_invoke( [&,ntile] { tprod( row_is, value_t(1), Q0, Usk, value_t(0), Uk, ntile ); },
                                [&,ntile] { tprod( col_is, value_t(1), Q1, Vsk, value_t(0), Vk, ntile ); } );

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
        hpro::TMatrix *                  A,
        const hpro::TTruncAcc &          acc,
        const size_t                     ntile )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        ::tbb::parallel_invoke(
            [&,ntile]
            {
                addlr( U, T, V, A00, acc, ntile );
            },
            
            [&,ntile]
            { 
                auto  [ U01, V01 ] = truncate( A01->row_is(), A01->col_is(),
                                               value_t(-1),
                                               U, T, V,
                                               A01->U(), A01->V(),
                                               acc, ntile );

                A01->set_lrmat( std::move( U01 ), std::move( V01 ) );
            },
            
            [&,ntile]
            { 
                auto  [ U10, V10 ] = truncate( A10->row_is(), A10->col_is(),
                                               value_t(-1),
                                               U, T, V,
                                               A10->U(), A10->V(),
                                               acc, ntile );
                
                A10->set_lrmat( std::move( U10 ), std::move( V10 ) );
            },

            [&,ntile]
            {
                addlr( U, T, V, A11, acc, ntile );
            } );
    }// if
    else
    {
        assert( U.contains( A->row_is() ) && V.contains( A->col_is() ) );
        
        auto        D = ptrcast( A, hpro::TDenseMatrix );
        const auto  W = blas::prod( value_t(1), U.at( A->row_is() ), T );

        blas::prod( value_t(-1), W, blas::adjoint( V.at( A->col_is() ) ),
                    value_t(1), hpro::blas_mat< value_t >( D ) );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const hpro::TMatrix *      U,
         tile_storage< value_t > &  X,
         const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U->id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, hpro::TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  U11 = BU->block( 1, 1 );

        trsmuh( U00, X, ntile );

        auto  T = dot( U01->row_is(), U01->U(), X, ntile );
        
        tprod( U01->col_is(), value_t(-1), U01->V(), T, value_t(1), X, ntile );

        trsmuh( U11, X, ntile );
    }// if
    else
    {
        assert( X.contains( U->row_is() ) );
        
        auto  DU = cptrcast( U, hpro::TDenseMatrix );

        auto               X_is = X.at( U->row_is() );
        matrix< value_t >  Y( X_is, hpro::copy_value );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( DU ) ), Y, value_t(0), X_is );

        // std::cout << "trsmu : " << blas::norm_F( X_is ) << std::endl;
    }// else
}

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const hpro::TMatrix *      L,
        tile_storage< value_t > &  X,
        const size_t               ntile )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L->id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, hpro::TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  L11 = BL->block( 1, 1 );

        trsml( L00, X, ntile );

        auto  T = dot( L10->col_is(), L10->V(), X, ntile );

        tprod( L10->row_is(), value_t(-1), L10->U(), T, value_t(1), X, ntile );

        trsml( L11, X, ntile );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //

        // std::cout << "trsml : " << blas::norm_F( X.at( L->row_is() ) ) << std::endl;
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
        auto  A01 = ptrcast( BA->block( 0, 1 ), tiled_lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), tiled_lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc, ntile );
        
        ::tbb::parallel_invoke( [&,ntile] { trsml(  A00, A01->U(), ntile ); },
                                [&,ntile] { trsmuh( A00, A10->V(), ntile ); } );

        // T = ( V(A_10)^H · U(A_01) )
        auto  T = dot( A10->col_is(), A10->V(), A01->U(), ntile ); 

        // std::cout << "dot : " << blas::norm_F( T ) << std::endl;
        
        addlr< value_t >( A10->U(), T, A01->V(), A11, acc, ntile );
        
        lu< value_t >( A11, acc, ntile );
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        blas::invert( hpro::blas_mat< value_t >( DA ) );

        // std::cout << "lu : " << norm_F( A ) << std::endl;
    }// else
}

}// namespace hodlr

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

namespace
{

template < typename value_t >
blas::Vector< value_t >
dot_vec ( const size_t                           lb,
          const size_t                           ub,
          const std::vector< indexset > &        is_vec,
          const tile_storage< value_t > &        M,
          const tiled_scalarvector< value_t > &  x )
{
    if ( ub - lb > 1 )
    {
        const auto  mid = (ub + lb) / 2;
        blas::Vector< value_t >  t0, t1;

        ::tbb::parallel_invoke( [&t0,lb,mid,&is_vec,&M,&x] () { t0 = dot_vec( lb, mid, is_vec, M, x ); },
                                [&t1,mid,ub,&is_vec,&M,&x] () { t1 = dot_vec( mid, ub, is_vec, M, x ); } );

        blas::add( value_t(1), t0, t1 );

        return t1;
    }// if
    else
    {
        return blas::mulvec( value_t(1), blas::adjoint( M.at( is_vec[lb] ) ), x.at( is_vec[lb] ) );
    }// else
}

}// namespace anonymous

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                          alpha,
          const hpro::matop_t                    op_M,
          const hpro::TMatrix &                  M,
          const tiled_scalarvector< value_t > &  x,
          tiled_scalarvector< value_t > &        y )
{
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [alpha,op_M,B,&x,&y] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j );
                        
                        if ( ! is_null( B_ij ) )
                            mul_vec( alpha, op_M, *B_ij, x, y );
                    }// for
                }// for
            } );
    }// if
    else if ( is_dense( M ) )
    {
        auto              D = cptrcast( &M, hpro::TDenseMatrix );
        std::scoped_lock  lock( y.tile_mtx( D->row_is( op_M ) ) );
        
        blas::mulvec( alpha,
                      blas::mat_view( op_M, hpro::blas_mat< value_t >( D ) ),
                      x.at( D->col_is( op_M ) ),
                      value_t(1),
                      y.at( D->row_is( op_M ) ) );
    }// if
    else if ( hlr::matrix::is_tiled_lowrank( M ) )
    {
        auto  R = cptrcast( &M, tiled_lrmatrix< value_t > );

        if ( op_M == hpro::apply_normal )
        {
            const auto  t = dot_vec( 0, R->V().tile_is().size(), R->V().tile_is(), R->V(), x );

            ::tbb::parallel_for< size_t >( 0, R->U().tile_is().size(),
                                           [alpha,R,&t,&y] ( const auto  i )
                                           {
                                               const auto        is = R->U().tile_is( i );
                                               std::scoped_lock  lock( y.tile_mtx( is ) );
                                               
                                               blas::mulvec( alpha, R->U().at( is ), t, value_t(1), y.at( is ) );
                                           } );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            assert( hpro::is_complex_type< value_t >::value == false );
            
            const auto  t = dot_vec( 0, R->U().tile_is().size(), R->U().tile_is(), R->U(), x ); // TODO: transposed !!!
            
            ::tbb::parallel_for< size_t >( 0, R->V().tile_is().size(),
                                           [alpha,R,&t,&y] ( const auto  i )
                                           {
                                               const auto        is = R->V().tile_is( i );
                                               std::scoped_lock  lock( y.tile_mtx( is ) );
                
                                               blas::mulvec( alpha, R->V().at( is ), t, value_t(1), y.at( is ) );
                                           } );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            const auto  t = dot_vec( 0, R->U().tile_is().size(), R->U().tile_is(), R->U(), x );
            
            ::tbb::parallel_for< size_t >( 0, R->V().tile_is().size(),
                                           [alpha,R,&t,&y] ( const auto  i )
                                           {
                                               const auto        is = R->V().tile_is( i );
                                               std::scoped_lock  lock( y.tile_mtx( is ) );
                
                                               blas::mulvec( alpha, R->V().at( is ), t, value_t(1), y.at( is ) );
                                           } );
        }// if
    }// if
    else
    {
        throw std::runtime_error( M.typestr() );
    }// else
}

}}}// namespace hlr::tbb::tiled2

#endif // __HLR_TBB_ARITH_TILE_HH
