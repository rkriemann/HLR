#ifndef __HLR_APPROX_HH
#define __HLR_APPROX_HH
//
// Project     : HLib
// File        : approx.hh
// Description : low-rank approximation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>

#include <blas/Matrix.hh>
#include <blas/Algebra.hh>

#include "approx.hh"

namespace LR
{

template < typename T >
std::pair< HLIB::BLAS::Matrix< T >, HLIB::BLAS::Matrix< T > >
approx_svd ( HLIB::BLAS::Matrix< T > &  M,
             const HLIB::TTruncAcc &    acc )
{
    using  value_t = T;
    using  real_t  = typename HLIB::real_type< value_t >::type_t;

    //
    // perform SVD of M
    //

    const HLIB::idx_t              n   = idx_t( M.nrows() );
    const HLIB::idx_t              m   = idx_t( M.ncols() );
    const HLIB::idx_t              mrc = std::min(n,m);
    HLIB::BLAS::Vector< real_t >   S( mrc );
    HLIB::BLAS::Matrix< value_t >  V( m, mrc );

    HLIB::BLAS::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const HLIB::BLAS::Range        row_is( 0, n-1 );
    const HLIB::BLAS::Range        col_is( 0, m-1 );
    HLIB::BLAS::Matrix< value_t >  Uk( M, row_is, HLIB::BLAS::Range( 0, k-1 ) );
    HLIB::BLAS::Matrix< value_t >  Vk( V, col_is, HLIB::BLAS::Range( 0, k-1 ) );
    
    HLIB::BLAS::Matrix< value_t >  A( n, k );
    HLIB::BLAS::Matrix< value_t >  B( m, k );

    HLIB::BLAS::copy( Uk, A );
    HLIB::BLAS::copy( Vk, B );

    if ( n < m ) prod_diag( A, S, k );
    else         prod_diag( B, S, k );

    return { std::move( A ), std::move( B ) };
}

template <typename T>
std::pair< HLIB::BLAS::Matrix< T >, HLIB::BLAS::Matrix< T > >
truncate_svd ( const HLIB::BLAS::Matrix< T > &  A,
               const HLIB::BLAS::Matrix< T > &  B,
               const HLIB::TTruncAcc &          acc )
{
    using  value_t = T;
    using  real_t  = typename HLIB::real_type< value_t >::type_t;

    if ( A.ncols() != B.ncols() )
        HERROR( ERR_MAT_SIZE, "(BLAS) truncate_svd", "rank in A and B differs" );

    const idx_t  n     = idx_t( A.nrows() );
    const idx_t  m     = idx_t( B.nrows() );
    const idx_t  irank = idx_t( A.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    HLIB::BLAS::Matrix< T >  OA, OB;
    
    if ( irank == 0 )
    {
        // reset matrices
        OA = std::move( HLIB::BLAS::Matrix< value_t >( n, 0 ) );
        OB = std::move( HLIB::BLAS::Matrix< value_t >( m, 0 ) );
    }// if

    if ( irank <= acc_rank )
    {
        OA = std::move( HLIB::BLAS::Matrix< value_t >( A, copy_value ) );
        OB = std::move( HLIB::BLAS::Matrix< value_t >( B, copy_value ) );
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    const idx_t  mrc   = std::min(n, m);
    idx_t        orank = 0;
        
    if ( acc_rank >= mrc )
    {
        //
        // build U = A*B^T
        //
            
        HLIB::BLAS::Matrix< value_t >  M( n, m );

        HLIB::BLAS::prod( value_t(1), A, adjoint(B), value_t(0), M );
            
        //
        // truncate to rank-k
        //

        HLIB::TTruncAcc  lacc( acc );

        lacc.set_max_rank( acc_rank );

        std::tie( OA, OB ) = LR::approx_svd( M, lacc );
    }// if
    else
    {
        //
        // do QR-factorisation of A and B
        //

        HLIB::BLAS::Matrix< value_t >  QA, QB, RA, RB;

        QA = std::move( HLIB::BLAS::Matrix< value_t >( A.nrows(), irank ) );
        RA = std::move( HLIB::BLAS::Matrix< value_t >( irank, irank ) );
        
        HLIB::BLAS::copy( A, QA );
        HLIB::BLAS::qr( QA, RA );
        
        QB = std::move( HLIB::BLAS::Matrix< value_t >( B.nrows(), irank ) );
        RB = std::move( HLIB::BLAS::Matrix< value_t >( irank, irank ) );
        
        HLIB::BLAS::copy( B, QB );
        HLIB::BLAS::qr( QB, RB );

        //
        // R = R_A · upper_triangular(QB)^H = R_B^H
        //
        
        HLIB::BLAS::Matrix< value_t >  R( irank, irank );

        HLIB::BLAS::prod( value_t(1), RA, adjoint(RB), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        HLIB::BLAS::Vector< real_t >   S( irank );
        HLIB::BLAS::Matrix< value_t >  U( std::move( R ) );
        HLIB::BLAS::Matrix< value_t >  V( std::move( RB ) );
            
        HLIB::BLAS::svd( U, S, V );
        
        // determine truncated rank based on singular values
        orank = idx_t( acc.trunc_rank( S ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < irank )
        {
            //
            // build new matrices A and B
            //

            const HLIB::BLAS::Range  irank_is( 0, irank-1 );
            const HLIB::BLAS::Range  orank_is( 0, orank-1 );

            // A := Q_A · U
            HLIB::BLAS::Matrix< value_t >  Urank( U, irank_is, orank_is );
            
            // U := U·S
            HLIB::BLAS::prod_diag( Urank, S, orank );
            OA = HLIB::BLAS::prod( value_t(1), QA, Urank );
            
            // B := Q_B · conj(V)
            HLIB::BLAS::Matrix< value_t >  Vrank( V, irank_is, orank_is );

            OB = HLIB::BLAS::prod( value_t(1), QB, Vrank );
        }// if
        else
        {
            OA = std::move( HLIB::BLAS::Matrix< value_t >( A, copy_value ) );
            OB = std::move( HLIB::BLAS::Matrix< value_t >( B, copy_value ) );
        }// else
    }// else

    return { OA, OB };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template< typename T >
std::pair< HLIB::BLAS::Matrix< T >, HLIB::BLAS::Matrix< T > >
approx_sum_svd ( const std::list< HLIB::BLAS::Matrix< T > > &  U,
                 const std::list< HLIB::BLAS::Matrix< T > > &  V,
                 const TTruncAcc &                             acc )
{
    using  value_t = T;

    if ( U.empty() || V.empty() )
        return { HLIB::BLAS::Matrix< value_t >(), HLIB::BLAS::Matrix< value_t >() };
    
    //
    // determine maximal rank
    //

    const size_t  nrows   = U.front().nrows();
    const size_t  ncols   = V.front().nrows();
    HLIB::uint    in_rank = 0;

    for ( auto &  U_i : U )
        in_rank += U_i.ncols();

    //
    // concatenate matrices
    //

    HLIB::BLAS::Matrix< value_t >  U_all( nrows, in_rank );
    HLIB::BLAS::Matrix< value_t >  V_all( ncols, in_rank );
    HLIB::idx_t                    ofs = 0;

    for ( auto &  U_i : U )
    {
        HLIB::BLAS::Matrix< value_t > U_all_i( U_all, HLIB::BLAS::Range::all, HLIB::BLAS::Range( ofs, ofs + U_i.ncols() - 1 ) );

        HLIB::BLAS::copy( U_i, U_all_i );
        ofs += U_i.ncols();
    }// for

    ofs = 0;
    
    for ( auto &  V_i : V )
    {
        HLIB::BLAS::Matrix< value_t > V_all_i( V_all, HLIB::BLAS::Range::all, HLIB::BLAS::Range( ofs, ofs + V_i.ncols() - 1 ) );

        HLIB::BLAS::copy( V_i, V_all_i );
        ofs += V_i.ncols();
    }// for

    //
    // truncate and return result
    //
    
    auto [ U_tr, V_tr ] = HLIB::BLAS::truncate2_svd( U_all, V_all, acc );

    return { std::move( U_tr ), std::move( V_tr ) };

    // HLIB::BLAS::truncate( U_all, V_all, acc );

    // return { std::move( U_all ), std::move( V_all ) };
}

}// namespace LR

#endif // __HLR_APPROX_HH
