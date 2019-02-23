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
#include <cassert>

#include <blas/Matrix.hh>
#include <blas/Algebra.hh>

namespace HLR
{

using namespace HLIB;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename T >
std::pair< BLAS::Matrix< T >, BLAS::Matrix< T > >
approx_svd ( BLAS::Matrix< T > &  M,
             const TTruncAcc &    acc )
{
    using  value_t = T;
    using  real_t  = typename real_type< value_t >::type_t;

    //
    // perform SVD of M
    //

    const idx_t              n   = idx_t( M.nrows() );
    const idx_t              m   = idx_t( M.ncols() );
    const idx_t              mrc = std::min(n,m);
    BLAS::Vector< real_t >   S( mrc );
    BLAS::Matrix< value_t >  V( m, mrc );

    BLAS::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const BLAS::Range        row_is( 0, n-1 );
    const BLAS::Range        col_is( 0, m-1 );
    BLAS::Matrix< value_t >  Uk( M, row_is, BLAS::Range( 0, k-1 ) );
    BLAS::Matrix< value_t >  Vk( V, col_is, BLAS::Range( 0, k-1 ) );
    
    BLAS::Matrix< value_t >  A( n, k );
    BLAS::Matrix< value_t >  B( m, k );

    BLAS::copy( Uk, A );
    BLAS::copy( Vk, B );

    if ( n < m ) prod_diag( A, S, k );
    else         prod_diag( B, S, k );

    return { std::move( A ), std::move( B ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template <typename T>
std::pair< BLAS::Matrix< T >, BLAS::Matrix< T > >
truncate_svd ( const BLAS::Matrix< T > &  A,
               const BLAS::Matrix< T > &  B,
               const TTruncAcc &          acc )
{
    using  value_t = T;
    using  real_t  = typename real_type< value_t >::type_t;

    assert( A.ncols() == B.ncols() );

    const idx_t  n     = idx_t( A.nrows() );
    const idx_t  m     = idx_t( B.nrows() );
    const idx_t  irank = idx_t( A.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    BLAS::Matrix< T >  OA, OB;
    
    if ( irank == 0 )
    {
        // reset matrices
        OA = std::move( BLAS::Matrix< value_t >( n, 0 ) );
        OB = std::move( BLAS::Matrix< value_t >( m, 0 ) );
    }// if

    if ( irank <= acc_rank )
    {
        OA = std::move( BLAS::Matrix< value_t >( A, copy_value ) );
        OB = std::move( BLAS::Matrix< value_t >( B, copy_value ) );
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
            
        BLAS::Matrix< value_t >  M( n, m );

        BLAS::prod( value_t(1), A, adjoint(B), value_t(0), M );
            
        //
        // truncate to rank-k
        //

        TTruncAcc  lacc( acc );

        lacc.set_max_rank( acc_rank );

        std::tie( OA, OB ) = HLR::approx_svd( M, lacc );
    }// if
    else
    {
        //
        // do QR-factorisation of A and B
        //

        BLAS::Matrix< value_t >  QA, QB, RA, RB;

        QA = std::move( BLAS::Matrix< value_t >( A.nrows(), irank ) );
        RA = std::move( BLAS::Matrix< value_t >( irank, irank ) );
        
        BLAS::copy( A, QA );
        BLAS::qr( QA, RA );
        
        QB = std::move( BLAS::Matrix< value_t >( B.nrows(), irank ) );
        RB = std::move( BLAS::Matrix< value_t >( irank, irank ) );
        
        BLAS::copy( B, QB );
        BLAS::qr( QB, RB );

        //
        // R = R_A · upper_triangular(QB)^H = R_B^H
        //
        
        BLAS::Matrix< value_t >  R( irank, irank );

        BLAS::prod( value_t(1), RA, adjoint(RB), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        BLAS::Vector< real_t >   S( irank );
        BLAS::Matrix< value_t >  U( std::move( R ) );
        BLAS::Matrix< value_t >  V( std::move( RB ) );
            
        BLAS::svd( U, S, V );
        
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

            const BLAS::Range  irank_is( 0, irank-1 );
            const BLAS::Range  orank_is( 0, orank-1 );

            // A := Q_A · U
            BLAS::Matrix< value_t >  Urank( U, irank_is, orank_is );
            
            // U := U·S
            BLAS::prod_diag( Urank, S, orank );
            OA = BLAS::prod( value_t(1), QA, Urank );
            
            // B := Q_B · conj(V)
            BLAS::Matrix< value_t >  Vrank( V, irank_is, orank_is );

            OB = BLAS::prod( value_t(1), QB, Vrank );
        }// if
        else
        {
            OA = std::move( BLAS::Matrix< value_t >( A, copy_value ) );
            OB = std::move( BLAS::Matrix< value_t >( B, copy_value ) );
        }// else
    }// else

    return { std::move( OA ), std::move( OB ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template< typename T >
std::pair< BLAS::Matrix< T >, BLAS::Matrix< T > >
approx_sum_svd ( const std::list< BLAS::Matrix< T > > &  U,
                 const std::list< BLAS::Matrix< T > > &  V,
                 const TTruncAcc &                       acc )
{
    using  value_t = T;

    if ( U.empty() || V.empty() )
        return { std::move( BLAS::Matrix< value_t >() ),
                 std::move( BLAS::Matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows   = U.front().nrows();
    const size_t  ncols   = V.front().nrows();
    uint    in_rank = 0;

    for ( auto &  U_i : U )
        in_rank += U_i.ncols();

    //
    // concatenate matrices
    //

    BLAS::Matrix< value_t >  U_all( nrows, in_rank );
    BLAS::Matrix< value_t >  V_all( ncols, in_rank );
    idx_t                    ofs = 0;

    for ( auto &  U_i : U )
    {
        BLAS::Matrix< value_t > U_all_i( U_all, BLAS::Range::all, BLAS::Range( ofs, ofs + U_i.ncols() - 1 ) );

        BLAS::copy( U_i, U_all_i );
        ofs += U_i.ncols();
    }// for

    ofs = 0;
    
    for ( auto &  V_i : V )
    {
        BLAS::Matrix< value_t > V_all_i( V_all, BLAS::Range::all, BLAS::Range( ofs, ofs + V_i.ncols() - 1 ) );

        BLAS::copy( V_i, V_all_i );
        ofs += V_i.ncols();
    }// for

    //
    // truncate and return result
    //
    
    return HLR::truncate_svd( U_all, V_all, acc );
}

}// namespace HLR

#endif // __HLR_APPROX_HH
