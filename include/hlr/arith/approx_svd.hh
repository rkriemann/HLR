#ifndef __HLR_ARITH_APPROX_SVD_HH
#define __HLR_ARITH_APPROX_SVD_HH
//
// Project     : HLib
// File        : approx_svd.hh
// Description : low-rank approximation functions using SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <cassert>

#include <hpro/blas/Matrix.hh>
#include <hpro/blas/Algebra.hh>

namespace hlr
{

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

using hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename T >
std::pair< blas::Matrix< T >, blas::Matrix< T > >
approx_svd ( blas::Matrix< T > &      M,
             const hpro::TTruncAcc &  acc )
{
    using  value_t = T;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    //
    // perform SVD of M
    //

    const idx_t              n   = idx_t( M.nrows() );
    const idx_t              m   = idx_t( M.ncols() );
    const idx_t              mrc = std::min(n,m);
    blas::Vector< real_t >   S( mrc );
    blas::Matrix< value_t >  V( m, mrc );

    blas::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const blas::Range        row_is( 0, n-1 );
    const blas::Range        col_is( 0, m-1 );
    blas::Matrix< value_t >  Uk( M, row_is, blas::Range( 0, k-1 ) );
    blas::Matrix< value_t >  Vk( V, col_is, blas::Range( 0, k-1 ) );
    
    blas::Matrix< value_t >  A( n, k );
    blas::Matrix< value_t >  B( m, k );

    blas::copy( Uk, A );
    blas::copy( Vk, B );

    if ( n < m ) prod_diag( A, S, k );
    else         prod_diag( B, S, k );

    return { std::move( A ), std::move( B ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template <typename T>
std::pair< blas::Matrix< T >, blas::Matrix< T > >
truncate_svd ( const blas::Matrix< T > &  A,
               const blas::Matrix< T > &  B,
               const hpro::TTruncAcc &    acc )
{
    using  value_t = T;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    assert( A.ncols() == B.ncols() );

    const idx_t  n     = idx_t( A.nrows() );
    const idx_t  m     = idx_t( B.nrows() );
    const idx_t  irank = idx_t( A.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    blas::Matrix< T >  OA, OB;
    
    if ( irank == 0 )
    {
        // reset matrices
        OA = std::move( blas::Matrix< value_t >( n, 0 ) );
        OB = std::move( blas::Matrix< value_t >( m, 0 ) );
    }// if

    if ( irank <= acc_rank )
    {
        OA = std::move( blas::Matrix< value_t >( A, hpro::copy_value ) );
        OB = std::move( blas::Matrix< value_t >( B, hpro::copy_value ) );
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
            
        blas::Matrix< value_t >  M( n, m );

        blas::prod( value_t(1), A, adjoint(B), value_t(0), M );
            
        //
        // truncate to rank-k
        //

        hpro::TTruncAcc  lacc( acc );

        lacc.set_max_rank( acc_rank );

        std::tie( OA, OB ) = hlr::approx_svd( M, lacc );
    }// if
    else
    {
        //
        // do QR-factorisation of A and B
        //

        blas::Matrix< value_t >  QA, QB, RA, RB;

        QA = std::move( blas::Matrix< value_t >( A.nrows(), irank ) );
        RA = std::move( blas::Matrix< value_t >( irank, irank ) );
        
        blas::copy( A, QA );
        blas::qr( QA, RA );
        
        QB = std::move( blas::Matrix< value_t >( B.nrows(), irank ) );
        RB = std::move( blas::Matrix< value_t >( irank, irank ) );
        
        blas::copy( B, QB );
        blas::qr( QB, RB );

        //
        // R = R_A · upper_triangular(QB)^H = R_B^H
        //
        
        blas::Matrix< value_t >  R( irank, irank );

        blas::prod( value_t(1), RA, adjoint(RB), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::Vector< real_t >   S( irank );
        blas::Matrix< value_t >  U( std::move( R ) );
        blas::Matrix< value_t >  V( std::move( RB ) );
            
        blas::svd( U, S, V );
        
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

            const blas::Range  irank_is( 0, irank-1 );
            const blas::Range  orank_is( 0, orank-1 );

            // A := Q_A · U
            blas::Matrix< value_t >  Urank( U, irank_is, orank_is );
            
            // U := U·S
            blas::prod_diag( Urank, S, orank );
            OA = blas::prod( value_t(1), QA, Urank );
            
            // B := Q_B · conj(V)
            blas::Matrix< value_t >  Vrank( V, irank_is, orank_is );

            OB = blas::prod( value_t(1), QB, Vrank );
        }// if
        else
        {
            OA = std::move( blas::Matrix< value_t >( A, hpro::copy_value ) );
            OB = std::move( blas::Matrix< value_t >( B, hpro::copy_value ) );
        }// else
    }// else

    return { std::move( OA ), std::move( OB ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template< typename T >
std::pair< blas::Matrix< T >, blas::Matrix< T > >
approx_sum_svd ( const std::list< blas::Matrix< T > > &  U,
                 const std::list< blas::Matrix< T > > &  V,
                 const hpro::TTruncAcc &                 acc )
{
    using  value_t = T;

    if ( U.empty() || V.empty() )
        return { std::move( blas::Matrix< value_t >() ),
                 std::move( blas::Matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows   = U.front().nrows();
    const size_t  ncols   = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  U_i : U )
        in_rank += U_i.ncols();

    if ( in_rank >= std::min( nrows, ncols ) )
    {
        //
        // perform dense approximation
        //

        blas::Matrix< value_t >  D( nrows, ncols );

        auto  u_i = U.cbegin();
        auto  v_i = V.cbegin();
        
        for ( ; u_i != U.cend(); ++u_i, ++v_i )
            blas::prod( value_t(1), *u_i, blas::adjoint( *v_i ), value_t(1), D );

        auto [ U_tr, V_tr ] = hlr::approx_svd( D, acc );

        return { std::move( U_tr ), std::move( V_tr ) };
    }// if
    else
    {
        //
        // concatenate matrices
        //

        blas::Matrix< value_t >  U_all( nrows, in_rank );
        blas::Matrix< value_t >  V_all( ncols, in_rank );
        idx_t                    ofs = 0;

        for ( auto &  U_i : U )
        {
            blas::Matrix< value_t > U_all_i( U_all, blas::Range::all, blas::Range( ofs, ofs + U_i.ncols() - 1 ) );

            blas::copy( U_i, U_all_i );
            ofs += U_i.ncols();
        }// for

        ofs = 0;
    
        for ( auto &  V_i : V )
        {
            blas::Matrix< value_t > V_all_i( V_all, blas::Range::all, blas::Range( ofs, ofs + V_i.ncols() - 1 ) );

            blas::copy( V_i, V_all_i );
            ofs += V_i.ncols();
        }// for

        //
        // truncate and return result
        //
    
        return hlr::truncate_svd( U_all, V_all, acc );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_APPROX_SVD_HH
