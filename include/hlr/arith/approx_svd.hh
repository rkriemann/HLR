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

#include <hlr/arith/blas.hh>

namespace hlr
{

namespace hpro = HLIB;

using hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename T >
std::pair< blas::matrix< T >, blas::matrix< T > >
approx_svd ( blas::matrix< T > &      M,
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
    blas::vector< real_t >   S( mrc );
    blas::matrix< value_t >  V( m, mrc );

    blas::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const blas::Range        row_is( 0, n-1 );
    const blas::Range        col_is( 0, m-1 );
    blas::matrix< value_t >  Uk( M, row_is, blas::Range( 0, k-1 ) );
    blas::matrix< value_t >  Vk( V, col_is, blas::Range( 0, k-1 ) );
    
    blas::matrix< value_t >  A( n, k );
    blas::matrix< value_t >  B( m, k );

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
std::pair< blas::matrix< T >, blas::matrix< T > >
truncate_svd ( const blas::matrix< T > &  U,
               const blas::matrix< T > &  V,
               const hpro::TTruncAcc &    acc )
{
    using  value_t = T;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    assert( U.ncols() == V.ncols() );

    const idx_t  n       = idx_t( U.nrows() );
    const idx_t  m       = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    blas::matrix< T >  OU, OV;
    
    if ( in_rank == 0 )
    {
        // reset matrices
        OU = std::move( blas::matrix< value_t >( n, 0 ) );
        OV = std::move( blas::matrix< value_t >( m, 0 ) );

        return { std::move( OU ), std::move( OV ) };
    }// if

    if ( in_rank <= acc_rank )
    {
        OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
        OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );

        return { std::move( OU ), std::move( OV ) };
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
        // build U = U·V^T
        //
            
        blas::matrix< value_t >  M( n, m );

        blas::prod( value_t(1), U, adjoint(V), value_t(0), M );
            
        //
        // truncate to rank-k
        //

        hpro::TTruncAcc  lacc( acc );

        lacc.set_max_rank( acc_rank );

        std::tie( OU, OV ) = hlr::approx_svd( M, lacc );
    }// if
    else
    {
        //
        // do QR-factorisation of U and V
        //

        blas::matrix< value_t >  QU, QV, RU, RV;

        QU = std::move( blas::matrix< value_t >( U.nrows(), in_rank ) );
        RU = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( U, QU );
        blas::qr( QU, RU );
        
        QV = std::move( blas::matrix< value_t >( V.nrows(), in_rank ) );
        RV = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( V, QV );
        blas::qr( QV, RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        blas::matrix< value_t >  R( in_rank, in_rank );

        blas::prod( value_t(1), RU, adjoint(RV), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) );
        blas::matrix< value_t >  Vs( std::move( RV ) );
            
        blas::svd( Us, Ss, Vs );
        
        // determine truncated rank based on singular values
        orank = idx_t( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < in_rank )
        {
            //
            // build new matrices U and V
            //

            const blas::Range  in_rank_is( 0, in_rank-1 );
            const blas::Range  orank_is( 0, orank-1 );

            // U := Q_U · U
            blas::matrix< value_t >  Urank( Us, in_rank_is, orank_is );
            
            // U := U·S
            blas::prod_diag( Urank, Ss, orank );
            OU = blas::prod( value_t(1), QU, Urank );
            
            // V := Q_V · conj(V)
            blas::matrix< value_t >  Vrank( Vs, in_rank_is, orank_is );

            OV = blas::prod( value_t(1), QV, Vrank );
        }// if
        else
        {
            OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
            OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );
        }// else
    }// else

    return { std::move( OU ), std::move( OV ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template< typename T >
std::pair< blas::matrix< T >, blas::matrix< T > >
approx_sum_svd ( const std::list< blas::matrix< T > > &  U,
                 const std::list< blas::matrix< T > > &  V,
                 const hpro::TTruncAcc &                 acc )
{
    assert( U.size() == V.size() );

    using  value_t = T;

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
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

        blas::matrix< value_t >  D( nrows, ncols );

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

        blas::matrix< value_t >  U_all( nrows, in_rank );
        blas::matrix< value_t >  V_all( ncols, in_rank );
        idx_t                    ofs = 0;

        for ( auto &  U_i : U )
        {
            blas::matrix< value_t > U_all_i( U_all, blas::Range::all, blas::Range( ofs, ofs + U_i.ncols() - 1 ) );

            blas::copy( U_i, U_all_i );
            ofs += U_i.ncols();
        }// for

        ofs = 0;
    
        for ( auto &  V_i : V )
        {
            blas::matrix< value_t > V_all_i( V_all, blas::Range::all, blas::Range( ofs, ofs + V_i.ncols() - 1 ) );

            blas::copy( V_i, V_all_i );
            ofs += V_i.ncols();
        }// for

        //
        // truncate and return result
        //
    
        return hlr::truncate_svd( U_all, V_all, acc );
    }// else
}

//
// compute low-rank approximation of a sum Σ_i U_i T_i V_i^H using SVD
//
template< typename value_t >
std::pair< blas::matrix< value_t >, blas::matrix< value_t > >
approx_sum_svd ( const std::list< blas::matrix< value_t > > &  U,
                 const std::list< blas::matrix< value_t > > &  T,
                 const std::list< blas::matrix< value_t > > &  V,
                 const hpro::TTruncAcc &                       acc )
{
    assert( U.size() == T.size() );
    assert( T.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows   = U.front().nrows();
    const size_t  ncols   = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  T_i : T )
        in_rank += T_i.ncols();

    if ( in_rank >= std::min( nrows, ncols ) )
    {
        //
        // perform dense approximation
        //

        blas::matrix< value_t >  D( nrows, ncols );

        auto  U_i = U.cbegin();
        auto  T_i = T.cbegin();
        auto  V_i = V.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i, ++V_i )
        {
            const auto  UT_i = blas::prod( value_t(1), *U_i, *T_i );
            
            blas::prod( value_t(1), UT_i, blas::adjoint( *V_i ), value_t(1), D );
        }// for

        return hlr::approx_svd( D, acc );
    }// if
    else
    {
        //
        // concatenate matrices
        //

        blas::matrix< value_t >  U_all( nrows, in_rank );
        blas::matrix< value_t >  V_all( ncols, in_rank );
        idx_t                    ofs = 0;

        auto  U_i = U.cbegin();
        auto  T_i = T.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i )
        {
            blas::matrix< value_t > U_all_i( U_all, blas::Range::all, blas::Range( ofs, ofs + T_i->ncols() - 1 ) );

            blas::prod( value_t(1), *U_i, *T_i, value_t(1), U_all_i );
            ofs += T_i.ncols();
        }// for

        ofs = 0;
    
        for ( auto &  V_i : V )
        {
            blas::matrix< value_t > V_all_i( V_all, blas::Range::all, blas::Range( ofs, ofs + V_i.ncols() - 1 ) );

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
