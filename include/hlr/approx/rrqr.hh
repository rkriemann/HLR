#pragma once
#ifndef __HLR_APPROX_RRQR_HH
#define __HLR_APPROX_RRQR_HH
//
// Project     : HLR
// Module      : approx/rrqr
// Description : low-rank approximation functions using rank revealing QR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>
#include <hlr/approx/tools.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using Hpro::idx_t;

namespace detail
{

//
// determine "singular values" of R by looking at
// norms of R(i:·,i:·) for all i
// - R is assumed to be upper triangular(!)
//
template < typename value_t >
blas::vector< real_type_t< value_t > >
singular_values ( const blas::matrix< value_t > &  R )
{
    // return blas::sv( R );
    
    using  real_t = real_type_t< value_t >;

    HLR_ASSERT( R.nrows() == R.ncols() );
    
    const idx_t  n   = idx_t( R.nrows() );
    auto         S   = blas::vector< real_t >( n );
    auto         sum = value_t(0);
    
    for ( int  i = n-1; i >= 0; --i )
    {
        for ( int  j = i; j < n; ++j )
            sum += math::square( R(i,j) );

        S(i) = math::sqrt( sum );
    }// for

    return S;
}

//
// return truncation rank of R
//
template < typename value_t >
int
trunc_rank ( const blas::matrix< value_t > &  R,
             const accuracy &                 acc )
{
    auto  S = singular_values( R );

    return acc.trunc_rank( S );
}

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
rrqr ( blas::matrix< value_t > &  M,
       const accuracy &           acc )
{
    //
    // algorithm only works for nrows >= ncols, so proceed with
    // transposed matrix if ncols > nrows
    //

    const idx_t  nrows = idx_t( M.nrows() );
    const idx_t  ncols = idx_t( M.ncols() );

    if ( ncols > nrows )
    {
        //
        // compute RRQR for M^H, e.g., M^H = U·V^H
        // and return V·U^H
        //
        
        auto  MH = blas::matrix< value_t >( ncols, nrows );

        blas::copy( blas::adjoint( M ), MH );

        auto [ U, V ] = rrqr( MH, acc );

        return { std::move( V ), std::move( U ) };
    }// if
    
    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( nrows, ncols ) );
    
    //
    // perform column pivoted QR of M
    //

    auto  R = blas::matrix< value_t >( ncols, ncols );
    auto  P = std::vector< int >( ncols, 0 );

    blas::qrp( M, R, P );

    auto  k = detail::trunc_rank( R, acc );
    
    //
    // restrict first k columns
    //

    // U = Q_k
    auto  Qk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );
    auto  U  = blas::copy( Qk );

    // copy first k columns of R' to V, i.e., first k rows of R
    auto  Rk = blas::matrix< value_t >( R, blas::range( 0, k-1 ), blas::range::all );
    auto  TV = blas::matrix< value_t >( ncols, k );
    
    blas::copy( blas::adjoint( Rk ), TV );
    
    // then permute rows of TV (do P·R') and copy to V
    auto  V = blas::matrix< value_t >( ncols, k );
    
    for ( int i = 0; i < ncols; ++i )
    {
        auto  j    = P[i];
        auto  TV_i = TV.row( i );
        auto  V_j  = V.row( j );

        copy( TV_i, V_j );
    }// for

    return { std::move( U ), std::move( V ) };
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
rrqr_ortho ( blas::matrix< value_t > &  M,
             const accuracy &           acc )
{
    auto [ U, V ] = rrqr( M, acc );

    return detail::make_ortho( U, V );
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename T >
std::pair< blas::matrix< T >, blas::matrix< T > >
rrqr ( const blas::matrix< T > &  U,
       const blas::matrix< T > &  V,
       const accuracy &           acc )
{
    using  value_t = T;

    HLR_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
    {
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };
    }// if

    if ( in_rank <= idx_t(acc.rank()) )
    {
        return { std::move( blas::copy( U ) ),
                 std::move( blas::copy( V ) ) };
    }// if

    //
    // if input rank is larger than maximal rank, use dense approximation
    //

    if ( in_rank > std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return rrqr( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );
    
        // [ QV, RV ] = qr( V )
        auto  QV = blas::copy( V );
        auto  RV = blas::matrix< value_t >( in_rank, in_rank );

        blas::qr( QV, RV );

        // compute column-pivoted QR of U·RV'
        auto  QU = blas::prod( value_t(1), U, adjoint(RV) );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        auto  P  = std::vector< int >( in_rank, 0 );

        blas::qrp( QU, RU, P );

        auto  out_rank = detail::trunc_rank( RU, acc );
        
        // U = QU(:,1:k)
        auto  Qk = blas::matrix< value_t >( QU, blas::range::all, blas::range( 0, out_rank-1 ) );
        auto  OU = blas::copy( Qk );
        
        // V = QV · P  (V' = P' · QV')
        auto  QV_P = blas::matrix< value_t >( nrows_V, in_rank );
        
        for ( int  i = 0; i < in_rank; ++i )
        {
            auto  j      = P[i];
            auto  QV_P_i = QV_P.column( i );
            auto  Q_j    = QV.column( j );

            blas::copy( Q_j, QV_P_i );
        }// for

        auto  Rk = blas::matrix< value_t >( RU, blas::range( 0, out_rank-1 ), blas::range( 0, in_rank-1 ) );
        auto  OV = blas::prod( value_t(1), QV_P, blas::adjoint( Rk ) );

        return { std::move( OU ), std::move( OV ) };
    }// else
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
rrqr_ortho ( const blas::matrix< value_t > &  U,
             const blas::matrix< value_t > &  V,
             const accuracy &                 acc )
{
    auto [ TU, TV ] = rrqr( U, V, acc );

    return detail::make_ortho( TU, TV );
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using RRQR
//
template< typename value_t >
std::pair< blas::matrix< value_t >, blas::matrix< value_t > >
rrqr ( const std::list< blas::matrix< value_t > > &  U,
       const std::list< blas::matrix< value_t > > &  V,
       const accuracy &                              acc )
{
    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows_U = U.front().nrows();
    const size_t  nrows_V = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  U_i : U )
        in_rank += U_i.ncols();

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        //
        // perform dense approximation
        //

        auto  M   = blas::matrix< value_t >( nrows_U, nrows_V );
        auto  u_i = U.cbegin();
        auto  v_i = V.cbegin();
        
        for ( ; u_i != U.cend(); ++u_i, ++v_i )
            blas::prod( value_t(1), *u_i, blas::adjoint( *v_i ), value_t(1), M );

        return rrqr( M, acc );
    }// if
    else
    {
        //
        // concatenate matrices
        //

        auto   U_all = blas::matrix< value_t >( nrows_U, in_rank );
        auto   V_all = blas::matrix< value_t >( nrows_V, in_rank );
        idx_t  ofs   = 0;

        for ( auto &  U_i : U )
        {
            auto  U_all_i = blas::matrix< value_t >( U_all, blas::range::all, blas::range( ofs, ofs + U_i.ncols() - 1 ) );

            blas::copy( U_i, U_all_i );
            ofs += U_i.ncols();
        }// for

        ofs = 0;
    
        for ( auto &  V_i : V )
        {
            auto  V_all_i = blas::matrix< value_t >( V_all, blas::range::all, blas::range( ofs, ofs + V_i.ncols() - 1 ) );

            blas::copy( V_i, V_all_i );
            ofs += V_i.ncols();
        }// for

        //
        // truncate and return result
        //
    
        return rrqr( U_all, V_all, acc );
    }// else
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
rrqr_ortho ( const std::list< blas::matrix< value_t > > &  U,
             const std::list< blas::matrix< value_t > > &  V,
             const accuracy &                              acc )
{
    auto [ TU, TV ] = rrqr( U, V, acc );

    return detail::make_ortho( TU, TV );
}

//
// compute low-rank approximation of a sum Σ_i U_i T_i V_i^H using RRQR
//
template< typename value_t >
std::pair< blas::matrix< value_t >, blas::matrix< value_t > >
rrqr ( const std::list< blas::matrix< value_t > > &  U,
       const std::list< blas::matrix< value_t > > &  T,
       const std::list< blas::matrix< value_t > > &  V,
       const accuracy &                              acc )
{
    HLR_ASSERT( U.size() == T.size() );
    HLR_ASSERT( T.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows_U = U.front().nrows();
    const size_t  nrows_V = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  T_i : T )
        in_rank += T_i.ncols();

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        //
        // perform dense approximation
        //

        auto  M   = blas::matrix< value_t >( nrows_U, nrows_V );
        auto  U_i = U.cbegin();
        auto  T_i = T.cbegin();
        auto  V_i = V.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i, ++V_i )
        {
            const auto  UT_i = blas::prod( value_t(1), *U_i, *T_i );
            
            blas::prod( value_t(1), UT_i, blas::adjoint( *V_i ), value_t(1), M );
        }// for

        return rrqr( M, acc );
    }// if
    else
    {
        //
        // concatenate matrices
        //

        auto   U_all = blas::matrix< value_t >( nrows_U, in_rank );
        auto   V_all = blas::matrix< value_t >( nrows_V, in_rank );
        idx_t  ofs   = 0;
        auto   U_i   = U.cbegin();
        auto   T_i   = T.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i )
        {
            auto  U_all_i = blas::matrix< value_t >( U_all, blas::range::all, blas::range( ofs, ofs + T_i->ncols() - 1 ) );

            blas::prod( value_t(1), *U_i, *T_i, value_t(1), U_all_i );
            ofs += T_i.ncols();
        }// for

        ofs = 0;
    
        for ( auto &  V_i : V )
        {
            auto  V_all_i = blas::matrix< value_t >( V_all, blas::range::all, blas::range( ofs, ofs + V_i.ncols() - 1 ) );

            blas::copy( V_i, V_all_i );
            ofs += V_i.ncols();
        }// for

        //
        // truncate and return result
        //
    
        return rrqr( U_all, V_all, acc );
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct RRQR
{
    using  value_t = T_value;
    using  real_t  = real_type_t< value_t >;
    
    // signal support for general lin. operators
    static constexpr bool supports_general_operator = false;
    
    //
    // matrix approximation routines
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const accuracy &           acc ) const
    {
        return hlr::approx::rrqr( M, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const 
    {
        return hlr::approx::rrqr( U, V, acc );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::rrqr( U, V, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::rrqr( U, T, V, acc );
    }

    //
    // matrix approximation routines (orthogonal version)
    //
    
    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( blas::matrix< value_t > &  M,
                   const accuracy &           acc ) const
    {
        return hlr::approx::rrqr_ortho( M, acc );
    }

    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const blas::matrix< value_t > &  U,
                   const blas::matrix< value_t > &  V,
                   const accuracy &                 acc ) const 
    {
        return hlr::approx::rrqr_ortho( U, V, acc );
    }
    
    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const std::list< blas::matrix< value_t > > &  U,
                   const std::list< blas::matrix< value_t > > &  V,
                   const accuracy &                              acc ) const
    {
        return hlr::approx::rrqr_ortho( U, V, acc );
    }

    //
    // compute (approximate) column basis
    //
    
    blas::matrix< value_t >
    column_basis ( blas::matrix< value_t > &  M,
                   const accuracy &           acc,
                   blas::vector< real_t > *   sv = nullptr ) const
    {
        // see "rrqr" above for comments

        const idx_t  ncols = idx_t( M.ncols() );

        // for update statistics
        HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), ncols ) );
    
        auto  R = blas::matrix< value_t >( ncols, ncols );
        auto  P = std::vector< int >( ncols, 0 );

        blas::qrp( M, R, P );

        auto  S  = detail::singular_values( R );
        auto  k  = std::min< idx_t >( M.ncols(), acc.trunc_rank( S ) ); // M might be adjusted(!)
        auto  Qk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );

        if ( ! is_null( sv ) )
        {
            if ( sv->length() != k )
                *sv = std::move( blas::vector< real_t >( k ) );
            
            for ( uint  i = 0; i < k; ++i )
                (*sv)(i) = S(i);
        }// if
        
        return blas::copy( Qk );
    }

    std::pair< blas::matrix< value_t >,
               blas::vector< real_type_t< value_t > > >
    column_basis ( const blas::matrix< value_t > &  M ) const
    {
        const idx_t  ncols = idx_t( M.ncols() );

        // for update statistics
        HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), ncols ) );

        auto  A = blas::copy( M );
        auto  R = blas::matrix< value_t >( ncols, ncols );
        auto  P = std::vector< int >( ncols, 0 );

        blas::qrp( A, R, P );

        auto  S  = detail::singular_values( R );

        return { std::move( A ), std::move( S ) };
    }
};

// signals, that T is of approximation type
template < typename T > struct is_approximation< RRQR< T > > { static const bool  value = true; };

}}// namespace hlr::approx

#endif // __HLR_APPROX_RRQR_HH
