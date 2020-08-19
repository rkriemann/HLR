#ifndef __HLR_APPROX_RRQR_HH
#define __HLR_APPROX_RRQR_HH
//
// Project     : HLib
// Module      : approx/rrqr
// Description : low-rank approximation functions using rank revealing QR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

namespace detail
{

//
// determine truncate rank of R by looking at
// norms of R(i:·,i:·) for all i
//
template < typename value_t >
int
trunc_rank ( const blas::matrix< value_t > &  R,
             const hpro::TTruncAcc &          acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    HLR_ASSERT( R.nrows() == R.ncols() );
    
    const idx_t             n = idx_t( R.nrows() );
    blas::vector< real_t >  s( n );
    
    for ( int  i = 0; i < n; ++i )
    {
        auto  rest = blas::range( i, n-1 );
        auto  R_i  = blas::matrix< value_t >( R, rest, rest );
        
        s( i ) = blas::normF( R_i );
    }// for

    return acc.trunc_rank( s );
}

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
rrqr ( blas::matrix< value_t > &  M,
       const hpro::TTruncAcc &    acc )
{
    //
    // algorithm only works for nrows >= ncols, so proceed with
    // transposed matrix if ncols > nrows
    //

    const idx_t  nrows = idx_t( M.nrows() );
    const idx_t  ncols = idx_t( M.ncols() );

    if ( ncols > nrows )
    {
        blas::matrix< value_t >  MH( ncols, nrows );

        blas::copy( blas::adjoint( M ), MH );

        auto [ U, V ] = rrqr( MH, acc );

        return { std::move( V ), std::move( U ) };
    }// if
    
    //
    // perform column pivoted QR of M
    //

    blas::matrix< value_t >  R( ncols, ncols );
    std::vector< int >       P( ncols, 0 );

    blas::qrp( M, R, P );

    auto  k = detail::trunc_rank( R, acc );
    
    //
    // restrict first k columns
    //

    // U = Q_k
    auto  Qk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );
    auto  U  = blas::matrix< value_t >( nrows, k );
    
    blas::copy( Qk, U );

    // copy first k columns of R^T to V, i.e., first k rows of R
    blas::matrix< value_t >  Rk( R, blas::range( 0, k-1 ), blas::range::all );
    blas::matrix< value_t >  TV( ncols, k );
    
    blas::copy( blas::adjoint( Rk ), TV );
    
    // then permute rows of TV (do P·R^T) and copy to V
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

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template <typename T>
std::pair< blas::matrix< T >, blas::matrix< T > >
rrqr ( const blas::matrix< T > &  U,
       const blas::matrix< T > &  V,
       const hpro::TTruncAcc &    acc )
{
    using  value_t = T;

    HLR_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows   = idx_t( U.nrows() );
    const idx_t  ncols   = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
    {
        return { std::move( blas::matrix< value_t >( nrows, 0 ) ),
                 std::move( blas::matrix< value_t >( ncols, 0 ) ) };
    }// if

    if ( in_rank <= idx_t(acc.rank()) )
    {
        return { std::move( blas::copy( U ) ),
                 std::move( blas::copy( V ) ) };
    }// if

    //
    // if input rank is larger than maximal rank, use dense approximation
    //

    if ( in_rank > std::min( nrows, ncols ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return std::move( rrqr( M, acc ) );
    }// if
    else
    {
        // [ QV, RV ] = qr( V )
        auto  QV = blas::copy( V );
        auto  RV = blas::matrix< value_t >( in_rank, in_rank );

        blas::qr_wrapper( QV, RV );

        // compute column-pivoted QR of U·RV'
        auto  QU = blas::prod( value_t(1), U, adjoint(RV) );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        auto  P  = std::vector< int >( in_rank, 0 );

        blas::qrp( QU, RU, P );

        auto  out_rank = detail::trunc_rank( RU, acc );
        
        // U = QU(:,1:k)
        auto  Qk = blas::matrix< value_t >( QU, blas::range::all, blas::range( 0, out_rank-1 ) );
        auto  OU = blas::matrix< value_t >( nrows, out_rank );
        
        blas::copy( Qk, OU );

        // V = QV · P  (V' = P' · QV')
        auto  QV_P = blas::matrix< value_t >( ncols, in_rank );
        
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

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using RRQR
//
template< typename value_t >
std::pair< blas::matrix< value_t >, blas::matrix< value_t > >
rrqr ( const std::list< blas::matrix< value_t > > &  U,
       const std::list< blas::matrix< value_t > > &  V,
       const hpro::TTruncAcc &                       acc )
{
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

        auto [ U_tr, V_tr ] = rrqr( D, acc );

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
    
        return rrqr( U_all, V_all, acc );
    }// else
}

//
// compute low-rank approximation of a sum Σ_i U_i T_i V_i^H using RRQR
//
template< typename value_t >
std::pair< blas::matrix< value_t >, blas::matrix< value_t > >
rrqr ( const std::list< blas::matrix< value_t > > &  U,
       const std::list< blas::matrix< value_t > > &  T,
       const std::list< blas::matrix< value_t > > &  V,
       const hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == T.size() );
    HLR_ASSERT( T.size() == V.size() );

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

        return rrqr( D, acc );
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
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc ) const
    {
        return std::move( hlr::approx::rrqr( M, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        return std::move( hlr::approx::rrqr( U, V, acc ) );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::rrqr( U, V, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::rrqr( U, T, V, acc ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_RRQR_HH
