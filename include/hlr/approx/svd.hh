#ifndef __HLR_APPROX_SVD_HH
#define __HLR_APPROX_SVD_HH
//
// Project     : HLR
// Module      : approx/svd
// Description : low-rank approximation functions using SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>
#include <hlr/approx/traits.hh>
#include <hlr/approx/pairwise.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using Hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( blas::matrix< value_t > &  M,
      const Hpro::TTruncAcc &    acc )
{
    using  real_t  = Hpro::real_type_t< value_t >;

    //
    // perform SVD of M
    //

    const idx_t  nrows_M = idx_t( M.nrows() );
    const idx_t  ncols_M = idx_t( M.ncols() );
    const idx_t  mrc     = std::min( nrows_M, ncols_M );
    auto         S       = blas::vector< real_t >( mrc );
    auto         V       = blas::matrix< value_t >( ncols_M, mrc );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( nrows_M, ncols_M ) );
    
    blas::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const auto  Uk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );
    const auto  Vk = blas::matrix< value_t >( V, blas::range::all, blas::range( 0, k-1 ) );
    auto        A = blas::copy( Uk );
    auto        B = blas::copy( Vk );

    if ( nrows_M < ncols_M )
        prod_diag( A, S, k );
    else
        prod_diag( B, S, k );

    return { std::move( A ), std::move( B ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( const blas::matrix< value_t > &  U,
      const blas::matrix< value_t > &  V,
      const Hpro::TTruncAcc &          acc )
{
    using  real_t  = typename Hpro::real_type< value_t >::type_t;

    HLR_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    if ( in_rank == 0 )
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };

    if ( in_rank <= acc_rank )
        return { std::move( blas::copy( U ) ), std::move( blas::copy( V ) ) };

    //
    // truncate given low-rank matrix
    //
    
    if ( std::max( in_rank, acc_rank ) >= std::min( nrows_U, nrows_V ) )
    {
        //
        // since rank is too large, build U = U·V^T and do full-SVD
        //
            
        auto  M    = blas::prod( value_t(1), U, adjoint(V) );
        auto  lacc = Hpro::TTruncAcc( acc );

        if ( acc_rank > 0 )
            lacc.set_max_rank( acc_rank );

        return svd( M, lacc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );
    
        #if 1

        //////////////////////////////////////////////////////////////
        //
        // QR-factorisation of U and V with explicit Q
        //

        auto  QU = blas::copy( U );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        
        blas::qr( QU, RU );
        
        auto  QV = blas::copy( V );
        auto  RV = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::qr( QV, RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  R = blas::prod( value_t(1), RU, adjoint(RV) );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) );  // reuse storage
        blas::matrix< value_t >  Vs( std::move( RV ) );
            
        blas::svd( Us, Ss, Vs );
        
        // determine truncated rank based on singular values
        const auto  orank = idx_t( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < in_rank )
        {
            //
            // build new matrices U and V
            //

            const blas::range  in_rank_is( 0, in_rank-1 );
            const blas::range  orank_is( 0, orank-1 );

            // U := Q_U · U
            blas::matrix< value_t >  Urank( Us, in_rank_is, orank_is );
            
            // U := U·S
            blas::prod_diag( Urank, Ss, orank );

            auto  OU = blas::prod( value_t(1), QU, Urank );
            
            // V := Q_V · conj(V)
            blas::matrix< value_t >  Vrank( Vs, in_rank_is, orank_is );

            auto  OV = blas::prod( value_t(1), QV, Vrank );

            return { std::move( OU ), std::move( OV ) };
        }// if
        else
        {
            // rank has not changed, so return original matrices
            return { std::move( blas::copy( U ) ), std::move( blas::copy( V ) ) };
        }// else

        #else

        //////////////////////////////////////////////////////////////
        //
        // QR-factorisation of U and V with implicit Q
        //
        
        auto  QU = blas::copy( U );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        auto  TU = std::vector< value_t >();
        
        blas::qr_impl( QU, RU, TU );
        
        auto  QV = blas::copy( V );
        auto  RV = blas::matrix< value_t >( in_rank, in_rank );
        auto  TV = std::vector< value_t >();
        
        blas::qr_impl( QV, RV, TV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  R = blas::prod( value_t(1), RU, adjoint(RV) );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) ); // reuse storage
        blas::matrix< value_t >  Vs( std::move( RV ) );
            
        blas::svd( Us, Ss, Vs );
        
        // determine truncated rank based on singular values
        const auto  orank = idx_t( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < in_rank )
        {
            //
            // build new matrices U and V
            //

            const blas::range  in_rank_is( 0, in_rank-1 );
            const blas::range  orank_is( 0, orank-1 );

            // U := U·S
            auto  Urank = blas::matrix< value_t >( Us, in_rank_is, orank_is );
            
            blas::prod_diag( Urank, Ss, orank );

            // U := Q_U · U
            auto  OU     = blas::matrix< value_t >( nrows_U, orank );
            auto  OU_sub = blas::matrix< value_t >( OU, in_rank_is, orank_is );

            blas::copy( Urank, OU_sub );
            blas::prod_Q( blas::from_left, Hpro::apply_normal, QU, TU, OU );

            // V := Q_V · conj(V)
            auto  Vrank  = blas::matrix< value_t >( Vs, in_rank_is, orank_is );
            auto  OV     = blas::matrix< value_t >( nrows_V, orank );
            auto  OV_sub = blas::matrix< value_t >( OV, in_rank_is, orank_is );

            blas::copy( Vrank, OV_sub );
            blas::prod_Q( blas::from_left, Hpro::apply_normal, QV, TV, OV );

            return { std::move( OU ), std::move( OV ) };
        }// if
        else
        {
            // rank has not changed, so return original matrices
            return { std::move( blas::copy( U ) ), std::move( blas::copy( V ) ) };
        }// else

        #endif
    }// else
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( const std::list< blas::matrix< value_t > > &  U,
      const std::list< blas::matrix< value_t > > &  V,
      const Hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == V.size() );

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
    {
        in_rank += U_i.ncols();

        HLR_DBG_ASSERT( U_i.nrows() == nrows );
    }// for

    #if ! defined(NDEBUG)
    for ( auto &  V_i : V )
    {
        HLR_DBG_ASSERT( V_i.nrows() == ncols );
    }// for
    #endif
    
    if ( in_rank >= std::min( nrows, ncols ) )
    {
        //
        // perform dense approximation
        //

        auto  M   = blas::matrix< value_t >( nrows, ncols );
        auto  u_i = U.cbegin();
        auto  v_i = V.cbegin();
        
        for ( ; u_i != U.cend(); ++u_i, ++v_i )
            blas::prod( value_t(1), *u_i, blas::adjoint( *v_i ), value_t(1), M );

        return svd( M, acc );
    }// if
    else
    {
        //
        // concatenate matrices
        //

        auto   U_all = blas::matrix< value_t >( nrows, in_rank );
        auto   V_all = blas::matrix< value_t >( ncols, in_rank );
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
    
        return svd( U_all, V_all, acc );
    }// else
}

//
// compute low-rank approximation of a sum Σ_i U_i T_i V_i^H using SVD
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( const std::list< blas::matrix< value_t > > &  U,
      const std::list< blas::matrix< value_t > > &  T,
      const std::list< blas::matrix< value_t > > &  V,
      const Hpro::TTruncAcc &                       acc )
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

        auto  M   = blas::matrix< value_t >( nrows, ncols );
        auto  U_i = U.cbegin();
        auto  T_i = T.cbegin();
        auto  V_i = V.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i, ++V_i )
        {
            const auto  UT_i = blas::prod( value_t(1), *U_i, *T_i );
            
            blas::prod( value_t(1), UT_i, blas::adjoint( *V_i ), value_t(1), M );
        }// for

        return svd( M, acc );
    }// if
    else
    {
        //
        // concatenate matrices
        //

        auto   U_all = blas::matrix< value_t >( nrows, in_rank );
        auto   V_all = blas::matrix< value_t >( ncols, in_rank );
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
    
        return svd( U_all, V_all, acc );
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct SVD
{
    using  value_t = T_value;
    using  real_t  = typename Hpro::real_type< value_t >::type_t;

    // signal support for general lin. operators
    static constexpr bool supports_general_operator = false;
    
    //
    // matrix approximation routines
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const Hpro::TTruncAcc &    acc ) const
    {
        return hlr::approx::svd( M, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const Hpro::TTruncAcc &          acc ) const 
    {
        return hlr::approx::svd( U, V, acc );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::svd( U, V, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::svd( U, T, V, acc );
    }

    //
    // compute (approximate) column basis
    //
    
    blas::matrix< value_t >
    column_basis ( blas::matrix< value_t > &  M,
                   const Hpro::TTruncAcc &    acc ) const
    {
        if ( M.ncols() > 2 * M.nrows() )
        {
            //
            // compute eigenvalues and eigenvectors of M·M'
            //
            
            auto  G        = blas::prod( M, blas::adjoint( M ) );
            auto  [ V, E ] = blas::eigen_herm( G );
            auto  perm     = std::vector< std::pair< value_t, uint > >( E.length() );
            
            for ( uint  i = 0; i < E.length(); ++i )
                perm.push_back({ E(i), i });
            
            std::sort( perm.begin(), perm.end(), [] ( auto  a, auto b ) { return a.first > b.first; } );
            
            for ( uint  i = 0; i < E.length(); ++i )
                E(i) = std::sqrt( perm[i].first ); // σ_i = √(λ_i)
            
            const auto  k  = acc.trunc_rank( E );
            auto        Vk = blas::matrix< value_t >( V.nrows(), k );
            
            for ( uint  i = 0; i < k; ++i )
            {
                auto  v1 = V.column( perm[i].second );
                auto  v2 = Vk.column( i );
                
                copy( v1, v2 );
            }// for

            return Vk;
        }// if
        else if ( M.ncols() > M.nrows() / 2 )
        {
            //
            // directly use first k column of U from M = U·S·V'
            // - V can be omitted as is does not contribute to basis
            //
            
            auto  S = blas::vector< real_t >();

            HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), M.ncols() ) );
        
            blas::svd( M, S );

            const auto  k  = acc.trunc_rank( S );
            const auto  Uk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );

            return  blas::copy( Uk );
        }// if
        else
        {
            //
            // M = Q·R = Q·U·S·V' with R = U·S·V'
            // - V can be omitted as is does not contribute to basis
            //
            
            auto  R = blas::matrix< value_t >();

            blas::qr( M, R );

            auto  S = blas::vector< real_t >();

            blas::svd( R, S );
            
            const auto  k  = acc.trunc_rank( S );
            const auto  Uk = blas::matrix< value_t >( R, blas::range::all, blas::range( 0, k-1 ) );

            return  blas::prod( M, Uk );
        }// else
    }

    //
    // compute column basis and return basis and singular values
    //
    
    std::pair< blas::matrix< value_t >,
               blas::vector< typename Hpro::real_type_t< value_t > > >
    column_basis ( blas::matrix< value_t > &  M ) const
    {
        if ( M.ncols() > 2 * M.nrows() )
        {
            //
            // compute eigenvalues and eigenvectors of M·M'
            //
            
            auto  G        = blas::prod( M, blas::adjoint( M ) );
            auto  [ V, E ] = blas::eigen_herm( G );
            auto  perm     = std::vector< std::pair< value_t, uint > >( E.length() );
            
            for ( uint  i = 0; i < E.length(); ++i )
                perm.push_back({ E(i), i });
            
            std::sort( perm.begin(), perm.end(), [] ( auto  a, auto b ) { return a.first > b.first; } );
            
            for ( uint  i = 0; i < E.length(); ++i )
                E(i) = std::sqrt( perm[i].first ); // σ_i = √(λ_i)

            // better: permute V
            auto  Vk = blas::matrix< value_t >( V.nrows(), V.ncols() );
            
            for ( uint  i = 0; i < V.ncols(); ++i )
            {
                auto  v1 = V.column( perm[i].second );
                auto  v2 = Vk.column( i );
                
                copy( v1, v2 );
            }// for

            return { std::move(Vk), std::move(E) };
        }// if
        else if ( M.ncols() > M.nrows() / 2 )
        {
            //
            // directly use first k column of U from M = U·S·V'
            // - V can be omitted as is does not contribute to basis
            //
            
            auto  S = blas::vector< real_t >();

            HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), M.ncols() ) );
        
            blas::svd( M, S );

            return  { std::move(blas::copy( M )), std::move(S) };
        }// if
        else
        {
            //
            // M = Q·R = Q·U·S·V' with R = U·S·V'
            // - V can be omitted as is does not contribute to basis
            //
            
            auto  R = blas::matrix< value_t >();

            blas::qr( M, R );

            auto  S = blas::vector< real_t >();

            blas::svd( R, S );
            
            return  { std::move(blas::prod( M, R )), std::move( S ) };
        }// else
    }
};

template < typename T_value >
struct PairSVD
{
    using  value_t = T_value;
    
    // signal support for general lin. operators
    static constexpr bool supports_general_operator = false;
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const Hpro::TTruncAcc &    acc ) const
    {
        return hlr::approx::svd( M, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const Hpro::TTruncAcc &          acc ) const 
    {
        return hlr::approx::svd( U, V, acc );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        auto  approx = SVD< value_t >();
        
        return pairwise( U, V, acc, approx );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::svd( U, T, V, acc );
    }
};

// signals, that T is of approximation type
template < typename T > struct is_approximation< SVD< T > > { static const bool  value = true; };

}}// namespace hlr::approx

#endif // __HLR_APPROX_SVD_HH
