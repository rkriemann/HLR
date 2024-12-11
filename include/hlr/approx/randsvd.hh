#ifndef __HLR_APPROX_RANDSVD_HH
#define __HLR_APPROX_RANDSVD_HH
//
// Project     : HLR
// Module      : approx/randsvd
// Description : low-rank approximation functions using randomized SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/approx/randlr.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using Hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename operator_t >
std::tuple< blas::matrix< typename operator_t::value_t >,
            blas::vector< real_type_t< typename operator_t::value_t > >,
            blas::matrix< typename operator_t::value_t > >
randsvd_ortho ( const operator_t &  M,
                const accuracy &    acc,
                const uint          power_steps,
                const uint          oversampling )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = Hpro::real_type_t< value_t >;

    const auto  nrows_M = nrows( M );
    const auto  ncols_M = ncols( M );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( nrows_M, ncols_M ) );
    
    // compute column basis
    auto  Q   = detail::rand_column_basis_full( M, acc, 4, power_steps, oversampling );
    auto  k   = Q.ncols();

    // B = Q^H · M  or B^H = M^H · Q
    auto  BT  = blas::matrix< value_t >( ncols_M, k );

    prod( value_t(1), Hpro::apply_adjoint, M, Q, BT );
    
    auto  R_B = blas::matrix< value_t >( k, k );
    auto  V   = blas::matrix< value_t >( k, k );
    auto  S   = blas::vector< real_t >( k );

    // B^T = Q_B R_B  (Q_B overwrites B)
    blas::qr( BT, R_B );

    // R_B = U·S·V^H
    blas::svd( R_B, S, V );

    // determine truncated rank based on singular values
    k = idx_t( acc.trunc_rank( S ) );

    // A = Y · V_k, B = B^T · U_k
    auto  rk = blas::range( 0, k-1 );
    auto  Uk = blas::matrix< value_t >( R_B, blas::range::all, rk );
    auto  Vk = blas::matrix< value_t >( V,   blas::range::all, rk );
    auto  Sk = blas::vector< value_t >( S,   rk );
    
    auto  OU = blas::prod( value_t(1), Q,  Vk );
    auto  OV = blas::prod( value_t(1), BT, Uk );

    return { std::move( OU ), std::move( blas::copy( Sk ) ), std::move( OV ) };
}

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename operator_t >
std::pair< blas::matrix< typename operator_t::value_t >,
           blas::matrix< typename operator_t::value_t > >
randsvd ( const operator_t &       M,
          const accuracy &         acc,
          const uint               power_steps,
          const uint               oversampling )
{
    auto  [ U, S, V ] = randsvd_ortho( M, acc, power_steps, oversampling );
    
    if ( U.nrows() < V.nrows() ) blas::prod_diag_ip( U, S );
    else                         blas::prod_diag_ip( V, S );

    return { std::move( U ), std::move( V ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
randsvd_ortho ( const blas::matrix< value_t > &  U,
                const blas::matrix< value_t > &  V,
                const accuracy &                 acc,
                const uint                       power_steps,
                const uint                       oversampling )
{
    using  real_t  = typename Hpro::real_type< value_t >::type_t;

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
                 std::move( blas::vector< real_t >( 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };
    }// if

    if ( in_rank <= idx_t(acc.rank()) )
    {
        HLR_ERROR( "TODO" );
        return { std::move( blas::copy( U ) ),
                 blas::vector< real_t >(),
                 std::move( blas::copy( V ) ) };
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return randsvd_ortho( M, acc, power_steps, oversampling );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        //
        // compute column basis
        //

        auto  Q      = detail::rand_column_basis( U, V, acc, 4, power_steps, oversampling );
        auto  k_base = idx_t(Q.ncols());

        // Q^H · U · V^H  = (V·U^H·Q)^H
        auto  UtQ    = blas::prod( value_t(1), blas::adjoint(U), Q );
        auto  VUtQ   = blas::prod( value_t(1), V, UtQ );

        auto  U_svd  = blas::matrix< value_t >( k_base, k_base );
        auto  V_svd  = blas::matrix< value_t >( k_base, k_base );
        auto  S      = blas::vector< real_t >( k_base );

        // (V·U^H·Q)^H = Q_B R
        blas::qr( VUtQ, U_svd );
        
        // R_V = U·S·V^H
        svd( U_svd, S, V_svd );
        
        // determine truncated rank based on singular values
        auto  out_rank = idx_t( acc.trunc_rank( S ) );

        // A = Y · V_k, B = B^T · U_k
        auto  rk = blas::range( 0, out_rank-1 );
        auto  Uk = blas::matrix< value_t >( U_svd, blas::range::all, rk );
        auto  Vk = blas::matrix< value_t >( V_svd, blas::range::all, rk );
        auto  Sk = blas::vector< value_t >( S, rk );

        auto  OU = blas::prod( value_t(1), Q,    Vk );
        auto  OV = blas::prod( value_t(1), VUtQ, Uk );

        return { std::move( OU ), std::move( blas::copy( Sk ) ), std::move( OV ) };
    }// else
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const blas::matrix< value_t > &  U,
          const blas::matrix< value_t > &  V,
          const accuracy &                 acc,
          const uint                       power_steps,
          const uint                       oversampling )
{
    auto  [ W, T, X ] = randsvd_ortho( U, V, acc, power_steps, oversampling );
    
    if ( W.nrows() < X.nrows() ) blas::prod_diag_ip( W, T );
    else                         blas::prod_diag_ip( X, T );

    return { std::move( W ), std::move( X ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
randsvd_ortho ( const std::list< blas::matrix< value_t > > &  U,
                const std::list< blas::matrix< value_t > > &  V,
                const accuracy &                              acc,
                const uint                                    power_steps,
                const uint                                    oversampling )
{
    using real_t = real_type_t< value_t >;
    
    HLR_ASSERT( U.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::vector< real_t >() ),
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

        return randsvd_ortho( M, acc, power_steps, oversampling );
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
    
        return randsvd_ortho( U_all, V_all, acc, power_steps, oversampling );
    }// else
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const std::list< blas::matrix< value_t > > &  U,
          const std::list< blas::matrix< value_t > > &  V,
          const accuracy &                              acc,
          const uint                                    power_steps,
          const uint                                    oversampling )
{
    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };

    auto  [ W, T, X ] = randsvd_ortho( U, V, acc, power_steps, oversampling );
    
    if ( W.nrows() < X.nrows() ) blas::prod_diag_ip( W, T );
    else                         blas::prod_diag_ip( X, T );

    return { std::move( W ), std::move( X ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i T_i V_i^H using SVD
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const std::list< blas::matrix< value_t > > &  U,
          const std::list< blas::matrix< value_t > > &  T,
          const std::list< blas::matrix< value_t > > &  V,
          const accuracy &                              acc,
          const uint                                    power_steps,
          const uint                                    oversampling )
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

        return randsvd( M, acc, power_steps, oversampling );
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
    
        return randsvd( U_all, V_all, acc, power_steps, oversampling );
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct RandSVD
{
    using  value_t = T_value;

    // signal support for general lin. operators
    static constexpr bool supports_general_operator = true;
    
    // number of steps in power iteration during construction of column basis
    const uint   power_steps  = 0;

    // oversampling parameter
    const uint   oversampling = 0;

    //
    // matrix approximation routines
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const accuracy &           acc ) const
    {
        return hlr::approx::randsvd( M, acc, power_steps, oversampling );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const 
    {
        auto  Uc = blas::copy( U );
        auto  Vc = blas::copy( V );
        
        return hlr::approx::randsvd( Uc, Vc, acc, power_steps, oversampling );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::randsvd( U, V, acc, power_steps, oversampling );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::randsvd( U, T, V, acc, power_steps, oversampling );
    }

    template < typename operator_t >
    std::pair< blas::matrix< typename operator_t::value_t >,
               blas::matrix< typename operator_t::value_t > >
    operator () ( const operator_t &       op,
                  const accuracy &         acc ) const
    {
        return hlr::approx::randsvd< operator_t >( op, acc, power_steps, oversampling );
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
        return hlr::approx::randsvd_ortho( M, acc, power_steps, oversampling );
    }

    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const blas::matrix< value_t > &  U,
                   const blas::matrix< value_t > &  V,
                   const accuracy &                 acc ) const 
    {
        return hlr::approx::randsvd_ortho( U, V, acc, power_steps, oversampling );
    }
    
    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const std::list< blas::matrix< value_t > > &  U,
                   const std::list< blas::matrix< value_t > > &  V,
                   const accuracy &                              acc ) const
    {
        return hlr::approx::randsvd_ortho( U, V, acc, power_steps, oversampling );
    }

    //
    // compute (approximate) column basis
    //
    
    template < typename value_t >
    blas::matrix< value_t >
    column_basis ( const blas::matrix< value_t > &  op,
                   const accuracy &                 acc,
                   blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr ) const
    {
        return detail::rand_column_basis( op, acc, 4, power_steps, oversampling, sv );
    }
    
    template < typename operator_t >
    blas::matrix< typename operator_t::value_t >
    column_basis ( const operator_t &  op,
                   const accuracy &    acc,
                   blas::vector< Hpro::real_type_t< typename operator_t::value_t > > *  sv = nullptr ) const
    {
        return detail::rand_column_basis< operator_t >( op, acc, 4, power_steps, oversampling, sv );
    }
};

//
// implements randomized SVD with full column basis search
//
template < typename T_value >
struct RandSVDFull
{
    using  value_t = T_value;

    // signal support for general lin. operators
    static constexpr bool supports_general_operator = true;
    
    // number of steps in power iteration during construction of column basis
    const uint   power_steps  = 0;

    // oversampling parameter
    const uint   oversampling = 0;

    //
    // matrix approximation routines
    //
    
    // std::pair< blas::matrix< value_t >,
    //            blas::matrix< value_t > >
    // operator () ( blas::matrix< value_t > &  M,
    //               const accuracy &           acc ) const
    // {
    //     return hlr::approx::randsvd( M, acc, power_steps, oversampling );
    // }

    // std::pair< blas::matrix< value_t >,
    //            blas::matrix< value_t > >
    // operator () ( const blas::matrix< value_t > &  U,
    //               const blas::matrix< value_t > &  V,
    //               const accuracy &                 acc ) const 
    // {
    //     auto  Uc = blas::copy( U );
    //     auto  Vc = blas::copy( V );
        
    //     return hlr::approx::randsvd( Uc, Vc, acc, power_steps, oversampling );
    // }
    
    // std::pair< blas::matrix< value_t >,
    //            blas::matrix< value_t > >
    // operator () ( const std::list< blas::matrix< value_t > > &  U,
    //               const std::list< blas::matrix< value_t > > &  V,
    //               const accuracy &                              acc ) const
    // {
    //     return hlr::approx::randsvd( U, V, acc, power_steps, oversampling );
    // }

    // std::pair< blas::matrix< value_t >,
    //            blas::matrix< value_t > >
    // operator () ( const std::list< blas::matrix< value_t > > &  U,
    //               const std::list< blas::matrix< value_t > > &  T,
    //               const std::list< blas::matrix< value_t > > &  V,
    //               const accuracy &                              acc ) const
    // {
    //     return hlr::approx::randsvd( U, T, V, acc, power_steps, oversampling );
    // }

    // template < typename operator_t >
    // std::pair< blas::matrix< typename operator_t::value_t >,
    //            blas::matrix< typename operator_t::value_t > >
    // operator () ( const operator_t &       op,
    //               const accuracy &         acc ) const
    // {
    //     return hlr::approx::randsvd< operator_t >( op, acc, power_steps, oversampling );
    // }

    // //
    // // matrix approximation routines (orthogonal version)
    // //
    
    // std::tuple< blas::matrix< value_t >,
    //             blas::vector< real_type_t< value_t > >,
    //             blas::matrix< value_t > >
    // approx_ortho ( blas::matrix< value_t > &  M,
    //                const accuracy &           acc ) const
    // {
    //     return hlr::approx::randsvd_ortho( M, acc, power_steps, oversampling );
    // }

    // std::tuple< blas::matrix< value_t >,
    //             blas::vector< real_type_t< value_t > >,
    //             blas::matrix< value_t > >
    // approx_ortho ( const blas::matrix< value_t > &  U,
    //                const blas::matrix< value_t > &  V,
    //                const accuracy &                 acc ) const 
    // {
    //     return hlr::approx::randsvd_ortho( U, V, acc, power_steps, oversampling );
    // }
    
    // std::tuple< blas::matrix< value_t >,
    //             blas::vector< real_type_t< value_t > >,
    //             blas::matrix< value_t > >
    // approx_ortho ( const std::list< blas::matrix< value_t > > &  U,
    //                const std::list< blas::matrix< value_t > > &  V,
    //                const accuracy &                              acc ) const
    // {
    //     return hlr::approx::randsvd_ortho( U, V, acc, power_steps, oversampling );
    // }

    //
    // compute (approximate) column basis
    //
    
    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::vector< real_type_t< value_t > > >
    column_basis ( const blas::matrix< value_t > &  op ) const
    {
        using  real_t = real_type_t< value_t >;
        
        auto  S = blas::vector< real_t >();
        auto  Q = detail::rand_column_basis_full_svd( op, 0, power_steps, oversampling, & S );

        return { std::move( Q ), std::move( S ) };
    }
    
    template < typename value_t >
    blas::matrix< value_t >
    column_basis ( const blas::matrix< value_t > &  op,
                   const accuracy &                 acc,
                   blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr ) const
    {
        return detail::rand_column_basis_full_svd( op, acc, 0, power_steps, oversampling, sv );
    }
    
    template < typename operator_t >
    blas::matrix< typename operator_t::value_t >
    column_basis ( const operator_t &  op,
                   const accuracy &    acc,
                   blas::vector< Hpro::real_type_t< typename operator_t::value_t > > *  sv = nullptr ) const
    {
        return detail::rand_column_basis_full_svd< operator_t >( op, acc, 0, power_steps, oversampling, sv );
    }
};

// signals, that T is of approximation type
template < typename T > struct is_approximation< RandSVD< T > >     { static const bool  value = true; };
template < typename T > struct is_approximation< RandSVDFull< T > > { static const bool  value = true; };

}}// namespace hlr::approx

#endif // __HLR_APPROX_RANDSVD_HH
