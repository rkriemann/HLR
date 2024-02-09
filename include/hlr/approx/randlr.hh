#ifndef __HLR_APPROX_RANDLR_HH
#define __HLR_APPROX_RANDLR_HH
//
// Project     : HLR
// Module      : approx/randlr
// Description : low-rank approximation functions using randomized low-rank approx.
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <list>
#include <deque>
#include <random>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>
#include <hlr/approx/traits.hh>
#include <hlr/approx/tools.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using Hpro::idx_t;

namespace detail
{

//
// compute basis for column space (range) of M
//
template < typename operator_t >
blas::matrix< typename operator_t::value_t >
rand_column_basis ( const operator_t &  M,
                    const accuracy &    acc,
                    const uint          block_size,
                    const uint          power_steps,
                    const uint          oversampling,
                    blas::vector< Hpro::real_type_t< typename operator_t::value_t > > *  sv = nullptr )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = Hpro::real_type_t< value_t >;
        
    const auto  nrows_M = nrows( M );
    const auto  ncols_M = ncols( M );

    std::random_device          rd{};
    std::mt19937                generator{ rd() };
    std::normal_distribution<>  distr{ 0, 1 };
    auto                        rand_norm = [&] () { return distr( generator ); };
    
    if ( acc.is_fixed_rank() )
    {
        const auto  k = acc.rank();
        auto        T = blas::matrix< value_t >( ncols_M, k + oversampling );

        blas::fill_fn( T, rand_norm );
        
        auto        Q = blas::matrix< value_t >( nrows_M, k + oversampling );

        prod( value_t(1), Hpro::apply_normal, M, T, Q );

        //
        // power iteration
        //
            
        auto  R = blas::matrix< value_t >( k + oversampling, k + oversampling );

        blas::qr( Q, R );

        if ( power_steps > 0 )
        {
            auto  MtQ = blas::matrix< value_t >( ncols_M, k + oversampling );
            
            for ( uint  j = 0; j < power_steps; ++j )
            {
                blas::scale( value_t(0), MtQ );
                prod( value_t(1), Hpro::apply_adjoint, M, Q, MtQ );
                blas::qr( MtQ, R );
                
                blas::scale( value_t(0), Q );
                prod( value_t(1), Hpro::apply_normal, M, MtQ, Q );
                blas::qr( Q, R );
            }// for
        }// if

        if ( ! is_null( sv ) )
        {
            auto  S = singular_values( R );

            if ( sv->length() != k )
                *sv = std::move( blas::vector< real_t >( k ) );
            
            for ( uint  i = 0; i < k; ++i )
                (*sv)(i) = S(i);
        }// if
        
        return Q;
    }// if
    else
    {
        real_t      norm_M  = real_t(0);
        const auto  rel_eps = acc.rel_eps();
        const auto  abs_eps = acc.abs_eps();
        const uint  bsize   = std::min< uint >( block_size, std::min< uint >( nrows_M, ncols_M ) );
        const uint  nblocks = std::min< uint >( nrows_M, ncols_M ) / bsize;
        auto        Qs      = std::list< blas::matrix< value_t > >();
        auto        T_i     = blas::matrix< value_t >( ncols_M, bsize );
        auto        QhQi    = blas::matrix< value_t >( bsize,   bsize );
        auto        TQ_i    = blas::matrix< value_t >( nrows_M, bsize );
        auto        R       = blas::matrix< value_t >( bsize,   bsize );
        auto        MtQ     = blas::matrix< value_t >( ncols_M, bsize );
        auto        S       = std::list< real_t >();
        
        for ( uint  i = 0; i < nblocks; ++i )
        {
            //
            // draw random matrix and compute approximation of remainder M - Σ_j Q_j·Q_j'·M
            //
            
            blas::fill_fn( T_i, rand_norm );
            
            auto  Q_i = blas::matrix< value_t >( nrows_M, bsize );

            prod( value_t(1), Hpro::apply_normal, M, T_i, Q_i );

            // subtract previous Q_j
            if ( ! Qs.empty() )
            {
                blas::copy( Q_i, TQ_i );
            
                for ( auto  Q_j : Qs )
                {
                    blas::prod( value_t(1), blas::adjoint( Q_j ), TQ_i, value_t(0), QhQi );
                    blas::prod( value_t(-1), Q_j, QhQi, value_t(1), Q_i );
                }// for
            }// if

            //
            // compute norm of remainder and update norm(M)
            //
            
            real_t  norm_Qi = real_t(0);

            for ( uint  j = 0; j < bsize; ++j )
            {
                const auto  Qi_j = Q_i.column( j );

                norm_Qi = std::max( norm_Qi, blas::norm2( Qi_j ) );
            }// for

            norm_M = std::sqrt( math::square( norm_M ) + math::square( norm_Qi ) );

            //
            // power iteration
            //
            
            blas::qr( Q_i, R );
            
            if ( power_steps > 0 )
            {
                for ( uint  j = 0; j < power_steps; ++j )
                {
                    blas::scale( value_t(0), MtQ );
                    prod( value_t(1), Hpro::apply_adjoint, M, Q_i, MtQ );
                    blas::qr( MtQ, R );
                    
                    blas::scale( value_t(0), Q_i );
                    prod( value_t(1), Hpro::apply_normal, M, MtQ, Q_i );
                    blas::qr( Q_i, R );
                }// for
            }// if
            
            //
            // project Q_i away from previous Q_j
            //
            //    Q_i = Q_i - [ Q_0 .. Q_i-1 ] [ Q_0 .. Q_i-1 ]^H Q_i = Q_i - Σ_j=0^i-1 Q_j Q_j^H Q_i
            //
                
            if ( i > 0 )
            {
                blas::copy( Q_i, TQ_i ); // auto  C_i   = blas::matrix< value_t >( Q_i, Hpro::copy_value );
                
                for ( const auto &  Q_j : Qs )
                {
                    blas::prod( value_t(1), blas::adjoint(Q_j), TQ_i, value_t(0), QhQi );
                    blas::prod( value_t(-1), Q_j, QhQi, value_t(1), Q_i );
                }// for
                
                blas::qr( Q_i, R );
            }// if
            
            //
            // M = M - Q_i Q_i^t M
            //

            Qs.push_back( std::move( Q_i ) );
            
            if (( norm_Qi <= abs_eps ) || (( norm_Qi ) <= rel_eps * norm_M ))
                break;

            if ( ! is_null( sv ) )
            {
                auto  S_i = singular_values( R );

                for ( uint  j = 0; j < S_i.length(); ++j )
                    S.push_back( S_i(j) );
            }// if
        }// for
        
        //
        // collect Q_i's into final result
        //

        auto   Q   = blas::matrix< value_t >( nrows_M, Qs.size() * bsize );
        idx_t  pos = 0;

        for ( const auto &  Q_i : Qs )
        {
            auto  Q_sub = blas::matrix< value_t >( Q, blas::range::all, blas::range( pos, pos+bsize-1 ) );

            blas::copy( Q_i, Q_sub );
            pos += bsize;
        }// for

        if ( ! is_null( sv ) )
        {
            if ( sv->length() != S.size() )
                *sv = std::move( blas::vector< real_t >( S.size() ) );

            uint  i = 0;
            
            for ( auto  s_i : S )
                (*sv)(i++) = s_i;
        }// if
        
        return Q;
    }// else
}

//
// computes column basis for U·V'
// - slightly faster than general version
//
template < typename value_t >
blas::matrix< value_t >
rand_column_basis ( const blas::matrix< value_t > &  U,
                    const blas::matrix< value_t > &  V,
                    const accuracy &                 acc,
                    const uint                       block_size,
                    const uint                       power_steps,
                    const uint                       oversampling )
{
    const idx_t  nrows = idx_t( U.nrows() );
    const idx_t  ncols = idx_t( V.nrows() );
    const idx_t  rank  = idx_t( U.ncols() );
    
    std::random_device          rd{};
    std::mt19937                generator{ rd() };
    std::normal_distribution<>  distr{ 0, 1 };
    auto                        fill_rand = [&] () { return distr( generator ); };
    
    if ( acc.is_fixed_rank() )
    {
        const idx_t  k   = idx_t(acc.rank());
        auto         T   = blas::matrix< value_t >( nrows, k + oversampling );

        blas::fill_fn( T, fill_rand );
        
        auto         VtT = blas::prod( value_t(1), blas::adjoint(V), T );
        auto         Q   = blas::prod( value_t(1), U, VtT );
        auto         R   = blas::matrix< value_t >( k + oversampling, k + oversampling );

        blas::qr( Q, R );

        //
        // power iteration
        //
        
        if ( power_steps > 0 )
        {
            auto  UtQ  = blas::matrix< value_t >( rank,  k + oversampling );
            auto  VUtQ = blas::matrix< value_t >( ncols, k + oversampling );

            for ( uint  j = 0; j < power_steps; ++j )
            {
                // [Q,R] = qr(Q); MtQ = M^H·Q = V·U^H·Q
                blas::prod( value_t(1), blas::adjoint(U), Q, value_t(0), UtQ );
                blas::prod( value_t(1), V, UtQ, value_t(0), VUtQ );
                blas::qr( VUtQ, R );
                
                // [Q,R] = qr(V·U^H·Q); Q = U·V^H·Q
                blas::prod( value_t(1), blas::adjoint(V), VUtQ, value_t(0), UtQ );
                blas::prod( value_t(1), U, UtQ, value_t(0), Q );
                blas::qr( Q, R );
            }// for
        }// if

        return Q;
    }// if
    else
    {
        auto        Uc      = blas::copy( U ); // need copy to be modified below
        const auto  norm_M  = lr_normF( Uc, V );
        const auto  rel_eps = acc.rel_eps();
        const auto  abs_eps = acc.abs_eps();
        const uint  bsize   = std::min< uint >( block_size, std::min< uint >( nrows, ncols ) );
        const uint  nblocks = std::min( nrows, ncols ) / bsize;
        auto        Qs      = std::list< blas::matrix< value_t > >();
        auto        T_i     = blas::matrix< value_t >( ncols, bsize );
        auto        VtT     = blas::matrix< value_t >( rank,  bsize );
        auto        TQ_i    = blas::matrix< value_t >( nrows, bsize );
        auto        UtQ     = blas::matrix< value_t >( rank,  bsize );
        auto        VUtQ    = blas::matrix< value_t >( ncols, bsize );
        auto        R       = blas::matrix< value_t >( bsize, bsize );
        auto        QjtQi   = blas::matrix< value_t >( bsize, bsize );
        auto        QtA     = blas::matrix< value_t >( bsize, rank );

        for ( uint  i = 0; i < nblocks; ++i )
        {
            blas::fill_fn( T_i, fill_rand );
            blas::prod( value_t(1), blas::adjoint(V), T_i, value_t(0), VtT );
            
            auto  Q_i = blas::prod( value_t(1), Uc, VtT );

            //
            // power iteration
            //
            
            blas::qr( Q_i, R );
            
            if ( power_steps > 0 )
            {
                for ( uint  j = 0; j < power_steps; ++j )
                {
                    blas::prod( value_t(1), blas::adjoint(Uc), Q_i, value_t(0), UtQ );
                    blas::prod( value_t(1), V, UtQ, value_t(0), VUtQ );
                    blas::qr( VUtQ, R );
                    
                    blas::prod( value_t(1), blas::adjoint(V), VUtQ, value_t(0), UtQ );
                    blas::prod( value_t(1), Uc, UtQ, value_t(0), Q_i );
                    blas::qr( Q_i, R );
                }// for
            }// if
            
            //
            // project Q_i away from previous Q_j
            //
                
            if ( i > 0 )
            {
                blas::copy( Q_i, TQ_i );
                
                for ( const auto &  Q_j : Qs )
                {
                    blas::prod( value_t(1), blas::adjoint(Q_j), TQ_i, value_t(0), QjtQi );
                    blas::prod( value_t(-1), Q_j, QjtQi, value_t(1), Q_i );
                }// for
                
                blas::qr( Q_i, R );
            }// if

            //
            // M = M - Q_i Q_i^T M = U·V^H - Q_i Q_i^T U·V^H = (U - Q_i Q_i^T U) V^H
            //

            blas::prod( value_t(1), blas::adjoint(Q_i), Uc, value_t(0), QtA );
            blas::prod( value_t(-1), Q_i, QtA, value_t(1), Uc );
            
            const auto  norm_Qi = blas::lr_normF( Uc, V );

            Qs.push_back( std::move( Q_i ) );
            
            if (( norm_Qi < abs_eps ) || ( norm_Qi <= rel_eps * norm_M ))
                break;
        }// for

        //
        // collect Q_i's into final result
        //

        auto   Q   = blas::matrix< value_t >( nrows, Qs.size() * bsize );
        idx_t  pos = 0;

        for ( const auto &  Q_i : Qs )
        {
            auto  Q_sub = blas::matrix< value_t >( Q, blas::range::all, blas::range( pos * bsize, (pos+1)*bsize - 1 ) );

            blas::copy( Q_i, Q_sub );
            ++pos;
        }// for

        return Q;
    }// else
}

//
// compute low-rank approximation of a given sum Σ_i M_i using randomized LR
// - only need matrix-vector evaluation of given operators
//
template < typename operator_t >
std::pair< blas::matrix< typename operator_t::value_t >,
           blas::matrix< typename operator_t::value_t > >
randlr  ( const operator_t &       M,
          const accuracy &         acc )
{
    using  value_t = typename operator_t::value_t;
    
    auto  W = rand_column_basis( M, acc, 2, 0, 0 );
    auto  X = blas::matrix< value_t >( ncols( M ), W.ncols() );

    // W·W'·M ≈ M ⇒ W·(M'·W)' =: W·X' ≈ M
    prod( value_t(1), Hpro::apply_adjoint, M, W, X );

    return { std::move( W ), std::move( X ) };
        
    // using  real_t  = typename Hpro::real_type< value_t >::type_t;

    // const auto  nrows_M = nrows( M );
    // const auto  ncols_M = ncols( M );
    
    // // value considered zero to avoid division by small values
    // const real_t  zero_val  = math::square( std::numeric_limits< real_t >::epsilon() );
    // auto          norm_M    = real_t(0);
    // auto          norm_R    = real_t(0);
    // uint          k         = 0;
    // auto          U         = std::deque< blas::vector< value_t > >();
    // auto          V         = std::deque< blas::vector< value_t > >();

    // // normal distributed random numbers
    // auto          rd        = std::random_device{};
    // auto          generator = std::mt19937{ rd() };
    // auto          distr     = std::normal_distribution<>{ real_t(0), real_t(1) };
    // auto          rand      = [&] () { return value_t( distr( generator ) ); };
    
    // do
    // {
    //     auto  W = blas::vector< value_t >( ncols_M );
    //     auto  L = blas::vector< value_t >( nrows_M );
        
    //     blas::fill_fn( W, rand );
    //     prod( value_t(1), Hpro::apply_normal, M, W, L );

    //     // subtract previously computed results
    //     if ( k > 0 )
    //     {
    //         auto  Lc = blas::copy( L );
        
    //         for ( auto &  u_i : U )
    //             blas::add( - blas::dot( u_i, Lc ), u_i, L );
    //     }// if

    //     const auto  norm_L = blas::norm2( L );

    //     // if norm is close to zero, matrix rest is zero as well
    //     if ( norm_L < zero_val )
    //         break;

    //     blas::scale( value_t( real_t(1) / norm_L ), L );

    //     blas::scale( value_t(0), W );
    //     prod( value_t(1), Hpro::apply_adjoint, M, L, W );

    //     //
    //     // update norm of M as norm( U·V^H ) and
    //     // norm of ( M - U·V^H ) by last vector pair
    //     //

    //     norm_R = blas::norm2( W );      // L is normalized!
        
    //     auto norm_upd = norm_R * norm_R;

    //     for ( uint  i = 0; i < k; ++i )
    //     {
    //         norm_upd += std::real( blas::dot( W, V[i] ) * blas::dot( U[i], L ) );
    //         norm_upd += std::real( blas::dot( V[i], W ) * blas::dot( L, U[i] ) );
    //     }// for

    //     norm_M = std::sqrt( norm_M * norm_M + norm_upd );

    //     //
    //     // and store new vectors/pivots
    //     //

    //     U.push_back( std::move( L ) );
    //     V.push_back( std::move( W ) );

    //     k++;
        
    //     //
    //     // check stop criterion
    //     //

    //     if (( acc.is_fixed_rank() && ( k      >= acc.rank() ) ) ||
    //         ( acc.is_fixed_prec() && ( norm_R <= acc.rel_eps() * norm_M )) ||
    //         ( acc.has_max_rank()  && ( k      >= acc.max_rank() ) ) ||
    //         ( norm_R <= acc.abs_eps() ))
    //         break;
             
    // } while ( k <= std::min( nrows_M, ncols_M ) ); // fallback 
    
    // //
    // // copy vectors to matrix
    // //

    // blas::matrix< value_t >  MU( nrows_M, k );
    // blas::matrix< value_t >  MV( ncols_M, k );

    // for ( uint  i = 0; i < k; ++i )
    // {
    //     auto  u_i = MU.column( i );
    //     auto  v_i = MV.column( i );

    //     blas::copy( U[i], u_i );
    //     blas::copy( V[i], v_i );
    // }// for
    
    // return { std::move( MU ), std::move( MV ) };
}

}// namespace detail

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( blas::matrix< value_t > &  M,
         const accuracy &           acc )
{
    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), M.ncols() ) );
    
    return detail::randlr( M, acc );
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
randlr_ortho ( blas::matrix< value_t > &  M,
               const accuracy &           acc )
{
    auto  [ U, V ] = randlr( M, acc );

    return detail::make_ortho( U, V );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const blas::matrix< value_t > &  U,
         const blas::matrix< value_t > &  V,
         const accuracy &                 acc )
{
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
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return std::move( detail::randlr( M, acc ) );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  W = detail::rand_column_basis( U, V, acc, 2, 0, 0 );
        auto  T = blas::prod( blas::adjoint( U ), W );
        auto  X = blas::prod( V, T );

        return { std::move( W ), std::move( X ) };
        // auto  op = operator_wrapper( U, V );

        // return std::move( detail::randlr( op, acc ) );
    }// else
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
randlr_ortho ( const blas::matrix< value_t > &  U,
               const blas::matrix< value_t > &  V,
               const accuracy &                 acc )
{
    auto  [ TU, TV ] = randlr( U, V, acc );

    return detail::make_ortho( TU, TV );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const std::list< blas::matrix< value_t > > &  U,
         const std::list< blas::matrix< value_t > > &  V,
         const accuracy &                              acc )
{
    HLR_ASSERT( U.size() == V.size() );

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

        return detail::randlr( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  op = operator_wrapper( U, V );
        
        return detail::randlr( op, acc );
    }// else
}

template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
randlr_ortho ( const std::list< blas::matrix< value_t > > &  U,
               const std::list< blas::matrix< value_t > > &  V,
               const accuracy &                              acc )
{
    auto  [ TU, TV ] = randlr( U, V, acc );

    return detail::make_ortho( TU, TV );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const std::list< blas::matrix< value_t > > &  U,
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

        return randlr( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  op = operator_wrapper( U, T, V );
        
        return randlr( op, acc );
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct RandLR
{
    using  value_t = T_value;
    
    // signal support for general lin. operators
    static constexpr bool supports_general_operator = true;
    
    //
    // matrix approximation routines
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const accuracy &           acc ) const
    {
        return hlr::approx::randlr( M, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const 
    {
        return hlr::approx::randlr( U, V, acc );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::randlr( U, V, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const accuracy &                              acc ) const
    {
        return hlr::approx::randlr( U, T, V, acc );
    }

    template < typename operator_t >
    std::pair< blas::matrix< typename operator_t::value_t >,
               blas::matrix< typename operator_t::value_t > >
    operator () ( const operator_t &       op,
                  const accuracy &         acc ) const
    {
        return detail::randlr< operator_t >( op, acc );
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
        return hlr::approx::randlr_ortho( M, acc );
    }

    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const blas::matrix< value_t > &  U,
                   const blas::matrix< value_t > &  V,
                   const accuracy &                 acc ) const 
    {
        return hlr::approx::randlr_ortho( U, V, acc );
    }
    
    std::tuple< blas::matrix< value_t >,
                blas::vector< real_type_t< value_t > >,
                blas::matrix< value_t > >
    approx_ortho ( const std::list< blas::matrix< value_t > > &  U,
                   const std::list< blas::matrix< value_t > > &  V,
                   const accuracy &                              acc ) const
    {
        return hlr::approx::randlr_ortho( U, V, acc );
    }

    //
    // compute (approximate) column basis
    //
    
    template < typename operator_t >
    blas::matrix< typename operator_t::value_t >
    column_basis ( const operator_t &       op,
                   const accuracy &         acc ) const
    {
        return detail::rand_column_basis< operator_t >( op, acc, 2, 0, 0 );
    }
};

// signals, that T is of approximation type
template < typename T > struct is_approximation< RandLR< T > > { static const bool  value = true; };

}}// namespace hlr::approx

#endif // __HLR_APPROX_RANDLR_HH
