#ifndef __HLR_APPROX_RANDSVD_HH
#define __HLR_APPROX_RANDSVD_HH
//
// Project     : HLib
// Module      : approx/randsvd
// Description : low-rank approximation functions using randomized SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <random>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

namespace detail
{

//
// compute basis for column space (range) of M
//
template < typename operator_t >
blas::matrix< typename operator_t::value_t >
column_basis ( const operator_t &       M,
               const hpro::TTruncAcc &  acc,
               const uint               power_steps,
               const uint               oversampling )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;
        
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

        prod( value_t(1), hpro::apply_normal, M, T, Q );

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
                prod( value_t(1), hpro::apply_adjoint, M, Q, MtQ );
                blas::qr( MtQ, R );
                
                blas::scale( value_t(0), Q );
                prod( value_t(1), hpro::apply_normal, M, MtQ, Q );
                blas::qr( Q, R );
            }// for
        }// if

        return Q;
    }// if
    else
    {
        real_t      norm_M  = real_t(0);
        const auto  rel_eps = acc.rel_eps();
        const auto  abs_eps = acc.abs_eps();
        const uint  bsize   = std::min< uint >( 4, std::min< uint >( nrows_M, ncols_M ) );
        const uint  nblocks = std::min< uint >( nrows_M, ncols_M ) / bsize;
        auto        Qs      = std::list< blas::matrix< value_t > >();
        auto        T_i     = blas::matrix< value_t >( ncols_M, bsize );
        auto        QhQi    = blas::matrix< value_t >( bsize,   bsize );
        auto        TQ_i    = blas::matrix< value_t >( nrows_M, bsize );
        auto        R       = blas::matrix< value_t >( bsize,   bsize );
        auto        MtQ     = blas::matrix< value_t >( ncols_M, bsize );

        for ( uint  i = 0; i < nblocks; ++i )
        {
            //
            // draw random matrix and compute approximation of remainder M - Σ_j Q_j·Q_j'·M
            //
            
            blas::fill_fn( T_i, rand_norm );
            
            auto  Q_i = blas::matrix< value_t >( nrows_M, bsize );

            prod( value_t(1), hpro::apply_normal, M, T_i, Q_i );

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
                    prod( value_t(1), hpro::apply_adjoint, M, Q_i, MtQ );
                    blas::qr( MtQ, R );
                    
                    blas::scale( value_t(0), Q_i );
                    prod( value_t(1), hpro::apply_normal, M, MtQ, Q_i );
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
                blas::copy( Q_i, TQ_i ); // auto  C_i   = blas::matrix< value_t >( Q_i, hpro::copy_value );
                
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

        return Q;
    }// else
}

//
// computes column basis for U·V'
// - slightly faster than general version
//
template < typename value_t >
blas::matrix< value_t >
column_basis ( const blas::matrix< value_t > &  U,
               const blas::matrix< value_t > &  V,
               const hpro::TTruncAcc &          acc,
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
        const uint  bsize   = std::min< uint >( 4, std::min< uint >( nrows, ncols ) );
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

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename operator_t >
std::pair< blas::matrix< typename operator_t::value_t >,
           blas::matrix< typename operator_t::value_t > >
randsvd ( const operator_t &       M,
          const hpro::TTruncAcc &  acc,
          const uint               power_steps,
          const uint               oversampling )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    const auto  nrows_M = nrows( M );
    const auto  ncols_M = ncols( M );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( nrows_M, ncols_M ) );
    
    // compute column basis
    auto  Q   = detail::column_basis( M, acc, power_steps, oversampling );
    auto  k   = Q.ncols();

    // B = Q^H · M  or B^H = M^H · Q
    auto  BT  = blas::matrix< value_t >( ncols_M, k );

    prod( value_t(1), hpro::apply_adjoint, M, Q, BT );
    
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
    auto  Uk = blas::matrix< value_t >( R_B, blas::range::all, blas::range( 0, k-1 ) );
    auto  Vk = blas::matrix< value_t >( V,   blas::range::all, blas::range( 0, k-1 ) );
    
    auto  OU = blas::prod( value_t(1), Q,  Vk );
    auto  OV = blas::prod( value_t(1), BT, Uk );

    if ( nrows_M < ncols_M )
        blas::prod_diag( OU, S, k );
    else
        blas::prod_diag( OV, S, k );

    return { std::move( OU ), std::move( OV ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const blas::matrix< value_t > &  U,
          const blas::matrix< value_t > &  V,
          const hpro::TTruncAcc &          acc,
          const uint                       power_steps,
          const uint                       oversampling )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;

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

        return randsvd( M, acc, power_steps, oversampling );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        //
        // compute column basis
        //

        auto  Q      = detail::column_basis( U, V, acc, power_steps, oversampling );
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
        auto  Uk = blas::matrix< value_t >( U_svd, blas::range::all, blas::range( 0, out_rank-1 ) );
        auto  Vk = blas::matrix< value_t >( V_svd, blas::range::all, blas::range( 0, out_rank-1 ) );

        auto  OU = blas::prod( value_t(1), Q,    Vk );
        auto  OV = blas::prod( value_t(1), VUtQ, Uk );

        if ( nrows_U < nrows_V )
            blas::prod_diag( OU, S, out_rank );
        else
            blas::prod_diag( OV, S, out_rank );

        return { std::move( OU ), std::move( OV ) };
    }// else
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const std::list< blas::matrix< value_t > > &  U,
          const std::list< blas::matrix< value_t > > &  V,
          const hpro::TTruncAcc &                       acc,
          const uint                                    power_steps,
          const uint                                    oversampling )
{
    assert( U.size() == V.size() );

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
    
        return randsvd( U_all, V_all, acc, power_steps, oversampling );
    }// else
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
          const hpro::TTruncAcc &                       acc,
          const uint                                    power_steps,
          const uint                                    oversampling )
{
    assert( U.size() == T.size() );
    assert( T.size() == V.size() );

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
                  const hpro::TTruncAcc &    acc ) const
    {
        return hlr::approx::randsvd( M, acc, power_steps, oversampling );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        auto  Uc = blas::copy( U );
        auto  Vc = blas::copy( V );
        
        return hlr::approx::randsvd( Uc, Vc, acc, power_steps, oversampling );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::randsvd( U, V, acc, power_steps, oversampling );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::randsvd( U, T, V, acc, power_steps, oversampling );
    }

    template < typename operator_t >
    std::pair< blas::matrix< typename operator_t::value_t >,
               blas::matrix< typename operator_t::value_t > >
    operator () ( const operator_t &       op,
                  const hpro::TTruncAcc &  acc ) const
    {
        return hlr::approx::randsvd< operator_t >( op, acc, power_steps, oversampling );
    }

    //
    // compute (approximate) column basis
    //
    
    template < typename operator_t >
    blas::matrix< typename operator_t::value_t >
    column_basis ( const operator_t &       op,
                   const hpro::TTruncAcc &  acc ) const
    {
        return detail::column_basis< operator_t >( op, acc, power_steps, oversampling );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_RANDSVD_HH
