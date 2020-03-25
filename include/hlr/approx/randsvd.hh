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

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

namespace detail
{

//
// compute basis for column space (range) of M
//
template < typename value_t >
blas::matrix< value_t >
column_basis ( const blas::matrix< value_t > &  M,
               const hpro::TTruncAcc &          acc,
               const uint                       power_steps,
               const uint                       oversampling )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;
        
    const idx_t  n = idx_t( M.nrows() );
    const idx_t  m = idx_t( M.ncols() );

    std::random_device          rd{};
    std::mt19937                generator{ rd() };
    std::normal_distribution<>  distr{ 0, 1 };
    auto                        fill_rand = [&] () { return distr( generator ); };
    
    if ( acc.is_fixed_rank() )
    {
        const auto  k = idx_t(acc.rank());
        auto        T = blas::matrix< value_t >( m, k + oversampling );

        blas::fill( T, fill_rand );
        
        auto        Y = blas::prod( value_t(1), M, T );

        //
        // power iteration
        //
            
        auto  MtQ = blas::matrix< value_t >( m, k + oversampling );
        auto  R   = blas::matrix< value_t >( k + oversampling, k + oversampling );
        
        for ( uint  j = 0; j < power_steps; ++j )
        {
            blas::qr( Y, R );
            blas::prod( value_t(1), blas::adjoint(M), Y, value_t(0), MtQ );

            blas::qr( MtQ, R );
            blas::prod( value_t(1), M, MtQ, value_t(0), Y );
        }// for

        blas::qr( Y, R );

        return Y;
    }// if
    else
    {
        auto        A       = std::move( blas::copy( M ) );
        auto        norm_M  = blas::normF( M );
        const auto  rel_eps = acc.rel_eps();
        const auto  abs_eps = acc.abs_eps();
        const uint  bsize   = std::min< uint >( 4, std::min< uint >( n, m ) );
        const uint  nblocks = std::min< uint >( n, m ) / bsize;
        auto        Qs      = std::list< blas::matrix< value_t > >();
        auto        T_i     = blas::matrix< value_t >( m, bsize );

        for ( uint  i = 0; i < nblocks; ++i )
        {
            blas::fill( T_i, fill_rand );
            
            auto  Q_i  = blas::prod( value_t(1), M, T_i ); // Y_i
            auto  TQ_i = std::move( blas::copy( Q_i ) );
            
            for ( auto  Q_j : Qs )
            {
                const auto  QhQi = blas::prod( value_t(1), blas::adjoint( Q_j ), TQ_i );

                blas::prod( value_t(-1), Q_j, QhQi, value_t(1), Q_i );
            }// for

            real_t  norm_Qi = real_t(0);

            for ( uint  j = 0; j < bsize; ++j )
            {
                const auto  Qi_j = Q_i.column( j );

                norm_Qi = std::max( norm_Qi, blas::norm2( Qi_j ) );
            }// for

            // use first approximation also as approximation of norm of M
            if ( i == 0 )
                norm_M = norm_Qi;

            //
            // power iteration
            //
            
            auto  R   = blas::matrix< value_t >( bsize, bsize );
            auto  AtQ = blas::matrix< value_t >( m, bsize );
            
            for ( uint  j = 0; j < power_steps; ++j )
            {
                blas::qr( Q_i, R );
                blas::prod( value_t(1), blas::adjoint(A), Q_i, value_t(0), AtQ );
                
                blas::qr( AtQ, R );
                blas::prod( value_t(1), A, AtQ, value_t(0), Q_i );  // Q_i = Y_i
            }// for
            
            qr( Q_i, R );
            
            //
            // project Q_i away from previous Q_j
            //
            //    Q_i = Q_i - [ Q_0 .. Q_i-1 ] [ Q_0 .. Q_i-1 ]^H Q_i = Q_i - Σ_j=0^i-1 Q_j Q_j^H Q_i
            //
                
            if ( i > 0 )
            {
                auto  C_i   = blas::matrix< value_t >( Q_i, hpro::copy_value );
                auto  QjtQi = blas::matrix< value_t >( bsize, bsize );
                
                for ( const auto &  Q_j : Qs )
                {
                    blas::prod( value_t(1), blas::adjoint(Q_j), C_i, value_t(0), QjtQi );
                    blas::prod( value_t(-1), Q_j, QjtQi, value_t(1), Q_i );
                }// for
                
                qr( Q_i, R );
            }// if
            
            //
            // A = A - Q_i Q_i^t A
            //

            Qs.push_back( std::move( Q_i ) );
            
            if (( norm_Qi <= abs_eps ) || (( norm_Qi ) <= rel_eps * norm_M ))
                break;
        }// for
        
        //
        // collect Q_i's into final result
        //

        auto   Q = blas::matrix< value_t >( n, Qs.size() * bsize );
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
// compute basis for column space (range) of M = A·B^H
// (same algorithm as above but in factorised form)
//
template < typename value_t >
blas::matrix< value_t >
column_basis ( const blas::matrix< value_t > &  IU,
               const blas::matrix< value_t > &  IV,
               const hpro::TTruncAcc &          acc,
               const uint                       power_steps,
               const uint                       oversampling )
{
    const idx_t  n    = idx_t( IU.nrows() );
    const idx_t  m    = idx_t( IV.nrows() );
    const idx_t  rank = idx_t( IU.ncols() );
    
    std::random_device          rd{};
    std::mt19937                generator{ rd() };
    std::normal_distribution<>  distr{ 0, 1 };
    auto                        fill_rand = [&] () { return distr( generator ); };
    
    if ( acc.is_fixed_rank() )
    {
        auto         U   = std::move( blas::copy( IU ) );
        auto         V   = std::move( blas::copy( IV ) );
        const idx_t  k   = idx_t(acc.rank());
        auto         T   = blas::matrix< value_t >( n, k + oversampling );

        blas::fill( T, fill_rand );
        
        auto         VtT = blas::prod( value_t(1), blas::adjoint(V), T );
        auto         Y   = blas::prod( value_t(1), U, VtT );

        //
        // power iteration
        //
        
        auto  UtQ  = blas::matrix< value_t >( rank, k + oversampling );
        auto  VUtQ = blas::matrix< value_t >( m, k + oversampling );
        auto  R    = blas::matrix< value_t >( k + oversampling, k + oversampling );
        
        for ( uint  j = 0; j < power_steps; ++j )
        {
            // [Y,R] = qr(Y); MtQ = M^H·Y = V·U^H·Y
            blas::qr( Y, R );
            blas::prod( value_t(1), blas::adjoint(U), Y, value_t(0), UtQ );
            blas::prod( value_t(1), V, UtQ, value_t(0), VUtQ );

            // [Q,R] = qr(V·U^H·Y); Y = U·V^H·Q
            blas::qr( VUtQ, R );
            blas::prod( value_t(1), blas::adjoint(V), VUtQ, value_t(0), UtQ );
            blas::prod( value_t(1), U, UtQ, value_t(0), Y );
        }// for

        blas::qr( Y, R );

        return Y;
    }// if
    else
    {
        auto        U       = std::move( blas::copy( IU ) );
        auto        V       = std::move( blas::copy( IV ) );
        const auto  norm_0  = lr_normF( U, V );
        const auto  rel_eps = acc.rel_eps();
        const auto  abs_eps = acc.abs_eps();
        const uint  bsize   = std::min< uint >( 4, std::min< uint >( n, m ) );
        const uint  nblocks = std::min( n, m ) / bsize;
        auto        Qs      = std::list< blas::matrix< value_t > >();
        auto        T_i     = blas::matrix< value_t >( m, bsize );
        auto        VUtQ    = blas::matrix< value_t >( m, bsize );

        for ( uint  i = 0; i < nblocks; ++i )
        {
            blas::fill( T_i, fill_rand );

            auto  VtT = blas::prod( value_t(1), blas::adjoint(V), T_i );
            auto  Q_i = blas::prod( value_t(1), U, VtT ); // Y_i

            //
            // power iteration
            //
            
            auto  R   = blas::matrix< value_t >( bsize, bsize );
            auto  UtQ = blas::matrix< value_t >( rank, bsize );
            
            for ( uint  j = 0; j < power_steps; ++j )
            {
                blas::qr( Q_i, R );
                blas::prod( value_t(1), blas::adjoint(U), Q_i, value_t(0), UtQ );
                blas::prod( value_t(1), V, UtQ, value_t(0), VUtQ );
                
                blas::qr( VUtQ, R );
                blas::prod( value_t(1), blas::adjoint(V), VUtQ, value_t(0), UtQ );
                blas::prod( value_t(1), U, UtQ, value_t(0), Q_i );  // Q_i = Y_i
            }// for
            
            blas::qr( Q_i, R );
            
            //
            // project Q_i away from previous Q_j
            //
                
            if ( i > 0 )
            {
                auto  C_i   = std::move( blas::copy( Q_i ) );
                auto  QjtQi = blas::matrix< value_t >( bsize, bsize );
                
                for ( const auto &  Q_j : Qs )
                {
                    blas::prod( value_t(1), blas::adjoint(Q_j), C_i, value_t(0), QjtQi );
                    blas::prod( value_t(-1), Q_j, QjtQi, value_t(1), Q_i );
                }// for
                
                blas::qr( Q_i, R );
            }// if

            //
            // M = M - Q_i Q_i^T M = U·V^H - Q_i Q_i^T U·V^H = (U - Q_i Q_i^T U) V^H
            //

            auto  QtA = blas::prod( value_t(1), blas::adjoint(Q_i), U );

            blas::prod( value_t(-1), Q_i, QtA, value_t(1), U );
            
            const auto  norm_i = blas::lr_normF( U, V );

            Qs.push_back( std::move( Q_i ) );
            
            if (( norm_i < abs_eps ) || (( norm_i / norm_0 ) < rel_eps ))
                break;
        }// for

        //
        // collect Q_i's into final result
        //

        auto   Q   = blas::matrix< value_t >( n, Qs.size() * bsize );
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
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( blas::matrix< value_t > &  M,
          const hpro::TTruncAcc &    acc,
          const uint                 power_steps,
          const uint                 oversampling )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    const idx_t  n   = idx_t( M.nrows() );
    const idx_t  m   = idx_t( M.ncols() );

    // compute column basis
    auto  Q = detail::column_basis( M, acc, power_steps, oversampling );
    auto  k = idx_t(Q.ncols());

    // B = Q^H · M  or B^H = M^H · Q
    auto  BT  = blas::prod( value_t(1), blas::adjoint(M), Q );
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
    
    auto  OU = prod( value_t(1), Q,  Vk );
    auto  OV = prod( value_t(1), BT, Uk );

    if ( n < m ) blas::prod_diag( OU, S, k );
    else         blas::prod_diag( OV, S, k );

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
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    if ( in_rank >= std::min( nrows, ncols ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return std::move( randsvd( M, acc, power_steps, oversampling ) );
    }// if
    else
    {
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

        if ( nrows < ncols ) blas::prod_diag( OU, S, out_rank );
        else                 blas::prod_diag( OV, S, out_rank );

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

        auto [ U_tr, V_tr ] = randsvd( D, acc, power_steps, oversampling );

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

        return randsvd( D, acc, power_steps, oversampling );
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

    // number of steps in power iteration during construction of column basis
    uint   power_steps;

    // oversampling parameter
    uint   oversampling;

    //
    // operators
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc ) const
    {
        return std::move( hlr::approx::randsvd( M, acc, power_steps, oversampling ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        return std::move( hlr::approx::randsvd( U, V, acc, power_steps, oversampling ) );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::randsvd( U, V, acc, power_steps, oversampling ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::randsvd( U, T, V, acc, power_steps, oversampling ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_RANDSVD_HH
