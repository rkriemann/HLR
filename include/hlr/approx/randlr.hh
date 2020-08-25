#ifndef __HLR_APPROX_RANDLR_HH
#define __HLR_APPROX_RANDLR_HH
//
// Project     : HLib
// Module      : approx/randlr
// Description : low-rank approximation functions using randomized low-rank approx.
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>
#include <deque>
#include <random>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

namespace detail
{

//
// compute low-rank approximation of a given sum Σ_i M_i using RANDLR
// - only need matrix-vector evaluation of given operators
//
template < typename operator_t >
std::pair< blas::matrix< typename operator_t::value_t >,
           blas::matrix< typename operator_t::value_t > >
randlr  ( const operator_t &       M,
          const hpro::TTruncAcc &  acc )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    const auto  nrows_M = nrows( M );
    const auto  ncols_M = ncols( M );
    
    // value considered zero to avoid division by small values
    const real_t  zero_val  = math::square( std::numeric_limits< real_t >::epsilon() );
    auto          norm_M    = real_t(0);
    auto          norm_R    = real_t(0);
    uint          k         = 0;
    auto          U         = std::deque< blas::vector< value_t > >();
    auto          V         = std::deque< blas::vector< value_t > >();

    // normal distributed random numbers
    auto          rd        = std::random_device{};
    auto          generator = std::mt19937{ rd() };
    auto          distr     = std::normal_distribution<>{ 0, 1 };
    auto          rand      = [&] () { return value_t( distr( generator ) ); };
    
    do
    {
        auto  W = blas::vector< value_t >( ncols_M );
        auto  L = blas::vector< value_t >( nrows_M );
        
        blas::fill( W, rand );
        prod( value_t(1), hpro::apply_normal, M, W, L );

        // subtract previously computed results
        if ( k > 0 )
        {
            auto  Lc = blas::copy( L );
        
            for ( auto &  u_i : U )
                blas::add( - blas::dot( u_i, Lc ), u_i, L );
        }// if

        const auto  norm_L = blas::norm2( L );

        // if norm is close to zero, matrix rest is zero as well
        if ( norm_L < zero_val )
            break;

        blas::scale( value_t( real_t(1) / norm_L ), L );

        blas::scale( value_t(0), W );
        prod( value_t(1), hpro::apply_adjoint, M, L, W );

        //
        // update norm of M as norm( U·V^H ) and
        // norm of ( M - U·V^H ) by last vector pair
        //

        norm_R = blas::norm2( W );      // L is normalized!
        
        auto norm_upd = norm_R * norm_R;

        for ( uint  i = 0; i < k; ++i )
        {
            norm_upd += std::real( blas::dot( W, V[i] ) * blas::dot( U[i], L ) );
            norm_upd += std::real( blas::dot( V[i], W ) * blas::dot( L, U[i] ) );
        }// for

        norm_M = std::sqrt( norm_M * norm_M + norm_upd );

        //
        // and store new vectors/pivots
        //

        U.push_back( std::move( L ) );
        V.push_back( std::move( W ) );

        k++;
        
        //
        // check stop criterion
        //

        if (( acc.is_fixed_rank() && ( k      >= acc.rank() ) ) ||
            ( acc.is_fixed_prec() && ( norm_R <= acc.rel_eps() * norm_M )) ||
            ( acc.has_max_rank()  && ( k      >= acc.max_rank() ) ) ||
            ( norm_R <= acc.abs_eps() ))
            break;
             
    } while ( k <= std::min( nrows_M, ncols_M ) ); // fallback 
    
    //
    // copy vectors to matrix
    //

    blas::matrix< value_t >  MU( nrows_M, k );
    blas::matrix< value_t >  MV( ncols_M, k );

    for ( uint  i = 0; i < k; ++i )
    {
        auto  u_i = MU.column( i );
        auto  v_i = MV.column( i );

        blas::copy( U[i], u_i );
        blas::copy( V[i], v_i );
    }// for
    
    return { std::move( MU ), std::move( MV ) };
}

}// namespace detail

////////////////////////////////////////////////////////////////////////////////
//
// functions for dense approximation and low-rank truncation
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( blas::matrix< value_t > &  M,
         const hpro::TTruncAcc &    acc )
{
    return std::move( detail::randlr( M, acc ) );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const blas::matrix< value_t > &  U,
         const blas::matrix< value_t > &  V,
         const hpro::TTruncAcc &          acc )
{
    auto  op = operator_wrapper( U, V );

    return std::move( detail::randlr( op, acc ) );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const std::list< blas::matrix< value_t > > &  U,
         const std::list< blas::matrix< value_t > > &  V,
         const hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    auto  op = operator_wrapper( U, V );
        
    return std::move( detail::randlr( op, acc ) );
        
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

        auto [ U_tr, V_tr ] = randlr( D, acc );

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
    
        return randlr( U_all, V_all, acc );
    }// else
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randlr ( const std::list< blas::matrix< value_t > > &  U,
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

        return detail::randlr( D, acc );
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
    
        return detail::randlr( U_all, V_all, acc );
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
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc ) const
    {
        return std::move( hlr::approx::randlr( M, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        return std::move( hlr::approx::randlr( U, V, acc ) );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::randlr( U, V, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::randlr( U, T, V, acc ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_RANDLR_HH
