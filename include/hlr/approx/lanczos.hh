#ifndef __HLR_APPROX_LANCZOS_HH
#define __HLR_APPROX_LANCZOS_HH
//
// Project     : HLib
// Module      : approx/lanczos
// Description : low-rank approximation functions using Lanczos Bidiagonalization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

namespace detail
{

//
// version of Lanczos bidiagonalization with generic operator
//
template < typename operator_t >
std::pair< blas::matrix< typename operator_t::value_t >,
           blas::matrix< typename operator_t::value_t > >
lanczos ( operator_t &             M,
          const hpro::TTruncAcc &  acc )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    const auto  nrowsM = nrows( M );
    const auto  ncolsM = ncols( M );

    auto  U     = std::deque< blas::vector< value_t > >();
    auto  V     = std::deque< blas::vector< value_t > >();
    auto  alpha = std::deque< real_t >();
    auto  beta  = std::deque< real_t >();

    //
    // start (step 0)
    //
    
    // random, normalised start vector
    auto  u = blas::random< value_t >( nrowsM );

    blas::scale( value_t(1) / blas::norm2(u), u );

    // v = M'·u / |M'·u|
    auto  v = blas::vector< value_t >( ncolsM );

    prod( value_t(1), hpro::apply_adjoint, M, u, v );
    alpha.push_back( blas::norm2(v) );
    blas::scale( value_t(1) / alpha[0], v );

    U.push_back( std::move( u ) );
    V.push_back( std::move( v ) );

    //
    // Lanczos iteration
    //

    uint    step   = 1;
    real_t  norm_M = alpha[0]; // approximation of |M|
    real_t  norm_R = alpha[0]; // approximation of |M-U·S·V'|
    
    do
    {
        //
        // check stop criterion
        //

        if (( acc.is_fixed_rank() && ( step >= acc.rank() ) ) ||
            ( acc.is_fixed_prec() && ( norm_R <= acc.rel_eps() * norm_M )) ||
            ( acc.has_max_rank()  && ( step >= acc.max_rank() ) ) ||
            ( norm_R <= acc.abs_eps() ))
            break;
             
        //
        // u = M·v - alpha(step) · U(:,step);
        //
        
        u = std::move( blas::vector< value_t >( nrowsM ) );
        prod( value_t(1), hpro::apply_normal, M, V[step-1], u );
        blas::add( - value_t(alpha[step-1]), U[step-1], u );

        // normalize
        beta.push_back( blas::norm2( u ) );
        blas::scale( value_t(1) / beta[step-1], u );

        // and orthogonalize u wrt. to U (except last u)
        for ( uint  i = 0; i < step; ++i )
            blas::add( - blas::dot( u, U[i] ), U[i], u );

        //
        // v = M'·u - beta_step · V(:,step);
        //

        v = std::move( blas::vector< value_t >( ncolsM ) );
        prod( value_t(1), hpro::apply_adjoint, M, u, v );
        blas::add( - value_t(beta[step-1]), V[step-1], v );

        // normalize
        alpha.push_back( blas::norm2( v ) );
        blas::scale( value_t(1) / alpha[step], v );
        
        // and orthogonalize v wrt. to V
        for ( uint  i = 0; i < step; ++i )
            blas::add( - blas::dot( v, V[i] ), V[i], v );

        U.push_back( std::move( u ) );
        V.push_back( std::move( v ) );

        //
        // update |M| = |diag(alpha) + diag(beta,-1)| and
        // norm of remainder (alpha[step])
        //
             
        norm_M = math::sqrt( math::square( norm_M ) + math::square( alpha[step] ) + math::square( beta[step-1] ) );
        norm_R = alpha[step];

        ++step;
        
    } while ( step <= std::min( nrowsM, ncolsM ) ); // fallback 


    //
    // U = U·diag(alpha) + diag(beta,-1), V remains unchanged
    //

    for ( uint  i = 0; i < step-1; ++i )
    {
        blas::scale( value_t(alpha[i]), U[i] );
        blas::add( value_t(beta[i]), U[i+1], U[i] );
    }// for
    
    blas::scale( value_t(alpha[step-1]), U[step-1] );

    //
    // form final low-rank matrices
    //

    auto  RU = blas::matrix< value_t >( nrowsM, step );
    auto  RV = blas::matrix< value_t >( ncolsM, step );

    for ( uint  i = 0; i < step; ++i )
    {
        auto  RU_i = RU.column( i );
        auto  RV_i = RV.column( i );

        blas::copy( U[i], RU_i );
        blas::copy( V[i], RV_i );
    }// for
    
    return { std::move( RU ), std::move( RV ) };
}

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
lanczos ( blas::matrix< value_t > &  M,
          const hpro::TTruncAcc &    acc )
{
    return std::move( detail::lanczos( M, acc ) );
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
lanczos ( const blas::matrix< value_t > &  U,
          const blas::matrix< value_t > &  V,
          const hpro::TTruncAcc &          acc )
{
    HLR_ASSERT( U.ncols() == V.ncols() );

    auto  op = operator_wrapper( U, V );

    return std::move( detail::lanczos( op, acc ) );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
lanczos ( const std::list< blas::matrix< value_t > > &  U,
          const std::list< blas::matrix< value_t > > &  V,
          const hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == V.size() );

    auto  op = operator_wrapper( U, V );

    return std::move( detail::lanczos( op, acc ) );
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
lanczos ( const std::list< blas::matrix< value_t > > &  U,
          const std::list< blas::matrix< value_t > > &  T,
          const std::list< blas::matrix< value_t > > &  V,
          const hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == V.size() );

    auto  op = operator_wrapper( U, T, V );

    return std::move( detail::lanczos( op, acc ) );
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct Lanczos
{
    using  value_t = T_value;

    //
    // operators
    //
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc ) const
    {
        return std::move( hlr::approx::lanczos( M, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        return std::move( hlr::approx::lanczos( U, V, acc ) );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::lanczos( U, V, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::lanczos( U, T, V, acc ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_LANCZOS_HH
