#ifndef __HLR_TBB_BLAS_HH
#define __HLR_TBB_BLAS_HH
//
// Project     : HLR
// Module      : tbb/blas
// Description : optimized BLAS algorithms using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <functional>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_invoke.h>
#include <tbb/blocked_range3d.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace tbb { namespace blas {

using namespace hlr::blas;

//
// dot product
//
template < tensor_type  tensorA_t,
           tensor_type  tensorB_t >
requires ( std::same_as< typename tensorA_t::value_t,
                         typename tensorB_t::value_t > )
typename tensorA_t::value_t
dot ( const tensorA_t &  A,
      const tensorB_t &  B )
{
    using value_t = typename tensorA_t::value_t;
    
    HLR_DBG_ASSERT( ( A.size(0) == B.size(0) ) &&
                    ( A.size(1) == B.size(1) ) &&
                    ( A.size(2) == B.size(2) ) );

    auto  d = value_t(0);

    ::tbb::parallel_reduce(
        ::tbb::blocked_range3d< size_t >( 0, A.size(0),
                                          0, A.size(1),
                                          0, A.size(2) ),
        [&A,&B] ( const auto &  r,
                  const auto    val )
        {
            auto  s = val;
            
            for ( size_t  l = r.pages().begin(); l < r.pages().end(); l++ )
                for ( size_t  j = r.cols().begin(); j < r.cols().end(); j++ )
                    for ( size_t  i = r.rows().begin(); i < r.rows().end(); i++ )
                        s += A(i,j,l) * B(i,j,l);

            return s;
        },
        std::plus< value_t >()
    );

    return d;
}

//
// Frobenius norm
//
template < tensor_type  tensor_t >
hlr::real_type_t< typename tensor_t::value_t >
norm_F ( const tensor_t &  t )
{
    return std::sqrt( std::abs( dot( t, t ) ) );
}

//
// compute B := α A + β B (element wise)
//
template < typename     alpha_t,
           tensor_type  tensorA_t,
           tensor_type  tensorB_t >
requires ( std::same_as< typename tensorA_t::value_t,
                         typename tensorB_t::value_t > &&
           std::convertible_to< alpha_t, typename tensorA_t::value_t > )
void
add ( const alpha_t      alpha,
      const tensorA_t &  A,
      tensorB_t &        B )
{
    using value_t = typename tensorA_t::value_t;
    
    HLR_DBG_ASSERT( ( A.size(0) == B.size(0) ) &&
                    ( A.size(1) == B.size(1) ) &&
                    ( A.size(2) == B.size(2) ) );
    
    ::tbb::parallel_for(
        ::tbb::blocked_range3d< size_t >( 0, A.size(0),
                                          0, A.size(1),
                                          0, A.size(2) ),
        [alpha,&A,&B] ( const auto &  r )
        {
            for ( size_t  l = r.pages().begin(); l < r.pages().end(); l++ )
                for ( size_t  j = r.cols().begin(); j < r.cols().end(); j++ )
                    for ( size_t  i = r.rows().begin(); i < r.rows().end(); i++ )
                        B(i,j,l) += value_t(alpha) * A(i,j,l);
        }
    );
}

////////////////////////////////////////////////////////////////
//
// truncation
//

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< blas::tensor3< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
hosvd ( const blas::tensor3< value_t > &  X,
        const accuracy &                  acc,
        const approx_t &                  apx )
{
    auto  U0 = blas::matrix< value_t >();
    auto  U1 = blas::matrix< value_t >();
    auto  U2 = blas::matrix< value_t >();

    ::tbb::parallel_invoke(
        [&] ()
        {
            auto  X0 = X.unfold( 0 );
            
            U0 = std::move( apx.column_basis( X0, acc ) );
        },
        
        [&] ()
        {
            auto  X1 = X.unfold( 1 );

            U1 = std::move( apx.column_basis( X1, acc ) );
        },
        
        [&] ()
        {
            auto  X2 = X.unfold( 2 );

            U2 = std::move( apx.column_basis( X2, acc ) );
        } );
        
    auto  Y0 = blas::tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = blas::tensor_product( Y0, adjoint( U1 ), 1 );
    auto  G  = blas::tensor_product( Y1, adjoint( U2 ), 2 );

    return { std::move(G), std::move(U0), std::move(U1), std::move(U2) };
}

template < typename  value_t >
std::tuple< blas::tensor3< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
hosvd ( const blas::tensor3< value_t > &  X,
        const accuracy &                  acc )
{
    const auto  apx = approx::SVD< value_t >();

    return hosvd( X, acc, apx );
}

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< blas::tensor3< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
greedy_hosvd ( const blas::tensor3< value_t > &  X,
               const accuracy &                  acc,
               const approx_t &                  apx )
{
    //
    // compute full column bases for unfolded matrices
    // for all dimensions
    //
    
    auto  U0 = blas::matrix< value_t >();
    auto  U1 = blas::matrix< value_t >();
    auto  U2 = blas::matrix< value_t >();
    auto  S0 = blas::vector< value_t >();
    auto  S1 = blas::vector< value_t >();
    auto  S2 = blas::vector< value_t >();

    ::tbb::parallel_invoke(
        [&] ()
        {
            auto  X0 = X.unfold( 0 );
            
            std::tie( U0, S0 ) = std::move( apx.column_basis( X0 ) );
        },
        
        [&] ()
        {
            auto  X1 = X.unfold( 1 );

            std::tie( U1, S1 ) = std::move( apx.column_basis( X1 ) );
        },
        
        [&] ()
        {
            auto  X2 = X.unfold( 2 );

            std::tie( U2, S2 ) = std::move( apx.column_basis( X2 ) );
        } );

    // for index-based access
    blas::matrix< value_t >  U[3] = { U0, U1, U2 };
    blas::vector< value_t >  S[3] = { S0, S1, S2 };

    //
    // iterate until error is met increasing rank of
    // dimension with highest error contribution, i.e.,
    // largest _next_ singular value
    //
    // error = √( Σ_d Σ_i>k_i σ²_d,i )
    //

    const auto  tol      = acc.abs_eps() * acc.abs_eps();
    value_t     error[3] = { 0, 0, 0 };
    size_t      k[3]     = { 1, 1, 1 }; // start with at least one rank per dimension

    // initial error
    for ( uint  d = 0; d < 3; ++d )
        for ( uint  i = k[d]; i < S[d].length(); ++i )
            error[d] += S[d](i) * S[d](i);

    // iteration
    while ( error[0] + error[1] + error[2] > tol )
    {
        int      max_dim = -1; // to signal error
        value_t  max_sig = 0;

        // look for maximal σ in all dimensions
        for ( uint  d = 0; d < 3; ++d )
        {
            // skip fully exhausted dimensions
            if ( k[d] == S[d].length() )
                continue;
            
            if ( S[d](k[d]) > max_sig )
            {
                max_sig = S[d](k[d]);
                max_dim = d;
            }// if
        }// for

        if ( max_dim < 0 )
        {
            // no unused singular values left; error should be zero
            break;
        }// if

        error[ max_dim ] -= max_sig * max_sig;
        k[ max_dim ]     += 1;
    }// while

    auto  U0k = blas::matrix( U0, range::all, range( 0, k[0]-1 ) );
    auto  U1k = blas::matrix( U1, range::all, range( 0, k[1]-1 ) );
    auto  U2k = blas::matrix( U2, range::all, range( 0, k[2]-1 ) );

    auto  W0  = blas::copy( U0k );
    auto  W1  = blas::copy( U1k );
    auto  W2  = blas::copy( U2k );
    
    auto  Y0 = blas::tensor_product( X,  adjoint( W0 ), 0 );
    auto  Y1 = blas::tensor_product( Y0, adjoint( W1 ), 1 );
    auto  G  = blas::tensor_product( Y1, adjoint( W2 ), 2 );
    
    return { std::move(G), std::move(W0), std::move(W1), std::move(W2) };
}

}}}// namespace hlr::tbb::blas

#endif // __HLR_TBB_BLAS_HH
