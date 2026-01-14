#pragma once
#ifndef __HLR_APPROX_ID_HH
#define __HLR_APPROX_ID_HH
//
// Project     : HLR
// Module      : approx/id
// Description : low-rank approximation functions using interpolative decomposition
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2025. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>
#include <hlr/approx/tools.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using Hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
id ( blas::matrix< value_t > &  M,
     const accuracy &           acc )
{
    using  real_t = real_type_t< value_t >;
    
    const idx_t  nrows = idx_t( M.nrows() );
    const idx_t  ncols = idx_t( M.ncols() );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( nrows, ncols ) );
    
    //
    // perform column pivoted QR of M
    //

    auto  A = blas::copy( M );
    auto  R = blas::matrix< value_t >( ncols, ncols );
    auto  P = std::vector< int >( ncols, 0 );

    blas::qrp( A, R, P );

    //
    // determine rank by looking at diagonal entries in R
    //

    auto  S = blas::vector< real_t >( ncols );

    for ( size_t  i = 0; i < ncols; ++i )
        S(i) = math::abs( R(i,i) );

    const auto  k = acc.trunc_rank( S );
    
    //
    // extract for k columns of M for C
    //

    auto  C = blas::matrix< value_t >( nrows, k );
    
    for ( size_t  i = 0; i < ncols; ++i )
    {
        auto  M_i = M.column( P[i] );
        auto  C_i = C.column( i );

        blas::copy( M_i, C_i );
    }// for

    //
    // form X = [ I_k R(1:k,1:k)^-1 R(1:k,k+1:ncols) ]
    //

    auto  R11 = blas::matrix( R, blas::range( 0, k-1 ), blas::range( 0, k-1 ) );
    auto  R12 = blas::matrix( R, blas::range( 0, k-1 ), blas::range( k, ncols-1 ) );
    auto  R1  = blas::copy( R11 );

    blas::invert( R1 );

    auto  T = blas::prod( R1, R12 );
    
    // then permute rows of R2 (do P·R') and copy to X
    auto  X = blas::matrix< value_t >( ncols, k );
    
    for ( int i = 0; i < ncols; ++i )
    {
        auto  T_i = T.row( P[i] );
        auto  X_i = V.column( i );

        blas::copy( blas::adjoint( T_i ), X_i );
    }// for

    return { std::move( C ), std::move( X ) };
}

}}// namespace hlr::approx

#endif // __HLR_APPROX_ID_HH
