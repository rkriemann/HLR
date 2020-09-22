#ifndef __HLR_TF_APPROX_HH
#define __HLR_TF_APPROX_HH
//
// Project     : HLib
// File        : tf/approx
// Description : approximation algorithms using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include "hlr/arith/magma.hh"
#include "hlr/approx/rrqr.hh"

namespace hlr { namespace tf {

namespace hpro = HLIB;

using namespace hpro;

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template <typename T>
std::pair< blas::matrix< T >,
           blas::matrix< T > >
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
        HLR_ERROR( "not implemented" );
    }// if
    else
    {
        // [ QV, RV ] = qr( V )
        auto  QV = blas::copy( V );
        auto  RV = blas::matrix< value_t >( in_rank, in_rank );

        blas::magma::qr( QV, RV );

        // compute column-pivoted QR of U·RV'
        auto  QU = blas::prod( value_t(1), U, adjoint(RV) );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        auto  P  = std::vector< int >( in_rank, 0 );

        blas::magma::qrp( QU, RU, P );

        auto  out_rank = hlr::approx::detail::trunc_rank( RU, acc );
        
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

}}// namespace hlr::tf

#endif // __HLR_TF_ARITH_HH
