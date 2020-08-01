#ifndef __HLR_APPROX_SVD_HH
#define __HLR_APPROX_SVD_HH
//
// Project     : HLib
// Module      : approx/svd
// Description : low-rank approximation functions using SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

using hpro::idx_t;

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( blas::matrix< value_t > &  M,
      const hpro::TTruncAcc &    acc )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    //
    // perform SVD of M
    //

    const idx_t              n   = idx_t( M.nrows() );
    const idx_t              m   = idx_t( M.ncols() );
    const idx_t              mrc = std::min(n,m);
    blas::vector< real_t >   S( mrc );
    blas::matrix< value_t >  V( m, mrc );

    blas::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const blas::Range        row_is( 0, n-1 );
    const blas::Range        col_is( 0, m-1 );
    blas::matrix< value_t >  Uk( M, row_is, blas::Range( 0, k-1 ) );
    blas::matrix< value_t >  Vk( V, col_is, blas::Range( 0, k-1 ) );
    
    blas::matrix< value_t >  A( n, k );
    blas::matrix< value_t >  B( m, k );

    blas::copy( Uk, A );
    blas::copy( Vk, B );

    if ( n < m ) prod_diag( A, S, k );
    else         prod_diag( B, S, k );

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
      const hpro::TTruncAcc &          acc )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    HLR_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    const idx_t  acc_rank = idx_t( acc.rank() );

    blas::matrix< value_t >  OU, OV;
    
    if ( in_rank == 0 )
    {
        // reset matrices
        OU = std::move( blas::matrix< value_t >( nrows_U, 0 ) );
        OV = std::move( blas::matrix< value_t >( nrows_V, 0 ) );

        return { std::move( OU ), std::move( OV ) };
    }// if

    if ( in_rank <= acc_rank )
    {
        OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
        OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );

        return { std::move( OU ), std::move( OV ) };
    }// if

    //
    // truncate given low-rank matrix
    //
    
    if ( std::max( in_rank, acc_rank ) >= std::min( nrows_U, nrows_V ) / 2 )
    {
        //
        // since rank is too large, build U = U·V^T and do full-SVD
        //
            
        auto  M    = blas::prod( value_t(1), U, adjoint(V) );
        auto  lacc = hpro::TTruncAcc( acc );

        if ( acc_rank > 0 )
            lacc.set_max_rank( acc_rank );

        std::tie( OU, OV ) = svd( M, lacc );
    }// if
    else
    {
        //
        // do QR-factorisation of U and V
        //

        #if 1

        blas::matrix< value_t >  QU, RU;

        QU = std::move( blas::matrix< value_t >( U.nrows(), in_rank ) );
        RU = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( U, QU );
        blas::qr2( QU, RU );
        
        blas::matrix< value_t >  QV, RV;
        
        QV = std::move( blas::matrix< value_t >( V.nrows(), in_rank ) );
        RV = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( V, QV );
        blas::qr2( QV, RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        blas::matrix< value_t >  R( in_rank, in_rank );

        blas::prod( value_t(1), RU, adjoint(RV), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) );
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

            const blas::Range  in_rank_is( 0, in_rank-1 );
            const blas::Range  orank_is( 0, orank-1 );

            // U := Q_U · U
            blas::matrix< value_t >  Urank( Us, in_rank_is, orank_is );
            
            // U := U·S
            blas::prod_diag( Urank, Ss, orank );
            OU = blas::prod( value_t(1), QU, Urank );
            
            // V := Q_V · conj(V)
            blas::matrix< value_t >  Vrank( Vs, in_rank_is, orank_is );

            OV = blas::prod( value_t(1), QV, Vrank );
        }// if
        else
        {
            OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
            OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );
        }// else

        #else

        blas::matrix< value_t >  QU, RU;
        std::vector< value_t >   TU;

        QU = std::move( blas::matrix< value_t >( U.nrows(), in_rank ) );
        RU = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( U, QU );
        blas::qr_impl( QU, RU, TU );
        
        blas::matrix< value_t >  QV, RV;
        std::vector< value_t >   TV;
        
        QV = std::move( blas::matrix< value_t >( V.nrows(), in_rank ) );
        RV = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::copy( V, QV );
        blas::qr_impl( QV, RV, TV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        blas::matrix< value_t >  R( in_rank, in_rank );

        blas::prod( value_t(1), RU, adjoint(RV), value_t(0), R );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) );
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

            const blas::Range  in_rank_is( 0, in_rank-1 );
            const blas::Range  orank_is( 0, orank-1 );

            // U := U·S
            blas::matrix< value_t >  Urank( Us, in_rank_is, orank_is );
            
            blas::prod_diag( Urank, Ss, orank );

            // U := Q_U · U
            OU = std::move( blas::matrix< value_t >( nrows_U, orank ) );

            auto  OU_sub = blas::matrix< value_t >( OU, in_rank_is, orank_is );

            blas::copy( Urank, OU_sub );
            blas::prod_Q( blas::from_left, hpro::apply_normal, QU, TU, OU );

            // V := Q_V · conj(V)
            blas::matrix< value_t >  Vrank( Vs, in_rank_is, orank_is );

            OV = std::move( blas::matrix< value_t >( nrows_V, orank ) );

            auto  OV_sub = blas::matrix< value_t >( OV, in_rank_is, orank_is );

            blas::copy( Vrank, OV_sub );
            blas::prod_Q( blas::from_left, hpro::apply_normal, QV, TV, OV );
        }// if
        else
        {
            OU = std::move( blas::matrix< value_t >( U, hpro::copy_value ) );
            OV = std::move( blas::matrix< value_t >( V, hpro::copy_value ) );
        }// else

        #endif
    }// else

    return { std::move( OU ), std::move( OV ) };
}

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H using SVD
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( const std::list< blas::matrix< value_t > > &  U,
      const std::list< blas::matrix< value_t > > &  V,
      const hpro::TTruncAcc &                       acc )
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

        auto [ U_tr, V_tr ] = svd( D, acc );

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

        return svd( D, acc );
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
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc ) const
    {
        return std::move( hlr::approx::svd( M, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const hpro::TTruncAcc &          acc ) const 
    {
        return std::move( hlr::approx::svd( U, V, acc ) );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::svd( U, V, acc ) );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const hpro::TTruncAcc &                       acc ) const
    {
        return std::move( hlr::approx::svd( U, T, V, acc ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_SVD_HH
