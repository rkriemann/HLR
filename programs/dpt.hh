//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <hlr/arith/blas_eigen.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_jac_bw ( blas::matrix< value_t > &                          M,
               const size_t                                       amax_sweeps = 0,
               const typename hpro::real_type< value_t >::type_t  atolerance  = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         bsize      = 4;
    const auto         nbrows     = nrows / bsize;
    const auto         nbcols     = ncols / bsize;

    HLR_ASSERT( ( nrows / bsize ) * bsize == nrows );
    HLR_ASSERT( ( ncols / bsize ) * bsize == ncols );
    
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15 );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;
    auto               V          = blas::matrix< value_t >( minrc, ncols );

    // initialise V with identity
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    hpro::DBG::write( M, "M0.mat", "M0" );
    
    while ( ! converged && ( sweep < max_sweeps ))
    {
        real_t  max_err = 0.0;
        
        sweep++;
        converged = true;
                
        for ( size_t  i = 0; i < nbrows-1; i++ )
        {
            for ( size_t j = i + 1; j < nbcols; j++ )
            {
                //
                // diagonalize ⎧ M_ii  M_ij ⎫, e.g.,
                //             ⎩ M_ji  M_jj ⎭
                //
                // compute
                //
                // ⎧ V_ii  V_ij ⎫' ⎧ M_ii  M_ij ⎫ ⎧ V_ii  V_ij ⎫ = ⎧ D_ii    0  ⎫
                // ⎩ V_ji  V_jj ⎭  ⎩ M_ji  M_jj ⎭ ⎩ V_ji  V_jj ⎭   ⎩   0   D_jj ⎭
                //

                auto  r_i  = blas::range( i*bsize, (i+1)*bsize-1 );
                auto  r_j  = blas::range( j*bsize, (j+1)*bsize-1 );
                auto  M_ii = blas::matrix< value_t >( M, r_i, r_i );
                auto  M_ij = blas::matrix< value_t >( M, r_i, r_j );
                auto  M_ji = blas::matrix< value_t >( M, r_j, r_i );
                auto  M_jj = blas::matrix< value_t >( M, r_j, r_j );

                // copy to local matrix for faster computation
                auto  A    = blas::matrix< value_t >( 2*bsize, 2*bsize );
                auto  r_0  = blas::range( 0, bsize-1 );
                auto  r_1  = blas::range( bsize, 2*bsize-1 );
                auto  A_ii = blas::matrix< value_t >( A, r_0, r_0 );
                auto  A_ij = blas::matrix< value_t >( A, r_0, r_1 );
                auto  A_ji = blas::matrix< value_t >( A, r_1, r_0 );
                auto  A_jj = blas::matrix< value_t >( A, r_1, r_1 );

                blas::copy( M_ii, A_ii );
                blas::copy( M_ij, A_ij );
                blas::copy( M_ji, A_ji );
                blas::copy( M_jj, A_jj );

                auto  [ EA, VA ] = blas::eigen_herm( A );

                hpro::DBG::write( VA, "V.mat", "V" );
                
                // apply V back to global matrix
                auto  VA_ii = blas::matrix< value_t >( VA, r_0, r_0 );
                auto  VA_ij = blas::matrix< value_t >( VA, r_0, r_1 );
                auto  VA_ji = blas::matrix< value_t >( VA, r_1, r_0 );
                auto  VA_jj = blas::matrix< value_t >( VA, r_1, r_1 );

                for ( size_t  l = 0; l < nbcols; ++l )
                {
                    auto  r_l  = blas::range( l*bsize, (l+1)*bsize-1 );
                    auto  M_il = blas::matrix< value_t >( M, r_i, r_l );
                    auto  M_jl = blas::matrix< value_t >( M, r_j, r_l );
                    
                    auto  T_il = blas::prod( value_t(1), blas::adjoint(VA_ii), M_il );
                    auto  T_jl = blas::prod( value_t(1), blas::adjoint(VA_ij), M_il );

                    blas::prod( value_t(1), blas::adjoint(VA_ji), M_jl, value_t(1), T_il );
                    blas::prod( value_t(1), blas::adjoint(VA_jj), M_jl, value_t(1), T_jl );

                    blas::copy( T_il, M_il );
                    blas::copy( T_jl, M_jl );
                }// for

                for ( size_t  l = 0; l < nbrows; ++l )
                {
                    auto  r_l  = blas::range( l*bsize, (l+1)*bsize-1 );
                    auto  M_li = blas::matrix< value_t >( M, r_l, r_i );
                    auto  M_lj = blas::matrix< value_t >( M, r_l, r_j );
                    
                    auto  T_li = blas::prod( value_t(1), M_li, VA_ii );
                    auto  T_lj = blas::prod( value_t(1), M_li, VA_ij );

                    blas::prod( value_t(1), M_lj, VA_ji, value_t(1), T_li );
                    blas::prod( value_t(1), M_lj, VA_jj, value_t(1), T_lj );

                    blas::copy( T_li, M_li );
                    blas::copy( T_lj, M_lj );
                }// for

                hpro::DBG::write( M, "M.mat", "M" );
            }// for
        }// for
    }// while

    //
    // extract eigenvalues as diagonal elements of M
    //

    blas::vector< value_t >  E( minrc );
    
    for ( size_t  i = 0; i < minrc; i++ )
        E(i) = M(i,i);

    return { std::move( E ), std::move( V ) };
}
    
//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = double;

    {
        auto  R  = blas::random< value_t >( n, n );
        auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );

        auto [ E, V ] = eigen_jac_bw( M );

        return;
    }
    
    std::cout << term::bullet << term::bold << "dense DPT eigen iteration ( " << impl_name
              << " )" << term::reset << std::endl;

    blas::eigen_stat  stat;
        
    for ( size_t  n = 128; n <= 512; n += 128 )
    {
        std::mutex  mtx;
        uint        nsweeps_min = 0;
        uint        nsweeps_jac = 0;

        ::tbb::parallel_for( uint(0), uint(10),
                             [&,n] ( const uint )
                             // for ( uint  i = 0; i < 10; ++i )
                             {
                                 auto  R  = blas::random< value_t >( n, n );
                                 auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );
                                 auto  Mc = blas::copy( M );

                                 for ( uint nsweeps = 1; nsweeps < n; ++nsweeps )
                                 {
                                     auto  Ms         = blas::copy< float >( M );
                                     auto  [ Ej, Vj ] = blas::eigen_jac( Ms, nsweeps, 1e-7 );
                                         
                                     auto  Wj         = blas::copy< double >( Vj );
                                     auto  VM         = blas::prod( value_t(1), blas::adjoint( Wj ), M );
                                     auto  VMV        = blas::prod( value_t(1), VM, Wj );
                                     auto  [ Ed, Vd ] = blas::eigen_dpt( VMV, 0, 1e-8, "fro", 0, & stat );
                                         
                                     if ( stat.converged )
                                     {
                                         // converged
                                         std::scoped_lock  lock( mtx );
                                             
                                         nsweeps_min = std::max( nsweeps_min, nsweeps+1 );
                                         break;
                                     }// if
                                 }// for

                                 auto  [ E, V ] = blas::eigen_jac( Mc, 100, 1e-14, & stat );

                                 {
                                     std::scoped_lock  lock( mtx );
                                         
                                     nsweeps_jac = std::max( nsweeps_jac, stat.nsweeps );
                                 }
                             } );

        std::cout << "n = " << n << "   " << nsweeps_min << "    " << nsweeps_jac << std::endl;
    }// for
}
