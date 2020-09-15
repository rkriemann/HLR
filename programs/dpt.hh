//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <random>

#include <tbb/parallel_for.h>
#include <hlr/arith/blas_eigen.hh>

#include <hlr/utils/tensor.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_jac_bw ( blas::matrix< value_t > &                          M,
               const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
               const size_t                                       amax_sweeps = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         bsize      = 128;
    const auto         nbrows     = nrows / bsize;
    const auto         nbcols     = ncols / bsize;

    HLR_ASSERT( ( nrows / bsize ) * bsize == nrows );
    HLR_ASSERT( ( ncols / bsize ) * bsize == ncols );
    
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15*minrc*minrc );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(100) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;
    auto               V          = blas::matrix< value_t >( minrc, ncols );
    auto               norms      = tensor2< real_t >( nbrows, nbcols );

    // initialise V with identity
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    // hpro::DBG::write( M, "M0.mat", "M0" );

    //
    // determine initial norm of blocks in M
    //

    for ( size_t  i = 0; i < nbrows; i++ )
    {
        for ( size_t j = 0; j < nbcols; j++ )
        {
            const auto  r_i  = blas::range( i*bsize, (i+1)*bsize-1 );
            const auto  r_j  = blas::range( j*bsize, (j+1)*bsize-1 );
            auto        M_ij = blas::matrix< value_t >( M, r_i, r_j );

            norms(i,j) = blas::norm_F( M_ij );
        }// for
    }// for

    //
    // block wise Jacobi
    //
    
    while ( ! converged && ( sweep < max_sweeps ))
    {
        sweep++;

        //
        // look for off-diagonal, maximal norm block
        //

        uint    max_i    = 0;
        uint    max_j    = 1;
        real_t  max_norm = norms( max_i, max_j );
        
        for ( size_t  i = 0; i < nbrows-1; i++ )
        {
            for ( size_t j = i + 1; j < nbcols; j++ )
            {
                if ( i == j )
                    continue;
                
                if ( norms(i,j) > max_norm )
                {
                    max_i    = i;
                    max_j    = j;
                    max_norm = norms(i,j);
                }// if
            }// for
        }// for

        if ( ( norms( max_i, max_j ) / std::sqrt( norms( max_i, max_i ) * norms( max_j, max_j ) ) ) < tolerance )
            break;

        std::cout << "sweep " << sweep-1 << " : " << format_error( norms( max_i, max_j ) / std::sqrt( norms( max_i, max_i ) * norms( max_j, max_j ) ) ) << std::endl;

        //
        // sort norms and determine index pairs with decreasing norm for
        // parallel handling
        //

        std::list< std::tuple< real_t, uint, uint > >  norm_idxs;

        for ( uint  i = 0; i < nbrows-1; i++ )
        {
            for ( uint j = i + 1; j < nbcols; j++ )
            {
                if ( i == j )
                    continue;

                norm_idxs.push_back( { norms(i,j), i, j } ); 
            }// for
        }// for

        norm_idxs.sort( [] ( const auto &  n1, const auto &  n2 )
                        {
                            return std::get<0>( n1 ) > std::get<0>( n2 ); // reverse order!
                        } );

        std::deque< std::pair< uint, uint > >  idx_pairs;

        if ( true )
        {
            while ( ! norm_idxs.empty() )
            {
                auto  first = norm_idxs.front();

                norm_idxs.pop_front();

                const auto  i = std::get<1>( first );
                const auto  j = std::get<2>( first );
            
                idx_pairs.push_back( { i, j } );

                // remove all other entries containing i/j
                for ( auto  it = norm_idxs.begin(); it != norm_idxs.end(); )
                {
                    auto  ti = std::get<1>( *it );
                    auto  tj = std::get<2>( *it );
                
                    if (( ti == i ) || ( tj == i ) || ( ti == j ) || ( tj == j ))
                        it = norm_idxs.erase( it );
                    else
                        ++it;
                }// for
            }// while
        }// if
        else
            idx_pairs.push_back( { max_i, max_j } );
        
        // for ( auto [ i, j ] : idx_pairs )
        //     std::cout << i << " / " << j << " : " << norms(i,j) << std::endl;
        
        //
        // diagonalize ⎧ M_ii  M_ij ⎫, e.g.,
        //             ⎩ M_ji  M_jj ⎭
        //
        // compute
        //
        // ⎧ V_ii  V_ij ⎫' ⎧ M_ii  M_ij ⎫ ⎧ V_ii  V_ij ⎫ = ⎧ D_ii    0  ⎫
        // ⎩ V_ji  V_jj ⎭  ⎩ M_ji  M_jj ⎭ ⎩ V_ji  V_jj ⎭   ⎩   0   D_jj ⎭
        //

        const auto                              npairs = idx_pairs.size();
        std::vector< blas::matrix< value_t > >  Vs( npairs );
        
        // for ( size_t  idx = 0; idx < npairs; ++idx )
        ::tbb::parallel_for< size_t >(
            0, npairs,
            [&,bsize] ( const auto  idx )
            {
                const auto  [ i, j ] = idx_pairs[ idx ];
                const auto  r_i  = blas::range( i*bsize, (i+1)*bsize-1 );
                const auto  r_j  = blas::range( j*bsize, (j+1)*bsize-1 );
                auto        M_ii = blas::matrix< value_t >( M, r_i, r_i );
                auto        M_ij = blas::matrix< value_t >( M, r_i, r_j );
                auto        M_ji = blas::matrix< value_t >( M, r_j, r_i );
                auto        M_jj = blas::matrix< value_t >( M, r_j, r_j );

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

                Vs[ idx ] = std::move( VA );
            } );

        //
        // apply local V's back to global matrix M (and global V)
        //
        
        // for ( size_t  idx = 0; idx < npairs; ++idx )
        ::tbb::parallel_for< size_t >(
            0, npairs,
            [&,bsize] ( const auto  idx )
            {
                const auto  [ i, j ] = idx_pairs[ idx ];
                const auto  VA       = Vs[ idx ];
                const auto  r_i      = blas::range( i*bsize, (i+1)*bsize-1 );
                const auto  r_j      = blas::range( j*bsize, (j+1)*bsize-1 );
                const auto  r_0      = blas::range( 0, bsize-1 );
                const auto  r_1      = blas::range( bsize, 2*bsize-1 );
                const auto  VA_ii    = blas::matrix< value_t >( VA, r_0, r_0 );
                const auto  VA_ij    = blas::matrix< value_t >( VA, r_0, r_1 );
                const auto  VA_ji    = blas::matrix< value_t >( VA, r_1, r_0 );
                const auto  VA_jj    = blas::matrix< value_t >( VA, r_1, r_1 );
                
                // for ( size_t  l = 0; l < nbcols; ++l )
                ::tbb::parallel_for< size_t >(
                    0, nbcols,
                    [&,i,j,bsize] ( const size_t  l )
                    {
                        auto  r_l  = blas::range( l*bsize, (l+1)*bsize-1 );
                        auto  M_il = blas::matrix< value_t >( M, r_i, r_l );
                        auto  M_jl = blas::matrix< value_t >( M, r_j, r_l );
                        auto  C_il = blas::copy( M_il );
                        auto  C_jl = blas::copy( M_jl );
                        
                        auto  T_il = blas::prod( value_t(1), blas::adjoint(VA_ii), C_il );
                        auto  T_jl = blas::prod( value_t(1), blas::adjoint(VA_ij), C_il );
                        
                        blas::prod( value_t(1), blas::adjoint(VA_ji), C_jl, value_t(1), T_il );
                        blas::prod( value_t(1), blas::adjoint(VA_jj), C_jl, value_t(1), T_jl );
                        
                        blas::copy( T_il, M_il );
                        blas::copy( T_jl, M_jl );
                    } );
            } );

        // for ( size_t  idx = 0; idx < npairs; ++idx )
        ::tbb::parallel_for< size_t >(
            0, npairs,
            [&,bsize] ( const auto  idx )
            {
                const auto  [ i, j ] = idx_pairs[ idx ];
                const auto  VA       = Vs[ idx ];
                const auto  r_i      = blas::range( i*bsize, (i+1)*bsize-1 );
                const auto  r_j      = blas::range( j*bsize, (j+1)*bsize-1 );
                const auto  r_0      = blas::range( 0, bsize-1 );
                const auto  r_1      = blas::range( bsize, 2*bsize-1 );
                const auto  VA_ii    = blas::matrix< value_t >( VA, r_0, r_0 );
                const auto  VA_ij    = blas::matrix< value_t >( VA, r_0, r_1 );
                const auto  VA_ji    = blas::matrix< value_t >( VA, r_1, r_0 );
                const auto  VA_jj    = blas::matrix< value_t >( VA, r_1, r_1 );
                
                // for ( size_t  l = 0; l < nbrows; ++l )
                ::tbb::parallel_for< size_t >(
                    0, nbrows,
                    [&,i,j,bsize] ( const size_t  l )
                    {
                        auto  r_l  = blas::range( l*bsize, (l+1)*bsize-1 );

                        // apply to M
                        auto  M_li = blas::matrix< value_t >( M, r_l, r_i );
                        auto  C_li = blas::copy( M_li );
                        auto  M_lj = blas::matrix< value_t >( M, r_l, r_j );
                        auto  C_lj = blas::copy( M_lj );
                        
                        auto  T_li = blas::prod( value_t(1), C_li, VA_ii );
                        auto  T_lj = blas::prod( value_t(1), C_li, VA_ij );
                        
                        blas::prod( value_t(1), C_lj, VA_ji, value_t(1), T_li );
                        blas::prod( value_t(1), C_lj, VA_jj, value_t(1), T_lj );
                        
                        norms(l,i) = blas::norm_F( C_li );
                        norms(l,j) = blas::norm_F( C_lj );
                        
                        blas::copy( T_li, M_li );
                        blas::copy( T_lj, M_lj );

                        // apply to V
                        auto  V_li = blas::matrix< value_t >( V, r_l, r_i );
                        auto  V_lj = blas::matrix< value_t >( V, r_l, r_j );
                        
                        auto  S_li = blas::prod( value_t(1), V_li, VA_ii );
                        auto  S_lj = blas::prod( value_t(1), V_li, VA_ij );
                        
                        blas::prod( value_t(1), V_lj, VA_ji, value_t(1), S_li );
                        blas::prod( value_t(1), V_lj, VA_jj, value_t(1), S_lj );
                        
                        blas::copy( S_li, V_li );
                        blas::copy( S_lj, V_lj );
                    } );
            } );
    }// while

    std::cout << sweep << std::endl;
    
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
        const auto                                 seed = 1593694284; // time( nullptr );
        std::default_random_engine                 generator( seed );
        std::uniform_real_distribution< double >   uniform_distr( -1.0, 1.0 );
        auto                                       random      = [&] () { return uniform_distr( generator ); };
        
        auto  R  = blas::matrix< value_t >( n, n );

        blas::fill_fn( R, random );
        
        auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );

        hpro::DBG::write( M, "M.mat", "M" );

        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            
            auto [ E1, V1 ] = eigen_jac_bw( M2, 0.000001 );
            
            std::cout << "done in " << format_time( timer::since( tic ) ) << std::endl;
            
            auto [ E2, V2 ] = blas::eigen_dpt( M2, V1, 0, 1e-14, "frobenius", 3 );
            
            auto toc      = timer::since( tic );

            std::cout << "done in " << format_time( toc ) << std::endl;
            hpro::DBG::write( V1, "V0.mat", "V0" );
            hpro::DBG::write( E1, "E0.mat", "E0" );
            hpro::DBG::write( V2, "V1.mat", "V1" );
            hpro::DBG::write( E2, "E1.mat", "E1" );
        }

        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            auto [ E, V ] = blas::eigen_herm( M2 );
            auto toc      = timer::since( tic );

            std::cout << "done in " << format_time( toc ) << std::endl;
            hpro::DBG::write( V, "V2.mat", "V2" );
            hpro::DBG::write( E, "E2.mat", "E2" );
        }

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
