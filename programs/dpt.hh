//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <mkl_service.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/blocked_range2d.h>

#include <hlr/arith/blas_eigen.hh>
#include <hlr/arith/cuda.hh>
#include <hlr/utils/tensor.hh>
#include <hlr/utils/io.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_bjac ( blas::matrix< value_t > &                          M,
             const size_t                                       block_size  = 128,
             const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
             const size_t                                       amax_sweeps = 0,
             const uint                                         verbosity   = 0,
             blas::eigen_stat *                                 stat        = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         nbrows     = nrows / block_size;
    const auto         nbcols     = ncols / block_size;

    HLR_ASSERT( ( nrows / block_size ) * block_size == nrows );
    HLR_ASSERT( ( ncols / block_size ) * block_size == ncols );
    
    const auto    minrc      = std::min( nrows, ncols );
    const size_t  max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15*minrc*minrc );
    const real_t  tolerance  = ( atolerance > 0 ? atolerance : real_t(100) * std::numeric_limits< real_t >::epsilon() );
    bool          converged  = false;
    uint          sweep      = 0;
    auto          V          = blas::matrix< value_t >( minrc, ncols );
    auto          norms      = blas::matrix< real_t >( nbrows, nbcols );

    // temporary workspace per block row/column
    auto  C0 = std::vector< blas::matrix< value_t > >( std::max( nbrows, nbcols ) );
    auto  C1 = std::vector< blas::matrix< value_t > >( std::max( nbrows, nbcols ) );
    auto  T0 = std::vector< blas::matrix< value_t > >( std::max( nbrows, nbcols ) );
    auto  T1 = std::vector< blas::matrix< value_t > >( std::max( nbrows, nbcols ) );
    auto  ap = ::tbb::affinity_partitioner();
    
    ::tbb::parallel_for(
        ::tbb::blocked_range< size_t >( 0, std::max( nbrows, nbcols ) ),
        [&,block_size] ( const auto  r )
        {
            for ( auto  i = r.begin(); i != r.end(); ++i )
            {
                C0 = std::move( blas::matrix< value_t >( block_size, block_size ) );
                C1 = std::move( blas::matrix< value_t >( block_size, block_size ) );
                T0 = std::move( blas::matrix< value_t >( block_size, block_size ) );
                T1 = std::move( blas::matrix< value_t >( block_size, block_size ) );
            }// for
        },
        ap );
            
    // initialise V with identity
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    //
    // determine initial norm of blocks in M
    //

    ::tbb::parallel_for(
        ::tbb::blocked_range2d< size_t >( 0, nbrows,
                                          0, nbcols ),
        [&] ( const auto  r )
        {
            for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
            {
                for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                {
                    const auto  r_i  = blas::range( i*block_size, (i+1)*block_size-1 );
                    const auto  r_j  = blas::range( j*block_size, (j+1)*block_size-1 );
                    auto        M_ij = blas::matrix< value_t >( M, r_i, r_j );
                    
                    norms(i,j) = blas::norm_F( M_ij );
                }// for
            }// for
        } );

    //
    // block wise Jacobi
    //

    real_t  norm_off = real_t(0);

    if ( verbosity > 0 )
        std::cout << "sweep       off        red        error        time " << std::endl;
    
    while ( ! converged && ( sweep < max_sweeps ))
    {
        sweep++;

        auto  tic = timer::now();
        
        //
        // sort norm of blocks
        //

        auto    norm_idxs = std::list< std::tuple< real_t, uint, uint > >();
        real_t  norm_sum  = real_t(0);

        for ( uint  i = 0; i < nbrows-1; i++ )
        {
            for ( uint  j = i + 1; j < nbcols; j++ )
            {
                norm_sum += 2 * math::square( norms(i,j) );
                norm_idxs.push_back( { norms(i,j), i, j } ); 
            }// for
        }// for

        norm_sum = math::sqrt( norm_sum );
        
        norm_idxs.sort( [] ( const auto &  n1, const auto &  n2 )
                        {
                            // reverse order for big to small!
                            return std::get<0>( n1 ) > std::get<0>( n2 );
                        } );

        if ( verbosity > 2 )
        {
            for ( auto [ n, i, j ] : norm_idxs )
                std::cout << i << " / " << j << " = " << n << std::endl;
        }// if
        
        //
        // check norm of off-diagonal part and stop if threshold was reached
        //
        
        const auto  max_norm = std::get<0>( norm_idxs.front() );
        const auto  max_i    = std::get<1>( norm_idxs.front() );
        const auto  max_j    = std::get<2>( norm_idxs.front() );
        const auto  error    = max_norm / std::sqrt( norms( max_i, max_i ) * norms( max_j, max_j ) );

        if ( verbosity > 0 )
            std::cout << boost::format( " %4d" ) % (sweep-1) << "  "
                      << format_norm( norm_sum ) << "  "
                      << format_norm( sweep > 1 ? norm_sum / norm_off : real_t(0) ) << "  "
                      << format_error( error ) << std::flush;

        norm_off = norm_sum;
        
        if ( error < tolerance )
            break;

        //
        // set up block index pairs by successively choosing maximal norm
        // and removing all pairs having same indices
        //
        
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
        
        //
        // diagonalize ⎧ M_ii  M_ij ⎫, e.g.,
        //             ⎩ M_ji  M_jj ⎭
        //
        // compute
        //
        // ⎧ V_ii  V_ij ⎫' ⎧ M_ii  M_ij ⎫ ⎧ V_ii  V_ij ⎫ = ⎧ D_ii    0  ⎫
        // ⎩ V_ji  V_jj ⎭  ⎩ M_ji  M_jj ⎭ ⎩ V_ji  V_jj ⎭   ⎩   0   D_jj ⎭
        //

      
        const auto  npairs = idx_pairs.size();
        auto        Vs     = std::vector< blas::matrix< value_t > >( npairs );
        
        // for ( size_t  idx = 0; idx < npairs; ++idx )
        ::tbb::parallel_for< size_t >(
            0, npairs,
            [&,block_size] ( const auto  idx )
            {
                const auto  [ i, j ] = idx_pairs[ idx ];
                const auto  r_i  = blas::range( i*block_size, (i+1)*block_size-1 );
                const auto  r_j  = blas::range( j*block_size, (j+1)*block_size-1 );
                auto        M_ii = blas::matrix< value_t >( M, r_i, r_i );
                auto        M_ij = blas::matrix< value_t >( M, r_i, r_j );
                auto        M_ji = blas::matrix< value_t >( M, r_j, r_i );
                auto        M_jj = blas::matrix< value_t >( M, r_j, r_j );

                if ( verbosity > 1 )
                    std::cout << i << " / " << j << std::endl;
                
                // copy to local matrix for faster computation
                auto        A    = blas::matrix< value_t >( 2*block_size, 2*block_size );
                const auto  r_0  = blas::range( 0, block_size-1 );
                const auto  r_1  = blas::range( block_size, 2*block_size-1 );
                auto        A_ii = blas::matrix< value_t >( A, r_0, r_0 );
                auto        A_ij = blas::matrix< value_t >( A, r_0, r_1 );
                auto        A_ji = blas::matrix< value_t >( A, r_1, r_0 );
                auto        A_jj = blas::matrix< value_t >( A, r_1, r_1 );

                blas::copy( M_ii, A_ii );
                blas::copy( M_ij, A_ij );
                blas::copy( M_ji, A_ji );
                blas::copy( M_jj, A_jj );

                auto  [ EA, VA ] = blas::eigen_herm( A );

                //
                // and apply to M and V
                //
                
                const auto  VA_ii = blas::matrix< value_t >( VA, r_0, r_0 );
                const auto  VA_ij = blas::matrix< value_t >( VA, r_0, r_1 );
                const auto  VA_ji = blas::matrix< value_t >( VA, r_1, r_0 );
                const auto  VA_jj = blas::matrix< value_t >( VA, r_1, r_1 );
                
                // for ( size_t  l = 0; l < nbcols; ++l )
                ::tbb::parallel_for(
                    ::tbb::blocked_range< size_t >( 0, nbcols ),
                    [&,i,j,block_size] ( const auto  r )
                    {
                        for ( auto  l = r.begin(); l != r.end(); ++l )
                        {
                            //
                            // ⎛A₀₀ A₀₁⎞' ⎛M_il⎞
                            // ⎝A₁₀ A₁₁⎠  ⎝M_jl⎠
                            //

                            const auto  r_l  = blas::range( l*block_size, (l+1)*block_size-1 );
                            auto        M_il = blas::matrix< value_t >( M, r_i, r_l );
                            auto        M_jl = blas::matrix< value_t >( M, r_j, r_l );

                            blas::copy( M_il, C0[l] );
                            blas::copy( M_jl, C1[l] );
                        
                            blas::prod( value_t(1), blas::adjoint(VA_ii), C0[l], value_t(0), T0[l] );
                            blas::prod( value_t(1), blas::adjoint(VA_ij), C0[l], value_t(0), T1[l] );
                        
                            blas::prod( value_t(1), blas::adjoint(VA_ji), C1[l], value_t(1), T0[l] );
                            blas::prod( value_t(1), blas::adjoint(VA_jj), C1[l], value_t(1), T1[l] );
                        
                            norms(i,l) = blas::norm_F( T0[l] );
                            norms(j,l) = blas::norm_F( T1[l] );
                        
                            blas::copy( T0[l], M_il );
                            blas::copy( T1[l], M_jl );
                        }// for
                    },
                    ap );

                // save for updates below
                Vs[ idx ] = std::move( VA );
            } );

        // for ( size_t  idx = 0; idx < npairs; ++idx )
        ::tbb::parallel_for< size_t >(
            0, npairs,
            [&,block_size] ( const auto  idx )
            {
                const auto  [ i, j ] = idx_pairs[ idx ];
                const auto  r_i      = blas::range( i*block_size, (i+1)*block_size-1 );
                const auto  r_j      = blas::range( j*block_size, (j+1)*block_size-1 );

                const auto  r_0      = blas::range( 0, block_size-1 );
                const auto  r_1      = blas::range( block_size, 2*block_size-1 );
                const auto  VA       = std::move( Vs[ idx ] );
                const auto  VA_ii    = blas::matrix< value_t >( VA, r_0, r_0 );
                const auto  VA_ij    = blas::matrix< value_t >( VA, r_0, r_1 );
                const auto  VA_ji    = blas::matrix< value_t >( VA, r_1, r_0 );
                const auto  VA_jj    = blas::matrix< value_t >( VA, r_1, r_1 );
                
                // for ( size_t  l = 0; l < nbrows; ++l )
                ::tbb::parallel_for< size_t >(
                    0, nbrows,
                    [&,i,j,block_size] ( const size_t  l )
                    {
                        const auto  r_l  = blas::range( l*block_size, (l+1)*block_size-1 );

                        //
                        // (M_li M_lj) ⎛A₀₀ A₀₁⎞
                        //             ⎝A₁₀ A₁₁⎠   
                        //

                        auto        M_li = blas::matrix< value_t >( M, r_l, r_i );
                        auto        M_lj = blas::matrix< value_t >( M, r_l, r_j );

                        const auto  C_li = blas::copy( M_li );
                        const auto  C_lj = blas::copy( M_lj );
                        
                        auto        T_li = blas::prod( value_t(1), C_li, VA_ii );
                        auto        T_lj = blas::prod( value_t(1), C_li, VA_ij );
                        
                        blas::prod( value_t(1), C_lj, VA_ji, value_t(1), T_li );
                        blas::prod( value_t(1), C_lj, VA_jj, value_t(1), T_lj );
                        
                        norms(l,i) = blas::norm_F( T_li );
                        norms(l,j) = blas::norm_F( T_lj );
                        
                        blas::copy( T_li, M_li );
                        blas::copy( T_lj, M_lj );

                        //
                        // (V_li V_lj) ⎛A₀₀, A₀₁⎞
                        //             ⎝A₁₀, A₁₁⎠   
                        //
                        
                        auto  V_li = blas::matrix< value_t >( V, r_l, r_i );
                        auto  V_lj = blas::matrix< value_t >( V, r_l, r_j );

                        C_li = blas::copy( V_li, C_li );
                        D_lj = blas::copy( V_lj, C_li );
                        
                        blas::prod( value_t(1), D_li, VA_ii, value_t(0), T_li );
                        blas::prod( value_t(1), D_li, VA_ij, value_t(0), T_lj );
                        
                        blas::prod( value_t(1), D_lj, VA_ji, value_t(1), S_li );
                        blas::prod( value_t(1), D_lj, VA_jj, value_t(1), S_lj );
                        
                        blas::copy( S_li, V_li );
                        blas::copy( S_lj, V_lj );
                    } );
            } );
        
        auto  toc = timer::since( tic );
        
        if ( verbosity > 0 )
            std::cout << "  " << format_time( toc ) << std::endl;
    }// while

    if ( verbosity > 0 )
        std::cout << std::endl;
    
    if ( ! is_null( stat ) )
    {
        stat->nsweeps   = sweep;
        stat->converged = converged;
    }// if
    
    //
    // extract eigenvalues as diagonal elements of M
    //

    blas::vector< value_t >  E( minrc );
    
    for ( size_t  i = 0; i < minrc; i++ )
        E(i) = M(i,i);
    
    return { std::move( E ), std::move( V ) };
}

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_bjac2 ( blas::matrix< value_t > &                          aM,
              const size_t                                       block_size  = 128,
              const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
              const size_t                                       amax_sweeps = 0,
              const uint                                         verbosity   = 0,
              blas::eigen_stat *                                 stat        = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const auto         nrows      = aM.nrows();
    const auto         ncols      = aM.ncols();
    const auto         nbrows     = nrows / block_size;
    const auto         nbcols     = ncols / block_size;

    HLR_ASSERT( ( nrows / block_size ) * block_size == nrows );
    HLR_ASSERT( ( ncols / block_size ) * block_size == ncols );
    
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15*minrc*minrc );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(100) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;

    //
    // represent M, V blockwise
    // - also determine initial norm of blocks in M
    //

    auto  M     = tensor2< blas::matrix< value_t > >( nbrows, nbcols );
    auto  V     = tensor2< blas::matrix< value_t > >( nbrows, nbcols );
    auto  norms = blas::matrix< real_t >( nbrows, nbcols );

    for ( size_t  i = 0; i < nbrows; ++i )
    {
        for ( size_t  j = 0; j < nbcols; ++j )
        {
            const auto  r_i  = blas::range( i*block_size, (i+1)*block_size-1 );
            const auto  r_j  = blas::range( j*block_size, (j+1)*block_size-1 );
            auto        M_ij = blas::matrix< value_t >( aM, r_i, r_j );

            M(i,j) = std::move( blas::matrix< value_t >( block_size, block_size ) );
            V(i,j) = std::move( blas::matrix< value_t >( block_size, block_size ) );
            
            copy( M_ij, M(i,j) );
            norms(i,j) = blas::norm_F( M(i,j) );
            
            if ( i == j )
            {
                for ( size_t  k = 0; k < block_size; ++k )
                    V(i,j)(k,k) = value_t(1);
            }// if
        }// for
    }// for
    
    //
    // block wise Jacobi
    //

    real_t  norm_off = real_t(0);

    if ( verbosity > 0 )
        std::cout << "sweep       off        red        error        time " << std::endl;
    
    while ( ! converged && ( sweep < max_sweeps ))
    {
        sweep++;

        auto  tic = timer::now();
        
        //
        // sort norm of blocks
        //

        auto    norm_idxs = std::list< std::tuple< real_t, uint, uint > >();
        real_t  norm_sum  = real_t(0);

        for ( uint  i = 0; i < nbrows-1; i++ )
        {
            for ( uint  j = i + 1; j < nbcols; j++ )
            {
                norm_sum += 2 * math::square( norms(i,j) );
                norm_idxs.push_back( { norms(i,j), i, j } ); 
            }// for
        }// for

        norm_sum = math::sqrt( norm_sum );
        
        norm_idxs.sort( [] ( const auto &  n1, const auto &  n2 )
                        {
                            // reverse order for big to small!
                            return std::get<0>( n1 ) > std::get<0>( n2 );
                        } );

        if ( verbosity > 2 )
        {
            for ( auto [ n, i, j ] : norm_idxs )
                std::cout << i << " / " << j << " = " << n << std::endl;
        }// if
        
        //
        // check norm of off-diagonal part and stop if threshold was reached
        //
        
        const auto  max_norm = std::get<0>( norm_idxs.front() );
        const auto  max_i    = std::get<1>( norm_idxs.front() );
        const auto  max_j    = std::get<2>( norm_idxs.front() );
        const auto  error    = max_norm / std::sqrt( norms( max_i, max_i ) * norms( max_j, max_j ) );

        if ( verbosity > 0 )
            std::cout << boost::format( " %4d" ) % (sweep-1) << "  "
                      << format_norm( norm_sum ) << "  "
                      << format_norm( sweep > 1 ? norm_sum / norm_off : real_t(0) ) << "  "
                      << format_error( error ) << std::flush;

        norm_off = norm_sum;
        
        if ( error < tolerance )
            break;

        //
        // set up block index pairs by successively choosing maximal norm
        // and removing all pairs having same indices
        //
        
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
        
        //
        // diagonalize ⎧ M_ii  M_ij ⎫, e.g.,
        //             ⎩ M_ji  M_jj ⎭
        //
        // compute
        //
        // ⎧ V_ii  V_ij ⎫' ⎧ M_ii  M_ij ⎫ ⎧ V_ii  V_ij ⎫ = ⎧ D_ii    0  ⎫
        // ⎩ V_ji  V_jj ⎭  ⎩ M_ji  M_jj ⎭ ⎩ V_ji  V_jj ⎭   ⎩   0   D_jj ⎭
        //

      
        const auto  npairs = idx_pairs.size();
        auto        Vs     = std::vector< blas::matrix< value_t > >( npairs );

        for ( size_t  idx = 0; idx < npairs; ++idx )
        {
            const auto  [ i, j ] = idx_pairs[ idx ];

            if ( verbosity > 1 )
                std::cout << i << " / " << j << std::endl;
                
            //
            // A = [ M_ii, M_ij ; M_ji, M_jj ]
            //
                
            auto        A    = blas::matrix< value_t >( 2*block_size, 2*block_size );
            const auto  r_0  = blas::range( 0, block_size-1 );
            const auto  r_1  = blas::range( block_size, 2*block_size-1 );
            auto        A_00 = blas::matrix< value_t >( A, r_0, r_0 );
            auto        A_01 = blas::matrix< value_t >( A, r_0, r_1 );
            auto        A_10 = blas::matrix< value_t >( A, r_1, r_0 );
            auto        A_11 = blas::matrix< value_t >( A, r_1, r_1 );

            blas::copy( M(i,i), A_00 );
            blas::copy( M(i,j), A_01 );
            blas::copy( M(j,i), A_10 );
            blas::copy( M(j,j), A_11 );

            //
            // compute eigenvalues of sub-system
            //

            auto  EA = blas::vector< real_t >( 2*block_size );
                
            blas::eigen_herm( A, EA );

            //
            // and apply to M and V
            //
                
            for ( size_t  l = 0; l < nbcols; ++l )
            {
                //
                // ⎛A₀₀ A₀₁⎞' ⎛M_il⎞
                // ⎝A₁₀ A₁₁⎠  ⎝M_jl⎠
                //

                const auto  T0 = blas::copy( M(i,l) );
                const auto  T1 = blas::copy( M(j,l) );
                    
                ::tbb::parallel_invoke(
                    [&,i,j,l] ()
                    { 
                        blas::prod( value_t(1), blas::adjoint(A_00), T0, value_t(0), M(i,l) );
                        blas::prod( value_t(1), blas::adjoint(A_10), T1, value_t(1), M(i,l) );

                        norms(i,l) = blas::norm_F( M(i,l) );
                    },
                    
                    [&,i,j,l] ()
                    { 
                        blas::prod( value_t(1), blas::adjoint(A_01), T0, value_t(0), M(j,l) );
                        blas::prod( value_t(1), blas::adjoint(A_11), T1, value_t(1), M(j,l) );
                        
                        norms(j,l) = blas::norm_F( M(j,l) );
                    } );
            }// for

            for ( size_t  l = 0; l < nbrows; ++l )
            {
                //
                // (M_li M_lj) ⎛A₀₀ A₀₁⎞
                //             ⎝A₁₀ A₁₁⎠   
                //

                ::tbb::parallel_invoke(
                    [&,i,j,l] ()
                    {
                        const auto  T0 = blas::copy( M(l,i) );
                        const auto  T1 = blas::copy( M(l,j) );
                    
                        ::tbb::parallel_invoke(
                            [&,i,j,l] ()
                            { 
                                blas::prod( value_t(1), T0, A_00, value_t(0), M(l,i) );
                                blas::prod( value_t(1), T1, A_10, value_t(1), M(l,i) );

                                norms(l,i) = blas::norm_F( M(l,i) );
                            },
                        
                            [&,i,j,l] ()
                            { 
                                blas::prod( value_t(1), T0, A_01, value_t(0), M(l,j) );
                                blas::prod( value_t(1), T1, A_11, value_t(1), M(l,j) );

                                norms(l,j) = blas::norm_F( M(l,j) );
                            } );
                    },
                    
                    //
                    // (V_li V_lj) ⎛A₀₀, A₀₁⎞
                    //             ⎝A₁₀, A₁₁⎠   
                    //
                    
                    [&,i,j,l] ()
                    {
                        const auto  T0 = blas::copy( V(l,i) );
                        const auto  T1 = blas::copy( V(l,j) );

                        ::tbb::parallel_invoke(
                            [&,i,j,l] ()
                            { 
                                blas::prod( value_t(1), T0, A_00, value_t(0), V(l,i) );
                                blas::prod( value_t(1), T1, A_10, value_t(1), V(l,i) );
                            },
                            
                            [&,i,j,l] ()
                            {
                                blas::prod( value_t(1), T0, A_01, value_t(0), V(l,j) );
                                blas::prod( value_t(1), T1, A_11, value_t(1), V(l,j) );
                            } );
                    } );
            }// for
        }// for
        
        auto  toc = timer::since( tic );
        
        if ( verbosity > 0 )
            std::cout << "  " << format_time( toc ) << std::endl;
    }// while

    if ( verbosity > 0 )
        std::cout << std::endl;
    
    if ( ! is_null( stat ) )
    {
        stat->nsweeps   = sweep;
        stat->converged = converged;
    }// if
    
    //
    // extract eigenvalues as diagonal elements of M
    //

    blas::vector< value_t >  rE( minrc );
    blas::matrix< value_t >  rV( nrows, ncols );
    
    for ( size_t  bi = 0; bi < nbrows; bi++ )
    {
        for ( size_t  i = 0; i < block_size; i++ )
            rE( bi * block_size + i ) = M(bi,bi)(i,i);
    }// for
    
    for ( size_t  i = 0; i < nbrows; ++i )
    {
        for ( size_t  j = 0; j < nbcols; ++j )
        {
            const auto  r_i  = blas::range( i*block_size, (i+1)*block_size-1 );
            const auto  r_j  = blas::range( j*block_size, (j+1)*block_size-1 );
            auto        V_ij = blas::matrix< value_t >( rV, r_i, r_j );

            copy( V(i,j), V_ij );
        }// for
    }// for

    return { std::move( rE ), std::move( rV ) };
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = double;

    if ( true )
    {
        const auto  seed = 1593694284; // time( nullptr );
        auto        M    = blas::matrix< value_t >();

        if ( cmdline::appl == "random" )
            M = std::move( blas::random_herm< value_t >( n, seed ) );
        else if ( cmdline::appl == "randcond" )
            M = std::move( blas::random_cond< value_t >( n, 1e10, seed ) );
        else if ( cmdline::appl == "randprob" )
            M = std::move( blas::random_probability< value_t >( n, 0.01, seed ) );
        else
            HLR_ERROR( "unsupported matrix content : " + cmdline::appl );
            
        if ( n <= 2048 )
            io::matlab::write( M, "M" );

        // M = std::move( // io::read_matlab< value_t >( "M.mat" ) );
        
        // {
        //     auto M2       = blas::copy( M );
        //     auto tic      = timer::now();
            
        //     auto [ E1, V1 ] = eigen_bjac( M2, cmdline::ntile, 1e-5, 100, cmdline::verbosity );
            
        //     std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;
        // }
        
        // {
        //     auto M2       = blas::copy< float >( M );
        //     auto tic      = timer::now();
            
        //     auto [ E1, V1 ] = eigen_bjac( M2, cmdline::ntile, 1e-5, 100, cmdline::verbosity );
            
        //     std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;
        // }

        // return;

        //
        // GPU
        //

        if ( true )
        {
            blas::cuda::init();

            // {
            //     auto  A1 = blas::random< value_t >( 16, 16 );
            //     auto  A2 = blas::random< value_t >( 16, 16 );
            //     auto  B1 = blas::random< value_t >( 16, 16 );
            //     auto  B2 = blas::random< value_t >( 16, 16 );
            //     auto  C1 = blas::random< value_t >( 16, 16 );
            //     auto  C2 = blas::random< value_t >( 16, 16 );

            //     io::matlab::write( A1, "A1" );
            //     io::matlab::write( A2, "A2" );
            //     io::matlab::write( B1, "B1" );
            //     io::matlab::write( B2, "B2" );
            //     io::matlab::write( C1, "C1" );
            //     io::matlab::write( C2, "C2" );
                
            //     auto  dev_A1 = blas::cuda::device_alloc< value_t >( 16*16 );
            //     auto  dev_A2 = blas::cuda::device_alloc< value_t >( 16*16 );
            //     auto  dev_B1 = blas::cuda::device_alloc< value_t >( 16*16 );
            //     auto  dev_B2 = blas::cuda::device_alloc< value_t >( 16*16 );
            //     auto  dev_C1 = blas::cuda::device_alloc< value_t >( 16*16 );
            //     auto  dev_C2 = blas::cuda::device_alloc< value_t >( 16*16 );

            //     blas::cuda::to_device( A1, dev_A1, 16 );
            //     blas::cuda::to_device( A2, dev_A2, 16 );
            //     blas::cuda::to_device( B1, dev_B1, 16 );
            //     blas::cuda::to_device( B2, dev_B2, 16 );
            //     blas::cuda::to_device( C1, dev_C1, 16 );
            //     blas::cuda::to_device( C2, dev_C2, 16 );

            //     value_t *  A_ptr[2] = { dev_A1, dev_A2 };
            //     value_t *  B_ptr[2] = { dev_B1, dev_B2 };
            //     value_t *  C_ptr[2] = { dev_C1, dev_C2 };

            //     auto  one   = blas::cuda::make_constant< value_t >( 1 );
            //     auto  zero  = blas::cuda::make_constant< value_t >( 0 );

            //     blas::cuda::prod_batched( blas::cuda::default_handle, CUBLAS_OP_C, CUBLAS_OP_N,
            //                               16, 16, 16, one, A_ptr, 16, B_ptr, 16, zero, C_ptr, 16, 2 );

            //     blas::cuda::from_device( dev_C1, 16, A1 );
            //     blas::cuda::from_device( dev_C2, 16, A2 );

            //     io::matlab::write( A1, "D1" );
            //     io::matlab::write( A2, "D2" );

            //     return;
            // }

            
            std::cout << term::bullet << term::bold << "CUDA eigenvalue algorithms" << term::reset << std::endl;
            
            {
                std::cout << term::bold << "  SYEVJ" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_jac( blas::cuda::default_handle, M2, 1e-14, 1000, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }

            {
                std::cout << term::bold << "  block-wise Jacobi" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;

                io::matlab::write( M, "M" );
                
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_bjac( blas::cuda::default_handle, M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;

                io::matlab::write( E, "E" );
                io::matlab::write( V, "V" );
            }

            {
                std::cout << term::bold << "  batched block-wise Jacobi" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;

                io::matlab::write( M, "M" );
                
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_bjac_batched( blas::cuda::default_handle, M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;

                io::matlab::write( E, "E" );
                io::matlab::write( V, "V" );
            }

            {
                std::cout << term::bold << "  multi-stream block-wise Jacobi" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;

                io::matlab::write( M, "M" );
                
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_bjac_stream( blas::cuda::default_handle, M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;

                io::matlab::write( E, "E" );
                io::matlab::write( V, "V" );
            }

            {
                std::cout << term::bold << "  SYEVD" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_herm( blas::cuda::default_handle, M2, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }

            {
                std::cout << term::bold << "  DPT" << term::reset << std::endl;
                
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::cuda::eigen_dpt( blas::cuda::default_handle, M2, 1e-14, 1000, "frobenius", cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "    done in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;

                if ( stat.converged )
                    std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
                else
                    std::cout << "    not converged" << std::endl;
            }
        }// if
        else
        {
            // {
            //     auto  tic = timer::now();
            //     auto  toc = timer::since( tic );

            //     blas::eigen_stat  stat;
            
            //     auto  M2  = blas::copy( M );
            
            //     tic = timer::now();
            
            //     auto [ E, V ] = blas::eigen_jac( M2, 1e-14, 1000, & stat );

            //     toc = timer::since( tic );
            
            //     std::cout << "Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
            //     std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            // }

            {
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = eigen_bjac( M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }

            {
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                io::matlab::write( M, "M" );
                
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = eigen_bjac2( M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

                toc = timer::since( tic );

                io::matlab::write( E, "E" );
                io::matlab::write( V, "V" );
                
                std::cout << "Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }

            {
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::eigen_herm( M2, & stat );

                toc = timer::since( tic );
            
                std::cout << "SYEVD in  " << format_time( toc ) << std::endl;
                std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }

            {
                auto  tic = timer::now();
                auto  toc = timer::since( tic );

                blas::eigen_stat  stat;
            
                auto  M2  = blas::copy( M );
            
                tic = timer::now();
            
                auto [ E, V ] = blas::eigen_dpt( M2, 1e-14, 1000, "frobenius", cmdline::verbosity, & stat );

                toc = timer::since( tic );
            
                std::cout << "DPT in    " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                
                if ( stat.converged )
                    std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            }
        }// else

        return;
        
        //
        // double Jacobi
        //

        {
            auto  tic = timer::now();
            auto  toc = timer::since( tic );

            blas::eigen_stat  stat;
            
            auto  M2  = blas::copy( M );
            
            auto  mkl_nthreads = mkl_set_num_threads_local( 1 );
            
            tic = timer::now();
            
            auto [ E, V ] = eigen_bjac( M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity, & stat );

            toc = timer::since( tic );
            
            std::cout << "Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;

            mkl_set_num_threads_local( mkl_nthreads );
        }
        
        //
        // single Jac + double Jac
        //
        
        {
            auto  tic = timer::now();
            auto  toc = timer::since( tic );

            blas::eigen_stat  stat;
            
            auto  mkl_nthreads = mkl_set_num_threads_local( 1 );

            tic = timer::now();
            
            auto  M2        = blas::copy< float >( M );
            auto [ Es, Vs ] = eigen_bjac( M2, cmdline::ntile, 1e-4, 1000, cmdline::verbosity, & stat );

            toc = timer::since( tic );

            std::cout << "single Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
            
            
            auto  Ed        = blas::copy< double >( Es );
            auto  Vd        = blas::copy< double >( Vs );

            std::cout << "    error = " << format_error( blas::everror( M, Ed, Vd ) ) << std::endl;

            tic = timer::now();
            
            auto  VM        = blas::prod( value_t(1), blas::adjoint( Vd ), M );
            auto  VMV       = blas::prod( value_t(1), VM, Vd );

            // auto  [ E, V ] = blas::eigen_dpt( VMV, 1e-14, 0, "frobenius", cmdline::verbosity );
            auto  [ E, V ]  = eigen_bjac( VMV, cmdline::ntile, 1e-14, 1000, cmdline::verbosity );
            
            toc = timer::since( tic );

            std::cout << "double Jacobi in " << format_time( toc ) << std::endl;

            auto  V2 = blas::prod( double(1), V, Vd );
            
            std::cout << "    error = " << format_error( blas::everror( M, E, V2 ) ) << std::endl;

            mkl_set_num_threads_local( mkl_nthreads );
            
            return;
        }

        //
        // Precond + Jac + DPT
        //
        
        {
            auto  M2  = blas::matrix< value_t >();
            auto  tic = timer::now();

            blas::eigen_stat  stat;

            if ( true )
            {
                auto  [ Q, R ] = blas::qr( M );
                auto  H        = blas::prod( value_t(1), R, Q );
                auto  Delta    = blas::prod( value_t(1), M, Q );

                blas::prod( value_t(-1), Q, H, value_t(1), Delta );
                blas::prod( value_t(1), blas::adjoint(Q), Delta, value_t(1), H );

                M2 = std::move( H );
            }// if
            else
                M2 = std::move( blas::copy( M ) );

            auto  toc      = timer::since( tic );

            std::cout << "precond in " << format_time( toc ) << std::endl;

            io::matlab::write( M2, "M2" );
            
            tic = timer::now();
            
            auto  mkl_nthreads = mkl_set_num_threads_local( 1 );
                
            auto [ E1, V1 ] = eigen_bjac( M2, cmdline::ntile, 1e-3, 1000, cmdline::verbosity, & stat );

            mkl_set_num_threads_local( mkl_nthreads );

            toc = timer::since( tic );
            
            std::cout << "Jacobi in " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E1, V1 ) ) << std::endl;

            tic = timer::now();
            
            auto  [ E, V2 ] = blas::eigen_dpt( M2, 1e-14, 0, "frobenius", cmdline::verbosity );

            toc = timer::since( tic );

            std::cout << "DPT in " << format_time( toc ) << std::endl;

            tic = timer::now();
            
            auto  V = blas::prod( double(1), V2, V1 );
            
            toc = timer::since( tic );

            std::cout << "V in   " << format_time( toc ) << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
        }

        return;
        
        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();

            auto  mkl_nthreads = mkl_set_num_threads_local( 1 );
                
            auto [ E1, V1 ] = eigen_bjac( M2, cmdline::ntile, 1e-14, 1000, cmdline::verbosity );

            mkl_set_num_threads_local( mkl_nthreads );

            auto  toc = timer::since( tic );
            
            std::cout << "Jacobi in " << format_time( toc ) << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E1, V1 ) ) << std::endl;

            tic = timer::now();
            
            auto  [ E, V2 ] = blas::eigen_dpt( M2, 1e-14, 0, "frobenius", cmdline::verbosity );

            toc = timer::since( tic );
            
            std::cout << "DPT in " << format_time( toc ) << std::endl;

            tic = timer::now();
            
            auto  V = blas::prod( double(1), V2, V1 );
            
            toc = timer::since( tic );

            std::cout << "V in   " << format_time( toc ) << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;

            if ( n <= 1024 )
                io::matlab::write( V, "V1" );
            io::matlab::write( E, "E1" );
        }

        // {
        //     auto M2       = blas::copy< float >( M );
        //     auto tic      = timer::now();
            
        //     auto [ E1, V1 ] = eigen_bjac( M2, cmdline::ntile, 1e-5, 1000, cmdline::verbosity );
            
        //     std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;

        //     auto  V1d = blas::copy< double >( V1 );
        //     auto  VM  = blas::prod( double(1), blas::adjoint(V1d), M );
        //     auto  VMV = blas::prod( double(1), VM, V1d );
            
        //     auto  [ E, V2 ] = blas::eigen_dpt( VMV, 1e-14, 0, "frobenius", cmdline::verbosity );
            
        //     std::cout << "DPT in " << format_time( timer::since( tic ) ) << std::endl;

        //     auto  V = blas::prod( double(1), V2, V1d );
            
        //     auto toc      = timer::since( tic );

        //     std::cout << "V   in " << format_time( toc ) << std::endl;

        //     if ( n <= 1024 )
        //         io::matlab::write( V, "V1" );
        //     io::matlab::write( E, "E1" );
        // }

        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            auto [ E, V ] = blas::eigen_herm( M2 );
            auto toc      = timer::since( tic );

            std::cout << "syev in " << format_time( toc ) << std::endl;
            std::cout << "    error = " << format_error( blas::everror( M, E, V ) ) << std::endl;
            
            if ( n <= 1024 )
                io::matlab::write( V, "V2" );
            io::matlab::write( E, "E2" );
        }

        return;
    }

    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    std::cout << term::bullet << term::bold << "dense DPT eigen iteration ( " << impl_name
              << " )" << term::reset << std::endl;

    blas::eigen_stat  stat;
        
    for ( size_t  n = 256; n <= 16384; )
    {
        real_t  tol_min     = real_t(1);
        uint    nsweeps_dpt = 0;
        double  time_dpt    = 0;
        double  time_jac    = 0;
        double  time_eig    = 0;

        for ( uint  i = 0; i < 10; ++i )
        {
            auto  R  = blas::random< value_t >( n, n );
            auto  M  = blas::copy( R );

            blas::add( value_t(1), blas::adjoint(R), M );
            blas::scale( value_t(0.5), M );
            
            // auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );

            const auto  eps = real_t(100) * std::numeric_limits< real_t >::epsilon();
            auto        Mc  = blas::copy( M );
            
            for ( real_t  tol = real_t(1); tol >= eps; tol /= real_t(2) )
            {
                auto  Ms           = blas::copy( M );

                auto  tic_dpt      = timer::now();
                auto  mkl_nthreads = mkl_set_num_threads_local( 1 );
                auto  [ Ej, Vj ]   = eigen_bjac( Ms, cmdline::ntile, tol, 1000, 0 );
                
                mkl_set_num_threads_local( mkl_nthreads );
                // auto  Wj         = blas::copy< double >( Vj );
                // auto  VM         = blas::prod( value_t(1), blas::adjoint( Wj ), M );
                // auto  VMV        = blas::prod( value_t(1), VM, Wj );
                // auto  [ Ed, Vd ] = blas::eigen_dpt( VMV, 0, 1e-14, "fro", 0, & stat );
                auto  [ Ed, Vd ] = blas::eigen_dpt( Ms, 1e-14, 0, "fro", 0, & stat );

                auto  toc_dpt = timer::since( tic_dpt );
                
                if ( stat.converged )
                {
                    // converged
                    tol_min     = std::min( tol_min, tol );
                    nsweeps_dpt = std::max( nsweeps_dpt, stat.nsweeps );

                    if ( i == 0 )
                        time_dpt = toc_dpt.seconds();
                    else
                        time_dpt = std::max( time_dpt, toc_dpt.seconds() );
                    
                    break;
                }// if
            }// for

            {
                // also test pure Jacobi
                auto  Ms       = blas::copy( Mc );
                auto  tic_jac  = timer::now();
                auto  [ E, V ] = eigen_bjac( Ms, cmdline::ntile, 1e-14, 1000, 0, & stat );
                auto  toc_jac  = timer::since( tic_jac );
            
                // nsweeps_jac = std::max( nsweeps_jac, stat.nsweeps );
                time_jac = toc_jac.seconds();
            }

            {
                // and syev
                auto  Ms       = blas::copy( Mc );
                auto  tic_eig  = timer::now();
                auto  [ E, V ] = blas::eigen_herm( Ms );
                auto  toc_eig  = timer::since( tic_eig );
                
                // nsweeps_jac = std::max( nsweeps_jac, stat.nsweeps );
                time_eig = toc_eig.seconds();
            }
        }// for
    
        std::cout << "n = " << n
                  << "   " << format_norm( tol_min )
                  << "   " << format_time( time_dpt )
                  << "   " << format_time( time_jac )
                  << "   " << format_time( time_eig )
                  << std::endl;

        if      ( n < 1024 ) n += 128;
        else if ( n < 4096 ) n += 512;
        else                 n += 1024;
    }// for
}
