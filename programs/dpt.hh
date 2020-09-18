//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <random>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hlr/arith/blas_eigen.hh>
#include <hlr/utils/tensor.hh>
#include <hlr/utils/io.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_jac_bw ( blas::matrix< value_t > &                          M,
               const size_t                                       block_size  = 128,
               const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
               const size_t                                       amax_sweeps = 0,
               const uint                                         verbosity   = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         nbrows     = nrows / block_size;
    const auto         nbcols     = ncols / block_size;

    HLR_ASSERT( ( nrows / block_size ) * block_size == nrows );
    HLR_ASSERT( ( ncols / block_size ) * block_size == ncols );
    
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15*minrc*minrc );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(100) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;
    auto               V          = blas::matrix< value_t >( minrc, ncols );
    auto               norms      = blas::matrix< real_t >( nbrows, nbcols );

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
                ::tbb::parallel_for< size_t >(
                    0, nbcols,
                    [&,i,j,block_size] ( const size_t  l )
                    {
                        const auto  r_l  = blas::range( l*block_size, (l+1)*block_size-1 );
                        auto        M_il = blas::matrix< value_t >( M, r_i, r_l );
                        auto        M_jl = blas::matrix< value_t >( M, r_j, r_l );
                        const auto  C_il = blas::copy( M_il );
                        const auto  C_jl = blas::copy( M_jl );
                        // auto  C_il = M_il;
                        // auto  C_jl = M_jl;
                        
                        auto        T_il = blas::prod( value_t(1), blas::adjoint(VA_ii), C_il );
                        auto        T_jl = blas::prod( value_t(1), blas::adjoint(VA_ij), C_il );
                        
                        blas::prod( value_t(1), blas::adjoint(VA_ji), C_jl, value_t(1), T_il );
                        blas::prod( value_t(1), blas::adjoint(VA_jj), C_jl, value_t(1), T_jl );
                        
                        norms(i,l) = blas::norm_F( T_il );
                        norms(j,l) = blas::norm_F( T_jl );
                        
                        blas::copy( T_il, M_il );
                        blas::copy( T_jl, M_jl );
                    } );

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

                        // apply to M
                        auto        M_li = blas::matrix< value_t >( M, r_l, r_i );
                        auto        M_lj = blas::matrix< value_t >( M, r_l, r_j );
                        const auto  C_li = blas::copy( M_li );
                        const auto  C_lj = blas::copy( M_lj );
                        // auto  C_li = M_li;
                        // auto  C_lj = M_lj;
                        
                        auto  T_li = blas::prod( value_t(1), C_li, VA_ii );
                        auto  T_lj = blas::prod( value_t(1), C_li, VA_ij );
                        
                        blas::prod( value_t(1), C_lj, VA_ji, value_t(1), T_li );
                        blas::prod( value_t(1), C_lj, VA_jj, value_t(1), T_lj );
                        
                        norms(l,i) = blas::norm_F( T_li );
                        norms(l,j) = blas::norm_F( T_lj );
                        
                        blas::copy( T_li, M_li );
                        blas::copy( T_lj, M_lj );

                        // apply to V
                        auto        V_li = blas::matrix< value_t >( V, r_l, r_i );
                        auto        V_lj = blas::matrix< value_t >( V, r_l, r_j );
                        const auto  D_li = blas::copy( V_li );
                        const auto  D_lj = blas::copy( V_lj );
                        // auto  D_li = V_li;
                        // auto  D_lj = V_lj;
                        
                        auto  S_li = blas::prod( value_t(1), D_li, VA_ii );
                        auto  S_lj = blas::prod( value_t(1), D_li, VA_ij );
                        
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

        #if 1
        
        auto  R  = blas::matrix< value_t >( n, n );

        blas::fill_fn( R, random );
        
        // auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );
        auto  M  = blas::copy( R );

        blas::add( value_t(1), blas::adjoint(R), M );
        blas::scale( value_t(0.5), M );

        #else

        auto  alpha = value_t(1e10);
        auto  D     = blas::matrix< value_t >( n, n );
        auto  Q     = blas::matrix< value_t >( n, n );
        auto  R     = blas::matrix< value_t >( n, n );

        // D with condition alpha
        for ( uint  i = 0; i < n; ++i )
            D(i,i) = std::pow( alpha, -(value_t(i)/value_t(n-1)) ); 

        // random, orthogonal Q
        blas::fill_fn( Q, random );
        blas::qr( Q, R );

        // M = Q·D·Q'
        blas::prod( value_t(1), Q,                D, value_t(0), R );
        blas::prod( value_t(1), R, blas::adjoint(Q), value_t(0), D );

        auto  M = std::move( D );
        
        #endif
        

        if ( n <= 2048 )
            io::write_matlab( M, "M" );

        // M = std::move( // io::read_matlab< value_t >( "M.mat" ) );
        
        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            
            auto [ E1, V1 ] = eigen_jac_bw( M2, cmdline::ntile, 1e-5, 100, cmdline::verbosity );
            
            std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;
        }
        
        {
            auto M2       = blas::copy< float >( M );
            auto tic      = timer::now();
            
            auto [ E1, V1 ] = eigen_jac_bw( M2, cmdline::ntile, 1e-5, 100, cmdline::verbosity );
            
            std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;
        }

        return;
        
        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            
            auto [ E1, V1 ] = eigen_jac_bw( M2, cmdline::ntile, 1e-5, 1000, cmdline::verbosity );
            
            std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;
            
            auto  [ E, V2 ] = blas::eigen_dpt( M2, 0, 1e-14, "frobenius", cmdline::verbosity );
            
            std::cout << "DPT in " << format_time( timer::since( tic ) ) << std::endl;

            auto  V = blas::prod( double(1), V2, V1 );
            
            auto toc      = timer::since( tic );

            std::cout << "done in " << format_time( toc ) << std::endl;

            if ( n <= 1024 )
                io::write_matlab( V, "V1" );
            io::write_matlab( E, "E1" );
        }

        {
            auto M2       = blas::copy< float >( M );
            auto tic      = timer::now();
            
            auto [ E1, V1 ] = eigen_jac_bw( M2, cmdline::ntile, 1e-5, 1000, cmdline::verbosity );
            
            std::cout << "Jacobi in " << format_time( timer::since( tic ) ) << std::endl;

            auto  V1d = blas::copy< double >( V1 );
            auto  VM  = blas::prod( double(1), blas::adjoint(V1d), M );
            auto  VMV = blas::prod( double(1), VM, V1d );
            
            auto  [ E, V2 ] = blas::eigen_dpt( VMV, 0, 1e-14, "frobenius", cmdline::verbosity );
            
            std::cout << "DPT in " << format_time( timer::since( tic ) ) << std::endl;

            auto  V = blas::prod( double(1), V2, V1d );
            
            auto toc      = timer::since( tic );

            std::cout << "V   in " << format_time( toc ) << std::endl;

            if ( n <= 1024 )
                io::write_matlab( V, "V1" );
            io::write_matlab( E, "E1" );
        }

        {
            auto M2       = blas::copy( M );
            auto tic      = timer::now();
            auto [ E, V ] = blas::eigen_herm( M2 );
            auto toc      = timer::since( tic );

            std::cout << "done in " << format_time( toc ) << std::endl;
            if ( n <= 1024 )
                io::write_matlab( V, "V2" );
            io::write_matlab( E, "E2" );
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
