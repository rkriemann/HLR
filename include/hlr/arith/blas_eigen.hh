#ifndef __HLR_ARITH_BLAS_EIGEN_HH
#define __HLR_ARITH_BLAS_EIGEN_HH
//
// Project     : HLR
// Module      : arith/blas
// Description : basic linear algebra functions for eigenvalue computations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas_eigen.hh>

namespace hlr { namespace blas {

//////////////////////////////////////////////////////////////////////
//
// functions for eigenvalue computations
//
//////////////////////////////////////////////////////////////////////

struct eigen_stat
{
    uint  nsweeps   = 0;
    bool  converged = false;
};

//
// compute eigenvalues and eigenvectors of hermitian matrix M
//
template < typename value_t >
std::pair< blas::vector< typename hpro::real_type< value_t >::type_t >,
           blas::matrix< value_t > >
eigen_herm ( const matrix< value_t > &  M )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    const blas_int_t        n = M.nrows();
    auto                    V = copy( M );
    blas::vector< real_t >  E( n );
    std::vector< real_t >   rwork( hpro::is_complex_type< value_t >::value ? 3*n-2 : 0 );
    value_t                 work_query = value_t(0);
    blas_int_t              lwork      = -1;
    blas_int_t              info       = 0;

    heev( 'V', 'L', n, V.data(), V.col_stride(), E.data(), & work_query, lwork, rwork.data(), info );

    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );

    heev( 'V', 'L', n, V.data(), V.col_stride(), E.data(), work.data(), work.size(), rwork.data(), info );

    return { std::move( E ), std::move( V ) };
}

//
// compute eigenvalues and eigenvectors of M using two-sided Jacobi iteration.
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_jac ( matrix< value_t > &                                M,
            const size_t                                       amax_sweeps = 0,
            const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
            eigen_stat *                                       stat        = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15 );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;
    matrix< value_t >  V( minrc, ncols );

    // initialise V with identity
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    while ( ! converged && ( sweep < max_sweeps ))
    {
        real_t  max_err = 0.0;
        
        sweep++;
        converged = true;
                
        for ( size_t  i = 0; i < nrows-1; i++ )
        {
            for ( size_t j = i + 1; j < ncols; j++ )
            {
                //
                // compute Jacobi rotation diagonalizing ⎧ M_ii  M_ij ⎫
                //                                       ⎩ M_ji  M_jj ⎭
                //

                const auto  c = M(i,j);

                if ( std::abs( c ) == value_t(0) )
                    continue;

                const auto  a   = M(i,i);
                const auto  b   = M(j,j);
                const auto  err = std::abs( c ) / std::real( std::sqrt( a*b ) );
                
                if (  err > tolerance )
                    converged = false;

                if ( ! std::isnormal( err ) )
                    std::cout << '.' << std::flush;
                
                max_err = std::max( err, max_err );
                
                //
                // compute Jacobi rotation which diagonalises │a c│
                //                                            │c b│
                //

                const auto  xi = (b - a) / ( value_t(2) * c );
                const auto  t  = ( math::sign( xi ) / ( std::abs(xi) + std::sqrt( 1.0 + xi*xi ) ) );
                const auto  cs = value_t(1) / std::sqrt( 1.0 + t*t );
                const auto  sn = cs * t;

                M(i,i) = a - c * t;
                M(j,j) = b + c * t;
                M(i,j) = M(j,i) = 0;
                
                //
                // update columns i and j of A (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    if (( k == i ) || ( k == j ))
                        continue;
                    
                    const auto  m_ik = M(i,k);
                    const auto  m_jk = M(j,k);

                    M(k,i) = M(i,k) = cs * m_ik - sn * m_jk;
                    M(k,j) = M(j,k) = sn * m_ik + cs * m_jk;
                }// for

                //
                // update V (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  v_ki = V(k,i);
                    const auto  v_kj = V(k,j);

                    V(k,i) = cs * v_ki - sn * v_kj;
                    V(k,j) = sn * v_ki + cs * v_kj;
                }// for
            }// for
        }// for

        //
        // determine diagonal dominance ( Σ_j≠i a_ij ) / a_ii
        //

        // real_t  diag_dom = real_t(0);
        // real_t  avg_dom  = real_t(0);
        
        // for ( size_t  i = 0; i < nrows-1; i++ )
        // {
        //     real_t  row_sum = real_t(0);
            
        //     for ( size_t j = 0; j < ncols; j++ )
        //     {
        //         if ( i != j )
        //             row_sum += std::abs( M(i,j) );
        //     }// for

        //     const auto  dom = row_sum / std::abs( M(i,i) );
            
        //     diag_dom = std::max( diag_dom, dom );
        //     avg_dom += dom;
        // }// for

        // avg_dom /= nrows;
        
        // std::cout << "sweeps " << sweep << " : "
        //           << "error = " << max_err << ", "
        //           << "diag_dom = " << diag_dom << ", "
        //           << "avg_dom = " << avg_dom 
        //           << std::endl;

        // if (( diag_dom <= 2.0 ) && ( avg_dom <= 0.05 ))
        //     break;
    }// while

    // std::cout << "#sweeps = " << sweep << std::endl;

    if ( ! is_null( stat ) )
    {
        stat->nsweeps   = sweep;
        stat->converged = converged;
    }// if
    
    //
    // extract eigenvalues as diagonal elements of M
    //

    vector< value_t >  E( minrc );
    
    for ( size_t  i = 0; i < minrc; i++ )
        E(i) = M(i,i);

    return { std::move( E ), std::move( V ) };
}

template < typename value_t >
void
make_diag_dom ( matrix< value_t > &                                M,
                const size_t                                       amax_sweeps = 0,
                const typename hpro::real_type< value_t >::type_t  atolerance  = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15 );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;

    while ( ! converged && ( sweep < max_sweeps ))
    {
        real_t  max_err = 0.0;
        
        sweep++;
        converged = true;

        //
        // look for m_ij with |m_ij| = max_k≠l |m_kl|
        //

        size_t  i     = 0;
        size_t  j     = 1;
        real_t  m_max = std::abs( M(i,j) );
        
        for ( size_t  k = 0; k < nrows-1; k++ )
        {
            for ( size_t l = k + 1; l < ncols; l++ )
            {
                if ( std::abs( M(k,l) ) > m_max )
                {
                    m_max = std::abs( M(k,l) );
                    i     = k;
                    j     = l;
                }// if
            }// for
        }// for
        
        //
        // compute Jacobi rotation diagonalizing ⎧ M_ii  M_ij ⎫
        //                                       ⎩ M_ji  M_jj ⎭
        //

        const auto  c = M(i,j);

        if ( std::abs( c ) == value_t(0) )
            break;
        
        const auto  a   = M(i,i);
        const auto  b   = M(j,j);
        const auto  err = std::abs( c ) / std::real( std::sqrt( a*b ) );
        
        if (  err > tolerance )
            converged = false;
        
        max_err = std::max( err, max_err );
        
        //
        // compute Jacobi rotation which diagonalises │a c│
        //                                            │c b│
        //
        
        const auto  xi = (b - a) / ( value_t(2) * c );
        const auto  t  = ( math::sign( xi ) / ( std::abs(xi) + std::sqrt( 1.0 + xi*xi ) ) );
        const auto  cs = value_t(1) / std::sqrt( 1.0 + t*t );
        const auto  sn = cs * t;
        
        M(i,i) = a - c * t;
        M(j,j) = b + c * t;
        M(i,j) = M(j,i) = 0;
        
        //
        // update columns i and j of A (apply rotation)
        //
        
        for ( size_t  k = 0; k < nrows; k++ )
        {
            if (( k == i ) || ( k == j ))
                continue;
            
            const auto  m_ik = M(i,k);
            const auto  m_jk = M(j,k);
            
            M(k,i) = M(i,k) = cs * m_ik - sn * m_jk;
            M(k,j) = M(j,k) = sn * m_ik + cs * m_jk;
        }// for

        //
        // determine diagonal dominance ( Σ_j≠i a_ij ) / a_ii
        //

        real_t  diag_dom = real_t(0);
        real_t  avg_dom  = real_t(0);
        
        for ( size_t  k = 0; k < nrows-1; k++ )
        {
            real_t  row_sum = real_t(0);
            
            for ( size_t l = 0; l < ncols; l++ )
            {
                if ( k != l )
                    row_sum += std::abs( M(k,l) );
            }// for

            const auto  dom = row_sum / std::abs( M(k,k) );
            
            diag_dom = std::max( diag_dom, dom );
            avg_dom += dom;
        }// for

        avg_dom /= nrows;
        
        std::cout << "sweeps " << sweep << " : "
                  << "error = " << max_err << ", "
                  << "diag_dom = " << diag_dom << ", "
                  << "avg_dom = " << avg_dom 
                  << std::endl;

        if (( diag_dom <= 2.0 ) && ( avg_dom <= 0.5 ))
            break;
    }// while

    std::cout << "#sweeps = " << sweep << std::endl;
}

//
// compute eigenvalues and eigenvectors of M using DPT iteration.
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_dpt ( matrix< value_t > &                                M,
            const size_t                                       amax_sweeps = 0,
            const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
            const std::string &                                error_type  = "frobenius",
            const int                                          verbosity   = 0,
            eigen_stat *                                       stat        = nullptr )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // assumption
    HLR_ASSERT( M.nrows() == M.ncols() );
    
    const auto    nrows      = M.nrows();
    const real_t  tolerance  = ( atolerance  > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    const uint    max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 100 );

    vector< value_t >  diag_T( nrows );
    vector< value_t >  diag_M( nrows );
    matrix< value_t >  Delta( M );  // reuse M
    matrix< value_t >  V( nrows, nrows );
    matrix< value_t >  T( nrows, nrows );

    for ( size_t  i = 0; i < nrows; ++i )
    {
        diag_M(i)     = M( i, i );
        Delta( i, i ) = value_t(0);     // diag(Δ) = 0
        V(i,i)        = value_t(1);     // V = I before iteration
    }// for

    //
    // compute I - Θ⊗M with Θ_ij = 1 / ( m_ii - m_jj )
    //
    auto  hmul_theta =  [&diag_M] ( matrix< value_t > &  A )
                        {
                            for ( size_t  j = 0; j < A.ncols(); ++j )
                                for ( size_t  i = 0; i < A.nrows(); ++i )
                                {
                                    if ( i == j )
                                        A(i,j)  =   value_t(1);
                                    else
                                        A(i,j) *= - value_t(1) / ( diag_M(i) - diag_M(j) );
                                }// for
                        };

    //
    // iteration
    //
    
    real_t  old_error = real_t(0);
    uint    sweep     = 0;

    if ( ! is_null( stat ) )
    {
        stat->nsweeps   = 0;
        stat->converged = false;
    }// if
    
    do
    {
        // T = Δ·V
        prod( value_t(1), Delta, V, value_t(0), T );
        
        // T = Δ·V - V·diag(Δ·V) = T - V·diag(T) 
        // computed as T(i,:) = T(i,:) - T(i,i) · V(i,:)
        for ( size_t  i = 0; i < nrows; ++i )
            diag_T(i) = T(i,i);
        
        for ( size_t  i = 0; i < nrows; ++i )
        {
            auto  V_i = V.column(i);
            auto  T_i = T.column(i);

            add( -diag_T(i), V_i, T_i );
        }// for

        // I - Θ ∗ (Δ·V - V·diag(Δ·V)) = I - Θ ∗ T
        hmul_theta( T );

        //
        // compute error ||V-T||_F
        //

        real_t  error = 0;

        if (( error_type == "frobenius" ) || ( error_type == "fro" ))
        {
            add( value_t(-1), T, V );
            error = norm_F( V );
        }// if
        else if (( error_type == "maximum" ) || ( error_type == "max" ))
        {
            add( value_t(-1), T, V );
            error = norm_max( V );
        }// if
        else if (( error_type == "residual" ) || ( error_type == "res" ))
        {
            // // extract eigenvalues as diag( M + Δ·V ) and eigenvectors as V
            // // (T holds new V)
            // std::vector< value_t >  E( n );

            // for ( int  i = 0; i < n; ++i )
            //     E[i] = diag_M[ i ] + dot( n, Delta + i, n, T.data() + i*n, 1 );

            // // copy diagonal back to M
            // copy( n, diag_M.data(), 1, M.data(), n+1 );
            // gemm( 'N', 'N', n, n, n, value_t(1), M.data(), n, T.data(), n, value_t(0), V.data(), n );
            // for ( int  i = 0; i < n; ++i )
            // {
            //     axpy( n, -E[i], T.data() + i*n, 1, V.data() + i*n, 1 );
            //     M[ i*n+i ] = 0.0; // reset diagonal for Delta
            // }// for
            
            // error = normF( n, n, V ) / ( M_norm * norm1( n, n, T ) );
        }// if
        else
            HLR_ERROR( "unknown error type" );

        //
        // test stop criterion
        //

        copy( T, V );

        if ( verbosity >= 1 )
        {
            std::cout << "    sweep " << sweep << " : error = " << error;

            if ( sweep > 0 )
                std::cout << ", reduction = " << error / old_error;
            
            std::cout << std::endl;
        }// if

        if (( sweep > 0 ) && ( error / old_error > 10.0 ))
            return { vector< value_t >(), matrix< value_t >() };
        
        old_error = error;
        
        ++sweep;

        if ( ! is_null( stat ) )
            stat->nsweeps = sweep;
        
        if ( error < tolerance )
        {
            if ( ! is_null( stat ) )
                stat->converged = true;
            
            break;
        }// if

        if ( ! std::isnormal( error ) )
            break;
        
    } while ( sweep < max_sweeps );

    //
    // eigenvalues  : diag( M + Δ·V )
    // eigenvectors : V
    //

    vector< value_t >  E( nrows );

    for ( size_t  i = 0; i < nrows; ++i )
    {
        auto  Delta_i = Delta.row( i );
        auto  V_i     = V.column( i );
        
        E(i) = diag_M( i ) + dot( Delta_i, V_i );
    }// for
    
    return { std::move( E ), std::move( V ) };
}

//
// compute singular value decomposition M = U S V^T of the
// nrows × ncols matrix M,
// - upon exit, M contains U
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
void
svd_jac ( matrix< value_t > &                                      M,
          vector< typename hpro::real_type< value_t >::type_t > &  S,
          matrix< value_t > &                                      V,
          const size_t                                             max_sweeps = 0,
          const typename hpro::real_type< value_t >::type_t        tolerance  = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto    nrows     = M.nrows();
    const auto    ncols     = M.ncols();
    const auto    minrc     = std::min( nrows, ncols );
    const real_t  tol       = ( tolerance > 0 ? tolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool          converged = false;
    uint          sweep     = 0;

    // initialise V with identity
    if (( V.nrows() != minrc ) || ( V.ncols() != ncols ) )
        V = std::move( matrix< value_t >( minrc, ncols ) );
    
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    while ( ! converged and (( max_sweeps > 0 ) && ( sweep < max_sweeps )) )
    {
        sweep++;
        converged = true;
                
        for ( size_t  i = 0; i < nrows-1; i++ )
        {
            for ( size_t j = i + 1; j < ncols; j++ )
            {
                //
                // compute |a c| = (i,j) submatrix of A^T A
                //         |c b|
                //

                auto  m_i = M.column( i );
                auto  m_j = M.column( j );
                
                const auto  a = dot( m_i, m_i );
                const auto  b = dot( m_j, m_j );
                const auto  c = dot( m_i, m_j );

                if ( std::abs( c ) / std::real( std::sqrt( a*b ) ) > tol )
                    converged = false;
                        
                //
                // compute Jacobi rotation which diagonalises │a c│
                //                                            │c b│
                //

                const auto  xi = (b - a) / ( value_t(2) * c );
                const auto  t  = ( math::sign( xi ) / ( std::abs(xi) + std::sqrt( 1.0 + xi*xi ) ) );
                const auto  cs = value_t(1) / std::sqrt( 1.0 + t*t );
                const auto  sn = cs * t;

                //
                // update columns i and j of A (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  m_ki = M(k,i);
                    const auto  m_kj = M(k,j);

                    M(k,i) = cs * m_ki - sn * m_kj;
                    M(k,j) = sn * m_ki + cs * m_kj;
                }// for

                //
                // update V (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  v_ki = V(k,i);
                    const auto  v_kj = V(k,j);

                    V(k,i) = cs * v_ki - sn * v_kj;
                    V(k,j) = sn * v_ki + cs * v_kj;
                }// for
            }// for
        }// for
    }// while

    //
    // extract singular values and update U
    //

    if ( S.length() != minrc )
        S = std::move( vector< real_t >( minrc ) );
    
    for ( size_t  i = 0; i < minrc; i++ )
    {
        auto  m_i = M.column(i);
        
        S(i) = norm2( m_i );

        if ( std::abs( S(i) ) > 1e-14 )
        {
            scale( value_t(1) / S(i), m_i );
        }// if
        else
        {
            S(i) = 0.0;
            fill( value_t(0), m_i );

            auto  v_i = V.column(i);

            fill( value_t(0), v_i );
        }// else
    }// for
}

}}// namespace hlr::blas

#endif // __HLR_ARITH_BLAS_EIGEN_HH
