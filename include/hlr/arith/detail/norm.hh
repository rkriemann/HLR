#ifndef __HLR_ARITH_DETAIL_NORM_HH
#define __HLR_ARITH_DETAIL_NORM_HH
//
// Project     : HLR
// Module      : norm
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <random>
#include <concepts>

#include <hlr/arith/blas.hh>
#include <hlr/arith/defaults.hh>
#include <hlr/arith/operator_wrapper.hh>

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/tiled_lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/convert.hh>

#include <hlr/utils/log.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/text.hh>
#include <hlr/utils/math.hh>
#include <hlr/utils/traits.hh>

namespace hlr { namespace norm { namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// Frobenius norm
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
long double
frobenius_squared ( const Hpro::TMatrix< value_t > &  A )
{
    // using  real_t   = Hpro::real_type_t< value_t >;
    using  result_t = long double;
    
    if ( is_blocked( A ) )
    {
        auto      B   = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        result_t  val = 0.0;
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( is_null( B->block( i, j ) ) )
                    continue;
                
                val += frobenius_squared( * B->block( i, j ) );
            }// for
        }// for

        return val;
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        auto  R = cptrcast( & A, hlr::matrix::lrmatrix< value_t > );
        
        //
        // version 1: compute singular values and sum up
        //

        auto  U = blas::copy( R->U() );
        auto  V = blas::copy( R->V() );
        auto  S = blas::vector< real_type_t< value_t > >( U.ncols() );

        blas::sv( U, V, S );
        
        auto  val = result_t(0);

        for ( size_t  i = 0; i < S.length(); ++i )
            val += math::square( S(i) );

        return val;

        //
        // version 2: use special lowrank method (has problems below 1e-8)
        //
        
        // return blas::sqnorm_F( R->U(), R->V() );
    }// if
    else if ( matrix::is_lowrank_sv( A ) )
    {
        auto  U = blas::prod_diag( cptrcast( &A, matrix::lrsvmatrix< value_t > )->U(),
                                   cptrcast( &A, matrix::lrsvmatrix< value_t > )->S() );
        
        return blas::sqnorm_F( U, cptrcast( &A, matrix::lrsvmatrix< value_t > )->V() );
    }// if
    else if ( hlr::matrix::is_tiled_lowrank( A ) )
    {
        //
        // ∑_ij (R_ij)² = ∑_ij (∑_k U_ik V_jk')²
        //              = ∑_ij (∑_k U_ik V_jk') (∑_l U_il V_jl')'
        //              = ∑_ij ∑_k ∑_l U_ik V_jk' U_il' V_jl
        //              = ∑_k ∑_l ∑_i U_ik U_il' ∑_j V_jk' V_jl
        //              = ∑_k ∑_l (U_l)^H · U_k  V_k^H · V_l
        //

        auto          R   = cptrcast( & A, hlr::matrix::tiled_lrmatrix< value_t > );
        const auto &  U   = R->U();
        const auto &  V   = R->V();
        result_t      val = 0.0;
        
        for ( size_t  l = 0; l < R->rank(); l++ )
        {
            for ( size_t  k = 0; k < R->rank(); k++ )
            {
                result_t  dot_U = 0;
                result_t  dot_V = 0;

                auto  U_i = U.cbegin();
                auto  V_i = V.cbegin();

                for ( ; ( U_i != U.cend() ) && ( V_i != V.cend() ); ++U_i, ++V_i )
                {
                    const auto  U_l = (*U_i).second.column( l );
                    const auto  V_l = (*V_i).second.column( l );
                
                    const auto  U_k = (*U_i).second.column( k );
                    const auto  V_k = (*V_i).second.column( k );
                    
                    dot_U += std::abs( blas::dot( U_l, U_k ) );
                    dot_V += std::abs( blas::dot( V_l, V_k ) );
                }// for

                val += dot_U * dot_V;
            }// for
        }// for

        return std::abs( val );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( A ) )
    {
        //
        // |A| = | U S V' | = |U||S||V| = |S| with orthogonal U/V
        //

        auto  R   = cptrcast( &A, hlr::matrix::uniform_lrmatrix< value_t > );
        auto  C   = R->coupling();
        auto  val = result_t(0);

        for ( size_t  i = 0; i < C.nrows()*C.ncols(); ++i )
            val += std::abs( math::square( C.data()[ i ] ) );

        return val;
    }// if
    else if ( hlr::matrix::is_h2_lowrank( A ) )
    {
        //
        // |A| = | U S V' | = |U||S||V| = |S| with orthogonal U/V
        //

        auto  R   = cptrcast( &A, hlr::matrix::h2_lrmatrix< value_t > );
        auto  C   = R->coupling();
        auto  val = result_t(0);

        for ( size_t  i = 0; i < C.nrows()*C.ncols(); ++i )
            val += std::abs( math::square( C.data()[ i ] ) );

        return val;
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  M   = cptrcast( &A, matrix::dense_matrix< value_t > )->mat();
        auto  val = result_t(0);

        for ( size_t  i = 0; i < M.nrows()*M.ncols(); ++i )
            val += std::abs( math::square( M.data()[ i ] ) );

        return val;
    }// if
    else
    {
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );
    }// else

    return 0;
}

//
// return Frobenius norm of αA+βB, e.g. |αA+βB|_F
//
template < general_number alpha_t,
           general_number beta_t,
           typename value_t >
long double
frobenius_squared ( const alpha_t                     alpha,
                    const Hpro::TMatrix< value_t > &  A,
                    const beta_t                      beta,
                    const Hpro::TMatrix< value_t > &  B )
{
    // using  real_t = Hpro::real_type_t< value_t >;
    using  result_t = long double;

    HLR_ASSERT( A.block_is() == B.block_is() );

    //
    // special lowrank function (see below)
    //
    // auto  lrdot = [] ( const auto &  U1,
    //                    const auto &  V1,
    //                    const auto &  U2,
    //                    const auto &  V2 )
    // {
    //     const auto  rank1 = U1.ncols();
    //     const auto  rank2 = U2.ncols();
    //     result_t    val   = 0.0;
                              
    //     for ( size_t  l = 0; l < rank1; l++ )
    //     {
    //         const auto  U1_l = U1.column( l );
    //         const auto  V1_l = V1.column( l );
                                  
    //         for ( size_t  k = 0; k < rank2; k++ )
    //         {
    //             const auto  U2_k = U2.column( k );
    //             const auto  V2_k = V2.column( k );
                                      
    //             val += blas::dot( U1_l, U2_k ) * blas::dot( V1_l, V2_k );
    //         }// for
    //     }// for

    //     return val;
    // };

    if ( hlr::is_blocked_all( A, B ) )
    {
        auto      BA  = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto      BB  = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        result_t  val = 0.0;

        HLR_ASSERT(( BA->nblock_rows() == BB->block_rows() ) &&
                   ( BA->nblock_cols() == BB->block_cols() ));
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( is_null( BA->block( i, j ) ) )
                {
                    if ( is_null( BB->block( i, j ) ) )
                        continue;
                    else
                        val += beta * frobenius_squared( * BB->block( i, j ) );
                }// if
                else
                {
                    if ( is_null( BB->block( i, j ) ) )
                        val += alpha * frobenius_squared( * BA->block( i, j ) );
                    else
                        val += frobenius_squared( alpha, * BA->block( i, j ),
                                                  beta,  * BB->block( i, j ) );
                }// else
            }// for
        }// for

        if ( ! std::isfinite( val ) )
        {
            if ( std::isinf( val ) ) std::cout << "(B) inf value for " << A.block_is() << std::endl;
            if ( std::isnan( val ) ) std::cout << "(B) nan value for " << A.block_is() << std::endl;
        }// if
        
        return std::abs( val );
    }// if
    else if (( matrix::is_lowrank(    A ) && matrix::is_lowrank(    B ) ) ||
             ( matrix::is_lowrank(    A ) && matrix::is_lowrank_sv( B ) ) ||
             ( matrix::is_lowrank_sv( A ) && matrix::is_lowrank(    B ) ) ||
             ( matrix::is_lowrank_sv( A ) && matrix::is_lowrank_sv( B ) ))
    {
        auto  UA = blas::matrix< value_t >();
        auto  VA = blas::matrix< value_t >();
        auto  UB = blas::matrix< value_t >();
        auto  VB = blas::matrix< value_t >();

        if ( matrix::is_lowrank( A )  )
        {
            UA = cptrcast( &A, matrix::lrmatrix< value_t > )->U();
            VA = cptrcast( &A, matrix::lrmatrix< value_t > )->V();
        }// if
        else if ( matrix::is_lowrank_sv( A ) )
        {
            auto  U = blas::prod_diag( cptrcast( &A, matrix::lrsvmatrix< value_t > )->U(),
                                       cptrcast( &A, matrix::lrsvmatrix< value_t > )->S() );
            
            UA = std::move( U );
            VA = cptrcast( &A, matrix::lrsvmatrix< value_t > )->V();
        }// if

        if ( matrix::is_lowrank( B )  )
        {
            UB = cptrcast( &B, matrix::lrmatrix< value_t > )->U();
            VB = cptrcast( &B, matrix::lrmatrix< value_t > )->V();
        }// if
        else if ( matrix::is_lowrank_sv( B ) )
        {
            auto  U = blas::prod_diag( cptrcast( &B, matrix::lrsvmatrix< value_t > )->U(),
                                       cptrcast( &B, matrix::lrsvmatrix< value_t > )->S() );
            
            UB = std::move( U );
            VB = cptrcast( &B, matrix::lrsvmatrix< value_t > )->V();
        }// if

        //
        // version 1: compute singular values and sum up
        //

        HLR_ASSERT(( UA.nrows() > 0 ) && ( UA.ncols() > 0 ) &&
                   ( VA.nrows() > 0 ) && ( VA.ncols() > 0 ) &&
                   ( UB.nrows() > 0 ) && ( UB.ncols() > 0 ) &&
                   ( VB.nrows() > 0 ) && ( VB.ncols() > 0 ) );
        
        auto  k1 = UA.ncols();
        auto  k2 = UB.ncols();
        auto  U  = blas::matrix< value_t >( UA.nrows(), k1 + k2 );
        auto  V  = blas::matrix< value_t >( VA.nrows(), k1 + k2 );

        auto  U_1 = blas::matrix< value_t >( U, blas::range::all, blas::range( 0, k1-1 ) );
        auto  V_1 = blas::matrix< value_t >( V, blas::range::all, blas::range( 0, k1-1 ) );
        auto  U_2 = blas::matrix< value_t >( U, blas::range::all, blas::range( k1, k1+k2-1 ) );
        auto  V_2 = blas::matrix< value_t >( V, blas::range::all, blas::range( k1, k1+k2-1 ) );

        blas::copy( UA, U_1 );
        blas::copy( VA, V_1 );
        blas::scale( alpha, U_1 );

        blas::copy( UB, U_2 );
        blas::copy( VB, V_2 );
        blas::scale( beta, U_2 );

        auto  S = blas::vector< real_type_t< value_t > >( k1 + k2 );

        blas::sv( U, V, S );
        
        auto  val = result_t(0);

        for ( size_t  i = 0; i < S.length(); ++i )
            val += math::square( S(i) );

        if ( ! std::isfinite( val ) )
        {
            if ( std::isinf( val ) ) std::cout << "(R) inf value for " << A.block_is() << std::endl;
            if ( std::isnan( val ) ) std::cout << "(R) nan value for " << A.block_is() << std::endl;
        }// if

        return val;

        //
        // version 2: use "lrdot" above (has problems below 1e-8)
        //
        
        // const auto  sqn = ( alpha * alpha * lrdot( UA, VA, UA, VA ) +
        //                     alpha * beta  * lrdot( UA, VA, UB, VB ) +
        //                     alpha * beta  * lrdot( UB, VB, UA, VA ) +
        //                     beta  * beta  * lrdot( UB, VB, UB, VB ) );
        
        // return std::abs( sqn );

        //
        // version 3: convert to dense (for debugging)
        //
        
        // auto  M1 = blas::prod( UA, blas::adjoint( VA ) );
        // auto  M2 = blas::prod( UB, blas::adjoint( VB ) );

        // blas::scale( value_t(alpha), M1 );
        // blas::add( beta, M2, M1 );

        // auto  val = result_t(0);

        // for ( size_t  i = 0; i < M1.nrows()*M1.ncols(); ++i )
        //     val += math::square( M1.data()[ i ] );

        // return val;
        
    }// if
    else if ( matrix::is_uniform_lowrank( A ) )
    {
        auto  RA = matrix::convert_to_lowrank( A );

        return frobenius_squared( alpha, *RA, beta, B );
    }// if
    else if ( matrix::is_uniform_lowrank( B ) )
    {
        auto  RB = matrix::convert_to_lowrank( B );

        return frobenius_squared( alpha, A, beta, *RB );
    }// if
    else if ( matrix::is_h2_lowrank( A ) )
    {
        auto  RA = matrix::convert_to_lowrank( A );

        return frobenius_squared( alpha, *RA, beta, B );
    }// if
    else if ( matrix::is_h2_lowrank( B ) )
    {
        auto  RB = matrix::convert_to_lowrank( B );

        return frobenius_squared( alpha, A, beta, *RB );
    }// if
    else if ( matrix::is_dense_all( A, B ) )
    {
        auto  MA = cptrcast( &A, matrix::dense_matrix< value_t > )->mat();
        auto  MB = cptrcast( &B, matrix::dense_matrix< value_t > )->mat();
        
        result_t     val = 0;
        const idx_t  n   = idx_t(MA.nrows());
        const idx_t  m   = idx_t(MA.ncols());
    
        for ( idx_t j = 0; j < m; ++j )
        {
            for ( idx_t i = 0; i < n; ++i )
            {
                const auto  a_ij = value_t(alpha) * MA(i,j) + value_t(beta) * MB(i,j);
                
                val += std::abs( math::square( a_ij ) );
            }// for
        }// for

        if ( ! std::isfinite( val ) )
        {
            if ( std::isinf( val ) ) std::cout << "(D) inf value for " << A.block_is() << std::endl;
            if ( std::isnan( val ) ) std::cout << "(D) nan value for " << A.block_is() << std::endl;
        }// if
            
        return val;
    }// if
    else if ( matrix::is_dense( A ) && matrix::is_lowrank( B ) )
    {
        auto  DB = matrix::convert_to_dense( B );
        
        return frobenius_squared( alpha, A, beta, *DB );
    }// if
    else if ( matrix::is_dense( B ) && matrix::is_lowrank( A ) )
    {
        auto  DA = matrix::convert_to_dense( A );
        
        return frobenius_squared( alpha, *DA, beta, B );
    }// if
    else
        HLR_ERROR( "unsupported matrix types: " + A.typestr() + " and " + B.typestr() );

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//
// Spectral norm (|·|₂)
//
////////////////////////////////////////////////////////////////////////////////

//
// compute spectral norm of A via power iteration
//
template < arithmetic_type arithmetic_t,
           typename operator_t >
Hpro::real_type_t< Hpro::value_type_t< operator_t > >
spectral ( arithmetic_t &&     arithmetic,
           const operator_t &  A,
           const double        atol,
           const size_t        amax_it,
           const bool          squared )
{
    using  value_t = Hpro::value_type_t< operator_t >;
    using  real_t  = Hpro::real_type_t< value_t >;

    const auto  nrows_A = nrows( A );
    const auto  ncols_A = ncols( A );
    
    auto  x = blas::vector< value_t >( ncols_A );
    auto  y = blas::vector< value_t >( ncols_A );
    auto  t = blas::vector< value_t >( nrows_A );

    //
    // x = rand with |x| = 1
    //
    
    auto  generator     = std::default_random_engine();
    auto  uniform_distr = std::uniform_real_distribution< real_t >( -1.0, 1.0 );
    auto  random        = [&] () { return uniform_distr( generator ); };
    
    blas::fill_fn( x, random );
    blas::scale( value_t(1) / blas::norm_2(x), x );

    const size_t  max_it  = ( amax_it == 0 ? std::max( size_t(5), std::min( nrows_A, ncols_A ) / 10 ) : amax_it );
    const real_t  tol     = ( atol    == 0 ? std::sqrt( std::numeric_limits< real_t >::epsilon() ) : atol );
    const real_t  abs_tol = std::min( real_t(10) * std::numeric_limits< real_t >::epsilon(), tol );
    const real_t  zero    = math::square( std::numeric_limits< real_t >::epsilon() );
    real_t        lambda  = 1.0;
    
    for ( size_t  i = 0; i < max_it; i++ )
    {
        real_t  lambda_new = 0;
        real_t  norm_y     = 0;
        
        if ( squared )
        {
            blas::fill( t, value_t(0) );
            arithmetic.prod( value_t(1), apply_normal,  A, x, t );
            blas::fill( y, value_t(0) );
            arithmetic.prod( value_t(1), apply_adjoint, A, t, y );

            lambda_new = math::sqrt( math::abs( blas::dot( x, y ) ) );
            norm_y     = blas::norm_2( y );
        }// if
        else
        {
            arithmetic.prod( value_t(1), apply_normal,  A, x, y );
            norm_y = lambda_new = blas::norm_2( y );
        }// else

        log( 6, "λ" + subscript( i ) + " = " + Hpro::to_string( "%.4e (%.4e)", lambda_new, math::abs( ( lambda_new - lambda ) / lambda ) ) );
        
        // test against given tolerance
        if ( math::abs( ( lambda_new - lambda ) / lambda ) < tol )
            return lambda_new;

        // test for machine precision
        if (( i > 5 ) && ( math::abs( lambda_new - lambda ) < abs_tol ))
            return lambda_new;

        if ( lambda_new < zero )
            return lambda_new;
        
        lambda = lambda_new;

        if ( norm_y <= zero )
            break;
        
        blas::scale( value_t(1) / norm_y, y );
        blas::copy( y, x );
    }// for

    return lambda;
}

// template < typename value_t >
// Hpro::real_type_t< value_t >
// spectral< Hpro::TLinearOperator< value_t > > ( const Hpro::TLinearOperator< value_t > &  A,
//                                                const bool                                squared,
//                                                const double                              atol,
//                                                const size_t                              amax_it )
// {
//     using  real_t  = Hpro::real_type_t< value_t >;

//     auto  x = A.domain_vector();
//     auto  y = A.domain_vector();
//     auto  t = A.range_vector();

//     // x = rand with |x| = 1
//     x->fill_rand( 0 );
//     x->scale( real_t(1) / x->norm2() );

//     const size_t  max_it  = ( amax_it == 0 ? std::max( size_t(5), std::min( A.range_dim(), A.domain_dim() ) / 10 ) : amax_it );
//     const real_t  tol     = ( atol    == 0 ? std::sqrt( std::numeric_limits< real >::epsilon() ) : atol );
//     const real_t  abs_tol = std::min( real_t(1e1) * std::numeric_limits< real >::epsilon(), tol );
//     const real_t  zero    = std::numeric_limits< real >::epsilon() * std::numeric_limits< real >::epsilon();
//     real          lambda  = 1.0;
    
//     for ( uint i = 0; i < max_it; i++ )
//     {
//         real  lambda_new = 0;
//         real  norm_y     = 0;
        
//         if ( squared )
//         {
//             A.apply( x.get(), t.get(), apply_normal );
//             A.apply( t.get(), y.get(), apply_adjoint );

//             lambda_new = math::sqrt( math::abs( Hpro::dot( x.get(), y.get() ) ) );
//             norm_y     = y->norm2();
//         }// if
//         else
//         {
//             A.apply( x.get(), y.get(), apply_normal );
//             norm_y = lambda_new = y->norm2();
//         }// else

//         log( 6, "λ" + subscript( i ) + " = " + Hpro::to_string( "%.8e (%.8e)", lambda_new, std::abs( ( lambda_new - lambda ) / lambda ) ) );
        
//         // test against given tolerance
//         if ( std::abs( ( lambda_new - lambda ) / lambda ) < tol )
//             return lambda_new;

//         // test for machine precision
//         if (( i > 5 ) && ( std::abs( lambda_new - lambda ) < abs_tol ))
//             return lambda_new;

//         if ( lambda_new < zero )
//             return lambda_new;
        
//         lambda = lambda_new;

//         if ( norm_y <= zero )
//             break;
        
//         y->scale( real_t(1) / norm_y );
//         y->copy_to( x.get() );
//     }// for

//     return lambda;
// }

// template < typename value_t >
// Hpro::real_type_t< value_t >
// spectral< Hpro::TMatrix< value_t > > ( const Hpro::TMatrix< value_t > &  A,
//                                        const bool                        squared,
//                                        const real                        atol,
//                                        const size_t                      amax_it )
// {
//     return spectral< Hpro::TLinearOperator< value_t > >( A, squared, atol, amax_it );
// }

//
// compute inversion error of A vs A^-1 in spectral norm, e.g. |A-A^-1|_2
//
// double
// inv_error_2 ( const TMatrix< value_t > &  A,
//               const TMatrix< value_t > &  A_inv )
// {
//     auto  x     = A->domain_vector();
//     auto  x_old = A->domain_vector();
//     auto  y     = A->range_vector();
    
//     x->fill_rand(1);
    
//     // normalise x
//     x->scale( real_t(1) / x->norm2() );

//     complex  lambda     = 0.0;
//     complex  lambda_old = 1.0;
    
//     for ( uint i = 0; i < _max_it; i++ )
//     {
//         x_old->assign( real_t(1), x.get() );

//         apply(               A, x.get(), y.get() );
//         apply_add( real_t(-1), B, x.get(), y.get() );

//         apply(               A, y.get(), x.get(), apply_adjoint );
//         apply_add( real_t(-1), B, y.get(), x.get(), apply_adjoint );

//         const auto  lambda = Math::abs( Math::sqrt( dot( x_old.get(), x.get() ) ) );

//         HLR_LOG( 4, Hpro::to_string( "%3d : %.6e", i, lambda ) );

//         const real  norm_x = x->norm2();
            
//         if ( norm_x <= math::square( Limits::epsilon< Hpro::real >() ); )
//             break;
        
//         x->scale( real_t(1) / norm_x );
        
//         if ( converged( lambda, lambda_old, i ) )
//             break;

//         lambda_old = lambda;
//     }// for

//     return Math::abs( Math::sqrt( lambda ) );
// }

}}}// namespace hlr::norm::detail

#endif // __HLR_ARITH_DETAIL_NORM_HH
