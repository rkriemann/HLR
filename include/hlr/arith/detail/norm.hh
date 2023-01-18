#ifndef __HLR_ARITH_DETAIL_NORM_HH
#define __HLR_ARITH_DETAIL_NORM_HH
//
// Project     : HLib
// Module      : norm
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <random>

#include "hlr/arith/blas.hh"
#include "hlr/arith/operator_wrapper.hh"

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"

#include "hlr/utils/log.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/text.hh"
#include "hlr/utils/math.hh"

namespace hlr { namespace norm { namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// Frobenius norm
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const Hpro::TMatrix< value_t > &  A )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    if ( is_blocked( A ) )
    {
        auto    B   = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        real_t  val = 0.0;
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( is_null( B->block( i, j ) ) )
                    continue;
                
                const auto  val_ij = frobenius( * B->block( i, j ) );

                val += val_ij * val_ij;
            }// for
        }// for

        return std::sqrt( val );
    }// if
    else if ( is_lowrank( A ) )
    {
        //
        // ∑_ij (R_ij)² = ∑_ij (∑_k U_ik V_jk')²
        //              = ∑_ij (∑_k U_ik V_jk') (∑_l U_il V_jl')'
        //              = ∑_ij ∑_k ∑_l U_ik V_jk' U_il' V_jl
        //              = ∑_k ∑_l ∑_i U_ik U_il' ∑_j V_jk' V_jl
        //              = ∑_k ∑_l (U_l)^H · U_k  V_k^H · V_l
        //

        auto  comp_lr = [] ( const blas::matrix< value_t > &  U,
                             const blas::matrix< value_t > &  V )
        {
            const uint  rank = U.ncols();
            real_t      val  = 0.0;
    
            for ( size_t  l = 0; l < rank; l++ )
            {
                const auto  U_l = U.column( l );
                const auto  V_l = V.column( l );
            
                for ( size_t  k = 0; k < rank; k++ )
                {
                    const auto  U_k = U.column( k );
                    const auto  V_k = V.column( k );
                
                    val += blas::dot( U_l, U_k ) * blas::dot( V_l, V_k );
                }// for
            }// for
        
            return std::sqrt( std::abs( val ) );
        };

        if ( matrix::is_compressed( A ) )
        {
            return comp_lr( cptrcast( &A, matrix::lrmatrix< value_t > )->U_decompressed(),
                            cptrcast( &A, matrix::lrmatrix< value_t > )->V_decompressed() );
        }// if
        else
        {
            return comp_lr( cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_A(),
                            cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_B() );
        }// if
    }// if
    // else if ( matrix::is_generic_lowrank( A ) )
    // {
    //     auto  R = cptrcast( &A, matrix::lrmatrix );
        
    //     //
    //     // ∑_ij (R_ij)² = ∑_ij (∑_k U_ik V_jk')²
    //     //              = ∑_ij (∑_k U_ik V_jk') (∑_l U_il V_jl')'
    //     //              = ∑_ij ∑_k ∑_l U_ik V_jk' U_il' V_jl
    //     //              = ∑_k ∑_l ∑_i U_ik U_il' ∑_j V_jk' V_jl
    //     //              = ∑_k ∑_l (U_l)^H · U_k  V_k^H · V_l
    //     //

    //     const auto  UV  = R->factors();

    //     return std::visit(
    //         [rank=R->rank()] ( auto &&  UV ) -> real_t
    //         {
    //             using  value_t = typename std::decay_t< decltype(UV) >::value_t;
                
    //             const auto  U    = UV.U;
    //             const auto  V    = UV.V;
    //             value_t     norm = value_t(0);
                
    //             for ( size_t  l = 0; l < rank; l++ )
    //             {
    //                 const auto  U_l = U.column( l );
    //                 const auto  V_l = V.column( l );
                    
    //                 for ( size_t  k = 0; k < rank; k++ )
    //                 {
    //                     const auto  U_k = U.column( k );
    //                     const auto  V_k = V.column( k );
                        
    //                     norm += blas::dot( U_l, U_k ) * blas::dot( V_l, V_k );
    //                 }// for
    //             }// for
                
    //             return std::real( std::sqrt( std::abs( norm ) ) );
    //         },
    //         R->factors()
    //     );
    // }// if
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
        real_t        val = 0.0;
        
        for ( size_t  l = 0; l < R->rank(); l++ )
        {
            for ( size_t  k = 0; k < R->rank(); k++ )
            {
                real_t  dot_U = 0;
                real_t  dot_V = 0;

                auto  U_i = U.cbegin();
                auto  V_i = V.cbegin();

                for ( ; ( U_i != U.cend() ) && ( V_i != V.cend() ); ++U_i, ++V_i )
                {
                    const auto  U_l = (*U_i).second.column( l );
                    const auto  V_l = (*V_i).second.column( l );
                
                    const auto  U_k = (*U_i).second.column( k );
                    const auto  V_k = (*V_i).second.column( k );
                    
                    dot_U += blas::dot( U_l, U_k );
                    dot_V += blas::dot( V_l, V_k );
                }// for

                val += dot_U * dot_V;
            }// for
        }// for

        return std::sqrt( std::abs( val ) );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( A ) )
    {
        //
        // |A| = | U S V' | = |U||S||V| = |S|
        //

        auto  R = cptrcast( &A, hlr::matrix::uniform_lrmatrix< value_t > );
        
        return blas::norm2( R->coeff() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( matrix::is_compressed( A ) )
            return blas::normF( cptrcast( &A, matrix::dense_matrix< value_t > )->mat_decompressed() );
        else
            return blas::normF( blas::mat( cptrcast( &A, Hpro::TDenseMatrix< value_t > ) ) );
    }// if
    // else if ( matrix::is_generic_dense( A ) )
    // {
    //     return std::visit( [] ( auto &&  M ) -> real_t { return blas::normF( M ); },
    //                        cptrcast( &A, matrix::dense_matrix )->matrix() ); 
    // }// if
    else
    {
        HLR_ASSERT( is_blocked( A ) || is_lowrank( A ) || is_dense( A ) );
    }// else

    return 0;
}

//
// return Frobenius norm of αA+βB, e.g. |αA+βB|_F
//
template < typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const value_t                     alpha,
            const Hpro::TMatrix< value_t > &  A,
            const value_t                     beta,
            const Hpro::TMatrix< value_t > &  B )
{
    using  real_t = Hpro::real_type_t< value_t >;

    assert( A.block_is()   == B.block_is() );
    
    if ( is_blocked_all( A, B ) )
    {
        auto    BA   = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto    BB   = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        real_t  val = 0.0;

        assert(( BA->nblock_rows() == BB->block_rows() ) &&
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
                        val += math::square( beta * frobenius( * BB->block( i, j ) ) );
                }// if
                else
                {
                    if ( is_null( BB->block( i, j ) ) )
                        val += math::square( alpha * frobenius( * BA->block( i, j ) ) );
                    else
                        val += math::square( frobenius( alpha, * BA->block( i, j ),
                                                        beta,  * BB->block( i, j ) ) );
                }// else
            }// for
        }// for

        return std::sqrt( val );
    }// if
    else if ( is_lowrank_all( A, B ) )
    {
        if ( A.is_complex() )
        {
            HLR_ERROR( "TODO" );
        }// if
        else
        {
            auto  lrdot = [] ( const auto &  U1,
                               const auto &  V1,
                               const auto &  U2,
                               const auto &  V2 )
                          {
                              const auto  rank1 = U1.ncols();
                              const auto  rank2 = U2.ncols();
                              real_t      val   = 0.0;
                              
                              for ( size_t  l = 0; l < rank1; l++ )
                              {
                                  const auto  U1_l = U1.column( l );
                                  const auto  V1_l = V1.column( l );
                                  
                                  for ( size_t  k = 0; k < rank2; k++ )
                                  {
                                      const auto  U2_k = U2.column( k );
                                      const auto  V2_k = V2.column( k );
                                      
                                      val += blas::dot( U1_l, U2_k ) * blas::dot( V1_l, V2_k );
                                  }// for
                              }// for

                              return val;
                          };

            auto  UA = blas::matrix< value_t >();
            auto  VA = blas::matrix< value_t >();
            auto  UB = blas::matrix< value_t >();
            auto  VB = blas::matrix< value_t >();
            
            if ( matrix::is_compressed( A )  )
            {
                if ( matrix::is_compressed( B ) )
                {
                    const auto  UA = cptrcast( &A, matrix::lrmatrix< value_t > )->U_decompressed();
                    const auto  VA = cptrcast( &A, matrix::lrmatrix< value_t > )->V_decompressed();
                    const auto  UB = cptrcast( &B, matrix::lrmatrix< value_t > )->U_decompressed();
                    const auto  VB = cptrcast( &B, matrix::lrmatrix< value_t > )->V_decompressed();
                    const auto  sqn = ( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                                        alpha * beta  * lrdot( UA, VA, UB, VB ) +
                                        alpha * beta  * lrdot( UB, VB, UA, VA ) +
                                        beta  * beta  * lrdot( UB, VB, UB, VB ) );

                    return std::sqrt( std::abs( sqn ) );
                }// if
                else
                {
                    const auto  UA = cptrcast( &A, matrix::lrmatrix< value_t > )->U_decompressed();
                    const auto  VA = cptrcast( &A, matrix::lrmatrix< value_t > )->V_decompressed();
                    const auto  UB = cptrcast( &B, Hpro::TRkMatrix< value_t > )->blas_mat_A();
                    const auto  VB = cptrcast( &B, Hpro::TRkMatrix< value_t > )->blas_mat_B();
                    const auto  sqn = ( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                                        alpha * beta  * lrdot( UA, VA, UB, VB ) +
                                        alpha * beta  * lrdot( UB, VB, UA, VA ) +
                                        beta  * beta  * lrdot( UB, VB, UB, VB ) );

                    return std::sqrt( std::abs( sqn ) );
                }// else
            }// if
            else
            {
                if ( matrix::is_compressed( B ) )
                {
                    const auto  UA = cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_A();
                    const auto  VA = cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_B();
                    const auto  UB = cptrcast( &B, matrix::lrmatrix< value_t > )->U_decompressed();
                    const auto  VB = cptrcast( &B, matrix::lrmatrix< value_t > )->V_decompressed();
                    const auto  sqn = ( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                                        alpha * beta  * lrdot( UA, VA, UB, VB ) +
                                        alpha * beta  * lrdot( UB, VB, UA, VA ) +
                                        beta  * beta  * lrdot( UB, VB, UB, VB ) );

                    return std::sqrt( std::abs( sqn ) );
                }// if
                else
                {
                    const auto  UA = cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_A();
                    const auto  VA = cptrcast( &A, Hpro::TRkMatrix< value_t > )->blas_mat_B();
                    const auto  UB = cptrcast( &B, Hpro::TRkMatrix< value_t > )->blas_mat_A();
                    const auto  VB = cptrcast( &B, Hpro::TRkMatrix< value_t > )->blas_mat_B();
                    const auto  sqn = ( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                                        alpha * beta  * lrdot( UA, VA, UB, VB ) +
                                        alpha * beta  * lrdot( UB, VB, UA, VA ) +
                                        beta  * beta  * lrdot( UB, VB, UB, VB ) );

                    return std::sqrt( std::abs( sqn ) );
                }// else
            }// else
        }// else
    }// if
    else if ( is_dense_all( A, B ) )
    {
        auto  comp_dense = [alpha,beta] ( const blas::matrix< value_t > &  MA,
                                          const blas::matrix< value_t > &  MB )
        {
            real_t       val = 0;
            const idx_t  n   = idx_t(MA.nrows());
            const idx_t  m   = idx_t(MA.ncols());
    
            for ( idx_t j = 0; j < m; ++j )
            {
                for ( idx_t i = 0; i < n; ++i )
                {
                    const auto  a_ij = alpha * MA(i,j) + beta * MB(i,j);
                    
                    val += std::real( math::conj( a_ij ) * a_ij );
                }// for
            }// for
            
            return std::sqrt( std::abs( val ) );
        };

        if ( matrix::is_compressed( A ) )
        {
            if ( matrix::is_compressed( B ) )
            {
                return comp_dense( cptrcast( &A, matrix::dense_matrix< value_t > )->mat_decompressed(),
                                   cptrcast( &B, matrix::dense_matrix< value_t > )->mat_decompressed() );
            }// if
            else
            {
                return comp_dense( cptrcast( &A, matrix::dense_matrix< value_t > )->mat_decompressed(),
                                   cptrcast( &B, Hpro::TDenseMatrix< value_t > )->blas_mat() );
            }// else
        }// if
        else
        {
            if ( matrix::is_compressed( B ) )
            {
                return comp_dense( cptrcast( &A, Hpro::TDenseMatrix< value_t > )->blas_mat(),
                                   cptrcast( &B, matrix::dense_matrix< value_t > )->mat_decompressed() );
            }// if
            else
            {
                return comp_dense( cptrcast( &A, Hpro::TDenseMatrix< value_t > )->blas_mat(),
                                   cptrcast( &B, Hpro::TDenseMatrix< value_t > )->blas_mat() );
            }// else
        }// else
    }// if
    else
    {
        HLR_ASSERT( is_blocked_all( A, B ) || is_lowrank_all( A, B ) || is_dense_all( A, B ) );
    }// else

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
template < typename arithmetic_t,
           typename operator_t >
requires provides_arithmetic< arithmetic_t >
Hpro::real_type_t< Hpro::value_type_t< operator_t > >
//std::enable_if_t< hlr::is_arithmetic_v< arithmetic_t >, Hpro::real_type_t< Hpro::value_type_t< operator_t > > >
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
