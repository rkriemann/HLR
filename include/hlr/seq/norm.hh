#ifndef __HLR_SEQ_NORM_HH
#define __HLR_SEQ_NORM_HH
//
// Project     : HLib
// File        : norm.hh
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/blas/Algebra.hh>

#include "hlr/arith/blas.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/text.hh"

namespace hlr { namespace seq { namespace norm {

namespace hpro = HLIB;

//
// return Frobenius norm of A, e.g. |A|_F
//
inline
double
norm_F ( const hpro::TMatrix &  A )
{
    if ( is_blocked( A ) )
    {
        auto    B   = cptrcast( &A, hpro::TBlockMatrix );
        double  val = 0.0;
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                const auto  val_ij = norm_F( * B->block( i, j ) );

                val += val_ij * val_ij;
            }// for
        }// for

        return std::sqrt( val );
    }// if
    else if ( is_lowrank( A ) )
    {
        auto  R = cptrcast( &A, hpro::TRkMatrix );
        
        //
        // ∑_ij (R_ij)² = ∑_ij (∑_k U_ik V_jk')²
        //              = ∑_ij (∑_k U_ik V_jk') (∑_l U_il V_jl')'
        //              = ∑_ij ∑_k ∑_l U_ik V_jk' U_il' V_jl
        //              = ∑_k ∑_l ∑_i U_ik U_il' ∑_j V_jk' V_jl
        //              = ∑_k ∑_l (U_l)^H · U_k  V_k^H · V_l
        //

        if ( R->is_complex() )
        {
            assert( false );
        }// if
        else
        {
            const auto  U   = hpro::blas_mat_A< hpro::real >( R );
            const auto  V   = hpro::blas_mat_B< hpro::real >( R );
            double      val = 0.0;
    
            for ( size_t  l = 0; l < R->rank(); l++ )
            {
                const auto  U_l = U.column( l );
                const auto  V_l = V.column( l );
                
                for ( size_t  k = 0; k < R->rank(); k++ )
                {
                    const auto  U_k = U.column( k );
                    const auto  V_k = V.column( k );
                    
                    val += blas::dot( U_l, U_k ) * blas::dot( V_l, V_k );
                }// for
            }// for

            return std::sqrt( val );
        }// else
    }// if
    else if ( IS_TYPE( &A, tiled_lrmatrix ) )
    {
        //
        // ∑_ij (R_ij)² = ∑_ij (∑_k U_ik V_jk')²
        //              = ∑_ij (∑_k U_ik V_jk') (∑_l U_il V_jl')'
        //              = ∑_ij ∑_k ∑_l U_ik V_jk' U_il' V_jl
        //              = ∑_k ∑_l ∑_i U_ik U_il' ∑_j V_jk' V_jl
        //              = ∑_k ∑_l (U_l)^H · U_k  V_k^H · V_l
        //

        if ( A.is_complex() )
        {
            assert( false );
        }// if
        else
        {
            auto          R   = cptrcast( & A, hlr::matrix::tiled_lrmatrix< hpro::real > );
            const auto &  U   = R->U();
            const auto &  V   = R->V();
            double        val = 0.0;
    
            for ( size_t  l = 0; l < R->rank(); l++ )
            {
                for ( size_t  k = 0; k < R->rank(); k++ )
                {
                    double  dot_U = 0;
                    double  dot_V = 0;

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

            return std::sqrt( val );
        }// else
    }// if
    else if ( is_dense( A ) )
    {
        if ( A.is_complex() )
            return blas::normF( hpro::blas_mat< hpro::complex >( cptrcast( &A, hpro::TDenseMatrix ) ) );
        else
            return blas::normF( hpro::blas_mat< hpro::real >( cptrcast( &A, hpro::TDenseMatrix ) ) ); 
    }// if
    else
    {
        HLR_ASSERT( is_blocked( A ) || is_lowrank( A ) || is_dense( A ) );
    }// else

    return 0;
}

//
// return Frobenius norm of αA+βB, e.g. |αA+βB|_F
//
inline
double
norm_F ( const double           alpha,
         const hpro::TMatrix &  A,
         const double           beta,
         const hpro::TMatrix &  B )
{
    assert( A.block_is()   == B.block_is() );
    assert( A.is_complex() == B.is_complex() );
    
    if ( is_blocked_all( A, B ) )
    {
        auto    BA   = cptrcast( &A, hpro::TBlockMatrix );
        auto    BB   = cptrcast( &B, hpro::TBlockMatrix );
        double  val = 0.0;

        assert(( BA->nblock_rows() == BB->block_rows() ) &&
               ( BA->nblock_cols() == BB->block_cols() ));
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                const auto  val_ij = norm_F( alpha, * BA->block( i, j ),
                                             beta,  * BB->block( i, j ) );

                val += val_ij * val_ij;
            }// for
        }// for

        return std::sqrt( val );
    }// if
    else if ( is_lowrank_all( A, B ) )
    {
        auto  RA = cptrcast( &A, hpro::TRkMatrix );
        auto  RB = cptrcast( &B, hpro::TRkMatrix );
        
        if ( RA->is_complex() )
        {
            assert( false );
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
                              double      val   = 0.0;
                              
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

            const auto  UA  = hpro::blas_mat_A< hpro::real >( RA );
            const auto  VA  = hpro::blas_mat_B< hpro::real >( RA );
            const auto  UB  = hpro::blas_mat_A< hpro::real >( RB );
            const auto  VB  = hpro::blas_mat_B< hpro::real >( RB );

            return std::sqrt( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                              alpha * beta  * lrdot( UA, VA, UB, VB ) +
                              alpha * beta  * lrdot( UB, VB, UA, VA ) +
                              beta  * beta  * lrdot( UB, VB, UB, VB ) );
        }// else
    }// if
    else if ( is_dense_all( A, B ) )
    {
        auto  DA = cptrcast( &A, hpro::TDenseMatrix );
        auto  DB = cptrcast( &B, hpro::TDenseMatrix );
        
        if ( A.is_complex() )
        {
            auto         MA  = hpro::blas_mat< hpro::complex >( DA );
            auto         MB  = hpro::blas_mat< hpro::complex >( DB );
            double       val = 0;
            const idx_t  n   = idx_t(MA.nrows());
            const idx_t  m   = idx_t(MA.ncols());
    
            for ( idx_t j = 0; j < m; ++j )
            {
                for ( idx_t i = 0; i < n; ++i )
                {
                    const auto  a_ij = alpha * MA(i,j) + beta * MB(i,j);
                    
                    val += re( hpro::conj( a_ij ) * a_ij );
                }// for
            }// for

            return std::sqrt( val );
        }// if
        else
        {
            auto         MA  = hpro::blas_mat< hpro::real >( DA );
            auto         MB  = hpro::blas_mat< hpro::real >( DB );
            double       val = 0;
            const idx_t  n   = idx_t(MA.nrows());
            const idx_t  m   = idx_t(MA.ncols());
    
            for ( idx_t j = 0; j < m; ++j )
            {
                for ( idx_t i = 0; i < n; ++i )
                {
                    const auto  a_ij = alpha * MA(i,j) + beta * MB(i,j);
                    
                    val += a_ij * a_ij;
                }// for
            }// for

            return std::sqrt( val );
        }// else
    }// if
    else
    {
        HLR_ASSERT( is_blocked_all( A, B ) || is_lowrank_all( A, B ) || is_dense_all( A, B ) );
    }// else

    return 0;
}

//
// compute spectral norm of A, e.g. |A|_2, via power iteration
//
template < typename matrix_t >
double
norm_2 ( const matrix_t &  A,
         const bool        squared = true,
         const real        atol    = 0,
         const size_t      amax_it = 0 )
{
    auto  x = A.domain_vector();
    auto  y = A.domain_vector();
    auto  t = A.range_vector();

    // x = rand with |x| = 1
    x->fill_rand( 0 );
    x->scale( real(1) / x->norm2() );

    const size_t  max_it  = ( amax_it == 0 ? std::min( A.range_dim(), A.domain_dim() ) / 10 : amax_it );
    const real    tol     = ( atol    == 0 ? std::sqrt( std::numeric_limits< real >::epsilon() ) : atol );
    const real    abs_tol = std::min( 1e1 * std::numeric_limits< real >::epsilon(), tol );
    const real    zero    = std::numeric_limits< real >::epsilon() * std::numeric_limits< real >::epsilon();
    real          lambda  = 1.0;
    
    for ( uint i = 0; i < max_it; i++ )
    {
        real  lambda_new = 0;
        real  norm_y     = 0;
        
        if ( squared )
        {
            A.apply( x.get(), t.get(), hpro::apply_normal );
            A.apply( t.get(), y.get(), hpro::apply_adjoint );

            lambda_new = hpro::Math::abs( hpro::Math::sqrt( hpro::dot( x.get(), y.get() ) ) );
            norm_y     = y->norm2();
        }// if
        else
        {
            A.apply( x.get(), y.get(), hpro::apply_normal );
            norm_y = lambda_new = y->norm2();
        }// else

        log( 6, "λ" + subscript( i ) + " = " + hpro::to_string( "%.4e (%.4e)", lambda_new, std::abs( ( lambda_new - lambda ) / lambda ) ) );
        
        // test against given tolerance
        if ( std::abs( ( lambda_new - lambda ) / lambda ) < tol )
            return lambda_new;

        // test for machine precision
        if (( i > 5 ) && ( std::abs( lambda_new - lambda ) < abs_tol ))
            return lambda_new;

        if ( lambda_new < zero )
            return lambda_new;
        
        lambda = lambda_new;

        if ( norm_y <= zero )
            break;
        
        y->scale( real(1) / norm_y );
        y->copy_to( x.get() );
    }// for

    return lambda;
}

//
// compute inversion error of A vs A^-1 in spectral norm, e.g. |A-A^-1|_2
//
// double
// inv_error_2 ( const TMatrix &  A,
//               const TMatrix &  A_inv )
// {
//     auto  x     = A->domain_vector();
//     auto  x_old = A->domain_vector();
//     auto  y     = A->range_vector();
    
//     x->fill_rand(1);
    
//     // normalise x
//     x->scale( real(1) / x->norm2() );

//     complex  lambda     = 0.0;
//     complex  lambda_old = 1.0;
    
//     for ( uint i = 0; i < _max_it; i++ )
//     {
//         x_old->assign( real(1), x.get() );

//         apply(               A, x.get(), y.get() );
//         apply_add( real(-1), B, x.get(), y.get() );

//         apply(               A, y.get(), x.get(), apply_adjoint );
//         apply_add( real(-1), B, y.get(), x.get(), apply_adjoint );

//         const auto  lambda = Math::abs( Math::sqrt( dot( x_old.get(), x.get() ) ) );

//         HLR_LOG( 4, hpro::to_string( "%3d : %.6e", i, lambda ) );

//         const real  norm_x = x->norm2();
            
//         if ( norm_x <= Math::square( Limits::epsilon< hpro::real >() ); )
//             break;
        
//         x->scale( real(1) / norm_x );
        
//         if ( converged( lambda, lambda_old, i ) )
//             break;

//         lambda_old = lambda;
//     }// for

//     return Math::abs( Math::sqrt( lambda ) );
// }

}}}// namespace hlr::seq::norm

#endif // __HLR_SEQ_NORM_HH
