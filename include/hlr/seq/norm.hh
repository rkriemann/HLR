#ifndef __HLR_SEQ_NORM_HH
#define __HLR_SEQ_NORM_HH
//
// Project     : HLib
// File        : norm.hh
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TMatrix.hh>
#include <blas/Algebra.hh>

#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/checks.hh"

namespace hlr { namespace seq { namespace norm {

using namespace HLIB;

//
// return Frobenius norm of A, e.g. |A|_F
//
double
norm_F ( const TMatrix &  A )
{
    if ( is_blocked( A ) )
    {
        auto    B   = cptrcast( &A, TBlockMatrix );
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
        auto  R = cptrcast( &A, TRkMatrix );
        
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
            const auto  U   = blas_mat_A< HLIB::real >( R );
            const auto  V   = blas_mat_B< HLIB::real >( R );
            double      val = 0.0;
    
            for ( size_t  l = 0; l < R->rank(); l++ )
            {
                const auto  U_l = U.column( l );
                const auto  V_l = V.column( l );
                
                for ( size_t  k = 0; k < R->rank(); k++ )
                {
                    const auto  U_k = U.column( k );
                    const auto  V_k = V.column( k );
                    
                    val += BLAS::dot( U_l, U_k ) * BLAS::dot( V_l, V_k );
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
            auto          R   = cptrcast( & A, hlr::matrix::tiled_lrmatrix< HLIB::real > );
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
                    
                        dot_U += BLAS::dot( U_l, U_k );
                        dot_V += BLAS::dot( V_l, V_k );
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
            return BLAS::normF( blas_mat< HLIB::complex >( cptrcast( &A, TDenseMatrix ) ) );
        else
            return BLAS::normF( blas_mat< HLIB::real >( cptrcast( &A, TDenseMatrix ) ) ); 
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
double
norm_F ( const double     alpha,
         const TMatrix &  A,
         const double     beta,
         const TMatrix &  B )
{
    assert( A.block_is()   == B.block_is() );
    assert( A.is_complex() == B.is_complex() );
    
    if ( is_blocked_all( A, B ) )
    {
        auto    BA   = cptrcast( &A, TBlockMatrix );
        auto    BB   = cptrcast( &B, TBlockMatrix );
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
        auto  RA = cptrcast( &A, TRkMatrix );
        auto  RB = cptrcast( &B, TRkMatrix );
        
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
                                      
                                      val += BLAS::dot( U1_l, U2_k ) * BLAS::dot( V1_l, V2_k );
                                  }// for
                              }// for

                              return val;
                          };

            const auto  UA  = blas_mat_A< HLIB::real >( RA );
            const auto  VA  = blas_mat_B< HLIB::real >( RA );
            const auto  UB  = blas_mat_A< HLIB::real >( RB );
            const auto  VB  = blas_mat_B< HLIB::real >( RB );

            return std::sqrt( alpha * alpha * lrdot( UA, VA, UA, VA ) +
                              alpha * beta  * lrdot( UA, VA, UB, VB ) +
                              alpha * beta  * lrdot( UB, VB, UA, VA ) +
                              beta  * beta  * lrdot( UB, VB, UB, VB ) );
        }// else
    }// if
    else if ( is_dense_all( A, B ) )
    {
        auto  DA = cptrcast( &A, TDenseMatrix );
        auto  DB = cptrcast( &B, TDenseMatrix );
        
        if ( A.is_complex() )
        {
            auto         MA  = blas_mat< HLIB::complex >( DA );
            auto         MB  = blas_mat< HLIB::complex >( DB );
            double       val = 0;
            const idx_t  n   = idx_t(MA.nrows());
            const idx_t  m   = idx_t(MA.ncols());
    
            for ( idx_t j = 0; j < m; ++j )
            {
                for ( idx_t i = 0; i < n; ++i )
                {
                    const auto  a_ij = alpha * MA(i,j) + beta * MB(i,j);
                    
                    val += re( HLIB::conj( a_ij ) * a_ij );
                }// for
            }// for

            return std::sqrt( val );
        }// if
        else
        {
            auto         MA  = blas_mat< HLIB::real >( DA );
            auto         MB  = blas_mat< HLIB::real >( DB );
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
// compute spectral norm of A, e.g. |A|_2
//
// double
// norm_2 ( const TMatrix &  A )
// {
//     BLAS::Vector< real >  x(     A.ncols() );
//     BLAS::Vector< real >  x_old( A.ncols() );
//     BLAS::Vector< real >  y(     A.nrows() );

//     std::random_device          rd{};
//     std::mt19937                generator{ rd() };
//     std::normal_distribution<>  distr{ 0, 1 };
    
//     BLAS::fill( x, [&] () { return distr( generator ); } );
    
//     // normalise x
//     BLAS::scale( real(1) / BLAS::norm_2( x ),  );
    
//     real  lambda_old = 1.0;
    
//     for ( uint i = 0; i < _max_it; i++ )
//     {
//         x_old->assign( real(1), x.get() );

//         mulvec( apply_normal,  A, x, y );
//         mulvec( apply_adjoint, A, y, x ); 

//         const auto  lambda = Math::abs( Math::sqrt( dot( x_old, x ) ) );

//         HLR_LOG( 4, HLIB::to_string( "%3d : %.6e", i, lambda ) );
        
//         const auto  norm_x = BLAS::norm_2( x );
        
//         if ( norm_x <= Math::square( Limits::epsilon< real >() ) )
//             break;
        
//         BLAS::scale( real(1) / norm_x, x );

//         if (( Math::abs( ( lambda - lambda_old ) / lambda_old ) < 1e-4 ) ||
//             (( i > 5 ) && ( Math::abs( lambda - lambda_old ) < Math::square( Limits::epsilon< real >() ) )))
//             return lambda;

//         lambda_old = lambda;
//     }// for

//     return lambda_old;
// }

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

//         HLR_LOG( 4, HLIB::to_string( "%3d : %.6e", i, lambda ) );

//         const real  norm_x = x->norm2();
            
//         if ( norm_x <= Math::square( Limits::epsilon< HLIB::real >() ); )
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
