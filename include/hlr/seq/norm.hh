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
                                  const auto  U_l = U1.column( l );
                                  const auto  V_l = V1.column( l );
                                  
                                  for ( size_t  k = 0; k < rank2; k++ )
                                  {
                                      const auto  U_k = U2.column( k );
                                      const auto  V_k = V2.column( k );
                                      
                                      val += BLAS::dot( U_l, U_k ) * BLAS::dot( V_l, V_k );
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
                    const auto  a_ij = alpha * MA(i,j) - beta * MB(i,j);
                    
                    val += re( HLIB::conj( a_ij ) * a_ij );
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

}}}// namespace hlr::seq::norm

#endif // __HLR_SEQ_NORM_HH
