#ifndef __HLR_OMP_NORM_HH
#define __HLR_OMP_NORM_HH
//
// Project     : HLR
// Module      : omp/norm
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <mutex>

#include "hlr/arith/norm.hh"

namespace hlr { namespace omp { namespace norm {

namespace detail
{

template < typename value_t >
long double
frobenius_squared ( const Hpro::TMatrix< value_t > &  A )
{
    // using  real_t = Hpro::real_type_t< value_t >;
    using  result_t = long double;
    
    if ( is_blocked( A ) )
    {
        auto      B   = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto      mtx = std::mutex();
        result_t  val = 0.0;
        
        #pragma omp taskloop collapse(2) default(shared)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                {
                    const auto  val_ij = frobenius_squared( * B->block( i, j ) );
                    
                    HLR_ASSERT( std::isfinite( val_ij ) );
                    
                    {
                        auto  lock = std::scoped_lock( mtx );
                        
                        val += val_ij;
                    }
                }// if
            }// for
        }// for
        
        return val;
    }// if
    else
        return hlr::norm::detail::frobenius_squared( A );
}

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
    
    if ( hlr::is_blocked_all( A, B ) )
    {
        auto         BA   = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto         BB   = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        result_t  val = 0.0;
        auto         mtx = std::mutex();

        HLR_ASSERT(( BA->nblock_rows() == BB->block_rows() ) &&
                   ( BA->nblock_cols() == BB->block_cols() ));
        
        #pragma omp taskloop collapse(2) default(shared)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                auto  val_ij = result_t(0);
                
                if ( is_null( BA->block( i, j ) ) )
                {
                    if ( is_null( BB->block( i, j ) ) )
                        continue;
                    else
                        val_ij = beta * frobenius_squared( * BB->block( i, j ) );
                }// if
                else
                {
                    if ( is_null( BB->block( i, j ) ) )
                        val_ij = alpha * frobenius_squared( * BA->block( i, j ) );
                    else
                        val_ij = frobenius_squared( alpha, * BA->block( i, j ),
                                                    beta,  * BB->block( i, j ) );
                }// else
                
                HLR_ASSERT( std::isfinite( val_ij ) );
                
                {
                    auto  lock = std::scoped_lock( mtx );
                    
                    val += val_ij;
                }
            }// for
        }// for

        return std::abs( val );
    }// if
    else
        return hlr::norm::detail::frobenius_squared( alpha, A, beta, B );
}

}// namespace detail

template < typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const Hpro::TMatrix< value_t > &  A )
{
    return std::sqrt( detail::frobenius_squared( A ) );
}

//
// return Frobenius norm of αA+βB, e.g. |αA+βB|_F
//
template < general_number alpha_t,
           general_number beta_t,
           typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const alpha_t                     alpha,
            const Hpro::TMatrix< value_t > &  A,
            const beta_t                      beta,
            const Hpro::TMatrix< value_t > &  B )
{
    return std::sqrt( detail::frobenius_squared( alpha, A, beta, B ) );
}

using hlr::norm::spectral;

}}}// namespace hlr::omp::norm

#endif // __HLR_OMP_NORM_HH
