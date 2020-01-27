#ifndef __HLR_BEM_ACA_HH
#define __HLR_BEM_ACA_HH
//
// Project     : HLR
// File        : aca.hh
// Description : various ACA algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>
#include <limits>

#include <boost/math/constants/constants.hpp>

#include <hpro/blas/Algebra.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

// represents array of pivot elements
using  pivot_arr_t = std::vector< std::pair< idx_t, idx_t > >;

//
// return pivot elements of ACA-Full applied to <M> with
// precision <eps>
//
template < typename value_t >
pivot_arr_t
aca_full_pivots  ( blas::Matrix< value_t > &                          M,
                   const typename hpro::real_type< value_t >::type_t  eps )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    //
    // perform ACA-Full on matrix, e.g. choosing maximal element of matrix
    // and compute next rank-1 matrix for low-rank approximation
    //
    
    const size_t             max_rank    = std::min( M.nrows(), M.ncols() );
    size_t                   k           = 0;
    const auto               almost_zero = std::numeric_limits< real_t >::epsilon();
    real_t                   apr         = eps;
    blas::Vector< value_t >  row( M.nrows() );
    blas::Vector< value_t >  col( M.ncols() );
    pivot_arr_t              pivots;
                
    pivots.reserve( max_rank );
    
    while ( k < max_rank )
    {
        //
        // look for maximal element
        //

        idx_t  pivot_row, pivot_col;

        blas::max_idx( M, pivot_row, pivot_col );

        const value_t  pivot_val = M( pivot_row, pivot_col );

        // stop if maximal element is almost 0
        if ( std::abs( pivot_val ) < almost_zero )
            return pivots;
        
        //
        // copy row and column into A/B and update M
        //

        const auto  M_row = M.row( pivot_row );
        const auto  M_col = M.column( pivot_col );

        blas::copy( M_row, row );
        blas::copy( M_col, col );
        
        blas::conj( row );
        blas::scale( value_t(1) / conj(pivot_val), row );
        
        //
        // look at norm of residual
        //
            
        const auto  norm = blas::norm2( col ) * blas::norm2( row );
                
        pivots.push_back( { pivot_row, pivot_col } );
        ++k;
            
        if ( k == 1 )
        {
            // adjust stop criterion with norm-estimate of initial matrix
            apr *= norm;
        }// if
        else if ( norm < apr ) 
        {
            return pivots;
        }// else

        //
        // update dense matrix
        //
        
        blas::add_r1( value_t(-1), col, row, M );
    }// while

    return pivots;
}

}}// namespace hlr::bem

#endif // __HLR_BEM_ACA_HH
