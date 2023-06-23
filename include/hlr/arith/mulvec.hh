#ifndef __HLR_ARITH_MULVEC_HH
#define __HLR_ARITH_MULVEC_HH
//
// Project     : HLR
// Module      : mul_vec
// Description : matrix-vector multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/h2.hh"
#include "hlr/vector/scalar_vector.hh"

namespace hlr
{

//
// compute y = y + α op( M ) x for blas::vector
//
template < typename value_t >
void
mul_vec ( const value_t                     alpha,
          const Hpro::matop_t               op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y )
{
    using namespace hlr::matrix;
    
    // assert( M.ncols( op_M ) == x.length() );
    // assert( M.nrows( op_M ) == y.length() );

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        const auto  row_ofs = M.row_is( op_M ).first();
        const auto  col_ofs = M.col_is( op_M ).first();
    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
            
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                    auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );
                
                    mul_vec( alpha, op_M, *B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else
        M.apply_add( alpha, x, y, op_M );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x for scalar vectors
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    mul_vec( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

// template < typename value_t >
// void
// mul_vec ( const value_t                           alpha,
//           const Hpro::matop_t                     op_M,
//           const Hpro::TMatrix< value_t > &        M,
//           const Hpro::TScalarVector< value_t > &  x,
//           Hpro::TScalarVector< value_t > &        y )
// {
//     mul_vec( alpha, op_M, M, Hpro::blas_vec< value_t >( x ), Hpro::blas_vec< value_t >( y ) );
// }

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH
