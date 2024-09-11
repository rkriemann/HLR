#ifndef __HLR_SEQ_ARITH_H2_HH
#define __HLR_SEQ_ARITH_H2_HH
//
// Project     : HLR
// Module      : seq/arith_h2
// Description : sequential arithmetic functions for H² matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/h2.hh>

namespace hlr { namespace seq { namespace h2 {

using hlr::h2::mul_vec;

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_mtx ( const value_t                             alpha,
              const matop_t                             op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y,
              cluster_basis_t &                         rowcb,
              cluster_basis_t &                         colcb )
{
    return mul_vec< value_t, cluster_basis_t >( alpha, op_M, M, x, y, rowcb, colcb );
}

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_row ( const value_t                             alpha,
              const matop_t                             op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y,
              cluster_basis_t &                         rowcb,
              cluster_basis_t &                         colcb )
{
    return mul_vec< value_t, cluster_basis_t >( alpha, op_M, M, x, y, rowcb, colcb );
}

}}}// namespace hlr::seq::h2

#endif // __HLR_SEQ_ARITH_H2_HH
