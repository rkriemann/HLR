#ifndef __HLR_SEQ_ARITH_UNIFORM_HH
#define __HLR_SEQ_ARITH_UNIFORM_HH
//
// Project     : HLR
// Module      : seq/arith_uniform
// Description : sequential arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/uniform.hh>

namespace hlr { namespace seq { namespace uniform {

using hlr::uniform::mul_vec;
using hlr::uniform::mul_vec_hier;
using hlr::uniform::mul_vec_row;
using hlr::uniform::build_id2cb;
using hlr::uniform::build_id2blocks;
using hlr::uniform::multiply;
using hlr::uniform::lu;

template < typename value_t >
void
mul_vec_mtx ( const value_t                              alpha,
              const hpro::matop_t                        op_M,
              const hpro::TMatrix< value_t > &           M,
              const vector::scalar_vector< value_t > &   x,
              vector::scalar_vector< value_t > &         y,
              matrix::shared_cluster_basis< value_t > &  rowcb,
              matrix::shared_cluster_basis< value_t > &  colcb )
{
    return mul_vec( alpha, op_M, M, x, y, rowcb, colcb );
}

template < typename value_t >
void
mul_vec_row ( const value_t                              alpha,
              const hpro::matop_t                        op_M,
              const hpro::TMatrix< value_t > &           M,
              const vector::scalar_vector< value_t > &   x,
              vector::scalar_vector< value_t > &         y,
              matrix::shared_cluster_basis< value_t > &  rowcb,
              matrix::shared_cluster_basis< value_t > &  colcb )
{
    return mul_vec( alpha, op_M, M, x, y, rowcb, colcb );
}

template < typename value_t >
void
mul_vec2 ( const value_t                              alpha,
           const hpro::matop_t                        op_M,
           const hpro::TMatrix< value_t > &           M,
           const vector::scalar_vector< value_t > &   x,
           vector::scalar_vector< value_t > &         y,
           matrix::shared_cluster_basis< value_t > &  rowcb,
           matrix::shared_cluster_basis< value_t > &  colcb )
{
    // standard version is faster
    return mul_vec( alpha, op_M, M, x, y, rowcb, colcb );
}

namespace accu
{

using hlr::uniform::accu::multiply;
using hlr::uniform::accu::multiply_cached;
using hlr::uniform::accu::lu;

}// namespace accu

namespace accu2 { using hlr::uniform::accu2::lu; }// namespace accu2
namespace accu3 { using hlr::uniform::accu3::lu; }// namespace accu3
namespace accu4 { using hlr::uniform::accu4::lu; }// namespace accu4

namespace tlr
{

using hlr::uniform::tlr::mul_vec;
// using hlr::uniform::tlr::addlr;
using hlr::uniform::tlr::multiply;
using hlr::uniform::tlr::lu;
using hlr::uniform::tlr::lu_sep;
using hlr::uniform::tlr::lu_lazy;
using hlr::uniform::tlr::ldu;

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
