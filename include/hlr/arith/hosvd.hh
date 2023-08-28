#ifndef __HLR_BLAS_HOSVD_HH
#define __HLR_BLAS_HOSVD_HH
//
// Project     : HLR
// Module      : blas/hosvd
// Description : implements different HOSVD algorithms for tensors
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/tensor.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/approx/svd.hh>

namespace hlr { namespace blas {

//
// standard HOSVD
//

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc,
        const approx_t &            apx );

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc );

//
// sequentially truncated HOSVD
//

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
          const accuracy &            acc,
          const approx_t &            apx );

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
          const accuracy &            acc );

//
// greedy HOSVD
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
greedy_hosvd ( const tensor3< value_t > &  X,
               const accuracy &            acc,
               const approx_t &            apx );

//
// recompress given tucker tensor
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
recompress ( tensor3< value_t > &  G,
             matrix< value_t > &   X0,
             matrix< value_t > &   X1,
             matrix< value_t > &   X2,
             const accuracy &      acc,
             const approx_t &      apx,
             hosvd_func_t &&       func );
    
//
// error of Tucker decomposition D - G ×₀ X₀ ×₁ X₁ ×₂ X₂ 
//
template < typename value_t >
Hpro::real_type_t< value_t >
tucker_error ( const tensor3< value_t > &  D,
               const tensor3< value_t > &  G,
               const matrix< value_t > &   X0,
               const matrix< value_t > &   X1,
               const matrix< value_t > &   X2 );

}}// namespace hlr::blas

#include <hlr/arith/detail/hosvd.hh>

#endif  // __HPRO_BLAS_TENSOR_HH
