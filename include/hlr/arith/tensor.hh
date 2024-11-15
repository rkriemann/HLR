#ifndef __HLR_BLAS_TENSOR_HH
#define __HLR_BLAS_TENSOR_HH
//
// Project     : HLR
// Module      : blas/tensor
// Description : implements dense tensor class
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <type_traits>
#include <vector>

#include <hpro/blas/MemBlock.hh>

#include <hlr/approx/traits.hh>
#include <hlr/arith/blas.hh>

namespace hlr
{

using Hpro::idx_t;

namespace blas
{

using Hpro::copy_policy_t;
using Hpro::copy_reference;
using Hpro::copy_value;
using Hpro::real_type_t;

// trait for giving access to tensor properties
template < typename T_tensor > struct tensor_trait;

// signals, that T is of tensor type
template < typename T > struct is_tensor { static const bool  value = false; };

template < typename T > inline constexpr bool is_tensor_v = is_tensor< T >::value;

// tensor type concept
template < typename T > concept tensor_type = is_tensor_v< T >;

}}// namespace hlr::blas

#include <hlr/arith/tensor3.hh>
#include <hlr/arith/tensor4.hh>

#endif  // __HPRO_BLAS_TENSOR_HH
