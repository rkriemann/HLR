#ifndef __HLR_APPROX_TRAITS_HH
#define __HLR_APPROX_TRAITS_HH
//
// Project     : HLR
// Module      : approx/traits
// Description : general traits for low-rank approximation objects
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

namespace hlr { namespace approx {

// signals, that T is of approximation type
template < typename T > struct is_approximation { static const bool  value = false; };

template < typename T > inline constexpr bool is_approximation_v = is_approximation< T >::value;

// approximation type concept
template < typename T > concept approximation_type = is_approximation_v< T >;

}}// namespace hlr::approx

#endif // __HLR_APPROX_TRAITS_HH
