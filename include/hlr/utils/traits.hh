#ifndef __HLR_UTILS_TRAITS_HH
#define __HLR_UTILS_TRAITS_HH
//
// Project     : HLR
// Module      : utils/traits
// Description : various type traits and concepts
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <concepts>
#include <complex>

namespace hlr
{

//
// requires types to have "value_t" as sub type
//
template < typename T >
concept has_value_type = requires ( T a )
{
    typename T::value_t;
};

// //
// // concept for types with value type
// //
// template < typename T >
// concept has_value_type = requires
// {
//     typename T::value_t;
// };
    
// access value type
template < typename T > struct value_type { using  type_t = typename T::value_t; };
template < typename T > using  value_type_t = typename value_type< T >::type_t;

//
// give access to underlying real valued type of T
//
template < typename T > struct real_type                       { using  type_t = T; };
template < typename T > struct real_type< std::complex< T > >  { using  type_t = T; };

template < typename T > using real_type_t = typename real_type< T >::type_t;

//
// general number, e.g., integral or floating point type
//
template < typename value_t >
concept general_number = std::floating_point< value_t > || std::integral< value_t >;

//
// floating point or complex number
//
template < typename value_t >
concept real_or_complex_number = std::floating_point< value_t > ||
                                 ( std::floating_point< real_type_t< value_t > > &&
                                   std::same_as< value_t, std::complex< real_type_t< value_t > > > );

}// namespace hlr

#endif // __HLR_UTILS_TRAITS_HH
