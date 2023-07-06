#ifndef __HLR_UTILS_TRAITS_HH
#define __HLR_UTILS_TRAITS_HH
//
// Project     : HLR
// Module      : utils/traits
// Description : various type traits and concepts
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

namespace hlr
{

//
// concept for general number
//
template < typename value_t >
concept general_number = std::floating_point< value_t > || std::integral< value_t >;

//
// requires types to have "value_t" as sub type
//
template < typename T >
concept with_value_type = requires ( T a )
{
    typename T::value_t;
};

// access value type
template < typename T > struct value_type { using  type_t = typename T::value_t; };
template < typename T > using  value_type_t = typename value_type< T >::type_t;

//
// give access to underlying real valued type of T
//
template < typename T > struct real_type                       { using  type_t = T; };
template < typename T > struct real_type< std::complex< T > >  { using  type_t = T; };

template < typename T > using real_type_t = typename real_type< T >::type_t;

}// namespace hlr

#endif // __HLR_UTILS_TRAITS_HH
