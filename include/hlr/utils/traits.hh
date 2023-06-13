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

}// namespace hlr

#endif // __HLR_UTILS_TRAITS_HH
