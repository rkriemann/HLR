#ifndef __HLR_TOOLS_MATH_HH
#define __HLR_TOOLS_MATH_HH
//
// Project     : HLR
// Module      : tools/math
// Description : basic mathematical functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/base/System.hh>

namespace hlr { namespace math {

//
// import functions from HLIBpro and adjust naming
//

namespace hpro = HLIB;

using namespace hpro::Math;

//
// conversion of floating point formats
//

// consistent naming scheme (float16 -> float32 -> float64)
//using float16 = half;
using float32 = float;
using float64 = double;

// increase precision one level
template < typename value_t >
struct increase_precision { using type_t = value_t; };

template < typename real_t >
struct increase_precision< std::complex< real_t > >
{
    using type_t = std::complex< typename increase_precision< real_t >::type_t >;
};

//template <> struct increase_precision< half >  { using type_t = float; };
template <> struct increase_precision< float >  { using type_t = double; };
template <> struct increase_precision< double > { using type_t = long double; };

template < typename value_t >
using increase_precision_t = typename increase_precision< value_t >::type_t;


// decrease precision one level
template < typename value_t >
struct decrease_precision { using type_t = value_t; };

template < typename real_t >
struct decrease_precision< std::complex< real_t > >
{
    using type_t = std::complex< typename decrease_precision< real_t >::type_t >;
};

//template <> struct decrease_precision< float >       { using type_t = half; };
template <> struct decrease_precision< double >      { using type_t = float; };
template <> struct decrease_precision< long double > { using type_t = double; };

template < typename value_t >
using decrease_precision_t = typename decrease_precision< value_t >::type_t;

}}// namespace hlr::math

#endif // __HLR_TOOLS_MATH_HH
