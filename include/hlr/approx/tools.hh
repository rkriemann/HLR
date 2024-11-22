#ifndef __HLR_APPROX_TOOLS_HH
#define __HLR_APPROX_TOOLS_HH
//
// Project     : HLR
// Module      : approx/tools
// Description : misc. functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>

namespace hlr { namespace approx { namespace detail {

//
// convert U·V' into W·S·X' with
// orthogonal W, X and singular values S
// - assuming U to be orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::vector< real_type_t< value_t > >,
            blas::matrix< value_t > >
make_ortho ( blas::matrix< value_t > &  U,
             blas::matrix< value_t > &  V )
{
    // U is orthonormal, so factorize V only
    // U·V' = U (Us·Ss·Vs')
    //      = (U·Vs) • Ss • Us
    //      =:  W    • Ss • X'
    auto  Us = V;
    auto  Vs = blas::matrix< value_t >();
    auto  Ss = blas::vector< real_type_t< value_t > >();

    blas::svd( Us, Ss, Vs );

    auto  W = blas::prod( U, Vs );

    return { std::move( W ), std::move( Ss ), std::move( Us ) };
}

}}}// namespace hlr::approx::detail

#endif // __HLR_APPROX_TOOLS_HH
