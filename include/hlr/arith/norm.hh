#ifndef __HLR_ARITH_NORM_HH
#define __HLR_ARITH_NORM_HH
//
// Project     : HLib
// Module      : norm
// Description : norm related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

#include "hlr/arith/defaults.hh"
#include "hlr/matrix/product.hh"
#include "hlr/matrix/sum.hh"
#include "hlr/matrix/identity.hh"

namespace hlr { namespace norm {

////////////////////////////////////////////////////////////////////////////////
//
// Frobenius norm
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const Hpro::TMatrix< value_t > &  A );

//
// return Frobenius norm of αA+βB, e.g. |αA+βB|_F
//
template < typename value_t >
Hpro::real_type_t< value_t >
frobenius ( const value_t                     alpha,
            const Hpro::TMatrix< value_t > &  A,
            const value_t                     beta,
            const Hpro::TMatrix< value_t > &  B );

////////////////////////////////////////////////////////////////////////////////
//
// Spectral norm (|·|₂)
//
////////////////////////////////////////////////////////////////////////////////

//
// compute spectral norm of A via power iteration
//
template < typename arithmetic_t,
           typename operator_t >
requires provides_arithmetic< arithmetic_t >
Hpro::real_type_t< Hpro::value_type_t< operator_t > >
spectral ( arithmetic_t &&     arithmetic,
           const operator_t &  A,
           const double        atol    = 1e-3,
           const size_t        amax_it = 50,
           const bool          squared = true );

template < typename operator_t >
Hpro::real_type_t< Hpro::value_type_t< operator_t > >
spectral ( const operator_t &  A,
           const double        atol    = 1e-3,
           const size_t        amax_it = 50,
           const bool          squared = true )
{
    return spectral( hlr::arithmetic, A, atol, amax_it, squared );
}

template < typename arithmetic_t,
           typename value_t >
requires provides_arithmetic< arithmetic_t >
Hpro::real_type_t< value_t >
inv_error_2 ( arithmetic_t &&                           arithmetic,
              const Hpro::TMatrix< value_t > &          A,
              const Hpro::TLinearOperator< value_t > &  A_inv )
{
    auto  AxInv   = matrix::product( A, A_inv );
    auto  I       = matrix::identity< value_t >( A.block_is() );
    auto  inv_err = matrix::sum( 1.0, *I, -1.0, *AxInv );

    return spectral( arithmetic, *inv_err );
}

template < typename value_t >
Hpro::real_type_t< value_t >
inv_error_2 ( const Hpro::TMatrix< value_t > &          A,
              const Hpro::TLinearOperator< value_t > &  A_inv )
{
    return inv_error_2( hlr::arithmetic, A, A_inv );
}

}}// namespace hlr::norm

#include <hlr/arith/detail/norm.hh>

#endif // __HLR_ARITH_NORM_HH
