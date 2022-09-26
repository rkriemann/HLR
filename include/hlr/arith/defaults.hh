#ifndef __HLR_ARITH_DEFAULTS_HH
#define __HLR_ARITH_DEFAULTS_HH
//
// Project     : HLib
// Module      : arith/defaults
// Description : default arithmetic definitions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlr/arith/mulvec.hh>

namespace hlr
{

//
// default collection of arithmetic functions
//
struct default_arithmetic
{
    //
    // matrix vector multiplication
    //
    
    template < typename value_t >
    void
    mul_vec ( const value_t                             alpha,
              const Hpro::matop_t                       op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y ) const
    {
        hlr::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    mul_vec ( const value_t                     alpha,
              const Hpro::matop_t               op_M,
              const Hpro::TMatrix< value_t > &  M,
              const blas::vector< value_t > &   x,
              blas::vector< value_t > &         y ) const
    {
        hlr::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    prod ( const value_t                             alpha,
           const matop_t                             op_M,
           const Hpro::TLinearOperator< value_t > &  M,
           const blas::vector< value_t > &           x,
           blas::vector< value_t > &                 y ) const
    {
        M.apply_add( alpha, x, y, op_M );
    }

    template < typename value_t >
    void
    prod ( const value_t                    alpha,
           const matop_t                    op_M,
           const blas::matrix< value_t > &  M,
           const blas::vector< value_t > &  x,
           blas::vector< value_t > &        y ) const
    {
        blas::mulvec( alpha, blas::mat_view( op_M, M ), x, value_t(1), y );
    }
};

constexpr default_arithmetic arithmetic{};

}// namespace hlr

#endif  // __HLR_ARITH_DEFAULTS_HH
