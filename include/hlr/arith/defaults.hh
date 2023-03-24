#ifndef __HLR_ARITH_DEFAULTS_HH
#define __HLR_ARITH_DEFAULTS_HH
//
// Project     : HLR
// Module      : arith/defaults
// Description : default arithmetic definitions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/mulvec.hh>
#include <hlr/arith/detail/solve_vec.hh>

namespace hlr
{

//
// define trait and concept for testing arithmetic
//
template < typename T > struct is_arithmetic { static constexpr bool value = false; };

template < typename T > inline constexpr bool is_arithmetic_v = is_arithmetic< T >::value;

template < typename T > concept arithmetic_type = is_arithmetic_v< T >;

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
              const matop_t                             op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y ) const
    {
        hlr::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    mul_vec ( const value_t                     alpha,
              const matop_t                     op_M,
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

    //
    // vector solves
    //

    template < typename value_t >
    void
    solve_lower_tri ( const matop_t                       op_L,
                      const Hpro::TMatrix< value_t > &    L,
                      vector::scalar_vector< value_t > &  v,
                      const Hpro::diag_type_t             diag_mode ) const
    {
        hlr::solve_lower_tri( op_L, L, v, diag_mode );
    }

    template < typename value_t >
    void
    solve_upper_tri ( const matop_t                       op_U,
                      const Hpro::TMatrix< value_t > &    U,
                      vector::scalar_vector< value_t > &  v,
                      const Hpro::diag_type_t             diag_mode ) const
    {
        hlr::solve_upper_tri( op_U, U, v, diag_mode );
    }
};

constexpr default_arithmetic arithmetic{};

template <> struct is_arithmetic<       default_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic< const default_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic<       default_arithmetic & > { static constexpr bool value = true; };
template <> struct is_arithmetic< const default_arithmetic & > { static constexpr bool value = true; };

}// namespace hlr

#endif  // __HLR_ARITH_DEFAULTS_HH
