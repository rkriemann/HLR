#ifndef __HLR_HPX_ARITH_UNIFORM_HH
#define __HLR_HPX_ARITH_UNIFORM_HH
//
// Project     : HLR
// Module      : hpx/arith_uniform
// Description : arithmetic functions for uniform matrices with HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>
#include <hlr/arith/uniform.hh>

#include <hlr/hpx/detail/uniform_mulvec.hh>
#include <hlr/hpx/detail/uniform_accu.hh>

namespace hlr { namespace hpx { namespace uniform {

namespace hpro = HLIB;

//
// chpxute mat-vec M·x = y with scalar vectors x,y.
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const matop_t                             op_M,
          const TMatrix &                           M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          hlr::matrix::cluster_basis< value_t > &   rowcb,
          hlr::matrix::cluster_basis< value_t > &   colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    detail::mutex_map_t  mtx_map;

    auto  ux = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::cluster_basis< value_t > > >();
    auto  uy = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::cluster_basis< value_t > > >();
    
    ux = std::move( detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x ) );
    uy = std::move( detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb ) );
    
    detail::build_mutex_map( rowcb, mtx_map );
    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y, mtx_map );
    detail::add_uniform_to_scalar( *uy, y );
}

//
// matrix multiplication (eager version)
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc )
{
    hlr::uniform::multiply< value_t >( alpha, op_A, A, op_B, B, C, acc );
}

//////////////////////////////////////////////////////////////////////
//
// accumulator version
//
//////////////////////////////////////////////////////////////////////

namespace accu
{

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const matop_t            op_A,
           const hpro::TMatrix &    A,
           const matop_t            op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    auto  pi_mtx     = std::mutex();
    auto  prod_inner = detail::inner_map_t();
    auto  accu       = detail::accumulator( & prod_inner, & pi_mtx );
    auto  basis_data = detail::rec_basis_data_t( C );

    accu.add_update( op_A, A, op_B, B );

    detail::multiply( alpha, C, accu, acc, approx, basis_data );
}

template < typename value_t,
           typename approx_t >
void
multiply_cached ( const value_t            alpha,
                  const matop_t            op_A,
                  const hpro::TMatrix &    A,
                  const matop_t            op_B,
                  const hpro::TMatrix &    B,
                  hpro::TMatrix &          C,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    auto  pi_mtx     = std::mutex();
    auto  prod_inner = detail::inner_map_t();
    auto  accu       = detail::accumulator( & prod_inner, & pi_mtx );
    auto  basis_data = detail::rec_basis_data_t( C );

    accu.add_update( op_A, A, op_B, B );

    detail::multiply( alpha, C, accu, acc, approx, basis_data );
}

}// namespace accu

}}}// namespace hlr::hpx::uniform

#endif // __HLR_HPX_ARITH_UNIFORM_HH
