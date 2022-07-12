#ifndef __HLR_TBB_ARITH_H2_HH
#define __HLR_TBB_ARITH_H2_HH
//
// Project     : HLib
// Module      : tbb/arith_h2.hh
// Description : arithmetic functions for H² matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#endif

#if defined(HAS_H2)

#include <hlr/arith/h2.hh>
#include <hlr/tbb/detail/h2_mvm.hh>

namespace hlr { namespace tbb { namespace h2 {

template < typename value_t >
using nested_cluster_basis = Hpro::TClusterBasis< value_t >;

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          nested_cluster_basis< value_t > &         rowcb,
          nested_cluster_basis< value_t > &         colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  mtx_map = detail::mutex_map_t();
    
    auto  ux = detail::scalar_to_uniform( op_M == Hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == Hpro::apply_normal ? rowcb : colcb );
    auto  s  = blas::vector< value_t >();

    detail::build_mutex_map( rowcb, mtx_map );
    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y, mtx_map );
    detail::add_uniform_to_scalar( *uy, y, s );
}

}}}// namespace hlr::tbb::h2

#endif // HAS_H2

#endif // __HLR_ARITH_H2_HH
