#ifndef __HLR_TBB_ARITH_H2_HH
#define __HLR_TBB_ARITH_H2_HH
//
// Project     : HLR
// Module      : tbb/arith_h2.hh
// Description : arithmetic functions for H² matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/h2.hh>
#include <hlr/tbb/detail/h2_mvm.hh>

namespace hlr { namespace tbb { namespace h2 {

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_mtx ( const value_t                             alpha,
              const Hpro::matop_t                       op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y,
              cluster_basis_t &                         rowcb,
              cluster_basis_t &                         colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( op_M == Hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform< value_t, cluster_basis_t >( op_M == Hpro::apply_normal ? rowcb : colcb );
    auto  s  = blas::vector< value_t >();

    auto  mtx_map = detail::mutex_map_t();
    
    detail::build_mutex_map( rowcb, mtx_map );
    detail::mul_vec_mtx( alpha, op_M, M, *ux, *uy, x, y, mtx_map );
    detail::add_uniform_to_scalar( *uy, y, s );
}

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_row ( const value_t                             alpha,
              const Hpro::matop_t                       op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y,
              cluster_basis_t &                         rowcb,
              cluster_basis_t &                         colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( op_M == Hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform< value_t, cluster_basis_t >( op_M == Hpro::apply_normal ? rowcb : colcb );
    auto  s  = blas::vector< value_t >();

    detail::mul_vec_row( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y, s );
}

//
// block row wise computation using IDs of involved objects
//
template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const hpro::matop_t                                                   op_M,
              const vector::scalar_vector< value_t > &                              x,
              vector::scalar_vector< value_t > &                                    y,
              const matrix::nested_cluster_basis< value_t > &                       rowcb,
              const matrix::nested_cluster_basis< value_t > &                       colcb,
              const std::vector< matrix::nested_cluster_basis< value_t > * > &      colcb_map,
              const std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap )
{
    if ( alpha == value_t(0) )
        return;

    auto  xcoeff = std::vector< blas::vector< value_t > >( colcb.id() + 1 );
    auto  ycoeff = blas::vector< value_t >();

    detail::scalar_to_uniform( colcb, x, xcoeff );
    detail::mul_vec_row< value_t >( alpha, op_M, rowcb, colcb_map, blockmap, xcoeff, ycoeff, x, y );
}

template < typename value_t >
std::vector< matrix::nested_cluster_basis< value_t > * >
build_id2cb ( matrix::nested_cluster_basis< value_t > &  cb )
{
    HLR_ASSERT( cb.id() != -1 );
    
    auto  idmap = std::vector< matrix::nested_cluster_basis< value_t > * >( cb.id() + 1 );

    detail::build_id2cb( cb, idmap );

    return idmap;
}

template < typename value_t >
std::vector< std::list< const Hpro::TMatrix< value_t > * > >
build_id2blocks ( const matrix::nested_cluster_basis< value_t > &  cb,
                  const Hpro::TMatrix< value_t > &                 M,
                  const bool                                       transposed )
{
    HLR_ASSERT( cb.id() != -1 );

    auto  blockmap = std::vector< std::list< const Hpro::TMatrix< value_t > * > >( cb.id() + 1 );

    detail::build_id2blocks( cb, M, blockmap, transposed );

    return blockmap;
}

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          cluster_basis_t &                         rowcb,
          cluster_basis_t &                         colcb )
{
    mul_vec_row( alpha, op_M, M, x, y, rowcb, colcb );
}

}}}// namespace hlr::tbb::h2

#endif // __HLR_ARITH_H2_HH
