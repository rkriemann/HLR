#ifndef __HLR_ARITH_H2_HH
#define __HLR_ARITH_H2_HH
//
// Project     : HLR
// Module      : arith/h2
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/detail/h2.hh>

namespace hlr { namespace h2 {

//
// mat-vec : y = y + α op( M ) x
//
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
    if ( alpha == value_t(0) )
        return;

    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( ( op_M == Hpro::apply_normal ? colcb : rowcb ), x );
    auto  uy = hlr::vector::make_uniform< value_t, cluster_basis_t >( ( op_M == Hpro::apply_normal ? rowcb : colcb ) );
    auto  s  = blas::vector< value_t >();

    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y, s );
}

// template < typename value_t >
// std::unique_ptr< detail::cluster_blocks_t< value_t > >
// build_cluster_blocks ( const matop_t                     op_M,
//                        const Hpro::TMatrix< value_t > &  M )
// {
//     auto  cb = std::make_unique< cluster_blocks_t< value_t > >( M.row_is( op_M ) );

//     detail::build_cluster_blocks( op_M, M, *cb );

//     return cb;
// }

// template < typename value_t,
//            typename cluster_basis_t >
// void
// mul_vec_cl ( const value_t                             alpha,
//              const Hpro::matop_t                       op_M,
//              const detail::cluster_blocks_t< value_t > &  cb,
//              const vector::scalar_vector< value_t > &  x,
//              vector::scalar_vector< value_t > &        y,
//              cluster_basis_t &                         rowcb,
//              cluster_basis_t &                         colcb )
// {
//     if ( alpha == value_t(0) )
//         return;

//     //
//     // construct uniform representation of x and y
//     //

//     auto  ux = detail::scalar_to_uniform( ( op_M == Hpro::apply_normal ? colcb : rowcb ), x );
//     auto  uy = hlr::vector::make_uniform< value_t, cluster_basis_t >( ( op_M == Hpro::apply_normal ? rowcb : colcb ) );
//     auto  s  = blas::vector< value_t >();

//     detail::mul_vec_cl( alpha, op_M, cb, *ux, *uy, x, y );
//     detail::add_uniform_to_scalar( *uy, y, s );
// }

//
// block row wise computation using IDs of involved objects
//
template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const Hpro::matop_t                                                   op_M,
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

//
// return FLOPs needed for computing y = y + α op( M ) x
// (implicit vectors)
//
template < typename value_t,
           typename cluster_basis_t >
flops_t
mul_vec_flops ( const Hpro::matop_t               op_M,
                const Hpro::TMatrix< value_t > &  M,
                const cluster_basis_t &           rowcb,
                const cluster_basis_t &           colcb )
{
    flops_t  flops = 0;

    flops += detail::scalar_to_uniform_flops( ( op_M == apply_normal ? colcb : rowcb ) );
    flops += detail::mul_vec_flops( op_M, M );
    flops += detail::add_uniform_to_scalar_flops( ( op_M == apply_normal ? rowcb : colcb ) );

    return flops;
}

//
// return size of data involved in computing y = y + α op( M ) x
//
template < typename value_t,
           typename cluster_basis_t >
flops_t
mul_vec_datasize ( const Hpro::matop_t               op_M,
                   const Hpro::TMatrix< value_t > &  M,
                   const cluster_basis_t &           rowcb,
                   const cluster_basis_t &           colcb )
{
    size_t  dsize = 0;

    // scalar to uniform: cluster basis and vector
    dsize += ( op_M == apply_normal ? colcb : rowcb ).data_byte_size() + sizeof( value_t ) * M.ncols( op_M );

    // actual matvec: matrix size
    dsize += M.data_byte_size();

    // uniform to scalar: cluster basis and vector
    dsize += ( op_M == apply_normal ? rowcb : colcb ).data_byte_size() + sizeof( value_t ) * M.nrows( op_M );

    return dsize;
}

}}// namespace hlr::h2

#endif // __HLR_ARITH_H2_HH
