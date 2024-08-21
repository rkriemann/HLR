#ifndef __HLR_TBB_ARITH_UNIFORM_HH
#define __HLR_TBB_ARITH_UNIFORM_HH
//
// Project     : HLR
// Module      : tbb/arith_uniform.hh
// Description : arithmetic functions for uniform matrices with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/level_hierarchy.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/tbb/detail/uniform_mvm.hh>
#include <hlr/tbb/detail/uniform_accu.hh>
#include <hlr/tbb/detail/uniform_accu_lu.hh>

namespace hlr { namespace tbb { namespace uniform {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                                   alpha,
          const matop_t                                   op_M,
          const Hpro::TMatrix< value_t > &                M,
          const vector::scalar_vector< value_t > &        x,
          vector::scalar_vector< value_t > &              y,
          hlr::matrix::shared_cluster_basis< value_t > &  rowcb,
          hlr::matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    #if 0
    
    auto  ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb );

    #else

    auto  ux = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();
    auto  uy = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();

    ::tbb::parallel_invoke(
        [&] () { ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x ); },
        [&] () { uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb ); }
    );

    #endif

    //
    // multiply
    //
    
    #if 0

    detail::mutex_map_t  mtx_map;
    
    detail::build_mutex_map( rowcb, mtx_map );
    detail::mul_vec_mtx( alpha, op_M, M, *ux, *uy, x, y, mtx_map );

    #else

    detail::mul_vec_row( alpha, op_M, M, *ux, *uy, x, y );
    
    #endif

    //
    // and add result to y
    //
    
    detail::add_uniform_to_scalar( *uy, y );
}

//
// MVM with all block parallelism and synchronization via mutexes
//
template < typename value_t >
void
mul_vec_mtx ( const value_t                                   alpha,
              const matop_t                                   op_M,
              const Hpro::TMatrix< value_t > &                M,
              const vector::scalar_vector< value_t > &        x,
              vector::scalar_vector< value_t > &              y,
              hlr::matrix::shared_cluster_basis< value_t > &  rowcb,
              hlr::matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();
    auto  uy = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();

    ::tbb::parallel_invoke(
        [&] () { ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x ); },
        [&] () { uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb ); }
    );

    //
    // multiply
    //
    
    detail::mutex_map_t  mtx_map;
    
    detail::build_mutex_map( rowcb, mtx_map );
    detail::mul_vec_mtx( alpha, op_M, M, *ux, *uy, x, y, mtx_map );

    //
    // and add result to y
    //
    
    detail::add_uniform_to_scalar( *uy, y );
}

//
// parallelism only via block rows without synchronization
//
template < typename value_t >
void
mul_vec_row ( const value_t                                   alpha,
              const matop_t                                   op_M,
              const Hpro::TMatrix< value_t > &                M,
              const vector::scalar_vector< value_t > &        x,
              vector::scalar_vector< value_t > &              y,
              hlr::matrix::shared_cluster_basis< value_t > &  rowcb,
              hlr::matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();
    auto  uy = std::unique_ptr< hlr::vector::uniform_vector< hlr::matrix::shared_cluster_basis< value_t > > >();

    ::tbb::parallel_invoke(
        [&] () { ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x ); },
        [&] () { uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb ); }
    );

    //
    // multiply
    //
    
    detail::mul_vec_row( alpha, op_M, M, *ux, *uy, x, y );

    //
    // and add result to y
    //
    
    detail::add_uniform_to_scalar( *uy, y );
}

//
// parallelism of all block rows per level without synchronization
//
template < typename value_t >
void
mul_vec_hier ( const value_t                                        alpha,
               const hpro::matop_t                                  op_M,
               const matrix::level_hierarchy< value_t > &           M,
               const vector::scalar_vector< value_t > &             x,
               vector::scalar_vector< value_t > &                   y,
               matrix::shared_cluster_basis_hierarchy< value_t > &  rowcb,
               matrix::shared_cluster_basis_hierarchy< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( ( op_M == hpro::apply_normal ? colcb : rowcb ), x );

    detail::mul_vec_hier( alpha, op_M, M, *ux, x, y, ( op_M == hpro::apply_normal ? rowcb : colcb ) );
}

//
// matrix multiplication (eager version)
//
template < typename value_t >
void
multiply ( const value_t                     alpha,
           const hpro::matop_t               op_A,
           const hpro::TMatrix< value_t > &  A,
           const hpro::matop_t               op_B,
           const hpro::TMatrix< value_t > &  B,
           hpro::TMatrix< value_t > &        C,
           const hpro::TTruncAcc &           acc )
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
multiply ( const value_t                     alpha,
           const matop_t                     op_A,
           const hpro::TMatrix< value_t > &  A,
           const matop_t                     op_B,
           const hpro::TMatrix< value_t > &  B,
           hpro::TMatrix< value_t > &        C,
           const hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    auto  pi_mtx     = std::mutex();
    auto  prod_inner = detail::inner_map_t< value_t >();
    auto  accu       = detail::accumulator< value_t >( & prod_inner, & pi_mtx );
    auto  mm         = detail::rec_matrix_mult( C );

    accu.add_update( op_A, A, op_B, B );

    mm.multiply( alpha, C, accu, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply_cached ( const value_t                     alpha,
                  const matop_t                     op_A,
                  const hpro::TMatrix< value_t > &  A,
                  const matop_t                     op_B,
                  const hpro::TMatrix< value_t > &  B,
                  hpro::TMatrix< value_t > &        C,
                  const hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    auto  pi_mtx     = std::mutex();
    auto  prod_inner = detail::inner_map_t< value_t >();
    auto  accu       = detail::accumulator< value_t >( & prod_inner, & pi_mtx );
    auto  mm         = detail::rec_matrix_mult< value_t >( C );

    accu.add_update( op_A, A, op_B, B );

    mm.multiply( alpha, C, accu, acc, approx );
}

}// namespace accu

namespace accu2
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix< value_t > &                      A,
     hpro::TMatrix< value_t > &                      L,
     hpro::TMatrix< value_t > &                      U,
     const hpro::TTruncAcc &                         acc,
     const approx_t &                                approx,
     hlr::matrix::shared_cluster_basis< value_t > &  rowcb_L,
     hlr::matrix::shared_cluster_basis< value_t > &  colcb_L,
     hlr::matrix::shared_cluster_basis< value_t > &  rowcb_U,
     hlr::matrix::shared_cluster_basis< value_t > &  colcb_U )
{
    auto  accu = hlr::tbb::uniform::accu::detail2::accumulator< value_t >();
    auto  lu   = hlr::tbb::uniform::accu::detail2::rec_lu_factorization< value_t >( L, U );

    lu.lu( A, L, U, accu, acc, approx, rowcb_L, colcb_L, rowcb_U, colcb_U );
}

}// namespace accu2

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                              alpha,
          const hpro::matop_t                        op_M,
          const hpro::TMatrix< value_t > &           M,
          const vector::scalar_vector< value_t > &   x,
          vector::scalar_vector< value_t > &         y,
          matrix::shared_cluster_basis< value_t > &  rowcb,
          matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    if ( op_M == apply_normal )
        detail::mul_vec( alpha, op_M, M, x, y, rowcb, colcb );
    else
        detail::mul_vec( alpha, op_M, M, x, y, colcb, rowcb );
}

}// namespace tlr

}}}// namespace hlr::tbb::uniform

#endif // __HLR_TBB_ARITH_UNIFORM_HH
