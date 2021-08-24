#ifndef __HLR_TBB_ARITH_UNIFORM_HH
#define __HLR_TBB_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : tbb/arith_uniform.hh
// Description : arithmetic functions for uniform matrices with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>
#include <hlr/arith/uniform.hh>
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
namespace detail
{

using hlr::matrix::cluster_basis;
using hlr::matrix::uniform_lrmatrix;
using hlr::vector::scalar_vector;
using hlr::vector::uniform_vector;

using  mutex_map_t = std::unordered_map< indexset, std::unique_ptr< std::mutex >, indexset_hash >;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec ( const value_t                                       alpha,
          const hpro::matop_t                                 op_M,
          const hpro::TMatrix &                               M,
          const uniform_vector< cluster_basis< value_t > > &  x,
          uniform_vector< cluster_basis< value_t > > &        y,
          const scalar_vector< value_t > &                    sx,
          scalar_vector< value_t > &                          sy,
          mutex_map_t &                                       mtx_map )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, TBlockMatrix );

        HLR_ASSERT(( B->nblock_rows( op_M ) == y.nblocks() ) &&
                   ( B->nblock_cols( op_M ) == x.nblocks() ));
            
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, B->nblock_rows( op_M ),
                                              0, B->nblock_cols( op_M ) ),
            [&,alpha,op_M,B] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j, op_M );
            
                        if ( ! is_null( B_ij ) )
                        {
                            auto  x_j = x.block( j );
                            auto  y_i = y.block( i );
                            
                            mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy, mtx_map );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D    = cptrcast( &M, TDenseMatrix );
        auto  x_i  = blas::vector< value_t >( blas_vec< value_t >( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas_vec< value_t >( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  mtx  = mtx_map[ M.row_is( op_M ) ].get();
        auto  lock = std::scoped_lock( *mtx );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas_mat< value_t >( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );
        
        if ( op_M == hpro::apply_normal )
        {
            std::scoped_lock  lock( y.mutex() );

            blas::mulvec( value_t(1), R->coeff(), x.coeffs(), value_t(1), y.coeffs() );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            std::scoped_lock  lock( y.mutex() );

            HLR_ASSERT( false );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            std::scoped_lock  lock( y.mutex() );
            
            blas::mulvec( value_t(1), blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
scalar_to_uniform ( const cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.is(), cb );

    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( blas_vec< value_t >( v ), cb.is() - v.ofs() );
        auto  s    = cb.transform_forward( v_cb );

        u->set_coeffs( std::move( s ) );
    }// if

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for( uint(0), cb.nsons(),
                             [&] ( const uint  i )
                             {
                                 u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
                             } );
    }// if

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
make_uniform ( const cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.is(), cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for( uint(0), cb.nsons(),
                             [&] ( const uint  i )
                             {
                                 u->set_block( i, make_uniform( *cb.son(i) ).release() );
                             } );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                          v )
{
    if ( u.basis().rank() > 0 )
    {
        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( blas_vec< value_t >( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        ::tbb::parallel_for( uint(0), u.nblocks(),
                             [&] ( const uint  i )
                             {
                                 add_uniform_to_scalar( *u.block(i), v );
                             } );
    }// if
}

//
// generate mapping of index set to mutices for leaf clusters
//
template < typename value_t >
void
build_mutex_map ( const cluster_basis< value_t > &  cb,
                  mutex_map_t &                     mtx_map )
{
    if ( cb.nsons() == 0 )
    {
        mtx_map[ cb.is() ] = std::make_unique< std::mutex >();
    }// if
    else
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            build_mutex_map( *cb.son(i), mtx_map );
    }// else
}

}// namespace detail

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
    
    auto  ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb );

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
    auto  mm         = detail::rec_matrix_mult( C );

    accu.add_update( op_A, A, op_B, B );

    mm.multiply( alpha, C, accu, acc, approx );
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
    auto  mm         = detail::rec_matrix_mult( C );

    accu.add_update( op_A, A, op_B, B );

    mm.multiply( alpha, C, accu, acc, approx );
}

}// namespace accu

namespace accu2
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &                          A,
     hpro::TMatrix &                          L,
     hpro::TMatrix &                          U,
     const hpro::TTruncAcc &                  acc,
     const approx_t &                         approx,
     hlr::matrix::cluster_basis< value_t > &  rowcb_L,
     hlr::matrix::cluster_basis< value_t > &  colcb_L,
     hlr::matrix::cluster_basis< value_t > &  rowcb_U,
     hlr::matrix::cluster_basis< value_t > &  colcb_U )
{
    auto  accu = hlr::tbb::uniform::accu::detail2::accumulator();
    auto  lu   = hlr::tbb::uniform::accu::detail2::rec_lu_factorization( L, U );

    lu.lu( A, L, U, accu, acc, approx, rowcb_L, colcb_L, rowcb_U, colcb_U );
}

}// namespace accu2

}}}// namespace hlr::tbb::uniform

#endif // __HLR_TBB_ARITH_UNIFORM_HH
