#ifndef __HLR_OMP_ARITH_DETAIL_UNIFORM_MULVEC_HH
#define __HLR_OMP_ARITH_DETAIL_UNIFORM_MULVEC_HH
//
// Project     : HLR
// Module      : omp/detail/uniform_mulvec.hh
// Description : matrix-vector multiplication for uniform matrices with OpenMP
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

namespace hlr { namespace omp { namespace uniform { namespace detail {

namespace hpro = HLIB;

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

        #pragma omp taskloop collapse(2) default(shared) firstprivate(alpha,op_M,B)
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j, op_M );
            
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x.block( j );
                    auto  y_i = y.block( i );
                    
                    mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy, mtx_map );
                }// if
            }// for
        }// omp taskloop for
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
        #pragma omp taskloop default(shared)
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
        }// omp taskloop for
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
        #pragma omp taskloop default(shared)
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            u->set_block( i, make_uniform( *cb.son(i) ).release() );
        }// omp taskloop for
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
        #pragma omp taskloop default(shared)
        for ( uint  i = 0; i < u.nblocks(); ++i )
        {
            add_uniform_to_scalar( *u.block(i), v );
        }// omp taskloop for
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

}}}}// namespace hlr::omp::uniform::detail

#endif // __HLR_OMP_ARITH_UNIFORM_HH
