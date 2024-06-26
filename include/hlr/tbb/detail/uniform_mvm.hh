#ifndef __HLR_TBB_ARITH_UNIFORM_MVM_HH
#define __HLR_TBB_ARITH_UNIFORM_MVM_HH
//
// Project     : HLR
// Module      : tbb/detail/uniform_mvm.hh
// Description : matrix-vector product for uniform matrices with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>

namespace hlr { namespace tbb { namespace uniform { namespace detail {

using hlr::matrix::shared_cluster_basis;
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
mul_vec_mtx ( const value_t                                              alpha,
              const Hpro::matop_t                                        op_M,
              const Hpro::TMatrix< value_t > &                           M,
              const uniform_vector< shared_cluster_basis< value_t > > &  x,
              uniform_vector< shared_cluster_basis< value_t > > &        y,
              const scalar_vector< value_t > &                           sx,
              scalar_vector< value_t > &                                 sy,
              mutex_map_t &                                              mtx_map )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

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
                            
                            mul_vec_mtx( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy, mtx_map );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( hlr::matrix::is_dense( M ) )
    {
        auto  x_i  = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  mtx  = mtx_map.at( M.row_is( op_M ) ).get();
        auto  lock = std::scoped_lock( *mtx );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R    = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  lock = std::scoped_lock( y.mutex() );
        
        if ( op_M == Hpro::apply_normal )
        {
            blas::mulvec( alpha, R->coeff(), x.coeffs(), value_t(1), y.coeffs() );
        }// if
        else if ( op_M == Hpro::apply_conjugate )
        {
            HLR_ASSERT( false );
        }// if
        else if ( op_M == Hpro::apply_transposed )
        {
            HLR_ASSERT( false );
        }// if
        else if ( op_M == Hpro::apply_adjoint )
        {
            blas::mulvec( alpha, blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t >
void
mul_vec_row ( const value_t                                              alpha,
              const Hpro::matop_t                                        op_M,
              const Hpro::TMatrix< value_t > &                           M,
              const uniform_vector< shared_cluster_basis< value_t > > &  x,
              uniform_vector< shared_cluster_basis< value_t > > &        y,
              const scalar_vector< value_t > &                           sx,
              scalar_vector< value_t > &                                 sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT(( B->nblock_rows( op_M ) == y.nblocks() ) &&
                   ( B->nblock_cols( op_M ) == x.nblocks() ));

        //
        // parallelise only block rows, then only one task will access y
        //
        
        ::tbb::parallel_for< uint >(
            0, B->nblock_rows( op_M ),
            [&,alpha,op_M,B] ( const auto  i )
            {
                auto  y_i = y.block( i );
                
                for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
                {
                    auto  B_ij = B->block( i, j, op_M );
                    
                    if ( ! is_null( B_ij ) )
                    {
                        auto  x_j = x.block( j );
                        
                        mul_vec_row( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                    }// if
                }// for
            } );
    }// if
    else if ( hlr::matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );

        #if defined(HLR_HAS_ZBLAS_DIRECT)
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                default               : HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     : blas::mulvec( alpha, R->coeff(), x.coeffs(), value_t(1), y.coeffs() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : blas::mulvec( alpha, blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() ); break;
                default               :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< shared_cluster_basis< value_t > > >
scalar_to_uniform ( const shared_cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &         v )
{
    auto  u = std::make_unique< uniform_vector< shared_cluster_basis< value_t > > >( cb );

    ::tbb::parallel_invoke(
        [&] ()
        {                    
            if ( cb.rank() > 0 )
            {
                auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
                auto  s    = cb.transform_forward( v_cb );
                
                u->set_coeffs( std::move( s ) );
            }// if
        },

        [&] ()
        {
            if ( cb.nsons() > 0 )
            {
                ::tbb::parallel_for(
                    uint(0), cb.nsons(),
                    [&] ( const uint  i )
                    {
                        u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
                    }
                );
            }// if
        } );

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< shared_cluster_basis< value_t > > >
make_uniform ( const shared_cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< shared_cluster_basis< value_t > > >( cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                u->set_block( i, make_uniform( *cb.son(i) ).release() );
            }
        );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t >
void
add_uniform_to_scalar ( const uniform_vector< shared_cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                                 v )
{
    if ( u.basis().rank() > 0 )
    {
        // auto  v_u = blas::vector< value_t >( blas::vec( v ), u.is() - v.ofs() );

        // u.basis().transform_backward( u.coeffs(), v_u );

        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( blas::vec( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        ::tbb::parallel_for(
            uint(0), u.nblocks(),
            [&] ( const uint  i )
            {
                add_uniform_to_scalar( *u.block(i), v );
            }
        );
    }// if
}

//
// generate mapping of index set to mutices for leaf clusters
//
template < typename value_t >
void
build_mutex_map ( const shared_cluster_basis< value_t > &  cb,
                  mutex_map_t &                            mtx_map )
{
    mtx_map[ cb.is() ] = std::make_unique< std::mutex >();

    for ( uint  i = 0; i < cb.nsons(); ++i )
        build_mutex_map( *cb.son(i), mtx_map );
}

}}}}// namespace hlr::tbb::uniform::detail

#endif // __HLR_TBB_ARITH_UNIFORM_MVM_HH
