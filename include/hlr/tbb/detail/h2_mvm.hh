#ifndef __HLR_TBB_DETAIL_ARITH_H2_HH
#define __HLR_TBB_DETAIL_ARITH_H2_HH
//
// Project     : HLR
// Module      : tbb/detail/h2_mvm.hh
// Description : matrix-vector multiplication for H² matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_for.h>

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)
#define HLR_HAS_H2
#endif

#if defined(HLR_HAS_H2)

#include <hlr/arith/h2.hh>

namespace hlr { namespace tbb { namespace h2 { namespace detail {

template < typename value_t >
using nested_cluster_basis = Hpro::TClusterBasis< value_t >;

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

using  mutex_map_t = std::unordered_map< indexset, std::unique_ptr< std::mutex >, indexset_hash >;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec ( const value_t                                              alpha,
          const Hpro::matop_t                                        op_M,
          const Hpro::TMatrix< value_t > &                           M,
          const uniform_vector< nested_cluster_basis< value_t > > &  x,
          uniform_vector< nested_cluster_basis< value_t > > &        y,
          const scalar_vector< value_t > &                           sx,
          scalar_vector< value_t > &                                 sy,
          mutex_map_t &                                              mtx_map )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
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
        auto  D    = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  x_i  = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  mtx  = mtx_map[ M.row_is( op_M ) ].get();
        auto  lock = std::scoped_lock( *mtx );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas::mat( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( Hpro::is_uniform( &M ) )
    {
        auto  R = cptrcast( &M, Hpro::TUniformMatrix< value_t > );
        
        if ( op_M == Hpro::apply_normal )
        {
            std::scoped_lock  lock( y.mutex() );
            
            blas::mulvec( alpha, Hpro::coeff( R ), x.coeffs(), value_t(1), y.coeffs() );
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
            std::scoped_lock  lock( y.mutex() );
            
            blas::mulvec( alpha, blas::adjoint( Hpro::coeff( R ) ), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< nested_cluster_basis< value_t > > >
scalar_to_uniform ( const nested_cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &         v )
{
    auto  u = std::make_unique< uniform_vector< nested_cluster_basis< value_t > > >( cb, cb );

    if ( cb.nsons() == 0 )
    {
        //
        // s ≔ V'·v
        //
        
        auto  v_cb = blas::vector< value_t >( blas::vec< value_t >( v ), cb - v.ofs() );
        auto  s    = blas::mulvec( blas::adjoint( cb.basis() ), v_cb );

        u->set_coeffs( std::move( s ) );
    }// if
    else
    {
        //
        // s ≔ V'·v = (∑_i V_i E_i)' v = ∑_i E_i' V_i' v
        //

        auto  s     = blas::vector< value_t >( cb.rank() );
        auto  s_mtx = std::mutex();
        
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                auto  u_i = scalar_to_uniform( *cb.son(i), v );
                
                if ( cb.rank() > 0 )
                {
                    auto  lock = std::scoped_lock( s_mtx );
                    
                    blas::mulvec( blas::adjoint( cb.transfer_mat(i) ), u_i->coeffs(), s );
                }// if
                
                u->set_block( i, u_i.release() );
            } );
        
        u->set_coeffs( std::move( s ) );
    }// else

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< nested_cluster_basis< value_t > > >
make_uniform ( const nested_cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< nested_cluster_basis< value_t > > >( cb, cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
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
add_uniform_to_scalar ( const uniform_vector< nested_cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                                 v,
                        blas::vector< value_t > &                                  s )
{
    if ( s.length() > 0 )
        blas::add( value_t(1), u.coeffs(), s );
    else
        s = blas::copy( u.coeffs() );

    if ( u.basis().nsons() == 0 )
    {
        auto  v_loc = blas::vector( blas::vec< value_t >( v ), u.basis() - v.ofs() );

        blas::mulvec( value_t(1), u.basis().basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        ::tbb::parallel_for< uint >(
            uint(0), u.nblocks(),
            [&] ( const uint  i )
            {
                auto  u_i = u.block( i );
                auto  s_i = u.basis().transfer_to_son( i, s );
                
                add_uniform_to_scalar( *u_i, v, s_i );
            } );
    }// else
}

//
// generate mapping of index set to mutices for leaf clusters
//
template < typename value_t >
void
build_mutex_map ( const nested_cluster_basis< value_t > &  cb,
                  mutex_map_t &                            mtx_map )
{
    if ( cb.nsons() == 0 )
    {
        mtx_map[ cb ] = std::make_unique< std::mutex >();
    }// if
    else
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            build_mutex_map( *cb.son(i), mtx_map );
    }// else
}

}}}} // namespace hlr::tbb::h2::detail

#endif // HLR_HAS_H2

#endif // __HLR_ARITH_H2_HH
