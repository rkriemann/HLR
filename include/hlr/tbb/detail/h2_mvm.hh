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

#include <hlr/arith/h2.hh>

namespace hlr { namespace tbb { namespace h2 { namespace detail {

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

using  mutex_map_t = std::unordered_map< indexset, std::unique_ptr< std::mutex >, indexset_hash >;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_mtx ( const value_t                              alpha,
              const Hpro::matop_t                        op_M,
              const Hpro::TMatrix< value_t > &           M,
              const uniform_vector< cluster_basis_t > &  x,
              uniform_vector< cluster_basis_t > &        y,
              const scalar_vector< value_t > &           sx,
              scalar_vector< value_t > &                 sy,
              mutex_map_t &                              mtx_map )
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
                            
                            mul_vec_mtx( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy, mtx_map );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i  = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  mtx  = mtx_map[ M.row_is( op_M ) ].get();
        auto  lock = std::scoped_lock( *mtx );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        
        if ( op_M == Hpro::apply_normal )
        {
            std::scoped_lock  lock( y.mutex() );
            
            blas::mulvec( alpha, R->coupling(), x.coeffs(), value_t(1), y.coeffs() );
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
            
            blas::mulvec( alpha, blas::adjoint( R->coupling() ), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    #if defined(HLR_HAS_H2)
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
    #endif
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_row ( const value_t                              alpha,
              const Hpro::matop_t                        op_M,
              const Hpro::TMatrix< value_t > &           M,
              const uniform_vector< cluster_basis_t > &  x,
              uniform_vector< cluster_basis_t > &        y,
              const scalar_vector< value_t > &           sx,
              scalar_vector< value_t > &                 sy )
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
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );

        #if HLR_COMPRESSOR == HLR_COMPRESSOR_AFLP || HLR_COMPRESSOR == HLR_COMPRESSOR_DFL
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     : compress::blas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoupling(), x.coeffs().data(), y.coeffs().data() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : compress::blas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoupling(), x.coeffs().data(), y.coeffs().data() ); break;
                default               : HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     : blas::mulvec( alpha, R->coupling(), x.coeffs(), value_t(1), y.coeffs() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : blas::mulvec( alpha, blas::adjoint(R->coupling()), x.coeffs(), value_t(1), y.coeffs() ); break;
                default               : HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t,
           typename cluster_basis_t >
std::unique_ptr< uniform_vector< cluster_basis_t > >
scalar_to_uniform ( const cluster_basis_t &           cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis_t > >( cb );

    if ( cb.nsons() == 0 )
    {
        auto  V = cb.basis();
        
        if ( V.nrows() == 0 )
            return u;
        
        //
        // s ≔ V'·v
        //
        
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
        auto  s    = blas::mulvec( blas::adjoint( V ), v_cb );

        u->set_coeffs( std::move( s ) );
    }// if
    else
    {
        //
        // s ≔ V'·v = (∑_i V_i E_i)' v = ∑_i E_i' V_i' v
        //

        #if 0

        //
        // summation of coefficients via mutex
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

        #else

        //
        // lock free with sequential summation of coefficients
        //
        
        auto  s  = blas::vector< value_t >( cb.rank() );
        auto  Si = std::vector< blas::vector< value_t > >( cb.nsons() );
        
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                auto  u_i = scalar_to_uniform( *cb.son(i), v );
                
                if ( cb.rank() > 0 )
                    Si[i] = std::move( blas::mulvec( blas::adjoint( cb.transfer_mat(i) ), u_i->coeffs() ) );
                
                u->set_block( i, u_i.release() );
            } );

        for ( auto  si : Si )
            blas::add( 1, si, s );
        
        #endif
        
        u->set_coeffs( std::move( s ) );
    }// else

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t,
           typename cluster_basis_t >
std::unique_ptr< uniform_vector< cluster_basis_t > >
make_uniform ( const cluster_basis_t &  cb )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis_t > >( cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                u->set_block( i, make_uniform< value_t, cluster_basis_t >( *cb.son(i) ).release() );
            } );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t,
           typename cluster_basis_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis_t > &  u,
                        scalar_vector< value_t > &                 v,
                        blas::vector< value_t > &                  s )
{
    if ( s.length() > 0 )
        blas::add( value_t(1), u.coeffs(), s );
    else
        s = blas::copy( u.coeffs() );

    auto &  cb = u.basis();

    if ( cb.nsons() == 0 )
    {
        auto  v_loc = blas::vector< value_t >( blas::vec< value_t >( v ), cb.is() - v.ofs() );

        blas::mulvec( value_t(1), cb.basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        ::tbb::parallel_for< uint >(
            uint(0), u.nblocks(),
            [&] ( const uint  i )
            {
                auto  u_i = u.block( i );
                auto  s_i = cb.transfer_to_son( i, s );
                
                add_uniform_to_scalar( *u_i, v, s_i );
            } );
    }// else
}

//
// generate mapping of index set to mutices for leaf clusters
//
template < typename cluster_basis_t >
void
build_mutex_map ( const cluster_basis_t &  cb,
                  mutex_map_t &            mtx_map )
{
    if ( cb.nsons() == 0 )
    {
        mtx_map[ cb.is() ] = std::make_unique< std::mutex >();
    }// if
    else
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            build_mutex_map< cluster_basis_t >( *cb.son(i), mtx_map );
    }// else
}

}}}} // namespace hlr::tbb::h2::detail

#endif // __HLR_ARITH_H2_HH
