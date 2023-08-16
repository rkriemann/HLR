#ifndef __HLR_ARITH_H2_HH
#define __HLR_ARITH_H2_HH
//
// Project     : HLR
// Module      : arith/h2
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)
#define HLR_HAS_H2
#endif

#if defined(HLR_HAS_H2)

#include <hpro/cluster/TClusterBasis.hh>
#include <hpro/matrix/TUniformMatrix.hh>
#include <hpro/vector/convert.hh>

#endif

#include <hlr/matrix/nested_cluster_basis.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/vector/scalar_vector.hh>

namespace hlr { namespace h2 {

namespace detail
{

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t,
           typename cluster_basis_t >
void
mul_vec ( const value_t                              alpha,
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

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            auto  y_i = y.block( i );
            
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j, op_M );
                auto  x_j  = x.block( j );
            
                if ( ! is_null( B_ij ) )
                {
                    mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                }// if
            }// for
        }// for
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );

        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );

        switch ( op_M )
        {
            case Hpro::apply_normal     : blas::mulvec( alpha, R->coupling(), x.coeffs(), value_t(1), y.coeffs() ); break;
            case Hpro::apply_conjugate  : HLR_ASSERT( false );
            case Hpro::apply_transposed : HLR_ASSERT( false );
            case Hpro::apply_adjoint    : blas::mulvec( alpha, blas::adjoint( R->coupling() ), x.coeffs(), value_t(1), y.coeffs() ); break;
            default                     : HLR_ERROR( "unsupported matrix operator" );
        }// switch
    }// if
    #if defined(HLR_HAS_H2)
    else if ( Hpro::is_uniform( &M ) )
    {
        auto  R = cptrcast( &M, Hpro::TUniformMatrix< value_t > );
        
        switch ( op_M )
        {
            case Hpro::apply_normal     : blas::mulvec( alpha, Hpro::coeff< value_t >( R ), x.coeffs(), value_t(1), y.coeffs() ); break;
            case Hpro::apply_conjugate  : HLR_ASSERT( false );
            case Hpro::apply_transposed : HLR_ASSERT( false );
            case Hpro::apply_adjoint    : blas::mulvec( alpha, blas::adjoint( Hpro::coeff< value_t >( R ) ), x.coeffs(), value_t(1), y.coeffs() ); break;
            default                     : HLR_ERROR( "unsupported matrix operator" );
        }// switch
    }// if
    #endif
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

        auto  s = blas::vector< value_t >( cb.rank() );
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto  u_i = scalar_to_uniform( *cb.son(i), v );

            if ( cb.rank() > 0 )
                blas::mulvec( blas::adjoint( cb.transfer_mat(i) ), u_i->coeffs(), s );
            
            u->set_block( i, u_i.release() );
        }// for

        u->set_coeffs( std::move( s ) );
    }// else

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
        auto  v_loc = blas::vector( blas::vec( v ), cb.is() - v.ofs() );

        blas::mulvec( value_t(1), cb.basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto  u_i = u.block( i );
            auto  s_i = cb.transfer_to_son( i, s );
            
            add_uniform_to_scalar( *u_i, v, s_i );
        }// for
    }// else
}

}// namespace detail

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

}}// namespace hlr::h2

#endif // __HLR_ARITH_H2_HH
