#ifndef __HLR_ARITH_H2_HH
#define __HLR_ARITH_H2_HH
//
// Project     : HLib
// Module      : arith/h2
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#endif

#if defined(HAS_H2)

#include <hpro/cluster/TClusterBasis.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TUniformMatrix.hh>
#include <hpro/vector/convert.hh>

#include <hlr/vector/uniform_vector.hh>
#include <hlr/vector/scalar_vector.hh>

namespace hlr { namespace h2 {

template < typename value_t >
using nested_cluster_basis = Hpro::TClusterBasis< value_t >;

namespace detail
{

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

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
          scalar_vector< value_t > &                                 sy )
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
    else if ( is_dense( M ) )
    {
        auto  D   = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas::mat< value_t >( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( Hpro::is_uniform( &M ) )
    {
        auto  R = cptrcast( &M, Hpro::TUniformMatrix< value_t > );
        
        if ( op_M == Hpro::apply_normal )
        {
            blas::mulvec( alpha, Hpro::coeff< value_t >( R ), x.coeffs(), value_t(1), y.coeffs() );
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
            blas::mulvec( alpha, blas::adjoint( Hpro::coeff< value_t >( R ) ), x.coeffs(), value_t(1), y.coeffs() );
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
        
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb - v.ofs() );
        auto  s    = blas::mulvec( blas::adjoint( cb.basis() ), v_cb );

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
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< nested_cluster_basis< value_t > > >
make_uniform ( const nested_cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< nested_cluster_basis< value_t > > >( cb, cb );

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, make_uniform( *cb.son(i) ).release() );
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
        auto  v_loc = blas::vector( blas::vec( v ), u.basis() - v.ofs() );

        blas::mulvec( value_t(1), u.basis().basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        for ( uint  i = 0; i < u.basis().nsons(); ++i )
        {
            auto  u_i = u.block( i );
            auto  s_i = u.basis().transfer_to_son( i, s );
            
            add_uniform_to_scalar( *u_i, v, s_i );
        }// for
    }// else
}

}// namespace detail

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

    auto  ux = detail::scalar_to_uniform( op_M == Hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == Hpro::apply_normal ? rowcb : colcb );
    auto  s  = blas::vector< value_t >();

    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y, s );
}

}}// namespace hlr::h2

#endif // HAS_H2

#endif // __HLR_ARITH_H2_HH
