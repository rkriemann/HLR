#ifndef __HLR_SEQ_ARITH_UNIFORM_HH
#define __HLR_SEQ_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : seq/arith_uniform.hh
// Description : sequential arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalarvector.hh>
#include <hlr/vector/uniform_vector.hh>

namespace hlr { namespace seq { namespace uniform {

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

using matrix::cluster_basis;
using vector::scalarvector;
using vector::uniform_vector;

template < typename value_t >
void
mul_vec ( const value_t                                       alpha,
          const hpro::matop_t                                 op_M,
          const hpro::TMatrix &                               M,
          const uniform_vector< cluster_basis< value_t > > &  x,
          uniform_vector< cluster_basis< value_t > > &        y )
{
    // if ( is_blocked( M ) )
    // {
    //     auto        B       = cptrcast( &M, TBlockMatrix );
    //     const auto  row_ofs = B->row_is( op_M ).first();
    //     const auto  col_ofs = B->col_is( op_M ).first();

    //     for ( uint  i = 0; i < B->nblock_rows(); ++i )
    //     {
    //         for ( uint  j = 0; j < B->nblock_cols(); ++j )
    //         {
    //             auto  B_ij = B->block( i, j );
                
    //             if ( ! is_null( B_ij ) )
    //             {
    //                 auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
    //                 auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );

    //                 mul_vec( alpha, op_M, *B_ij, x_j, y_i );
    //             }// if
    //         }// for
    //     }// for
    // }// if
    // else if ( is_dense( M ) )
    // {
    //     auto  D = cptrcast( &M, TDenseMatrix );
        
    //     blas::mulvec( alpha, blas::mat_view( op_M, blas_mat< value_t >( D ) ), x, value_t(1), y );
    // }// if
    // else if ( matrix::is_uniform_lowrank( M ) )
    // {
    //     auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        
    //     if ( op_M == hpro::apply_normal )
    //     {
    //         blas::mulvec( value_t(1), R->coeff(), t );
    //     }// if
    //     else if ( op_M == hpro::apply_transposed )
    //     {
    //         auto  s = blas::mulvec( value_t(1), blas::transposed(R->coeff()), t );
    //     }// if
    //     else if ( op_M == hpro::apply_adjoint )
    //     {
    //         auto  s = blas::mulvec( value_t(1), blas::adjoint(R->coeff()), t );
    //     }// if
    // }// if
    // else
    //     assert( false );
}

template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
copy_scalar_to_uniform ( const cluster_basis< value_t > &  cb,
                         const scalarvector &              v )
{
}
    
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
make_uniform ( const cluster_basis< value_t > &  cb )
{
}
    
template < typename value_t >
void
add_uniform ( const uniform_vector< cluster_basis< value_t > > &  x,
              scalarvector &                                      y )
{
}
    
}// namespace detail

template < typename value_t >
void
mul_vec ( const value_t                       alpha,
          const matop_t                       op_M,
          const TMatrix &                     M,
          const hpro::TVector &               x,
          hpro::TVector &                     y,
          matrix::cluster_basis< value_t > &  rowcb,
          matrix::cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( vector::is_scalar( x ) && vector::is_scalar( y ) );
    
    //
    // construct uniform representation of y
    //

    auto  sx = cptrcast( & x, vector::scalarvector );
    auto  sy = ptrcast(  & y, vector::scalarvector );

    auto  ux = detail::copy_scalar_to_uniform( op_M == hpro::apply_normal ? rowcb : colcb, * sx );
    auto  uy = detail::make_uniform( op_M == hpro::apply_normal ? colcb : rowcb );

    detail::mul_vec( alpha, op_M, M, *ux, *uy );
    detail::add_uniform( *uy, *sy );
}

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
