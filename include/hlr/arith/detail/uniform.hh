#ifndef __HLR_ARITH_DETAIL_UNIFORM_HH
#define __HLR_ARITH_DETAIL_UNIFORM_HH
//
// Project     : HLib
// Module      : arith/uniform
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>

namespace hlr { namespace uniform { namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// mat-vec : y = y + α op( M ) x
//
////////////////////////////////////////////////////////////////////////////////

using matrix::cluster_basis;
using vector::scalar_vector;
using vector::uniform_vector;

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
          scalar_vector< value_t > &                          sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

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
        auto  D   = cptrcast( &M, hpro::TDenseMatrix );
        auto  x_i = blas::vector< value_t >( blas::vec< value_t >( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec< value_t >( sy ), M.row_is( op_M ) - sy.ofs() );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas::mat< value_t >( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        
        if ( op_M == hpro::apply_normal )
        {
            blas::mulvec( value_t(1), R->coeff(), x.coeffs(), value_t(1), y.coeffs() );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            HLR_ASSERT( false );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
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
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( blas::vec< value_t >( v ), cb.cluster() - v.ofs() );
        auto  s    = cb.transform_forward( v_cb );

        u->set_coeffs( std::move( s ) );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
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
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

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
add_uniform_to_scalar ( const uniform_vector< cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                          v )
{
    if ( u.basis().rank() > 0 )
    {
        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( blas::vec< value_t >( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        for ( uint  i = 0; i < u.nblocks(); ++i )
            add_uniform_to_scalar( *u.block(i), v );
    }// if
}

}// namespace detail


////////////////////////////////////////////////////////////
//
// LU factorization
//
////////////////////////////////////////////////////////////

using  indexset = hpro::TIndexSet;

namespace detail
{

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

// //
// // matrix multiplication C := α·A·B + C
// //
// template < typename value_t >
// void
// multiply ( const value_t               alpha,
//            const hpro::matop_t         op_A,
//            const hpro::TBlockMatrix &  A,
//            const hpro::matop_t         op_B,
//            const hpro::TBlockMatrix &  B,
//            hpro::TBlockMatrix &        C,
//            const hpro::TTruncAcc &     acc,
//            const uniform_map_t &       rowmap,
//            const uniform_map_t &       colmap )
// {
//     for ( uint  i = 0; i < C.nblock_rows(); ++i )
//     {
//         for ( uint  j = 0; j < C.nblock_cols(); ++j )
//         {
//             HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
//             for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
//             {
//                 if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
//                     multiply< value_t >( alpha,
//                                          op_A, *A.block( i, l, op_A ),
//                                          op_B, *B.block( l, j, op_B ),
//                                          *C.block( i, j ), acc, rowmap, colmap );
//             }// if       
//         }// for
//     }// for
// }

// template < typename value_t >
// void
// multiply ( const value_t                          alpha,
//            const hpro::matop_t                    op_A,
//            const hpro::TBlockMatrix &             A,
//            const hpro::matop_t                    op_B,
//            const hpro::TBlockMatrix &             B,
//            matrix::uniform_lrmatrix< value_t > &  C,
//            const hpro::TTruncAcc &                acc,
//            const uniform_map_t &                  rowmap,
//            const uniform_map_t &                  colmap )
// {
// }

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const uniform_map_t &    rowmap,
           const uniform_map_t &    colmap )
{
    // if ( is_blocked( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha, 
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else if ( matrix::is_uniform_lowrank( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else if ( is_dense( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else
    //     HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
template < typename value_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap );

template < typename value_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const uniform_map_t &       rowmap,
                  const uniform_map_t &       colmap )
{
    HLR_LOG( 4, hpro::to_string( "svltr( B %d, B %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint i = 0; i < M.nblock_rows(); ++i )
        {
            const auto  L_ii = L.block( i, i );

            HLR_ASSERT( ! is_null( L_ii ) );
            
            for ( uint j = 0; j < M.nblock_cols(); ++j )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, rowmap, colmap );
            }// for

            for ( uint  k = i+1; k < M.nblock_rows(); ++k )
            {
                for ( uint  j = 0; j < M.nblock_cols(); ++j )
                {
                    if ( ! is_null_any( L.block(k,i), M.block(i,j) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(k,j) ) );
                        
                        multiply< value_t >( value_t(-1),
                                             apply_normal, *L.block(k,i),
                                             apply_normal, *M.block(i,j),
                                             *M.block(k,j),
                                             acc, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const hpro::TMatrix &                  L,
                  matrix::uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &                acc,
                  const uniform_map_t &                  rowmap,
                  const uniform_map_t &                  colmap )
{
    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L W T X' = U S V'
        // as L W = U
        //

        auto  U = blas::copy( M.row_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.row_is(), hpro::is( 0, U.ncols()-1 ), U );

        hlr::solve_lower_tri< value_t >( side, diag, L, D );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap )
{
    if ( is_blocked( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( matrix::is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( matrix::is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
template < typename value_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap );

template < typename value_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const uniform_map_t &       rowmap,
                  const uniform_map_t &       colmap )
{
    HLR_LOG( 4, hpro::to_string( "svutr( B %d, B %d )", U.id(), M.id() ) );
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        for ( uint j = 0; j < M.nblock_cols(); ++j )
        {
            const auto  U_jj = U.block( j, j );

            HLR_ASSERT( ! is_null( U_jj ) );
            
            for ( uint i = 0; i < M.nblock_rows(); ++i )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, rowmap, colmap );
            }// for
            
            for ( uint  k = j+1; k < M.nblock_cols(); ++k )
            {
                for ( uint  i = 0; i < M.nblock_rows(); ++i )
                {
                    if ( ! is_null_any( M.block(i,j), U.block(j,k) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(i,k) ) );
                        
                        multiply< value_t >( value_t(-1),
                                             apply_normal, *M.block(i,j),
                                             apply_normal, *U.block(j,k),
                                             *M.block(i,k),
                                             acc, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const hpro::TMatrix &                  U,
                  matrix::uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &                acc,
                  const uniform_map_t &                  rowmap,
                  const uniform_map_t &                  colmap )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve W T X' R = U S V', e.g., X' R = V', as R' X = V
        //

        auto  V = blas::copy( M.col_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.col_is(), hpro::is( 0, V.ncols()-1 ), V );

        hlr::solve_upper_tri< value_t >( from_left, diag, U, D );
    }// else
}

    template < typename value_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap )
{
    if ( is_blocked( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( matrix::is_uniform_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

//
// recursive LU factorization
//
template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const uniform_map_t &    rowmap,
     const uniform_map_t &    colmap )
{
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, hpro::TBlockMatrix );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), acc, rowmap, colmap );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag,
                                                *BA->block( i, i ), *BA->block( j, i ),
                                                acc, rowmap, colmap );
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag,
                                                *BA->block( i, i ), *BA->block( i, j ),
                                                acc, rowmap, colmap );
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                {
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                    {
                        HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                    
                        multiply( value_t(-1),
                                  apply_normal, *BA->block( j, i ),
                                  apply_normal, *BA->block( i, l ),
                                  *BA->block( j, l ),
                                  acc, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else if ( is_dense( A ) )
    {
        auto  D = ptrcast( &A, hpro::TDenseMatrix );

        invert< value_t >( *D );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}}}// namespace hlr::uniform::detail

#endif // __HLR_ARITH_DETAIL_UNIFORM_HH
