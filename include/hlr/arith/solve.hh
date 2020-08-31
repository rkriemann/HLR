#ifndef __HLR_ARITH_SOLVE_HH
#define __HLR_ARITH_SOLVE_HH
//
// Project     : HLib
// File        : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/utils/log.hh>

namespace hlr
{

namespace hpro = HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with lower triangular matrix L
//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
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
                    solve_lower_tri( side, *L_ii, *M_ij, acc, approx );
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
                                             M.block(k,j), acc, approx );
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

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  L,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = TDenseMatrix( M.row_is(), hpro::is( 0, M.rank() ), blas::mat_U< value_t >( M ) );

        solve_lower_tri( side, L, U, acc, approx );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  L,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint i = 0; i < L.nblock_cols(); ++i )
        {
            const auto  L_ii = L.block( i, i );

            HLR_ASSERT( ! is_null( L_ii ) );

            auto  M_i = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                 L_ii->col_is() - L.col_ofs(),
                                                 blas::range::all );
            auto  D_i = hpro::TDenseMatrix( M_i, L_ii->col_is(), M.col_is() );
                
            solve_lower_tri( side, *L_ii, D_i, acc, approx );

            for ( uint  k = i+1; k < L.nblock_rows(); ++k )
            {
                auto  L_ki = L.block( k, i );
                
                if ( ! is_null( L_ki ) )
                {
                    auto  M_k = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                         L_ki->row_is() - L.row_ofs(),
                                                         blas::range::all );
                    auto  D_i = hpro::TDenseMatrix( M_i, L_ki->row_is(), M.col_is() );
                    
                    multiply< value_t >( value_t(-1),
                                         apply_normal, *L_ki,
                                         apply_normal, M_i,
                                         *M_ki, acc, approx );
                }// for
            }// for
        }// for
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        HLR_ERROR( "todo" );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  L,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = TDenseMatrix( M.row_is(), hpro::is( 0, M.rank() ), blas::mat_U< value_t >( M ) );

        solve_lower_tri( side, L, U, acc, approx );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  L,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        auto  Mc = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), hpro::blas_mat< value_t >( L ), Mc, value_t(0), hpro::blas_mat< value_t >( M ) );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

//
// matrix type based deduction of special versions
//
template < typename value_t,
           typename approx_t,
           typename matrix_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const matrix_t &         L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked( M ) )
        solve_lower_tri( side, L, * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
    else if ( is_lowrank( M ) )
        solve_lower_tri( side, L, * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
    else if ( is_dense( M ) )
        solve_lower_tri( side, L, * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
    else
        HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked( L ) )
        solve_lower_tri( side, * cptrcast( & L, hpro::TBlockMatrix ), M, acc, approx );
    else if ( is_dense( L ) )
        solve_lower_tri( side, * cptrcast( & L, hpro::TDenseMatrix ), M, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower triangular matrix L
//
// solve op(L) x = v 
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

void
trsvl ( const hpro::matop_t      op_L,
        const hpro::TMatrix &    L,
        hpro::TScalarVector &    v,
        const hpro::diag_type_t  diag_mode )
{
    HLR_LOG( 4, HLIB::to_string( "trsvl( %d )", L.id() ) );
        
    if ( is_blocked( L ) )
    {
        auto        BL  = cptrcast( & L, hpro::TBlockMatrix );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();
            
        if ( op_L == hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->col_is() );
                
                    trsvl( op_L, *L_ii, v_i, diag_mode );
                }// if
            
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );

                    if ( ! is_null( L_ji ) )
                    {
                        auto  v_j = v.sub_vector( L_ji->row_is() );
                        auto  v_i = v.sub_vector( L_ji->col_is() );
                    
                        mul_vec< real >( real(-1), op_L, *L_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min( nbr, nbc )-1; i >= 0; --i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->row_is() );
                
                    trsvl( op_L, *L_ii, v_i, diag_mode );
                }// if

                for ( int  j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        auto  v_i = v.sub_vector( L_ij->col_is() );
                        auto  v_j = v.sub_vector( L_ij->row_is() );
                    
                        mul_vec( real(-1), op_L, *L_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( L ) )
    {
        if ( diag_mode == hpro::general_diag )
        {
            //
            // assuming L contains inverse (store_inverse!)
            //

            auto  vc = hpro::TScalarVector( v );

            v.scale( 0 );
            mul_vec( real(1), op_L, L, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with upper triangular matrix U
//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
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
            
            for ( int i = 0; i < M.nblock_rows(); ++i )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_upper_tri( side, U_jj, M_ij, acc, approx );
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
                                             M.block(i,k), acc, approx );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  U,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = TDenseMatrix( M.col_is(), hpro::is( 0, M.rank() ), blas::mat_V< value_t >( M ) );

        solve_upper_tri( from_left, U, V, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TBlockMatrix &  U,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  U,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = TDenseMatrix( M.col_is(), hpro::is( 0, M.rank() ), blas::mat_V< value_t >( M ) );

        solve_upper_tri( from_left, U, V, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const hpro::TDenseMatrix &  U,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    if ( side == from_left )
    {
        // assuming U' X = M
        auto  Mc = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( U ) ), Mc, value_t(0), hpro::blas_mat< value_t >( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), Mc, hpro::blas_mat< value_t >( U ), value_t(0), hpro::blas_mat< value_t >( M ) );
    }// else
}

//
// matrix type based deduction of special versions
//
template < typename value_t,
           typename approx_t,
           typename matrix_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const matrix_t &         U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked( M ) )
        solve_upper_tri( side, U, * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
    else if ( is_lowrank( M ) )
        solve_upper_tri( side, U, * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
    else if ( is_dense( M ) )
        solve_upper_tri( side, U, * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
    else
        HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked( U ) )
        solve_upper_tri( side, * cptrcast( & U, hpro::TBlockMatrix ), M, acc, approx );
    else if ( is_dense( U ) )
        solve_upper_tri( side, * cptrcast( & U, hpro::TDenseMatrix ), M, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with upper triangular matrix U
//
// solve op(U) x = v
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

void
trsvu ( const hpro::matop_t      op_U,
        const hpro::TMatrix &    U,
        hpro::TScalarVector &    v,
        const hpro::diag_type_t  diag_mode )
{
    HLR_LOG( 4, HLIB::to_string( "trsvu( %d )", U.id() ) );
        
    if ( is_blocked( U ) )
    {
        auto        BU  = cptrcast( & U, hpro::TBlockMatrix );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();
            
        if ( op_U == hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min<int>( nbr, nbc )-1; i >= 0; --i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->col_is() );
                
                    trsvu( op_U, *U_ii, v_i, diag_mode );
                }// if
            
                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );

                    if ( ! is_null( U_ji ) )
                    {
                        auto  v_j = v.sub_vector( U_ji->row_is() );
                        auto  v_i = v.sub_vector( U_ji->col_is() );
                    
                        mul_vec( real(-1), op_U, *U_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->row_is() );
                
                    trsvu( op_U, *U_ii, v_i, diag_mode );
                }// if

                for ( uint  j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        auto  v_i = v.sub_vector( U_ij->col_is() );
                        auto  v_j = v.sub_vector( U_ij->row_is() );
                    
                        HLR_LOG( 4, HLIB::to_string( "update( %d, ", U_ij->id() ) + v_j.is().to_string() + ", " + v_i.is().to_string() );
                        mul_vec( real(-1), op_U, *U_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( U ) )
    {
        if ( diag_mode == hpro::general_diag )
        {
            //
            // assuming U contains inverse (store_inverse!)
            //

            auto  vc = hpro::TScalarVector( v );

            v.scale( 0 );
            mul_vec( real(1), op_U, U, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const hpro::TDenseMatrix *  U,
         hpro::TMatrix *             X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d, %d )", U->id(), X->id() ) );
    
    if ( is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, hpro::TRkMatrix );
        auto  Y  = blas::copy( hpro::blas_mat_B< value_t >( RX ) );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( U ) ), Y,
                    value_t(0), hpro::blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, hpro::TDenseMatrix );
        auto  Y  = copy( hpro::blas_mat< value_t >( DX ) );
    
        blas::prod( value_t(1), Y, hpro::blas_mat< value_t >( U ),
                    value_t(0), hpro::blas_mat< value_t >( DX ) );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_SOLVE_HH
