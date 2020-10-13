#ifndef __HLR_SEQ_ARITH_HH
#define __HLR_SEQ_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_fac.hh>
#include <hpro/algebra/solve_tri.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/mulvec.hh"
#include "hlr/arith/solve.hh"
#include "hlr/arith/invert.hh"
#include "hlr/approx/svd.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/vector/scalar_vector.hh"
#include "hlr/seq/matrix.hh"
#include "hlr/seq/norm.hh"

namespace hlr { namespace seq {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const matop_t                    op_M,
          const hpro::TMatrix &            M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    hlr::mul_vec< value_t >( alpha, op_M, M, x, y );
}

//
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const matop_t                             op_M,
          const hpro::TMatrix &                     M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );

    mul_vec( alpha, op_M, M, hpro::blas_vec< value_t >( x ), hpro::blas_vec< value_t >( y ) );
}

//
// compute C = C + α op( A ) op( B )
//
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
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                }// for
            }// for
        }// for
    }// if
    else
    {
        hlr::multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx );
    }// else
}

//
// compute C = C + α · op( A ) · op( D ) · op( B )
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const matop_t            op_A,
           const hpro::TMatrix &    A,
           const matop_t            op_D,
           const hpro::TMatrix &    D,
           const matop_t            op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    hlr::multiply< value_t >( alpha, op_A, A, op_D, D, op_B, B, C, acc, approx );
}

//
// compute C = C + α op( A ) op( B )
//
template < typename value_t >
void
multiply_apx ( const value_t            alpha,
               const matop_t            op_A,
               const hpro::TMatrix &    A,
               const matop_t            op_B,
               const hpro::TMatrix &    B,
               hpro::TMatrix &          C,
               const hpro::TTruncAcc &  acc,
               typename hpro::real_type< value_t >::type_t  tol )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_rows( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_apx< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, tol );
                }// for
            }// for
        }// for
    }// if
    else
    {
        if ( is_lowrank( C ) )
        {
            //
            // look for Frobenius norm of factors and return if too small
            //

            const auto  norm_A = norm::norm_F( A );
            const auto  norm_B = norm::norm_F( B );

            if ( norm_A * norm_B < tol )
                return;
        }// if
        
        hpro::multiply< value_t >( alpha, op_A, &A, op_B, &B, value_t(1), &C, acc );
    }// else
}

//
// compute Hadamard product A = α A*B 
//
template < typename value_t,
           typename approx_t >
void
multiply_hadamard ( const value_t            alpha,
                    hpro::TMatrix &          A,
                    const hpro::TMatrix &    B,
                    const hpro::TTruncAcc &  acc,
                    const approx_t &         approx )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = ptrcast( &A,  hpro::TBlockMatrix );
        auto  BB = cptrcast( &B, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                auto  A_ij = BA->block( i, j );
                auto  B_ij = BB->block( i, j );
                
                HLR_ASSERT( ! is_null_any( A_ij, B_ij ) );
            
                multiply_hadamard< value_t >( alpha, *A_ij, *B_ij, acc, approx );
            }// for
        }// for
    }// if
    else if ( is_dense_all( A, B ) )
    {
        auto        DA     = ptrcast( &A,  hpro::TDenseMatrix );
        auto        DB     = cptrcast( &B, hpro::TDenseMatrix );
        auto        blas_A = hpro::blas_mat< value_t >( DA );
        auto        blas_B = hpro::blas_mat< value_t >( DB );
        const auto  nrows  = DA->nrows();
        const auto  ncols  = DA->ncols();

        for ( size_t  i = 0; i < nrows*ncols; ++i )
            blas_A.data()[i] *= alpha * blas_B.data()[i];
    }// if
    else if ( is_lowrank_all( A, B ) )
    {
        auto  RA = ptrcast( &A,  hpro::TRkMatrix );
        auto  RB = cptrcast( &B, hpro::TRkMatrix );

        //
        // construct product with rank rank(A)·rank(B) and fill
        // new low-rank vectors based on hadamard product
        //
        //  a_ij · b_ij = ( Σ_l u^l_i · v^l_j ) ( Σ_k w^k_i · x^k_j )
        //              = Σ_l Σ_k ( u^l_i · w^k_i ) ( v^l_j · x^k_j )
        //
        //  i.e., C = Y·Z' with y^p_i = u^l_i · w^k_i and
        //                      z^p_j = v^l_j · x^k_j
        //
        //  with p = l·rank(B)+k
        //

        auto  rank_A = RA->rank();
        auto  rank_B = RB->rank();
        auto  rank   = rank_A * rank_B;

        auto  nrows = RA->nrows();
        auto  ncols = RA->ncols();

        auto  U = blas::mat_U< value_t >( RA );
        auto  V = blas::mat_V< value_t >( RA );
        auto  W = blas::mat_U< value_t >( RB );
        auto  X = blas::mat_V< value_t >( RB );
        auto  Y = blas::matrix< value_t >( nrows, rank );
        auto  Z = blas::matrix< value_t >( ncols, rank );

        uint  p = 0;
        
        for ( uint  l = 0; l < rank_A; ++l )
        {
            auto  u_l = U.column( l );
            auto  v_l = V.column( l );
                
            for ( uint  k = 0; k < rank_B; ++k, ++p )
            {
                auto  w_k = W.column( k );
                auto  x_k = X.column( k );
                auto  y_p = Y.column( p );
                auto  z_p = Z.column( p );

                for ( size_t  i = 0; i < nrows; ++i )
                    y_p(i) = alpha * u_l(i) * w_k(i);

                for ( size_t  j = 0; j < ncols; ++j )
                    z_p(j) = v_l(j) * x_k(j);
            }// for
        }// for

        //
        // truncate Y·Z and copy back to A
        //

        auto [ Y_acc, Z_acc ] = approx( Y, Z, acc );
        
        RA->set_lrmat( std::move( Y_acc ), std::move( Z_acc ) );
    }// if
}

//
// LU factorization
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, hpro::TBlockMatrix );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), acc, approx );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag, *BA->block( i, i ), *BA->block( i, j ), acc, approx );
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                {
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                    {
                        HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                    
                        multiply( value_t(-1), apply_normal, *BA->block( j, i ), apply_normal, *BA->block( i, l ),
                                  *BA->block( j, l ), acc, approx );
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
     

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
inline void
gauss_elim ( hpro::TMatrix *          A,
             hpro::TMatrix *          T,
             const hpro::TTruncAcc &  acc )
{
    assert( ! is_null_any( A, T ) );
    assert( A->type() == T->type() );

    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( A, hpro::TBlockMatrix );
        auto  BT = ptrcast( T, hpro::TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        hlr::seq::gauss_elim( MA(0,0), MT(0,0), acc );
        // hlr::log( 0, hpro::to_string( "                               %d = %.8e", MA(0,0)->id(), norm_F( MA(0,0) ) ) );

        // T_01 = A_00⁻¹ · A_01
        hpro::multiply( 1.0, hpro::apply_normal, MA(0,0), hpro::apply_normal, MA(0,1), 0.0, MT(0,1), acc );
        // seq::matrix::clear( *MT(0,1) );
        // multiply( 1.0, MA(0,0), MA(0,1), MT(0,1), acc );
        
        // T_10 = A_10 · A_00⁻¹
        hpro::multiply( 1.0, hpro::apply_normal, MA(1,0), hpro::apply_normal, MA(0,0), 0.0, MT(1,0), acc );
        // seq::matrix::clear( *MT(1,0) );
        // multiply( 1.0, MA(1,0), MA(0,0), MT(1,0), acc );

        // A_11 = A_11 - T_10 · A_01
        hpro::multiply( -1.0, hpro::apply_normal, MT(1,0), hpro::apply_normal, MA(0,1), 1.0, MA(1,1), acc );
        // multiply( -1.0, MT(1,0), MA(0,1), MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        hlr::seq::gauss_elim( MA(1,1), MT(1,1), acc );
        // hlr::log( 0, hpro::to_string( "                               %d = %.8e", MA(1,1)->id(), norm_F( MA(1,1) ) ) );

        // A_01 = - T_01 · A_11
        hpro::multiply( -1.0, hpro::apply_normal, MT(0,1), hpro::apply_normal, MA(1,1), 0.0, MA(0,1), acc );
        // seq::matrix::clear( *MA(0,1) );
        // multiply( -1.0, MT(0,1), MA(1,1), MA(0,1), acc );
            
        // A_10 = - A_11 · T_10
        hpro::multiply( -1.0, hpro::apply_normal, MA(1,1), hpro::apply_normal, MT(1,0), 0.0, MA(1,0), acc );
        // seq::matrix::clear( *MA(1,0) );
        // multiply( -1.0, MA(1,1), MT(1,0), MA(1,0), acc );

        // A_00 = T_00 - A_01 · T_10
        hpro::multiply( -1.0, hpro::apply_normal, MA(0,1), hpro::apply_normal, MT(1,0), 1.0, MA(0,0), acc );
        // multiply( -1.0, MA(0,1), MT(1,0), MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        if ( A->is_complex() ) blas::invert( DA->blas_cmat() );
        else                   blas::invert( DA->blas_rmat() );
    }// if
    else
        HLR_ASSERT( false );

    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d )", A->id() ) );
}

namespace tlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
// 
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
            
        blas::invert( hpro::blas_mat< value_t >( A_ii ) );

        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal !!!
            // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
            trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( j, i ),
                                               hpro::apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
// 
template < typename value_t,
           typename approx_t >
void
ldu ( hpro::TMatrix &          A,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );

        HLR_ASSERT( is_dense( A_ii ) );
        
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat< value_t >( ptrcast( A_ii, hpro::TDenseMatrix ) );
            
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //

        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  L_ji = BA->block( j, i );

            
            // auto  DT_ji = hlr::seq::matrix::convert_to_dense< value_t >( *L_ji );
            // auto  DC_ji = blas::copy( blas::mat< value_t >( DT_ji ) );

            // blas::prod( value_t(1), DC_ji, D_ii, value_t(0), blas::mat< value_t >( DT_ji ) );

            
            if ( is_lowrank( L_ji ) )
            {
                // L_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                // ⟶ W = U, X = D_ii^-T·V
                auto  R_ji = ptrcast( L_ji, hpro::TRkMatrix );
                auto  V    = blas::copy( blas::mat_V< value_t >( R_ji ) );

                blas::prod( value_t(1), blas::adjoint( D_ii ), V, value_t(0), blas::mat_V< value_t >( R_ji ) );
            }// if
            else if ( is_dense( L_ji ) )
            {
                auto  D_ji = ptrcast( L_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// else


            
            // auto  TT_ji = hlr::seq::matrix::convert_to_dense< value_t >( *L_ji );

            // blas::add( value_t(-1), blas::mat< value_t >( *DT_ji ), blas::mat< value_t >( *TT_ji ) );

            // std::cout << L_ji->id() << " : " << blas::norm_F( blas::mat< value_t >( *TT_ji ) ) << std::endl;
        }// for

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  U_ij = BA->block( i, j );


            // auto  DT_ij = hlr::seq::matrix::convert_to_dense< value_t >( *U_ij );
            // auto  DC_ij = blas::copy( blas::mat< value_t >( DT_ij ) );

            // blas::prod( value_t(1), D_ii, DC_ij, value_t(0), blas::mat< value_t >( DT_ij ) );

            
            
            if ( is_lowrank( U_ij ) )
            {
                // U_ij = W·X' = D_ii^-1·U·V' = D_ii^-1·A_ij
                // ⟶ W = D_ii^-1·U, X = V
                auto  R_ij = ptrcast( U_ij, hpro::TRkMatrix );
                auto  U    = blas::copy( blas::mat_U< value_t >( R_ij ) );

                blas::prod( value_t(1), D_ii, U, value_t(0), blas::mat_U< value_t >( R_ij ) );
            }// if
            else if ( is_dense( U_ij ) )
            {
                auto  D_ij = ptrcast( U_ij, hpro::TDenseMatrix );
                auto  T_ij = blas::copy( blas::mat< value_t >( D_ij ) );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), blas::mat< value_t >( D_ij ) );
            }// else


            // auto  TT_ij = hlr::seq::matrix::convert_to_dense< value_t >( *U_ij );

            // blas::add( value_t(-1), blas::mat< value_t >( *DT_ij ), blas::mat< value_t >( *TT_ij ) );

            // std::cout << U_ij->id() << " : " << blas::norm_F( blas::mat< value_t >( *TT_ij ) ) << std::endl;
        }// for

        //
        // update trailing sub matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               hpro::apply_normal, *BA->block( j, i ),
                                               hpro::apply_normal, *T_ii,
                                               hpro::apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tlr

namespace hodlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const hpro::TMatrix &      L,
        blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsml( %d )", L.id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( &L, hpro::TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), hpro::TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        blas::matrix< value_t >  X0( X, L00->row_is() - L.row_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, L11->row_is() - L.row_ofs(), blas::range::all );
            
        trsml( *L00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( L10 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( *L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, TDenseMatrix );
        
        // blas::matrix< value_t >  Y( X, copy_value );

        // blas::prod( value_t(1), blas_mat< value_t >( DL ), Y, value_t(0), X );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const hpro::TMatrix &      U,
         blas::matrix< value_t > &  X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d )", U.id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( &U, hpro::TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), hpro::TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        blas::matrix< value_t >  X0( X, U00->col_is() - U.col_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, U11->col_is() - U.col_ofs(), blas::range::all );
            
        trsmuh( *U00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( U01 ) ), X0 );
        
        blas::prod( value_t(-1), hpro::blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( *U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( &U, hpro::TDenseMatrix );
        
        blas::matrix< value_t >  Y( X, hpro::copy_value );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t,
           typename approx_t >
void
addlr ( blas::matrix< value_t > &  U,
        blas::matrix< value_t > &  V,
        hpro::TMatrix &            A,
        const hpro::TTruncAcc &    acc,
        const approx_t &           approx )
{
    HLR_LOG( 4, hpro::to_string( "addlr( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A.col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A.col_ofs(), blas::range::all );

        addlr( U0, V0, *A00, acc, approx );
        addlr( U1, V1, *A11, acc, approx );

        {
            auto [ U01, V01 ] = approx( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                        { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                        acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = approx( { hpro::blas_mat_A< value_t >( A10 ), U1 },
                                        { hpro::blas_mat_B< value_t >( A10 ), V0 },
                                        acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( &A, hpro::TDenseMatrix );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), hpro::blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( &A00, acc, approx );
        
        trsml(  A00, hpro::blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, hpro::blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A10 ) ), hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( A10 ), T );

        seq::hodlr::addlr< value_t >( UT, hpro::blas_mat_B< value_t >( A01 ), A11, acc, approx );
        
        seq::hodlr::lu< value_t >( &A11, acc, approx );
    }// if
    else
    {
        auto  DA = ptrcast( &A, hpro::TDenseMatrix );
        
        blas::invert( hpro::blas_mat< value_t >( DA ) );
    }// else
}

}// namespace hodlr

namespace tileh
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        hpro::LU::factorise_rec( BA->block( i, i ), acc );

        for ( uint j = i+1; j < nbr; ++j )
        {
            solve_upper_right( BA->block( j, i ),
                               BA->block( i, i ), nullptr, acc,
                               hpro::solve_option_t( hpro::block_wise, hpro::general_diag, hpro::store_inverse ) );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            solve_lower_left( hpro::apply_normal, BA->block( i, i ), nullptr,
                              BA->block( i, l ), acc,
                              hpro::solve_option_t( hpro::block_wise, hpro::unit_diag, hpro::store_inverse ) );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply( value_t(-1),
                                    hpro::apply_normal, * BA->block( j, i ),
                                    hpro::apply_normal, * BA->block( i, l ),
                                    * BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tileh

}}// namespace hlr::seq

#endif // __HLR_SEQ_ARITH_HH
