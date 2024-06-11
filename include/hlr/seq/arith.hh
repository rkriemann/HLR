#ifndef __HLR_SEQ_ARITH_HH
#define __HLR_SEQ_ARITH_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include <hlr/arith/defaults.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/add.hh>
#include <hlr/arith/multiply.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/arith/lu.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/invert.hh>
#include <hlr/approx/svd.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/seq/matrix.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr { namespace seq {

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α · op( M ) · x
//
using hlr::mul_vec;

using hlr::cluster_block_map_t;
using hlr::cluster_blocks_t;
using hlr::mul_vec_cl;
using hlr::mul_vec_cl2;
using hlr::mul_vec_hier;
using hlr::realloc;
using hlr::setup_cluster_block_map;
using hlr::build_cluster_blocks;
using hlr::build_cluster_matrix;

//
// compute C = C + α · A
//
using hlr::add;

//
// compute C = C + α · op( A ) · op( B ) 
// and     C = C + α · op( A ) · op( D ) · op( B )
//
using hlr::multiply;

//
// compute C = C + α · op( A ) · op( B ) with additional approximation
// by omitting sub products based on Frobenius norm of factors
//
using hlr::multiply_apx;

//
// compute Hadamard product A = α A*B 
//
using hlr::multiply_hadamard;

//
// matrix factorizations
//
using hlr::lu;
using hlr::ldu;

//
// LU factorization for nested dissection type matrices
//
template < typename value_t,
           typename approx_t >
void
lu_nd ( Hpro::TMatrix< value_t > &  A,
        const Hpro::TTruncAcc &     acc,
        const approx_t &            approx )
{
    lu< value_t >( A, acc, approx );
}
    
//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
template < typename value_t >
void
gauss_elim ( Hpro::TMatrix< value_t > &  A,
             Hpro::TMatrix< value_t > &  T,
             const Hpro::TTruncAcc &     acc )
{
    assert( ! is_null_any( &A, &T ) );
    assert( A.type() == T.type() );

    HLR_LOG( 4, Hpro::to_string( "gauss_elim( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BT = ptrcast( &T, Hpro::TBlockMatrix< value_t > );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };
        auto  apx = approx::SVD< value_t >();

        // A_00 = A_00⁻¹
        hlr::seq::gauss_elim< value_t >( *MA(0,0), *MT(0,0), acc );
        // hlr::log( 0, Hpro::to_string( "                               %d = %.8e", MA(0,0)->id(), norm_F( MA(0,0) ) ) );

        // T_01 = A_00⁻¹ · A_01
        // Hpro::multiply( 1.0, apply_normal, MA(0,0), apply_normal, MA(0,1), 0.0, MT(0,1), acc );
        seq::matrix::clear( *MT(0,1) );
        multiply< value_t >( value_t(1), apply_normal, *MA(0,0), apply_normal, *MA(0,1), *MT(0,1), acc, apx );
        
        // T_10 = A_10 · A_00⁻¹
        // Hpro::multiply( 1.0, apply_normal, MA(1,0), apply_normal, MA(0,0), 0.0, MT(1,0), acc );
        seq::matrix::clear( *MT(1,0) );
        multiply< value_t >( value_t(1), apply_normal, *MA(1,0), apply_normal, *MA(0,0), *MT(1,0), acc, apx );

        // A_11 = A_11 - T_10 · A_01
        // Hpro::multiply( -1.0, apply_normal, MT(1,0), apply_normal, MA(0,1), 1.0, MA(1,1), acc );
        multiply< value_t >( value_t(-1), apply_normal, *MT(1,0), apply_normal, *MA(0,1), *MA(1,1), acc, apx );
    
        // A_11 = A_11⁻¹
        hlr::seq::gauss_elim< value_t >( *MA(1,1), *MT(1,1), acc );
        // hlr::log( 0, Hpro::to_string( "                               %d = %.8e", MA(1,1)->id(), norm_F( MA(1,1) ) ) );

        // A_01 = - T_01 · A_11
        // Hpro::multiply( -1.0, apply_normal, MT(0,1), apply_normal, MA(1,1), 0.0, MA(0,1), acc );
        seq::matrix::clear( *MA(0,1) );
        multiply( value_t(-1), apply_normal, *MT(0,1), apply_normal, *MA(1,1), *MA(0,1), acc, apx );
            
        // A_10 = - A_11 · T_10
        // Hpro::multiply( -1.0, apply_normal, MA(1,1), apply_normal, MT(1,0), 0.0, MA(1,0), acc );
        seq::matrix::clear( *MA(1,0) );
        multiply< value_t >( value_t(-1), apply_normal, *MA(1,1), apply_normal, *MT(1,0), *MA(1,0), acc, apx );

        // A_00 = T_00 - A_01 · T_10
        // Hpro::multiply( -1.0, apply_normal, MA(0,1), apply_normal, MT(1,0), 1.0, MA(0,0), acc );
        multiply< value_t >( value_t(-1), apply_normal, *MA(0,1), apply_normal, *MT(1,0), *MA(0,0), acc, apx );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        blas::invert( DD );
        
        if ( D->is_compressed() )
            D->set_matrix( std::move( DD ), acc );
    }// if
    else
        HLR_ASSERT( false );

    HLR_LOG( 4, Hpro::to_string( "gauss_elim( %d )", A.id() ) );
}

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
// 
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );
            
        blas::invert( Hpro::blas_mat< value_t >( A_ii ) );

        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal !!!
            // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
            trsmuh< value_t >( *A_ii, *BA->block( j, i ) ); // A10->blas_rmat_B() );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               apply_normal, *BA->block( j, i ),
                                               apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
lu_lazy ( Hpro::TMatrix< value_t > &  A,
          const Hpro::TTruncAcc &     acc,
          const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    HLR_ASSERT( is_blocked( A ) );

    auto  BA  = ptrcast( & A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        for ( int  k = 0; k < int(i); k++ )
            hlr::seq::multiply< value_t >( value_t(-1),
                                           apply_normal, *BA->block( i, k ),
                                           apply_normal, *BA->block( k, i ),
                                           *BA->block( i, i ), acc, approx );
        
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );
        auto  D_ii = A_ii->mat();
            
        HLR_ASSERT( ! A_ii->is_compressed() );
        
        blas::invert( D_ii );

        //
        // solve with L, e.g. L_ii X_ij = M_ij
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  A_ij = BA->block( i, j );

            // only update block as L = I
            for ( int  k = 0; k < int(i); k++ )
                hlr::seq::multiply< value_t >( value_t(-1),
                                               apply_normal, *BA->block( i, k ),
                                               apply_normal, *BA->block( k, j ),
                                               *A_ij, acc, approx );
        }// for
        
        //
        // solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            for ( int  k = 0; k < int(i); k++ )
                hlr::seq::multiply< value_t >( value_t(-1),
                                               apply_normal, *BA->block( j, k ),
                                               apply_normal, *BA->block( k, i ),
                                               *A_ji, acc, approx );

            if ( matrix::is_lowrank( A_ji ) )
            {
                // A_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                // ⟶ W = U, X = D_ii^-T·V
                auto  R_ji = ptrcast( A_ji, matrix::lrmatrix< value_t > );
                auto  V_ji = blas::prod( value_t(1), blas::adjoint( D_ii ), R_ji->V() );

                R_ji->set_V( std::move( V_ji ), acc );
            }// if
            else if ( matrix::is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, matrix::dense_matrix< value_t > );
                auto  T_ji = blas::copy( D_ji->mat() );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + A_ji->typestr() );
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
// 
template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > &  A,
      const Hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        HLR_ASSERT( matrix::is_dense( BA->block( i, i ) ) );
        
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );

        HLR_ASSERT( ! A_ii.is_compressed() );
        
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = A_ii->mat_direct();
            
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //

        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  L_ji = BA->block( j, i );

            if ( matrix::is_lowrank( L_ji ) )
            {
                // L_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                // ⟶ W = U, X = D_ii^-T·V
                auto  R_ji = ptrcast( L_ji, matrix::lrmatrix< value_t > );
                auto  V_ji = blas::prod( blas::adjoint( D_ii ), R_ji->V_direct() );

                if ( R_ji.is_compressed() )
                    R_ji->set_V( std::move( V_ji ), acc );
            }// if
            else if ( matrix::is_dense( L_ji ) )
            {
                auto  D_ji  = ptrcast( L_ji, matrix::dense_matrix< value_t > );
                auto  DD_ji = D_ji->mat_direct();
                auto  T_ji  = blas::copy( DD_ji );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), DD_ji );
            }// else
        }// for

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  U_ij = BA->block( i, j );

            if ( matrix::is_lowrank( U_ij ) )
            {
                // U_ij = W·X' = D_ii^-1·U·V' = D_ii^-1·A_ij
                // ⟶ W = D_ii^-1·U, X = V
                auto  R_ij = ptrcast( U_ij, matrix::lrmatrix< value_t > );
                auto  U    = blas::copy( R_ij->U_direct() );

                blas::prod( value_t(1), D_ii, U, value_t(0), R_ij->U_direct() );
            }// if
            else if ( matrix::is_dense( U_ij ) )
            {
                auto  D_ij = ptrcast( U_ij, matrix::dense_matrix< value_t > );
                auto  T_ij = blas::copy( D_ij->mat_direct() );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), D_ij->mat_direct() );
            }// else
        }// for

        //
        // update trailing sub matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::seq::multiply< value_t >( value_t(-1),
                                               apply_normal, *BA->block( j, i ),
                                               apply_normal, *T_ii,
                                               apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace hodlr
{

//
// solve L X = M
// - on input, X = M
//
template < typename value_t >
void
trsml ( const Hpro::TMatrix< value_t > &  L,
        blas::matrix< value_t > &         X )
{
    HLR_LOG( 4, Hpro::to_string( "trsml( %d )", L.id() ) );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  L11 = BL->block( 1, 1 );

        auto  X0 = blas::matrix< value_t >( X, L00->row_is() - L.row_ofs(), blas::range::all );
        auto  X1 = blas::matrix< value_t >( X, L11->row_is() - L.row_ofs(), blas::range::all );
            
        trsml( *L00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( L10->V_direct() ), X0 );
        
        blas::prod( value_t(-1), L10->U_direct(), T, value_t(1), X1 );

        trsml( *L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, matrix::dense_matrix< value_t > );
        
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
trsmuh ( const Hpro::TMatrix< value_t > &  U,
         blas::matrix< value_t > &         X )
{
    HLR_LOG( 4, Hpro::to_string( "trsmuh( %d )", U.id() ) );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  U11 = BU->block( 1, 1 );

        blas::matrix< value_t >  X0( X, U00->col_is() - U.col_ofs(), blas::range::all );
        blas::matrix< value_t >  X1( X, U11->col_is() - U.col_ofs(), blas::range::all );
            
        trsmuh( *U00, X0 );

        auto  T = blas::prod( value_t(1), blas::adjoint( U01->U_direct() ), X0 );
        
        blas::prod( value_t(-1), U01->V_direct(), T, value_t(1), X1 );

        trsmuh( *U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( &U, matrix::dense_matrix< value_t > );
        
        auto  Y = blas::copy( X );

        blas::prod( value_t(1), blas::adjoint( DU->mat_direct() ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t,
           typename approx_t >
void
addlr ( blas::matrix< value_t > &   U,
        blas::matrix< value_t > &   V,
        Hpro::TMatrix< value_t > &  A,
        const Hpro::TTruncAcc &     acc,
        const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "addlr( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A.col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A.col_ofs(), blas::range::all );

        addlr( U0, V0, *A00, acc, approx );
        addlr( U1, V1, *A11, acc, approx );

        {
            auto [ U01, V01 ] = approx( { A01->U_direct(), U0 },
                                        { A01->V_direct(), V1 },
                                        acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = approx( { A10->U_direct(), U1 },
                                        { A10->V_direct(), V0 },
                                        acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), DA->mat_direct() );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( *A00, acc, approx );
        
        trsml(  *A00, A01->U_direct() );
        trsmuh( *A00, A10->V_direct() );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( A10->V_direct() ), A01->U_direct() ); 
        auto  UT = blas::prod( value_t(-1), A10->U_direct(), T );

        seq::hodlr::addlr< value_t >( UT, A01->V_direct(), *A11, acc, approx );
        
        seq::hodlr::lu< value_t >( *A11, acc, approx );
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
        
        blas::invert( DA->mat_direct() );
    }// else
}

}// namespace hodlr

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace tileh
{

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        hlr::seq::lu< value_t >( *BA->block( i, i ), acc, approx );

        for ( uint j = i+1; j < nbr; ++j )
        {
            hlr::solve_upper_tri< value_t >( from_right, general_diag, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            hlr::solve_lower_tri< value_t >( from_left, unit_diag, *BA->block( i, i ), *BA->block( i, l ), acc, approx );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::multiply( value_t(-1),
                               apply_normal, *BA->block( j, i ),
                               apply_normal, *BA->block( i, l ),
                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
    }// for
}

}// namespace tileh

//
// collection of arithmetic functions
//
struct seq_arithmetic
{
    //
    // matrix vector multiplication
    //
    
    template < typename value_t >
    void
    mul_vec ( const value_t                             alpha,
              const Hpro::matop_t                       op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y ) const
    {
        hlr::seq::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    mul_vec ( const value_t                     alpha,
              const Hpro::matop_t               op_M,
              const Hpro::TMatrix< value_t > &  M,
              const blas::vector< value_t > &   x,
              blas::vector< value_t > &         y ) const
    {
        hlr::seq::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    prod ( const value_t                             alpha,
           const matop_t                             op_M,
           const Hpro::TLinearOperator< value_t > &  M,
           const blas::vector< value_t > &           x,
           blas::vector< value_t > &                 y ) const
    {
        if ( is_matrix( M ) )
            hlr::seq::mul_vec( alpha, op_M, *cptrcast( &M, Hpro::TMatrix< value_t > ), x, y );
        else
            M.apply_add( alpha, x, y, op_M );
    }
};

constexpr seq_arithmetic arithmetic{};

}

template <> struct is_arithmetic<       seq::seq_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic< const seq::seq_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic<       seq::seq_arithmetic & > { static constexpr bool value = true; };
template <> struct is_arithmetic< const seq::seq_arithmetic & > { static constexpr bool value = true; };

}// namespace hlr::seq

#endif // __HLR_SEQ_ARITH_HH
