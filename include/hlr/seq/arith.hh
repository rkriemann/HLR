#ifndef __HLR_SEQ_ARITH_HH
#define __HLR_SEQ_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"

namespace hlr
{

using namespace HLIB;

namespace seq
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// LU factorization for TLR block format
// 
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
        BLAS::invert( blas_mat< value_t >( A_ii ) );

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
                multiply< value_t >( value_t(-1), BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
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
trsml ( const TMatrix *            L,
        BLAS::Matrix< value_t > &  X )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "trsml( %d )", L->id() );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), BLAS::Range::all );
            
        trsml( L00, X0 );

        auto  T = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_B< value_t >( L10 ) ), X0 );
        
        BLAS::prod( value_t(-1), blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, TDenseMatrix );
        
        // BLAS::Matrix< value_t >  Y( X, copy_value );

        // BLAS::prod( value_t(1), blas_mat< value_t >( DL ), Y, value_t(0), X );
    }// else
}

//
// solve X U = M
// - on input, X = M
//
template < typename value_t >
void
trsmuh ( const TMatrix *            U,
         BLAS::Matrix< value_t > &  X )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "trsmuh( %d )", U->id() );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        BLAS::Matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), BLAS::Range::all );
            
        trsmuh( U00, X0 );

        auto  T = BLAS::prod( value_t(1), BLAS::adjoint( blas_mat_A< value_t >( U01 ) ), X0 );
        
        BLAS::prod( value_t(-1), blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( U, TDenseMatrix );
        
        BLAS::Matrix< value_t >  Y( X, copy_value );

        BLAS::prod( value_t(1), BLAS::adjoint( blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

//
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( BLAS::Matrix< value_t > &  U,
        BLAS::Matrix< value_t > &  V,
        TMatrix *               A,
        const TTruncAcc &       acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "addlr( %d )", A->id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        BLAS::Matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), BLAS::Range::all );
        BLAS::Matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), BLAS::Range::all );

        addlr( U0, V0, A00, acc );
        addlr( U1, V1, A11, acc );

        {
            auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                { blas_mat_B< value_t >( A01 ), V1 },
                                                                acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                { blas_mat_B< value_t >( A10 ), V0 },
                                                                acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );

        BLAS::prod( value_t(1), U, BLAS::adjoint( V ), value_t(1), blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        seq::hodlr::lu< value_t >( A00, acc );
        
        trsml(  A00, blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = BLAS::prod(  value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = BLAS::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        seq::hodlr::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        
        seq::hodlr::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        BLAS::invert( blas_mat< value_t >( DA ) );
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
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        LU::factorise_rec( BA->block( i, i ), acc );

        for ( uint j = i+1; j < nbr; ++j )
        {
            solve_upper_right( BA->block( j, i ),
                               BA->block( i, i ), nullptr, acc,
                               solve_option_t( block_wise, general_diag, store_inverse ) );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            solve_lower_left( apply_normal, BA->block( i, i ), nullptr,
                              BA->block( i, l ), acc,
                              solve_option_t( block_wise, unit_diag, store_inverse ) );
        }// for
            
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                multiply( -1.0, BA->block( j, i ), BA->block( i, l ), 1.0, BA->block( j, l ), acc );
            }// for
        }// for
    }// for
}

}// namespace tileh

}// namespace seq

}// namespace hlr

#endif // __HLR_SEQ_ARITH_HH
