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

namespace HLR
{

using namespace HLIB;

namespace SEQ
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace TLR
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
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  nbr = BA->nblock_rows();
        auto  nbc = BA->nblock_cols();

        for ( uint  i = 0; i < nbr; ++i )
        {
            auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
            SEQ::TLR::lu< value_t >( A_ii, acc );

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
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

}// namespace TLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace HODLR
{

//
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( B::Matrix< value_t > &  U,
        B::Matrix< value_t > &  V,
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

        B::Matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), B::Range::all );
        B::Matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), B::Range::all );
        B::Matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), B::Range::all );
        B::Matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), B::Range::all );

        addlr( U0, V0, A00, acc );
        addlr( U1, V1, A11, acc );

        {
            auto [ U01, V01 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                               { blas_mat_B< value_t >( A01 ), V1 },
                                                               acc );

            A01->set_lrmat( U01, V01 );
        }

        {
            auto [ U10, V10 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                               { blas_mat_B< value_t >( A10 ), V0 },
                                                               acc );
            A10->set_lrmat( U10, V10 );
        }
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );

        B::prod( value_t(1), U, B::adjoint( V ), value_t(1), blas_mat< value_t >( DA ) );
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

        SEQ::HODLR::lu< value_t >( A00, acc );
        
        trsml(  A00, blas_mat_A< value_t >( A01 ) );
        trsmuh( A00, blas_mat_B< value_t >( A10 ) );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = B::prod(  value_t(1), B::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = B::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        SEQ::HODLR::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        
        SEQ::HODLR::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( blas_mat< value_t >( DA ) );
    }// else
}

}// namespace HODLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace TileH
{

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( is_blocked( A ) )
    {
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
    }// if
    else
    {
        LU::factorise_rec( A, acc );
    }// else
}

}// namespace TileH

}// namespace SEQ

}// namespace HLR

#endif // __HLR_SEQ_ARITH_HH
