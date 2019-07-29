#ifndef __HLR_TBB_ARITH_HH
#define __HLR_TBB_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hlib.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

namespace hlr
{

using namespace HLIB;

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace tbb
{

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
    assert( is_blocked( A ) );
    
    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
        BLAS::invert( blas_mat< value_t >( A_ii ) );

        ::tbb::parallel_for( i+1, nbc,
                             [A_ii,BA,i] ( uint  j )
                             {
                                 // L is unit diagonal !!!
                                 // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                                 trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
                             } );

        ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                             [BA,i,&acc] ( const ::tbb::blocked_range2d< uint > & r )
                             {
                                 for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                 {
                                     for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                     {
                                         multiply< value_t >( value_t(-1), BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
                                     }// for
                                 }// for
                             } );
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
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( BLAS::Matrix< value_t > &  U,
        BLAS::Matrix< value_t > &  V,
        TMatrix *                  A,
        const TTruncAcc &          acc )
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

        ::tbb::parallel_invoke( [&U0,&V0,A00,&acc] () { addlr( U0, V0, A00, acc ); },
                                [&U1,&V1,A11,&acc] () { addlr( U1, V1, A11, acc ); },
                                [&U0,&V1,A01,&acc] ()
                                {
                                    auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                        { blas_mat_B< value_t >( A01 ), V1 },
                                                                                        acc );
                                    A01->set_lrmat( U01, V01 );
                                },
                                [&U1,&V0,A10,&acc] ()
                                {
                                    auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                                        { blas_mat_B< value_t >( A10 ), V0 },
                                                                                        acc );
                                    A10->set_lrmat( U10, V10 );
                                } );
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

        tbb::hodlr::lu< value_t >( A00, acc );

        ::tbb::parallel_invoke( [A00,A01] () { seq::hodlr::trsml(  A00, blas_mat_A< value_t >( A01 ) ); },
                                [A00,A10] () { seq::hodlr::trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = BLAS::prod(  value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = BLAS::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        tbb::hodlr::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        
        tbb::hodlr::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        BLAS::invert( DA->blas_rmat() );
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
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        HLIB::LU::factorise_rec( BA->block( i, i ), acc );

        ::tbb::parallel_invoke(
            [BA,i,nbr,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbr,
                                     [BA,i,&acc] ( uint  j )
                                     {
                                         solve_upper_right( BA->block( j, i ),
                                                            BA->block( i, i ), nullptr, acc,
                                                            solve_option_t( block_wise, general_diag, store_inverse ) );
                                     } );
            },
                
            [BA,i,nbc,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbc,
                                     [BA,i,&acc] ( uint  l )
                                     {
                                         solve_lower_left( apply_normal, BA->block( i, i ), nullptr,
                                                           BA->block( i, l ), acc,
                                                           solve_option_t( block_wise, unit_diag, store_inverse ) );
                                     } );
            } );

        ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                             [BA,i,&acc] ( const ::tbb::blocked_range2d< uint > & r )
                             {
                                 for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                 {
                                     for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                     {
                                         multiply( -1.0, BA->block( j, i ), BA->block( i, l ), 1.0, BA->block( j, l ), acc );
                                     }// for
                                 }// for
                             } );
    }// for
}

}// namespace tileh

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
inline void
gauss_elim ( HLIB::TMatrix *    A,
             HLIB::TMatrix *    T,
             const TTruncAcc &  acc )
{
    assert( ! is_null_any( A, T ) );
    assert( A->type() == T->type() );
    
    HLR_LOG( 4, HLIB::to_string( "gauss_elim( %d ) {", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );
        auto  BT = ptrcast( T, TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        tbb::gauss_elim( MA(0,0), MT(0,0), acc );

        ::tbb::parallel_invoke(
            [&]
            { 
                // T_01 = A_00⁻¹ · A_01
                multiply( 1.0, apply_normal, MA(0,0), apply_normal, MA(0,1), 0.0, MT(0,1), acc );
            },

            [&]
            {
                // T_10 = A_10 · A_00⁻¹
                multiply( 1.0, apply_normal, MA(1,0), apply_normal, MA(0,0), 0.0, MT(1,0), acc );
            } );

        // A_11 = A_11 - T_10 · A_01
        multiply( -1.0, apply_normal, MT(1,0), apply_normal, MA(0,1), 1.0, MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        tbb::gauss_elim( MA(1,1), MT(1,1), acc );

        ::tbb::parallel_invoke(
            [&]
            { 
                // A_01 = - T_01 · A_11
                multiply( -1.0, apply_normal, MT(0,1), apply_normal, MA(1,1), 0.0, MA(0,1), acc );
            },
            
            [&]
            { 
                // A_10 = - A_11 · T_10
                multiply( -1.0, apply_normal, MA(1,1), apply_normal, MT(1,0), 0.0, MA(1,0), acc );
            } );

        // A_00 = T_00 - A_01 · T_10
        multiply( -1.0, apply_normal, MA(0,1), apply_normal, MT(1,0), 1.0, MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        if ( A->is_complex() ) HLIB::BLAS::invert( DA->blas_cmat() );
        else                   HLIB::BLAS::invert( DA->blas_rmat() );
    }// if
    else
        assert( false );

    HLR_LOG( 4, HLIB::to_string( "} gauss_elim( %d )", A->id() ) );
}

}// namespace tbb

}// namespace hlr

#endif // __HLR_TBB_ARITH_HH
