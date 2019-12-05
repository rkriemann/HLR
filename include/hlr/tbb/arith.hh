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

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

#include "hlr/tbb/arith_tiled.hh"
#include "hlr/tbb/arith_tiled_v2.hh"

namespace hlr { namespace tbb {

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

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
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    assert( is_blocked( A ) );
    
    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
            
        blas::invert( hpro::blas_mat< value_t >( A_ii ) );

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
addlr ( blas::Matrix< value_t > &  U,
        blas::Matrix< value_t > &  V,
        hpro::TMatrix *            A,
        const hpro::TTruncAcc &    acc )
{
    HLR_LOG( 5, hpro::to_string( "addlr( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        blas::Matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), blas::Range::all );
        blas::Matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), blas::Range::all );
        blas::Matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), blas::Range::all );
        blas::Matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), blas::Range::all );

        ::tbb::parallel_invoke( [&U0,&V0,A00,&acc] () { addlr( U0, V0, A00, acc ); },
                                [&U1,&V1,A11,&acc] () { addlr( U1, V1, A11, acc ); },
                                [&U0,&V1,A01,&acc] ()
                                {
                                    auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                                                                        { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                                                                        acc );
                                    A01->set_lrmat( U01, V01 );
                                },
                                [&U1,&V0,A10,&acc] ()
                                {
                                    auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( A10 ), U1 },
                                                                                        { hpro::blas_mat_B< value_t >( A10 ), V0 },
                                                                                        acc );
                                    A10->set_lrmat( U10, V10 );
                                } );
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), hpro::blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t >
void
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc );

        ::tbb::parallel_invoke( [A00,A01] () { seq::hodlr::trsml(  A00, hpro::blas_mat_A< value_t >( A01 ) ); },
                                [A00,A10] () { seq::hodlr::trsmuh( A00, hpro::blas_mat_B< value_t >( A10 ) ); } );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A10 ) ), hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( A10 ), T );

        addlr< value_t >( UT, hpro::blas_mat_B< value_t >( A01 ), A11, acc );
        
        lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        blas::invert( DA->blas_rmat() );
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
lu ( hpro::TMatrix *          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A->id() ) );

    assert( is_blocked( A ) );

    auto  BA  = ptrcast( A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        hpro::LU::factorise_rec( BA->block( i, i ), acc );

        ::tbb::parallel_invoke(
            [BA,i,nbr,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbr,
                                     [BA,i,&acc] ( uint  j )
                                     {
                                         hpro::solve_upper_right( BA->block( j, i ),
                                                                  BA->block( i, i ), nullptr, acc,
                                                                  hpro::solve_option_t( hpro::block_wise, hpro::general_diag, hpro::store_inverse ) );
                                     } );
            },
                
            [BA,i,nbc,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbc,
                                     [BA,i,&acc] ( uint  l )
                                     {
                                         hpro::solve_lower_left( hpro::apply_normal, BA->block( i, i ), nullptr,
                                                                 BA->block( i, l ), acc,
                                                                 hpro::solve_option_t( hpro::block_wise, hpro::unit_diag, hpro::store_inverse ) );
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
                                         hpro::multiply( -1.0, BA->block( j, i ), BA->block( i, l ), 1.0, BA->block( j, l ), acc );
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
// compute y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                    alpha,
          const hpro::matop_t              op_M,
          const hpro::TMatrix *            M,
          const blas::Vector< value_t > &  x,
          blas::Vector< value_t > &        y )
{
    assert( ! is_null( M ) );
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( M, hpro::TBlockMatrix );
        const auto  row_ofs = B->row_is( op_M ).first();
        const auto  col_ofs = B->col_is( op_M ).first();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                    auto  y_i = x( B_ij->row_is( op_M ) - row_ofs );

                    mul_vec( alpha, op_M, B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = cptrcast( M, hpro::TDenseMatrix );
        
        blas::mulvec( alpha, blas::mat_view( op_M, hpro::blas_mat< value_t >( D ) ), x, value_t(1), y );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( M, hpro::TRkMatrix );

        if ( op_M == hpro::apply_normal )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_A< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            assert( is_complex_type< value_t >::value == false );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( hpro::blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( hpro::blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, hpro::blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
    }// if
    else
        assert( false );
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
    
    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d ) {", A->id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( A, hpro::TBlockMatrix );
        auto  BT = ptrcast( T, hpro::TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        tbb::gauss_elim( MA(0,0), MT(0,0), acc );

        ::tbb::parallel_invoke(
            [&]
            { 
                // T_01 = A_00⁻¹ · A_01
                hpro::multiply( 1.0, hpro::apply_normal, MA(0,0), hpro::apply_normal, MA(0,1), 0.0, MT(0,1), acc );
            },

            [&]
            {
                // T_10 = A_10 · A_00⁻¹
                hpro::multiply( 1.0, hpro::apply_normal, MA(1,0), hpro::apply_normal, MA(0,0), 0.0, MT(1,0), acc );
            } );

        // A_11 = A_11 - T_10 · A_01
        hpro::multiply( -1.0, hpro::apply_normal, MT(1,0), hpro::apply_normal, MA(0,1), 1.0, MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        gauss_elim( MA(1,1), MT(1,1), acc );

        ::tbb::parallel_invoke(
            [&]
            { 
                // A_01 = - T_01 · A_11
                hpro::multiply( -1.0, hpro::apply_normal, MT(0,1), hpro::apply_normal, MA(1,1), 0.0, MA(0,1), acc );
            },
            
            [&]
            { 
                // A_10 = - A_11 · T_10
                hpro::multiply( -1.0, hpro::apply_normal, MA(1,1), hpro::apply_normal, MT(1,0), 0.0, MA(1,0), acc );
            } );

        // A_00 = T_00 - A_01 · T_10
        hpro::multiply( -1.0, hpro::apply_normal, MA(0,1), hpro::apply_normal, MT(1,0), 1.0, MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        if ( A->is_complex() ) blas::invert( DA->blas_cmat() );
        else                   blas::invert( DA->blas_rmat() );
    }// if
    else
        assert( false );

    HLR_LOG( 4, hpro::to_string( "} gauss_elim( %d )", A->id() ) );
}

}}// namespace hlr::tbb

#endif // __HLR_TBB_ARITH_HH
