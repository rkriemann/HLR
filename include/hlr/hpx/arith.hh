#ifndef __HLR_HPX_ARITH_HH
#define __HLR_HPX_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpx/parallel/task_block.hpp>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

namespace hlr
{

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

namespace hpx
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
    assert( is_blocked( A ) );
    
    auto  BA  = ptrcast( A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );

        blas::invert( blas_mat< value_t >( A_ii ) );
        
        ::hpx::parallel::v2::define_task_block(
            [i,nbc,A_ii,BA] ( auto &  tb )
            {
                for ( uint  j = i+1; j < nbc; ++j )
                    tb.run( [A_ii,BA,j,i] { trsmuh< value_t >( A_ii, BA->block( j, i ) ); } );
            } );

        ::hpx::parallel::v2::define_task_block(
            [BA,i,nbr,nbc,&acc] ( auto &  tb )
            {
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  A_ji = BA->block( j, i );
                                   
                    for ( uint  l = i+1; l < nbc; ++l )
                    {
                        auto  A_il = BA->block( i, l );
                        auto  A_jl = BA->block( j, l );
                                       
                        tb.run( [A_ji,A_il,A_jl,&acc] { multiply< value_t >( value_t(-1), A_ji, A_il, A_jl, acc ); } );
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
    if ( hpro::verbose( 4 ) )
        DBG::printf( "addlr( %d )", A->id() );
    
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

        auto  task_00 = ::hpx::async( [&,A00] () { addlr( U0, V0, A00, acc ); } );
        auto  task_11 = ::hpx::async( [&,A11] () { addlr( U1, V1, A11, acc ); } );
        auto  task_01 = ::hpx::async( [&,A01] ()
                        {
                            auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                                                                { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                                                                acc );
                            A01->set_lrmat( U01, V01 );
                        } );
        auto  task_10 = ::hpx::async( [&,A10] ()
                        {
                            auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { hpro::blas_mat_A< value_t >( A10 ), U1 },
                                                                                { hpro::blas_mat_B< value_t >( A10 ), V0 },
                                                                                acc );
                            A10->set_lrmat( U10, V10 );
                        } );
        auto  all = ::hpx::when_all( task_00, task_01, task_10, task_11 );

        all.wait();
    }// if
    else
    {
        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ) );
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
    if ( hpro::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, hpro::TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), hpro::TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), hpro::TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( A00, acc );

        auto  solve_01 = ::hpx::async( [A00,A01] () { seq::hodlr::trsml(  A00, hpro::blas_mat_A< value_t >( A01 ) ); } );
        auto  solve_10 = ::hpx::async( [A00,A10] () { seq::hodlr::trsmuh( A00, hpro::blas_mat_B< value_t >( A10 ) ); } );
        auto  solve    = ::hpx::when_all( solve_01, solve_10 );

        solve.wait();
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( hpro::blas_mat_B< value_t >( A10 ) ), hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), hpro::blas_mat_A< value_t >( A10 ), T );

        addlr< value_t >( UT, hpro::blas_mat_B< value_t >( A01 ), A11, acc );
        
        lu< value_t >( A11, acc );
    }// if
    else
    {
        blas::invert( blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ) );
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
gauss_elim ( hpro::TMatrix *    A,
             hpro::TMatrix *    T,
             const TTruncAcc &  acc )
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

        ::hpx::parallel::v2::define_task_block(
            [&] ( auto &  tb )
            {
                // T_01 = A_00⁻¹ · A_01
                tb.run( [&] () { hpro::multiply( 1.0, apply_normal, MA(0,0), apply_normal, MA(0,1), 0.0, MT(0,1), acc ); } );
        
                // T_10 = A_10 · A_00⁻¹
                tb.run( [&] () { hpro::multiply( 1.0, apply_normal, MA(1,0), apply_normal, MA(0,0), 0.0, MT(1,0), acc ); } );
            } );

        // A_11 = A_11 - T_10 · A_01
        hpro::multiply( -1.0, apply_normal, MT(1,0), apply_normal, MA(0,1), 1.0, MA(1,1), acc );
    
        // A_11 = A_11⁻¹
        hlr::seq::gauss_elim( MA(1,1), MT(1,1), acc );

        ::hpx::parallel::v2::define_task_block(
            [&] ( auto &  tb )
            {
                // A_01 = - T_01 · A_11
                tb.run( [&] () { hpro::multiply( -1.0, apply_normal, MT(0,1), apply_normal, MA(1,1), 0.0, MA(0,1), acc ); } );
            
                // A_10 = - A_11 · T_10
                tb.run( [&] () { hpro::multiply( -1.0, apply_normal, MA(1,1), apply_normal, MT(1,0), 0.0, MA(1,0), acc ); } );
            } );
        
        // A_00 = T_00 - A_01 · T_10
        hpro::multiply( -1.0, apply_normal, MA(0,1), apply_normal, MT(1,0), 1.0, MA(0,0), acc );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( A, hpro::TDenseMatrix );
        
        if ( A->is_complex() ) blas::invert( DA->blas_cmat() );
        else                   blas::invert( DA->blas_rmat() );
    }// if
    else
        assert( false );

    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d )", A->id() ) );
}

}// namespace hpx

}// namespace hlr

#endif // __HLR_HPX_ARITH_HH
