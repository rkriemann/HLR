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

#include <hlib.hh>

#include "common/multiply.hh"
#include "common/solve.hh"
#include "seq/arith.hh"

namespace HLR
{

using namespace HLIB;

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace TLR
{

namespace HPX
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
        
        hpx::parallel::v2::define_task_block(
            [i,nbc,A_ii,BA] ( auto &  tb )
            {
                for ( uint  j = i+1; j < nbc; ++j )
                    tb.run( [A_ii,BA,j,i] { trsmuh< value_t >( A_ii, BA->block( j, i ) ); } );
            } );

        hpx::parallel::v2::define_task_block(
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

}// namespace HPX

}// namespace TLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace HODLR
{

namespace HPX
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

        auto  task_00 = hpx::async( [&,A00] () { addlr( U0, V0, A00, acc ); } );
        auto  task_11 = hpx::async( [&,A11] () { addlr( U1, V1, A11, acc ); } );
        auto  task_01 = hpx::async( [&,A01] ()
                        {
                            auto [ U01, V01 ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                { blas_mat_B< value_t >( A01 ), V1 },
                                                                                acc );
                            A01->set_lrmat( U01, V01 );
                        } );
        auto  task_10 = hpx::async( [&,A10] ()
                        {
                            auto [ U10, V10 ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                                { blas_mat_B< value_t >( A10 ), V0 },
                                                                                acc );
                            A10->set_lrmat( U10, V10 );
                        } );
        auto  all = hpx::when_all( task_00, task_01, task_10, task_11 );

        all.wait();
    }// if
    else
    {
        B::prod( value_t(1), U, B::adjoint( V ), value_t(1), blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ) );
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

        HODLR::HPX::lu< value_t >( A00, acc );

        auto  solve_01 = hpx::async( [A00,A01] () { HLR::HODLR::Seq::trsml(  A00, blas_mat_A< value_t >( A01 ) ); } );
        auto  solve_10 = hpx::async( [A00,A10] () { HLR::HODLR::Seq::trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );
        auto  solve    = hpx::when_all( solve_01, solve_10 );

        solve.wait();
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = B::prod(  value_t(1), B::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = B::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        HODLR::HPX::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        
        HODLR::HPX::lu< value_t >( A11, acc );
    }// if
    else
    {
        BLAS::invert( blas_mat< value_t >( ptrcast( A, TDenseMatrix ) ) );
    }// else
}

}// namespace HPX

}// namespace HODLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace TileH
{

namespace HPX
{


}// namespace HPX

}// namespace TileH

}// namespace HLR

#endif // __HLR_HPX_ARITH_HH
