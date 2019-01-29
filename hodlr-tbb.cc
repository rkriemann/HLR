//
// Project     : HLib
// File        : hodlr-lu.cc
// Description : HODLR arithmetic with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

#include "approx.hh"
#include "hodlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace HODLR
{

namespace TBB
{

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

        tbb::parallel_invoke( [&U0,&V0,A00,&acc] () { addlr( U0, V0, A00, acc ); },
                              [&U1,&V1,A11,&acc] () { addlr( U1, V1, A11, acc ); },
                              [&U0,&V1,A01,&acc] () { auto [ U01, V01 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                                         { blas_mat_B< value_t >( A01 ), V1 },
                                                                                                         acc );
                                                      A01->set_rank( U01, V01 );
                              },
                              [&U1,&V0,A10,&acc] () { auto [ U10, V10 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                                                         { blas_mat_B< value_t >( A10 ), V0 },
                                                                                                         acc );
                                                      A10->set_rank( U10, V10 );
                              } );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );

        B::prod( value_t(1), U, B::adjoint( V ), value_t(1), blas_mat< value_t >( DA ) );
    }// else
}

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

        HODLR::TBB::lu< value_t >( A00, acc );

        tbb::parallel_invoke( [A00,A01] () { trsml(  A00, blas_mat_A< value_t >( A01 ) ); },
                              [A00,A10] () { trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = B::prod(  value_t(1), B::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); 
        auto  UT = B::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        HODLR::TBB::addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), BA->block( 1, 1 ), acc );
        
        HODLR::TBB::lu< value_t >( A11, acc );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

template
void
lu< HLIB::real > ( TMatrix *          A,
                   const TTruncAcc &  acc );

}// namespace TBB

}// namespace HODLR
