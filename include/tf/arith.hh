#ifndef __HLR_TF_ARITH_HH
#define __HLR_TF_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : sequential arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <taskflow/taskflow.hpp>

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

namespace TF
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

    tf::Taskflow  tf;
    
    auto                 nbr = BA->nblock_rows();
    auto                 nbc = BA->nblock_cols();
    tensor2< tf::Task >  fs_tasks( nbr, nbc );
    tensor3< tf::Task >  u_tasks( nbr, nbr, nbc );
    tensor3< char >      has_u_task( nbr, nbr, nbc, false );

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );

        fs_tasks(i,i) = tf.silent_emplace( [A_ii] ()
                                           {
                                               TScopedLock  lock( *A_ii );
                                               
                                               BLAS::invert( blas_mat< value_t >( A_ii ) );
                                           } );
            
        for ( uint  l = 0; l < i; ++l )
            if ( has_u_task(l,i,i) )
                u_tasks(l,i,i).precede( fs_tasks(i,i) );
            
        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is identity; task only for ensuring correct execution order
            fs_tasks(i,j) = tf.silent_emplace( [A_ii,BA,i,j] ()
                                               {
                                                   auto         A_ij = BA->block(i,j);
                                                   TScopedLock  lock( *A_ij );
                                               } );
            fs_tasks(i,i).precede( fs_tasks(i,j) );

            for ( uint  l = 0; l < i; ++l )
                if ( has_u_task(l,i,j) )
                    u_tasks(l,i,j).precede( fs_tasks(i,j) );
            
            fs_tasks(j,i) = tf.silent_emplace( [A_ii,BA,i,j] ()
                                               {
                                                   auto         A_ji = BA->block(j,i);
                                                   TScopedLock  lock( *A_ji );
                                                   
                                                   trsmuh< value_t >( A_ii, A_ji );
                                               } );
            fs_tasks(i,i).precede( fs_tasks(j,i) );

            for ( uint  l = 0; l < i; ++l )
                if ( has_u_task(l,j,i) )
                    u_tasks(l,j,i).precede( fs_tasks(j,i) );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  A_ji = BA->block( j, i );
                
            for ( uint  l = i+1; l < nbc; ++l )
            {
                auto  A_il = BA->block( i, l );
                auto  A_jl = BA->block( j, l );

                u_tasks(i,j,l)    = tf.silent_emplace( [A_ji,A_il,A_jl,&acc] ()
                                                       {
                                                           TScopedLock  lock( *A_jl );
                                                           
                                                           multiply< value_t >( value_t(-1), A_ji, A_il, A_jl, acc );
                                                       } );
                has_u_task(i,j,l) = true;
                
                // ensures non-simultanous writes
                // if ( i > 0 )
                //     u_tasks(i-1,j,l).precede( u_tasks(i,j,l) );
                
                fs_tasks(j,i).precede( u_tasks(i,j,l) );
                fs_tasks(i,l).precede( u_tasks(i,j,l) );
            }// for
        }// for
    }// for
    
    tf.wait_for_all();
}

}// namespace TF

}// namespace TLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace HODLR
{

namespace TF
{

//
// add U·V' to matrix A
//
template < typename value_t >
void
addlr ( const B::Matrix< value_t > &  U,
        const B::Matrix< value_t > &  V,
        TMatrix *                     A,
        const TTruncAcc &             acc,
        tf::SubflowBuilder &          sf )
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

        auto  add_00 = sf.silent_emplace( [&U0,&V0,A00,&acc] ( auto &  sf ) { addlr( U0, V0, A00, acc, sf ); } );
        auto  add_11 = sf.silent_emplace( [&U1,&V1,A11,&acc] ( auto &  sf ) { addlr( U1, V1, A11, acc, sf ); } );
        auto  add_01 = sf.silent_emplace( [&U0,&V1,A01,&acc] ()
                                          {
                                              auto [ U01, V01 ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                                  { blas_mat_B< value_t >( A01 ), V1 },
                                                                                                  acc );
                                              A01->set_lrmat( U01, V01 );
                                          } );
        auto  add_10 = sf.silent_emplace( [&U1,&V0,A10,&acc] ()
                                          {
                                              auto [ U10, V10 ] = HLR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                                                  { blas_mat_B< value_t >( A10 ), V0 },
                                                                                                  acc );
                                              A10->set_lrmat( U10, V10 );
                                          } );
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
lu ( TMatrix *             A,
     const TTruncAcc &     acc,
     tf::SubflowBuilder &  sf )
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

        auto  task_00 = sf.silent_emplace( [A00,&acc] ( auto &  sf ) { HODLR::TF::lu< value_t >( A00, acc, sf ); } );
        auto  task_01 = sf.silent_emplace( [A00,A01] () { HODLR::Seq::trsml(  A00, blas_mat_A< value_t >( A01 ) ); } );
        auto  task_10 = sf.silent_emplace( [A00,A10] () { HODLR::Seq::trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

        task_00.precede( { task_01, task_10 } );
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  [ task_T,   T ] = sf.emplace( [A10,A01] () { return B::prod(  value_t(1), B::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); } );
        auto  [ task_UT, UT ] = sf.emplace( [A10,&T]  () { return B::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T.get() ); } );

        task_01.precede( task_T );
        task_10.precede( task_T );
        task_T.precede( task_UT );
        
        auto  task_add11      = sf.silent_emplace( [A01,A11,&UT,&acc] ( auto &  sf )
                                                   { HODLR::TF::addlr< value_t >( UT.get(), blas_mat_B< value_t >( A01 ), A11, acc, sf ); } );

        task_UT.precede( task_add11 );
        
        auto  task_11         = sf.silent_emplace( [A11,&acc] ( auto &  sf ) { HODLR::TF::lu< value_t >( A11, acc, sf ); } );

        task_add11.precede( task_11 );
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    tf::Taskflow  tf( 1 );

    auto  lu_A  = tf.silent_emplace( [A,&acc] ( auto &  sf ) { lu< value_t >( A, acc, sf ); } );

    tf.wait_for_all();
}

}// namespace TF

}// namespace HODLR

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace TileH
{

namespace TF
{

//
// compute LU factorization of A
//

}// namespace TF

}// namespace TileH

}// namespace HLR

#endif // __HLR_TF_ARITH_HH
