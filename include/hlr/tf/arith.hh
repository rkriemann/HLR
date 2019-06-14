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

namespace tf
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

    ::tf::Taskflow  tf;
    
    auto                   nbr = BA->nblock_rows();
    auto                   nbc = BA->nblock_cols();
    tensor2< ::tf::Task >  fs_tasks( nbr, nbc );
    tensor3< ::tf::Task >  u_tasks( nbr, nbr, nbc );
    tensor3< char >        has_u_task( nbr, nbr, nbc, false );

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
addlr ( const BLAS::Matrix< value_t > &  U,
        const BLAS::Matrix< value_t > &  V,
        TMatrix *                        A,
        const TTruncAcc &                acc )
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

        ::tf::Taskflow  tf;
        
        auto  add_00 = tf.silent_emplace( [&U0,&V0,A00,&acc] () { addlr( U0, V0, A00, acc ); } );
        auto  add_11 = tf.silent_emplace( [&U1,&V1,A11,&acc] () { addlr( U1, V1, A11, acc ); } );
        auto  add_01 = tf.silent_emplace( [&U0,&V1,A01,&acc] ()
                                          {
                                              auto [ U01, V01 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                                  { blas_mat_B< value_t >( A01 ), V1 },
                                                                                                  acc );
                                              A01->set_lrmat( U01, V01 );
                                          } );
        auto  add_10 = tf.silent_emplace( [&U1,&V0,A10,&acc] ()
                                          {
                                              auto [ U10, V10 ] = hlr::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
                                                                                                  { blas_mat_B< value_t >( A10 ), V0 },
                                                                                                  acc );
                                              A10->set_lrmat( U10, V10 );
                                          } );

        tf.wait_for_all();
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
lu ( TMatrix *             A,
     const TTruncAcc &     acc )
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

        #if  0

        //
        // all function calls wrapped in tasks
        //
        
        ::tf::Taskflow  tf;
        
        auto  task_00 = tf.silent_emplace( [A00,&acc] () { lu< value_t >( A00, acc ); } );
        auto  task_01 = tf.silent_emplace( [A00,A01]  () { seq::hodlr::trsml(  A00, blas_mat_A< value_t >( A01 ) ); } );
        auto  task_10 = tf.silent_emplace( [A00,A10]  () { seq::hodlr::trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

        task_00.precede( { task_01, task_10 } );
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  [ task_T,   T ] = tf.emplace( [A10,A01] () { return BLAS::prod(  value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); } );
        auto  [ task_UT, UT ] = tf.emplace( [A10,&T]  () { return BLAS::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T.get() ); } );

        task_01.precede( task_T );
        task_10.precede( task_T );
        task_T.precede( task_UT );
        
        auto  task_add11      = tf.silent_emplace( [A01,A11,&UT,&acc] ()
                                                   { addlr< value_t >( UT.get(), blas_mat_B< value_t >( A01 ), A11, acc ); } );

        task_UT.precede( task_add11 );
        
        auto  task_11         = tf.silent_emplace( [A11,&acc] () { lu< value_t >( A11, acc ); } );

        task_add11.precede( task_11 );

        tf.wait_for_all();

        #else

        //
        // only tasks for the two parallel calls
        //
        
        lu< value_t >( A00, acc );

        {
            ::tf::Taskflow  tf;
        
            auto  task_01 = tf.silent_emplace( [A00,A01]  () { seq::hodlr::trsml(  A00, blas_mat_A< value_t >( A01 ) ); } );
            auto  task_10 = tf.silent_emplace( [A00,A10]  () { seq::hodlr::trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

            tf.wait_for_all();
        }
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = BLAS::prod(  value_t(1), BLAS::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) );
        auto  UT = BLAS::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), A11, acc );
        lu< value_t >( A11, acc );

        #endif
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

}// namespace tileh

}// namespace tf

}// namespace hlr

#endif // __HLR_TF_ARITH_HH
