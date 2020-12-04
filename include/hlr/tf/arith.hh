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

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

#include <hlr/dag/lu.hh>
#include <hlr/tf/dag.hh>

#include "hlr/tf/arith_tiled.hh"

namespace hlr { namespace tf {

namespace hpro = HLIB;

using namespace hpro;

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
          const matop_t                    op_M,
          const TMatrix &                  M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    // assert( ! is_null( M ) );
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, TBlockMatrix );
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
                    auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );

                    mul_vec( alpha, op_M, *B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = cptrcast( &M, TDenseMatrix );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas_mat< value_t >( D ) ), x, value_t(1), y );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, TRkMatrix );

        if ( op_M == apply_normal )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( blas_mat_B< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_A< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_transposed )
        {
            assert( is_complex_type< value_t >::value == false );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
        else if ( op_M == apply_adjoint )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( blas_mat_A< value_t >( R ) ), x );

            blas::mulvec( alpha, blas_mat_B< value_t >( R ), t, value_t(1), y );
        }// if
    }// if
    else
        assert( false );
}

//
// compute C = C + α op( A ) op( B )
//
namespace detail
{

template < typename value_t,
           typename approx_t >
void
multiply ( ::tf::SubflowBuilder &   tf,
           const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = cptrcast( &B, TBlockMatrix );
        auto  BC = ptrcast(  &C, TBlockMatrix );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    tf.emplace(
                        [=,&acc,&approx] ( auto &  sf )
                        {
                            multiply< value_t >( sf, alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                        } );
                    
                    // multiply< value_t >( tf, alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc );
                }// for
            }// for
        }// for
    }// if
    else
    {
        // tf.emplace(
        //     [=,&A,&B,&C,&acc] ()
        //     {
        hlr::multiply( alpha, op_A, A, op_B, B, C, acc, approx );
            // } );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    ::tf::Taskflow  tf;
    
    tf.emplace( [=,&A,&B,&C,&acc,&approx] ( auto &  sf ) { detail::multiply( sf, alpha, op_A, A, op_B, B, C, acc, approx ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
namespace detail
{

inline void
gauss_elim ( ::tf::SubflowBuilder &  tf,
             hpro::TMatrix &         A,
             hpro::TMatrix &         T,
             const TTruncAcc &       acc )
{
    assert( ! is_null_any( &A, &T ) );
    assert( A.type() == T.type() );
    
    HLR_LOG( 4, hpro::to_string( "gauss_elim( %d ) {", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, TBlockMatrix );
        auto  BT = ptrcast( &T, TBlockMatrix );

        // A_00 = A_00⁻¹
        auto  inv_a00 = tf.emplace( [BA,BT,&acc] ( auto &  sf ) { detail::gauss_elim( sf, *BA->block(0,0), *BT->block(0,0), acc ); } );

        // T_01 = A_00⁻¹ · A_01
        auto  upd_t01 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( 1.0, apply_normal, BA->block(0,0), apply_normal, BA->block(0,1), 0.0, BT->block(0,1), acc ); } );
        inv_a00.precede( upd_t01 );
        
        // T_10 = A_10 · A_00⁻¹
        auto  upd_t10 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( 1.0, apply_normal, BA->block(1,0), apply_normal, BA->block(0,0), 0.0, BT->block(1,0), acc ); } );
        inv_a00.precede( upd_t10 );

        // A_11 = A_11 - T_10 · A_01
        auto  upd_a11 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( -1.0, apply_normal, BT->block(1,0), apply_normal, BA->block(0,1), 1.0, BA->block(1,1), acc ); } );
        upd_t10.precede( upd_a11 );
    
        // A_11 = A_11⁻¹
        auto  inv_a11 = tf.emplace( [BA,BT,&acc] ( auto &  sf ) { detail::gauss_elim( sf, *BA->block(1,1), *BT->block(1,1), acc ); } );
        upd_a11.precede( inv_a11 );

        // A_01 = - T_01 · A_11
        auto  upd_a01 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( -1.0, apply_normal, BT->block(0,1), apply_normal, BA->block(1,1), 0.0, BA->block(0,1), acc ); } );
        upd_t01.precede( upd_a01 );
        inv_a11.precede( upd_a01 );
            
        // A_10 = - A_11 · T_10
        auto  upd_a10 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( -1.0, apply_normal, BA->block(1,1), apply_normal, BT->block(1,0), 0.0, BA->block(1,0), acc ); } );
        upd_t10.precede( upd_a10 );
        inv_a11.precede( upd_a10 );

        // A_00 = T_00 - A_01 · T_10
        auto  upd_a00 = tf.emplace( [BA,BT,&acc] () { hpro::multiply( -1.0, apply_normal, BA->block(0,1), apply_normal, BT->block(1,0), 1.0, BA->block(0,0), acc ); } );
        upd_t10.precede( upd_a00 );
        upd_a01.precede( upd_a00 );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( &A, TDenseMatrix );
        
        if ( A.is_complex() ) blas::invert( DA->blas_cmat() );
        else                  blas::invert( DA->blas_rmat() );
    }// if
    else
        assert( false );

    HLR_LOG( 4, hpro::to_string( "} gauss_elim( %d )", A.id() ) );
}

}// namespace detail

inline void
gauss_elim ( hpro::TMatrix &    A,
             hpro::TMatrix &    T,
             const TTruncAcc &  acc )
{
    ::tf::Taskflow  tf;

    tf.emplace( [&A,&T,&acc] ( auto &  sf ) { detail::gauss_elim( sf, A, T, acc ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

namespace tlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

//
// LU factorization for TLR block format
// 
template < typename value_t,
           typename approx_t >
void
lu ( TMatrix &          A,
     const TTruncAcc &  acc,
     const approx_t &   approx )
{
    assert( is_blocked( A ) );
    
    auto  BA  = ptrcast( &A, TBlockMatrix );

    ::tf::Taskflow  tf;
    
    auto                   nbr = BA->nblock_rows();
    auto                   nbc = BA->nblock_cols();
    tensor2< ::tf::Task >  fs_tasks( nbr, nbc );
    tensor3< ::tf::Task >  u_tasks( nbr, nbr, nbc );
    tensor3< char >        has_u_task( nbr, nbr, nbc, false );

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );

        fs_tasks(i,i) = tf.emplace( [A_ii] ()
                                    {
                                        TScopedLock  lock( *A_ii );
                                        
                                        blas::invert( blas_mat< value_t >( A_ii ) );
                                    } );
            
        for ( uint  l = 0; l < i; ++l )
            if ( has_u_task(l,i,i) )
                u_tasks(l,i,i).precede( fs_tasks(i,i) );
            
        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is identity; task only for ensuring correct execution order
            fs_tasks(i,j) = tf.emplace( [A_ii,BA,i,j] ()
                                        {
                                            auto         A_ij = BA->block(i,j);
                                            TScopedLock  lock( *A_ij );
                                        } );
            fs_tasks(i,i).precede( fs_tasks(i,j) );

            for ( uint  l = 0; l < i; ++l )
                if ( has_u_task(l,i,j) )
                    u_tasks(l,i,j).precede( fs_tasks(i,j) );
            
            fs_tasks(j,i) = tf.emplace( [A_ii,BA,i,j] ()
                                        {
                                            auto         A_ji = BA->block(j,i);
                                            TScopedLock  lock( *A_ji );
                                            
                                            trsmuh< value_t >( *A_ii, *A_ji );
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

                u_tasks(i,j,l)    = tf.emplace( [A_ji,A_il,A_jl,&acc,&approx] ()
                                                {
                                                    TScopedLock  lock( *A_jl );
                                                    
                                                    hlr::tf::multiply< value_t >( value_t(-1),
                                                                                  hpro::apply_normal, *A_ji,
                                                                                  hpro::apply_normal, *A_il,
                                                                                  *A_jl, acc, approx );
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
    
    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

}// namespace tlr

namespace hodlr
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

//
// add U·V' to matrix A
//
template < typename value_t,
           typename approx_t >
void
addlr ( const blas::matrix< value_t > &  U,
        const blas::matrix< value_t > &  V,
        TMatrix &                        A,
        const TTruncAcc &                acc,
        const approx_t &                 approx )
{
    static_assert( std::is_same< value_t, typename approx_t::value_t >::value,
                   "matrices and approximation object need to have same value type" );
    
    if ( hpro::verbose( 4 ) )
        DBG::printf( "addlr( %d )", A.id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        auto  U0  = blas::matrix< value_t >( U, A00->row_is() - A.row_ofs(), blas::range::all );
        auto  U1  = blas::matrix< value_t >( U, A11->row_is() - A.row_ofs(), blas::range::all );
        auto  V0  = blas::matrix< value_t >( V, A00->col_is() - A.col_ofs(), blas::range::all );
        auto  V1  = blas::matrix< value_t >( V, A11->col_is() - A.col_ofs(), blas::range::all );

        ::tf::Taskflow  tf;
        
        auto  add_00 = tf.emplace( [&] () { addlr( U0, V0, *A00, acc, approx ); } );
        auto  add_11 = tf.emplace( [&] () { addlr( U1, V1, *A11, acc, approx ); } );
        auto  add_01 = tf.emplace( [&] ()
                                   {
                                       auto [ U01, V01 ] = approx( { blas_mat_A< value_t >( A01 ), U0 },
                                                                   { blas_mat_B< value_t >( A01 ), V1 },
                                                                   acc );
                                       A01->set_lrmat( U01, V01 );
                                   } );
        auto  add_10 = tf.emplace( [&] ()
                                   {
                                       auto [ U10, V10 ] = approx( { blas_mat_A< value_t >( A10 ), U1 },
                                                                   { blas_mat_B< value_t >( A10 ), V0 },
                                                                   acc );
                                       A10->set_lrmat( U10, V10 );
                                   } );

        ::tf::Executor  executor;
    
        executor.run( tf ).wait();
    }// if
    else
    {
        auto  DA = ptrcast( &A, TDenseMatrix );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( TMatrix &          A,
     const TTruncAcc &  acc,
     const approx_t &   approx )
{
    static_assert( std::is_same< value_t, typename approx_t::value_t >::value,
                   "matrices and approximation object need to have same value type" );
    
    if ( hpro::verbose( 4 ) )
        DBG::printf( "lu( %d )", A.id() );

    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );

        #if  0

        //
        // all function calls wrapped in tasks
        //
        
        ::tf::Taskflow  tf;
        
        auto  task_00 = tf.emplace( [A00,&acc] () { lu< value_t >( *A00, acc ); } );
        auto  task_01 = tf.emplace( [A00,A01]  () { seq::hodlr::trsml(  *A00, blas_mat_A< value_t >( A01 ) ); } );
        auto  task_10 = tf.emplace( [A00,A10]  () { seq::hodlr::trsmuh( *A00, blas_mat_B< value_t >( A10 ) ); } );

        task_00.precede( { task_01, task_10 } );
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  [ task_T,   T ] = tf.emplace( [A10,A01] () { return blas::prod(  value_t(1), blas::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) ); } );
        auto  [ task_UT, UT ] = tf.emplace( [A10,&T]  () { return blas::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T.get() ); } );

        task_01.precede( task_T );
        task_10.precede( task_T );
        task_T.precede( task_UT );
        
        auto  task_add11      = tf.emplace( [A01,A11,&UT,&acc] ()
                                            { addlr< value_t >( UT.get(), blas_mat_B< value_t >( A01 ), *A11, acc ); } );

        task_UT.precede( task_add11 );
        
        auto  task_11         = tf.emplace( [A11,&acc] () { lu< value_t >( *A11, acc ); } );

        task_add11.precede( task_11 );

        tf.wait_for_all();

        #else

        //
        // only tasks for the two parallel calls
        //
        
        lu< value_t >( *A00, acc, approx );

        {
            ::tf::Taskflow  tf;
        
            auto  task_01 = tf.emplace( [A00,A01]  () { seq::hodlr::trsml(  *A00, blas_mat_A< value_t >( A01 ) ); } );
            auto  task_10 = tf.emplace( [A00,A10]  () { seq::hodlr::trsmuh( *A00, blas_mat_B< value_t >( A10 ) ); } );

            ::tf::Executor  executor;
    
            executor.run( tf ).wait();
        }
        
        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( blas_mat_B< value_t >( A10 ) ), blas_mat_A< value_t >( A01 ) );
        auto  UT = blas::prod( value_t(-1), blas_mat_A< value_t >( A10 ), T );

        addlr< value_t >( UT, blas_mat_B< value_t >( A01 ), *A11, acc, approx );
        lu< value_t >( *A11, acc, approx );

        #endif
    }// if
    else
    {
        auto  DA = ptrcast( &A, TDenseMatrix );
        
        blas::invert( DA->blas_rmat() );
    }// else
}

}// namespace hodlr

namespace tileh
{

///////////////////////////////////////////////////////////////////////
//
// arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( TMatrix &          A,
     const TTruncAcc &  acc,
     const approx_t &   approx )
{
    static_assert( std::is_same< value_t, typename approx_t::value_t >::value,
                   "matrices and approximation object need to have same value type" );
    
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    ::tf::Taskflow         tf;
    tensor2< ::tf::Task >  finished( nbr, nbc );
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        finished( i, i ) = tf.emplace(
            [=,&acc] () 
            {
                auto  dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *(BA->block( i, i )),
                                                                      128,
                                                                      tf::dag::refine ) );
                
                hlr::tf::dag::run( dag, acc );
            } );
        
        for ( uint j = i+1; j < nbr; ++j )
        {
            finished( j, i ) = tf.emplace(
                [=,&acc] () 
                {
                    auto  dag = std::move( hlr::dag::gen_dag_solve_upper( *BA->block( i, i ),
                                                                          *BA->block( j, i ),
                                                                          128,
                                                                          tf::dag::refine ) );
                    
                    hlr::tf::dag::run( dag, acc );
                } );

            finished( i, i ).precede( finished( j, i ) );
        }// for
            
        for ( uint  l = i+1; l < nbc; ++l )
        {
            finished( i, l ) = tf.emplace(
                [=,&acc] () 
                {
                    auto  dag = std::move( hlr::dag::gen_dag_solve_lower( *BA->block( i, i ),
                                                                          *BA->block( i, l ),
                                                                          128,
                                                                          tf::dag::refine ) );
                    
                    hlr::tf::dag::run( dag, acc );
                } );

            finished( i, i ).precede( finished( i, l ) );
        }// for
    }// for
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                auto  update = tf.emplace(
                    [=,&acc,&approx] ( auto &  sf ) 
                    {
                        hlr::tf::detail::multiply( sf,
                                                   value_t(-1),
                                                   apply_normal, * BA->block( j, i ),
                                                   apply_normal, * BA->block( i, l ),
                                                   * BA->block( j, l ), acc, approx );
                    } );

                finished( j, i ).precede( update );
                finished( i, l ).precede( update );
                update.precede( finished( j, l ) );
            }// for
        }// for
    }// for

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

}// namespace tileh

}}// namespace hlr::tf

#endif // __HLR_TF_ARITH_HH
