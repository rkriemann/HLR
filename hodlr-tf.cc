//
// Project     : HLib
// File        : hodlr-lu.cc
// Description : HODLR arithmetic with cpp-taskflow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <taskflow/taskflow.hpp>

#include "approx.hh"
#include "common.inc"
#include "hodlr.hh"
#include "hodlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace HODLR
{

namespace TF
{

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
                                              auto [ U01, V01 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A01 ), U0 },
                                                                                                 { blas_mat_B< value_t >( A01 ), V1 },
                                                                                                 acc );
                                              A01->set_rank( U01, V01 );
                                          } );
        auto  add_10 = sf.silent_emplace( [&U1,&V0,A10,&acc] ()
                                          {
                                              auto [ U10, V10 ] = LR::approx_sum_svd< value_t >( { blas_mat_A< value_t >( A10 ), U1 },
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
        auto  task_01 = sf.silent_emplace( [A00,A01] () { trsml(  A00, blas_mat_A< value_t >( A01 ) ); } );
        auto  task_10 = sf.silent_emplace( [A00,A10] () { trsmuh( A00, blas_mat_B< value_t >( A10 ) ); } );

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


//
// main function
//
void
mymain ( int argc, char ** argv )
{
    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = HODLR::cluster( coord.get(), ntile );
    
    if ( verbose( 3 ) )
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  A   = problem->build_matrix( bct.get(), fixed_rank( k ) );
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "hlrtest_A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( HODLR TF )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        HODLR::TF::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }
}
