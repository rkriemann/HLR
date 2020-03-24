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
#include <tbb/blocked_range3d.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

#include <hlr/dag/lu.hh>
#include <hlr/tbb/dag.hh>

#include <hlr/tbb/arith_impl.hh>

namespace hlr { namespace tbb {

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
          const hpro::matop_t              op_M,
          const hpro::TMatrix &            M,
          const blas::vector< value_t > &  x,
          blas::vector< value_t > &        y )
{
    auto        mtx_map = detail::mutex_map_t();
    const auto  is      = M.row_is( op_M );

    for ( idx_t  i = is.first() / detail::CHUNK_SIZE; i <= idx_t(is.last() / detail::CHUNK_SIZE); ++i )
        mtx_map[ i ] = std::make_unique< std::mutex >();
    
    detail::mul_vec_chunk( alpha, op_M, M, x, y, M.row_is( op_M ).first(), M.col_is( op_M ).first(), mtx_map );
}

template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const matop_t                             op_M,
          const TMatrix &                           M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );

    mul_vec( alpha, op_M, M, hpro::blas_vec< value_t >( x ), hpro::blas_vec< value_t >( y ) );
}

//
// compute C = C + α op( A ) op( B )
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = cptrcast( &B, TBlockMatrix );
        auto  BC = ptrcast(  &C, TBlockMatrix );

        ::tbb::parallel_for(
            ::tbb::blocked_range3d< size_t >( 0, BC->nblock_rows(),
                                              0, BC->nblock_cols(),
                                              0, BA->nblock_cols( op_A ) ),
            [=,&acc] ( const auto &  r )
            {
                for ( auto  i = r.pages().begin(); i != r.pages().end(); ++i )
                {
                    for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                    {
                        for ( auto  l = r.cols().begin(); l != r.cols().end(); ++l )
                        {
                            auto  C_ij = BC->block( i, j );
                            auto  A_il = BA->block( i, l, op_A );
                            auto  B_lj = BB->block( l, j, op_B );
                
                            if ( is_null_any( A_il, B_lj ) )
                                continue;
                    
                            HLR_ASSERT( ! is_null( C_ij ) );
            
                            multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc );
                        }// for
                    }// for
                }// for
            } );
    }// if
    else
        hpro::multiply< value_t >( alpha, op_A, &A, op_B, &B, value_t(1), &C, acc );
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
                                         hlr::tbb::multiply< value_t >( value_t(-1),
                                                                        hpro::apply_normal, *BA->block( j, i ),
                                                                        hpro::apply_normal, *BA->block( i, l ),
                                                                        *BA->block( j, l ), acc );
                                     }// for
                                 }// for
                             } );
    }// for
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
template < typename value_t >
void
addlr ( blas::matrix< value_t > &  U,
        blas::matrix< value_t > &  V,
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

        blas::matrix< value_t >  U0( U, A00->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A->row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A->col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A->col_ofs(), blas::range::all );

        ::tbb::parallel_invoke( [&U0,&V0,A00,&acc] () { addlr( U0, V0, A00, acc ); },
                                [&U1,&V1,A11,&acc] () { addlr( U1, V1, A11, acc ); },
                                [&U0,&V1,A01,&acc] ()
                                {
                                    auto [ U01, V01 ] = hlr::approx::svd< value_t >( { hpro::blas_mat_A< value_t >( A01 ), U0 },
                                                                                     { hpro::blas_mat_B< value_t >( A01 ), V1 },
                                                                                     acc );
                                    A01->set_lrmat( U01, V01 );
                                },
                                [&U1,&V0,A10,&acc] ()
                                {
                                    auto [ U10, V10 ] = hlr::approx::svd< value_t >( { hpro::blas_mat_A< value_t >( A10 ), U1 },
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
        {
            auto  dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *(BA->block( i, i )),
                                                                  128,
                                                                  tbb::dag::refine ) );

            hlr::tbb::dag::run( dag, acc );

            // hpro::LU::factorise_rec( BA->block( i, i ), acc );
        }

        ::tbb::parallel_invoke(
            [BA,i,nbr,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbr,
                                     [BA,i,&acc] ( uint  j )
                                     {
                                         auto  dag = std::move( hlr::dag::gen_dag_solve_upper( BA->block( i, i ),
                                                                                               BA->block( j, i ),
                                                                                               128,
                                                                                               tbb::dag::refine ) );
                                                     
                                         hlr::tbb::dag::run( dag, acc );
                                         // hpro::solve_upper_right( BA->block( j, i ),
                                         //                          BA->block( i, i ), nullptr, acc,
                                         //                          hpro::solve_option_t( hpro::block_wise, hpro::general_diag, hpro::store_inverse ) ); 
                                     } );
            },
                
            [BA,i,nbc,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbc,
                                     [BA,i,&acc] ( uint  l )
                                     {
                                         auto  dag = std::move( hlr::dag::gen_dag_solve_lower( BA->block( i, i ),
                                                                                               BA->block( i, l ),
                                                                                               128,
                                                                                               tbb::dag::refine ) );
                                                     
                                         hlr::tbb::dag::run( dag, acc );
                                         // hpro::solve_lower_left( hpro::apply_normal, BA->block( i, i ), nullptr,
                                         //                         BA->block( i, l ), acc,
                                         //                         hpro::solve_option_t( hpro::block_wise, hpro::unit_diag, hpro::store_inverse ) );
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
                                         hlr::tbb::multiply( -1.0,
                                                             apply_normal, * BA->block( j, i ),
                                                             apply_normal, * BA->block( i, l ),
                                                             * BA->block( j, l ), acc );
                                     }// for
                                 }// for
                             } );
    }// for
}

}// namespace tileh

}}// namespace hlr::tbb

#endif // __HLR_TBB_ARITH_HH
