#ifndef __HLR_OMP_ARITH_HH
#define __HLR_OMP_ARITH_HH
//
// Project     : HLR
// Module      : omp/arith
// Description : arithmetic functions using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/matrix/lrmatrix.hh"
#include "hlr/matrix/dense_matrix.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/solve.hh"
#include "hlr/seq/arith.hh"

#include <hlr/dag/lu.hh>
#include <hlr/omp/dag.hh>

namespace hlr { namespace omp {

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
mul_vec ( const value_t                     alpha,
          const matop_t                     op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y )
{
    // HLR_ASSERT( ! is_null( M ) );
    // HLR_ASSERT( M->ncols( op_M ) == x.length() );
    // HLR_ASSERT( M->nrows( op_M ) == y.length() );

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
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
    else if ( hlr::matrix::is_dense( M ) )
    {
        auto  D = cptrcast( &M, hlr::matrix::dense_matrix< value_t > );
        
        blas::mulvec( alpha, blas::mat_view( op_M, D->mat() ), x, value_t(1), y );
    }// if
    else if ( hlr::matrix::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, hlr::matrix::lrmatrix< value_t > );

        if ( op_M == apply_normal )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( R->V() ), x );

            blas::mulvec( alpha, R->U(), t, value_t(1), y );
        }// if
        else if ( op_M == apply_transposed )
        {
            HLR_ASSERT( Hpro::is_complex_type_v< value_t > == false );
            
            auto  t = blas::mulvec( value_t(1), blas::transposed( R->U() ), x );

            blas::mulvec( alpha, R->V(), t, value_t(1), y );
        }// if
        else if ( op_M == apply_adjoint )
        {
            auto  t = blas::mulvec( value_t(1), blas::adjoint( R->U() ), x );

            blas::mulvec( alpha, R->V(), t, value_t(1), y );
        }// if
    }// if
    else
        HLR_ASSERT( false );
}

template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const matop_t                             op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    mul_vec( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

namespace detail
{

//
// compute C = C + α op( A ) op( B )
//
template < typename value_t,
           typename approx_t >
void
multiply_task ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B,
                Hpro::TMatrix< value_t > &        C,
                const Hpro::TTruncAcc &           acc,
                const approx_t &                  approx )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop collapse(3) default(shared) firstprivate(alpha,op_A,op_B)
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                {
                    auto  C_ij = BC->block(i,j);
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_task< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                }// for
            }// for
        }// omp taskloop for
    }// if
    else
        hlr::multiply( alpha, op_A, A, op_B, B, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply_parfor ( const value_t                    alpha,
                  const Hpro::matop_t              op_A,
                  const Hpro::TMatrix< value_t > & A,
                  const Hpro::matop_t              op_B,
                  const Hpro::TMatrix< value_t > & B,
                  Hpro::TMatrix< value_t > &       C,
                  const Hpro::TTruncAcc &          acc,
                  const approx_t &                 approx )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );

        #pragma omp for collapse(3) firstprivate(alpha,op_A,op_B)
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                {
                    auto  C_ij = BC->block( i, j );
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_parfor< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                }// for
            }// for
        }// for
    }// if
    else
        hlr::multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx );
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                    alpha,
           const Hpro::matop_t              op_A,
           const Hpro::TMatrix< value_t > & A,
           const Hpro::matop_t              op_B,
           const Hpro::TMatrix< value_t > & B,
           Hpro::TMatrix< value_t > &       C,
           const Hpro::TTruncAcc &          acc,
           const approx_t &                 approx )
{
    #pragma omp parallel
    {
        // #pragma omp single
        // {
        //     #pragma omp task
        //     {
        //         detail::multiply_task( alpha, op_A, A, op_B, B, C, acc, approx );
        //     }// omp task
        // }// omp single
        
        detail::multiply_parfor( alpha, op_A, A, op_B, B, C, acc, approx );
    }// omp parallel
}

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
namespace detail
{

template < typename value_t,
           typename approx_t >
void
gauss_elim_task ( Hpro::TMatrix< value_t > & A,
                  Hpro::TMatrix< value_t > & T,
                  const Hpro::TTruncAcc &    acc,
                  const approx_t &           approx )
{
    HLR_ASSERT( ! is_null_any( &A, &T ) );
    HLR_ASSERT( A.type() == T.type() );

    HLR_LOG( 4, Hpro::to_string( "gauss_elim( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BT = ptrcast( &T, Hpro::TBlockMatrix< value_t > );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        gauss_elim_task( *MA(0,0), *MT(0,0), acc, approx );

        // T_01 = A_00⁻¹ · A_01
        #pragma omp taskgroup
        {
            #pragma omp task
            {
                MT(0,1)->scale( 0.0 );
                hlr::multiply( 1.0, Hpro::apply_normal, MA(0,0), Hpro::apply_normal, MA(0,1), MT(0,1), acc, approx );
            }// omp task
        
            // T_10 = A_10 · A_00⁻¹
            #pragma omp task
            {
                MT(1,0)->scale( 0.0 );
                hlr::multiply( 1.0, Hpro::apply_normal, MA(1,0), Hpro::apply_normal, MA(0,0), MT(1,0), acc, approx );
            }// omp task
        }// taskgroup
        
        // A_11 = A_11 - T_10 · A_01
        hlr::multiply( -1.0, Hpro::apply_normal, MT(1,0), Hpro::apply_normal, MA(0,1), MA(1,1), acc, approx );
    
        // A_11 = A_11⁻¹
        gauss_elim_task( *MA(1,1), *MT(1,1), acc, approx );

        #pragma omp taskgroup
        {
            // A_01 = - T_01 · A_11
            #pragma omp task
            {
                MA(0,1)->scale( 0.0 );
                hlr::multiply( -1.0, Hpro::apply_normal, MT(0,1), Hpro::apply_normal, MA(1,1), MA(0,1), acc, approx );
            }// omp task
            
            // A_10 = - A_11 · T_10
            #pragma omp task
            {
                MA(1,0)->scale( 0.0 );
                hlr::multiply( -1.0, Hpro::apply_normal, MA(1,1), Hpro::apply_normal, MT(1,0), MA(1,0), acc, approx );
            }// omp task
        }// taskgroup

        // A_00 = T_00 - A_01 · T_10
        hlr::multiply( -1.0, Hpro::apply_normal, MA(0,1), Hpro::apply_normal, MT(1,0), MA(0,0), acc, approx );
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
        
        if ( A.is_complex() ) blas::invert( DA->blas_cmat() );
        else                  blas::invert( DA->blas_rmat() );
    }// if
    else
        HLR_ASSERT( false );

    HLR_LOG( 4, Hpro::to_string( "gauss_elim( %d )", A.id() ) );
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
gauss_elim ( Hpro::TMatrix< value_t > & A,
             Hpro::TMatrix< value_t > & T,
             const Hpro::TTruncAcc &    acc,
             const approx_t &           approx )
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                detail::gauss_elim_task( A, T, acc, approx );
            }// # task
        }// single 
    }// omp
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
lu ( Hpro::TMatrix< value_t > & A,
     const Hpro::TTruncAcc &    acc,
     const approx_t &           approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    HLR_ASSERT( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );
            
        blas::invert( Hpro::blas_mat< value_t >( A_ii ) );

        #pragma omp parallel for
        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal !!!
            // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
            trsmuh< value_t >( *A_ii, *BA->block( j, i ) ); // A10->blas_rmat_B() );
        }// for

        #pragma omp parallel for collapse(2)
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                hlr::omp::multiply< value_t >( value_t(-1),
                                               Hpro::apply_normal, *BA->block( j, i ),
                                               Hpro::apply_normal, *BA->block( i, l ),
                                               *BA->block( j, l ), acc, approx );
            }// for
        }// for
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
template < typename value_t,
           typename approx_t >
void
addlr ( blas::Matrix< value_t > &  U,
        blas::Matrix< value_t > &  V,
        Hpro::TMatrix< value_t > &            A,
        const Hpro::TTruncAcc &    acc,
        const approx_t &           approx )
{
    HLR_LOG( 4, Hpro::to_string( "addlr( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        auto  U0  = blas::matrix< value_t >( U, A00->row_is() - A.row_ofs(), blas::range::all );
        auto  U1  = blas::matrix< value_t >( U, A11->row_is() - A.row_ofs(), blas::range::all );
        auto  V0  = blas::matrix< value_t >( V, A00->col_is() - A.col_ofs(), blas::range::all );
        auto  V1  = blas::matrix< value_t >( V, A11->col_is() - A.col_ofs(), blas::range::all );

        #pragma omp parallel sections
        {
            #pragma omp section
            { addlr( U0, V0, *A00, acc, approx ); }

            #pragma omp section
            { addlr( U1, V1, *A11, acc, approx ); }

            #pragma omp section
            {
                auto [ U01, V01 ] = approx( { Hpro::blas_mat_A< value_t >( A01 ), U0 },
                                            { Hpro::blas_mat_B< value_t >( A01 ), V1 },
                                            acc );
                
                A01->set_lrmat( U01, V01 );
            }
            
            #pragma omp section
            {
                auto [ U10, V10 ] = approx( { Hpro::blas_mat_A< value_t >( A10 ), U1 },
                                            { Hpro::blas_mat_B< value_t >( A10 ), V0 },
                                            acc );
                A10->set_lrmat( U10, V10 );
            }
        }
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), Hpro::blas_mat< value_t >( DA ) );
    }// else
}

//
// compute LU factorization of A
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        lu< value_t >( *A00, acc, approx );

        #pragma omp parallel sections
        {
            #pragma omp section
            { seq::hodlr::trsml(  *A00, Hpro::blas_mat_A< value_t >( A01 ) ); }
            
            #pragma omp section
            { seq::hodlr::trsmuh( *A00, Hpro::blas_mat_B< value_t >( A10 ) ); }
        }

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( Hpro::blas_mat_B< value_t >( A10 ) ), Hpro::blas_mat_A< value_t >( A01 ) ); 
        auto  UT = blas::prod( value_t(-1), Hpro::blas_mat_A< value_t >( A10 ), T );

        addlr< value_t >( UT, Hpro::blas_mat_B< value_t >( A01 ), *A11, acc, approx );
        
        lu< value_t >( *A11, acc, approx );
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
        
        blas::invert( Hpro::blas_mat< value_t >( DA ) );
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
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_ASSERT( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    #pragma omp parallel
    {
        for ( uint  i = 0; i < nbr; ++i )
        {
            #pragma omp single
            {
                auto  dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *(BA->block( i, i )),
                                                                      128,
                                                                      omp::dag::refine ) );

                hlr::omp::dag::run( dag, acc );

                // Hpro::LU::factorise_rec( BA->block( i, i ), acc );
            }// omp single

            // #pragma omp sections
            // {
            //     #pragma omp section
            //     {
                    #pragma omp for
                    for ( uint j = i+1; j < nbr; ++j )
                    {
                        auto  dag = std::move( hlr::dag::gen_dag_solve_upper( *BA->block( i, i ),
                                                                              *BA->block( j, i ),
                                                                              128,
                                                                              omp::dag::refine ) );
                    
                        hlr::omp::dag::run( dag, acc );
                    
                        // Hpro::solve_upper_right( BA->block( j, i ),
                        //                          BA->block( i, i ), nullptr, acc,
                        //                          Hpro::solve_option_t( Hpro::block_wise, Hpro::general_diag, Hpro::store_inverse ) );
                    }// for
                // }// omp section
            
                // #pragma omp section
                // {
                    #pragma omp for
                    for ( uint  l = i+1; l < nbc; ++l )
                    {
                        auto  dag = std::move( hlr::dag::gen_dag_solve_lower( *BA->block( i, i ),
                                                                              *BA->block( i, l ),
                                                                              128,
                                                                              omp::dag::refine ) );
                                                     
                        hlr::omp::dag::run( dag, acc );
                    
                        // Hpro::solve_lower_left( Hpro::apply_normal, BA->block( i, i ), nullptr,
                        //                         BA->block( i, l ), acc,
                        //                         Hpro::solve_option_t( Hpro::block_wise, Hpro::unit_diag, Hpro::store_inverse ) );
                    }// for
            //     }// omp section
            // }// omp sections
            
            #pragma omp for collapse(2)
            for ( uint  j = i+1; j < nbr; ++j )
            {
                for ( uint  l = i+1; l < nbc; ++l )
                {
                    hlr::omp::detail::multiply_parfor( value_t(-1),
                                                       apply_normal, * BA->block( j, i ),
                                                       apply_normal, * BA->block( i, l ),
                                                       * BA->block( j, l ), acc, approx );
                }// for
            }// for
        }// for
    }// omp parallel
}

}// namespace tileh

}}// namespace hlr::omp

#endif // __HLR_OMP_ARITH_HH
