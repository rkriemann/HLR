#ifndef __HLR_TBB_ARITH_HH
#define __HLR_TBB_ARITH_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include <hlr/arith/defaults.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/multiply.hh>
#include <hlr/arith/solve.hh>
#include <hlr/seq/arith.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/luinv_eval.hh>

#include <hlr/dag/lu.hh>
#include <hlr/tbb/dag.hh>

#include <hlr/tbb/detail/arith.hh>

namespace hlr { namespace tbb {

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
          const Hpro::matop_t               op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y )
{
    auto        mtx_map = detail::mutex_map_t();
    const auto  is      = M.row_is( op_M );

    for ( idx_t  i = is.first() / detail::CHUNK_SIZE; i <= idx_t(is.last() / detail::CHUNK_SIZE); ++i )
        mtx_map[ i ] = std::make_unique< std::mutex >();
    
    detail::mul_vec_chunk( alpha, op_M, M, x, y, M.row_is( op_M ).first(), M.col_is( op_M ).first(), mtx_map );
}

template < typename value_t >
void
mul_vec_chunk ( const value_t                             alpha,
                const matop_t                             op_M,
                const Hpro::TMatrix< value_t > &          M,
                const vector::scalar_vector< value_t > &  x,
                vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );

    mul_vec( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

template < typename value_t >
void
mul_vec_row ( const value_t                             alpha,
              const matop_t                             op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );

    detail::mul_vec_row( alpha, op_M, M, x, y );
}

template < typename value_t >
void
mul_vec_reduce ( const value_t                             alpha,
                 const matop_t                             op_M,
                 const Hpro::TMatrix< value_t > &          M,
                 const vector::scalar_vector< value_t > &  x,
                 vector::scalar_vector< value_t > &        y )
{
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( Hpro::is_complex_type< value_t >::value == y.is_complex() );

    // just for now
    HLR_ASSERT( op_M == apply_normal );
    
    detail::mul_vec_reduce( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const matop_t                             op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    // mul_vec_chunk( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
    mul_vec_row( alpha, op_M, M, x, y );
    // mul_vec_reduce( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

//
// compute C := C + α A with different types of A/C
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const Hpro::TMatrix< value_t > &    A,
      Hpro::TMatrix< value_t > &          C,
      const Hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    if ( alpha == value_t(0) )
        return;

    if ( is_blocked_all( A, C ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BC =  ptrcast( &C, Hpro::TBlockMatrix< value_t > );
        
        HLR_ASSERT(( BA->block_rows() == BC->nblock_rows() ) &&
                   ( BA->block_cols() == BC->nblock_cols() ));

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BA->block( i, j ) ) )
                    continue;
                
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                add( alpha, * BA->block( i, j ), * BC->block( i, j ), acc, approx );
            }// for
        }// for
    }// if
    else
        hlr::add( alpha, A, C, acc, approx );
}

//
// compute C = C + α op( A ) op( B )
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                     alpha,
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
            
                            multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx );
                        }// for
                    }// for
                }// for
            } );
    }// if
    else
        hlr::multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx );
}

//
// compute Hadamard product A = α A*B 
//
template < typename value_t,
           typename approx_t >
void
multiply_hadamard ( const value_t                     alpha,
                    Hpro::TMatrix< value_t > &        A,
                    const Hpro::TMatrix< value_t > &  B,
                    const Hpro::TTruncAcc &           acc,
                    const approx_t &                  approx )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = ptrcast( &A,  Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, BA->nblock_rows(),
                                              0, BA->nblock_cols() ),
            [=,&acc] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  A_ij = BA->block( i, j );
                        auto  B_ij = BB->block( i, j );
                
                        HLR_ASSERT( ! is_null_any( A_ij, B_ij ) );
            
                        multiply_hadamard< value_t >( alpha, *A_ij, *B_ij, acc, approx );
                    }// for
                }// for
            } );
    }// if
    else
    {
        hlr::seq::multiply_hadamard< value_t >( alpha, A, B, acc, approx );
    }// if
}

//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M,
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( is_nd( L ) && is_blocked( M ) )
    {
        auto  BL  = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  BM  =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  nbr = BM->nblock_rows();
        auto  nbc = BM->nblock_cols();
        
        if ( side == from_left )
        {
            HLR_ASSERT( ( BL->nblock_rows() == nbr ) && ( BL->nblock_cols() == nbr ) );
            
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< size_t >( 0, nbr-1,
                                                  0, nbc ),
                [=,&acc,&approx] ( const auto &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            auto  L_ii = BL->block( i, i );
                            auto  M_ij = BM->block( i, j );

                            HLR_ASSERT( ! is_null( L_ii ) );
                            
                            if ( ! is_null( M_ij ) )
                            {
                                solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, approx );

                                if ( ! is_null( BL->block( nbr-1, i ) ) )
                                {
                                    HLR_ASSERT( ! is_null( BM->block( nbr-1, j ) ) );
                                    
                                    multiply< value_t >( value_t(-1),
                                                         apply_normal, *BL->block( nbr-1, i ),
                                                         apply_normal, *M_ij,
                                                         *BM->block( nbr-1, j ), acc, approx );
                                }// if
                            }// if
                        }// for
                    }// for
                } );
                
            HLR_ASSERT( ! is_null( BL->block( nbr-1, nbr-1 ) ) );
            
            ::tbb::parallel_for< uint >(
                0, nbc,
                [=,&acc,&approx] ( const uint  j )
                {
                    auto  M_ij = BM->block( nbr-1, j );
                    
                    if ( ! is_null( M_ij ) )
                        solve_lower_tri< value_t >( side, diag, *BL->block( nbr-1, nbr-1 ), *M_ij, acc, approx );
                } );
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        if ( side == from_left )
        {
            for ( uint i = 0; i < BL->nblock_cols(); ++i )
            {
                const auto  L_ii = BL->block( i, i );

                HLR_ASSERT( ! is_null( L_ii ) );
            
                ::tbb::parallel_for< uint >( 0, BM->nblock_cols(),
                                             [&,i,BL,BM] ( const uint  j )
                                             {
                                                 auto  M_ij = BM->block( i, j );
                
                                                 if ( ! is_null( M_ij ) )
                                                     solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, approx );
                                             } );

                ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, BM->nblock_rows(),
                                                                       0, BM->nblock_cols() ),
                                     [&,i,BL,BM] ( const auto  r )
                                     {
                                         for ( auto  k = r.rows().begin(); k != r.rows().end(); ++k )
                                         {
                                             for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                                             {
                                                 if ( ! is_null_any( BL->block(k,i), BM->block(i,j) ) )
                                                 {
                                                     HLR_ASSERT( ! is_null( BM->block(k,j) ) );
                                                     
                                                     multiply< value_t >( value_t(-1),
                                                                          apply_normal, *BL->block(k,i),
                                                                          apply_normal, *BM->block(i,j),
                                                                          *BM->block(k,j), acc, approx );
                                                 }// if
                                             }// for
                                         }// for
                                     } );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else
        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
}

//
// solve U·X = M (side = from_left) or X·U = M (side = from_right)
// with upper triangular matrix U
// - on exit, M contains X
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M,
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( is_nd( U ) && is_blocked( M ) )
    {
        auto  BU  = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  BM  =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  nbr = BM->nblock_rows();
        auto  nbc = BM->nblock_cols();
        
        if ( side == from_left )
        {
            HLR_ASSERT( false );
        }// if
        else
        {
            HLR_ASSERT( ( BU->nblock_rows() == nbc ) && ( BU->nblock_cols() == nbc ) );
            
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< size_t >( 0, nbr,
                                                  0, nbc-1 ),
                [=,&acc,&approx] ( const auto &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            auto  U_jj = BU->block( j, j );
                            auto  M_ij = BM->block( i, j );

                            HLR_ASSERT( ! is_null( U_jj ) );
                            
                            if ( ! is_null( M_ij ) )
                            {
                                solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, approx );

                                if ( ! is_null( BU->block( j, nbc-1 ) ) )
                                {
                                    HLR_ASSERT( ! is_null( BM->block( i, nbc-1 ) ) );
                                    
                                    multiply< value_t >( value_t(-1),
                                                         apply_normal, *M_ij,
                                                         apply_normal, *BU->block( j, nbc-1 ),
                                                         *BM->block( i, nbc-1 ), acc, approx );
                                }// if
                            }// if
                        }// for
                    }// for
                } );
                
            HLR_ASSERT( ! is_null( BU->block( nbc-1, nbc-1 ) ) );
            
            ::tbb::parallel_for< uint >(
                0, nbr,
                [=,&acc,&approx] ( const uint  i )
                {
                    auto  M_ij = BM->block( i, nbc-1 );
                    
                    if ( ! is_null( M_ij ) )
                        solve_upper_tri< value_t >( side, diag, *BU->block( nbc-1, nbc-1 ), *M_ij, acc, approx );
                } );
        }// else
    }// if
    else if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        if ( side == from_left )
        {
            HLR_ASSERT( false );
        }// if
        else
        {
            for ( uint j = 0; j < BM->nblock_cols(); ++j )
            {
                const auto  U_jj = BU->block( j, j );

                HLR_ASSERT( ! is_null( U_jj ) );
            
                ::tbb::parallel_for< uint >( 0, BM->nblock_rows(),
                                             [&,j,BU,BM] ( const uint  i )
                                             {
                                                 auto  M_ij = BM->block( i, j );
                
                                                 if ( ! is_null( M_ij ) )
                                                     solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, approx );
                                             } );
            
                ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( j+1, BM->nblock_cols(),
                                                                       0, BM->nblock_rows() ),
                                     [&,j,BU,BM] ( const auto  r )
                                     {
                                         for ( auto  k = r.rows().begin(); k != r.rows().end(); ++k )
                                         {
                                             for ( auto  i = r.cols().begin(); i != r.cols().end(); ++i )
                                             {
                                                 if ( ! is_null_any( BM->block(i,j), BU->block(j,k) ) )
                                                 {
                                                     HLR_ASSERT( ! is_null( BM->block(i,k) ) );
                                                     
                                                     multiply< value_t >( value_t(-1),
                                                                          apply_normal, *BM->block(i,j),
                                                                          apply_normal, *BU->block(j,k),
                                                                          *BM->block(i,k), acc, approx );
                                                 }// if
                                             }// for
                                         }// for
                                     } );
            }// for
        }// else
    }// if
    else
    {
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

//
// LU factorization
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), acc, approx );

            ::tbb::parallel_invoke(
                [&,i,BA] ()
                {
                    ::tbb::parallel_for< uint >( i+1, BA->nblock_rows(),
                                                 [&,i,BA] ( const uint  j )
                                                 {
                                                     if ( ! is_null( BA->block( j, i ) ) )
                                                         solve_upper_tri< value_t >( from_right, general_diag,
                                                                                     *BA->block( i, i ), *BA->block( j, i ),
                                                                                     acc, approx );
                                                 } );
                },

                [&,i,BA] ()
                {
                    ::tbb::parallel_for< uint >( i+1, BA->nblock_cols(),
                                                 [&,i,BA] ( const uint  j )
                                                 {
                                                     if ( ! is_null( BA->block( i, j ) ) )
                                                         solve_lower_tri< value_t >( from_left, unit_diag,
                                                                                     *BA->block( i, i ), *BA->block( i, j ),
                                                                                     acc, approx );
                                                 } );
                } );

            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( i+1, BA->nblock_rows(),
                                                i+1, BA->nblock_cols() ),
                [&,i,BA] ( const auto  r )
                {
                    for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                    {
                        for ( auto  l = r.cols().begin(); l != r.cols().end(); ++l )
                        {
                            if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                            {
                                HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                                
                                multiply( value_t(-1), apply_normal, *BA->block( j, i ), apply_normal, *BA->block( i, l ),
                                          *BA->block( j, l ), acc, approx );
                            }// if
                        }// for
                    }// for
                } );
        }// for
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        blas::invert( DD );
        
        if ( D->is_compressed() )
            D->set_matrix( std::move( DD ), acc );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

//
// LU factorization for nested dissection type matrices
//
template < typename value_t,
           typename approx_t >
void
lu_nd ( Hpro::TMatrix< value_t > &  A,
        const Hpro::TTruncAcc &     acc,
        const approx_t &            approx )
{
    if ( is_nd( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( BA->nblock_rows() == BA->nblock_cols() );

        const auto  last = BA->nblock_rows()-1;
        
        ::tbb::parallel_for< uint >( 0, BA->nblock_rows()-1,
                                     [=,&acc,&approx] ( const uint  i )
                                     {
                                         auto  A_ii = BA->block( i, i );
                                         
                                         lu_nd< value_t >( *A_ii, acc, approx );

                                         ::tbb::parallel_invoke(
                                             [&,A_ii,BA] () {
                                                 solve_upper_tri< value_t >( from_right, general_diag, *A_ii, *BA->block( last, i ), acc, approx );
                                             },
                                             [&,A_ii,BA] () {
                                                 solve_lower_tri< value_t >( from_left,  unit_diag,    *A_ii, *BA->block( i, last ), acc, approx );
                                             }
                                         );
                                         
                                         multiply( value_t(-1), apply_normal, *BA->block( last, i ), apply_normal, *BA->block( i, last ),
                                                   *BA->block( last, last ), acc, approx );
                                     } );

        lu< value_t >( *BA->block( last, last ), acc, approx );
    }// if
    else if ( is_blocked( A ) )
    {
        hlr::lu< value_t >( A, acc, approx );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        blas::invert( DD );
        
        if ( D->is_compressed() )
            D->set_matrix( std::move( DD ), acc );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

//
// Gaussian elimination of A, e.g. A = A^-1
// - T is used as temporary space and has to have the same
//   structure as A
//
template < typename value_t,
           typename approx_t >
void
gauss_elim ( Hpro::TMatrix< value_t > &  A,
             Hpro::TMatrix< value_t > &  T,
             const Hpro::TTruncAcc &     acc,
             const approx_t &            approx )
{
    HLR_ASSERT( ! is_null_any( &A, &T ) );
    HLR_ASSERT( A.type() == T.type() );
    
    HLR_LOG( 4, Hpro::to_string( "gauss_elim( %d ) {", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BT = ptrcast( &T, Hpro::TBlockMatrix< value_t > );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        // A_00 = A_00⁻¹
        tbb::gauss_elim( *MA(0,0), *MT(0,0), acc, approx );

        ::tbb::parallel_invoke(
            [&]
            { 
                // T_01 = A_00⁻¹ · A_01
                MT(0,1)->scale( 0.0 );
                hlr::multiply( 1.0, apply_normal, MA(0,0), apply_normal, MA(0,1), MT(0,1), acc, approx );
            },

            [&]
            {
                // T_10 = A_10 · A_00⁻¹
                MT(1,0)->scale( 0.0 );
                hlr::multiply( 1.0, apply_normal, MA(1,0), apply_normal, MA(0,0), MT(1,0), acc, approx );
            } );

        // A_11 = A_11 - T_10 · A_01
        hlr::multiply( -1.0, apply_normal, MT(1,0), apply_normal, MA(0,1), MA(1,1), acc, approx );
    
        // A_11 = A_11⁻¹
        gauss_elim( *MA(1,1), *MT(1,1), acc, approx );

        ::tbb::parallel_invoke(
            [&]
            { 
                // A_01 = - T_01 · A_11
                MA(0,1)->scale( 0.0 );
                hlr::multiply( -1.0, apply_normal, MT(0,1), apply_normal, MA(1,1), MA(0,1), acc, approx );
            },
            
            [&]
            { 
                // A_10 = - A_11 · T_10
                MA(1,0)->scale( 0.0 );
                hlr::multiply( -1.0, apply_normal, MA(1,1), apply_normal, MT(1,0), MA(1,0), acc, approx );
            } );

        // A_00 = T_00 - A_01 · T_10
        hlr::multiply( -1.0, apply_normal, MA(0,1), apply_normal, MT(1,0), MA(0,0), acc, approx );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        blas::invert( DD );
        
        if ( D->is_compressed() )
            D->set_matrix( std::move( DD ), acc );
    }// if
    else
        HLR_ASSERT( false );

    HLR_LOG( 4, Hpro::to_string( "} gauss_elim( %d )", A.id() ) );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower/upper triangular matrix
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_lower_tri ( const Hpro::matop_t               op_L,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    hlr::tbb::detail::solve_lower_tri( op_L, L, v, diag_mode );
}

template < typename value_t >
void
solve_upper_tri ( const Hpro::matop_t               op_U,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    hlr::tbb::detail::solve_upper_tri( op_U, U, v, diag_mode );
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
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_ASSERT( is_blocked( A ) );
    
    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );
        auto  D_ii = A_ii->mat_direct();
            
        blas::invert( D_ii );

        ::tbb::parallel_for( i+1, nbc,
                             [A_ii,BA,i] ( uint  j )
                             {
                                 // L is unit diagonal !!!
                                 // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                                 trsmuh< value_t >( *A_ii, *BA->block( j, i ) ); // A10->blas_rmat_B() );
                             } );

        ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                             [&,BA,i] ( const ::tbb::blocked_range2d< uint > & r )
                             {
                                 for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                 {
                                     for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                     {
                                         hlr::tbb::multiply< value_t >( value_t(-1),
                                                                        apply_normal, *BA->block( j, i ),
                                                                        apply_normal, *BA->block( i, l ),
                                                                        *BA->block( j, l ), acc, approx );
                                     }// for
                                 }// for
                             } );
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
// 
template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > &          A,
      const Hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, Hpro::to_string( "ldu( %d )", A.id() ) );
    
    HLR_ASSERT( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        HLR_ASSERT( matrix::is_dense( BA->block( i, i ) ) );
        
        auto  A_ii = ptrcast( BA->block( i, i ), matrix::dense_matrix< value_t > );
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = A_ii->mat_direct();
            
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //

        ::tbb::parallel_for( i+1, nbc,
                             [D_ii,BA,i] ( uint  j )
                             {
                                 auto  L_ji = BA->block( j, i );

                                 if ( matrix::is_lowrank( L_ji ) )
                                 {
                                     // L_ji = W·X' = U·V'·D_ii^-1 = A_ji·D_ii^-1
                                     // ⟶ W = U, X = D_ii^-T·V
                                     auto  R_ji = ptrcast( L_ji, matrix::lrmatrix< value_t > );
                                     auto  V    = blas::copy( blas::mat_V( R_ji ) );
                                     
                                     blas::prod( value_t(1), blas::adjoint( D_ii ), V, value_t(0), blas::mat_V( R_ji ) );
                                 }// if
                                 else if ( matrix::is_dense( L_ji ) )
                                 {
                                     auto  D_ji = ptrcast( L_ji, matrix::dense_matrix< value_t > );
                                     auto  T_ji = blas::copy( D_ji->mat_direct() );
                                     
                                     blas::prod( value_t(1), T_ji, D_ii, value_t(0), D_ji->mat_direct() );
                                 }// else
                             } );

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        ::tbb::parallel_for( i+1, nbr,
                             [D_ii,BA,i] ( uint  j )
                             {
                                 auto  U_ij = BA->block( i, j );
                                 
                                 if ( matrix::is_lowrank( U_ij ) )
                                 {
                                     // U_ij = W·X' = D_ii^-1·U·V' = D_ii^-1·A_ij
                                     // ⟶ W = D_ii^-1·U, X = V
                                     auto  R_ij = ptrcast( U_ij, matrix::lrmatrix< value_t > );
                                     auto  U    = blas::copy( blas::mat_U( R_ij ) );
                                     
                                     blas::prod( value_t(1), D_ii, U, value_t(0), blas::mat_U( R_ij ) );
                                 }// if
                                 else if ( matrix::is_dense( U_ij ) )
                                 {
                                     auto  D_ij = ptrcast( U_ij, matrix::dense_matrix< value_t > );
                                     auto  T_ij = blas::copy( D_ij->mat_direct() );
                                     
                                     blas::prod( value_t(1), D_ii, T_ij, value_t(0), D_ij->mat_direct() );
                                 }// else
                             } );

        //
        // update trailing sub matrix
        //
        
        ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                             [&,BA,i] ( const ::tbb::blocked_range2d< uint > & r )
                             {
                                 for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                 {
                                     for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                     {
                                         hlr::seq::multiply< value_t >( value_t(-1),
                                                                        apply_normal, *BA->block( j, i ),
                                                                        apply_normal, *T_ii,
                                                                        apply_normal, *BA->block( i, l ),
                                                                        *BA->block( j, l ), acc, approx );
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
template < typename value_t,
           typename approx_t >
void
addlr ( blas::matrix< value_t > &  U,
        blas::matrix< value_t > &  V,
        Hpro::TMatrix< value_t > &            A,
        const Hpro::TTruncAcc &    acc,
        const approx_t &           approx )
{
    HLR_LOG( 5, Hpro::to_string( "addlr( %d )", A.id() ) );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  A00 = BA->block( 0, 0 );
        auto  A01 = ptrcast( BA->block( 0, 1 ), matrix::lrmatrix< value_t > );
        auto  A10 = ptrcast( BA->block( 1, 0 ), matrix::lrmatrix< value_t > );
        auto  A11 = BA->block( 1, 1 );

        blas::matrix< value_t >  U0( U, A00->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  U1( U, A11->row_is() - A.row_ofs(), blas::range::all );
        blas::matrix< value_t >  V0( V, A00->col_is() - A.col_ofs(), blas::range::all );
        blas::matrix< value_t >  V1( V, A11->col_is() - A.col_ofs(), blas::range::all );

        ::tbb::parallel_invoke( [&] () { addlr( U0, V0, *A00, acc, approx ); },
                                [&] () { addlr( U1, V1, *A11, acc, approx ); },
                                [&] ()
                                {
                                    auto [ U01, V01 ] = approx( { A01->U_direct(), U0 },
                                                                { A01->V_direct(), V1 },
                                                                acc );
                                    A01->set_lrmat( U01, V01 );
                                },
                                [&] ()
                                {
                                    auto [ U10, V10 ] = approx( { A10->U_direct(), U1 },
                                                                { A10->V_direct(), V0 },
                                                                acc );
                                    A10->set_lrmat( U10, V10 );
                                } );
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), DA->mat_direct() );
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

        ::tbb::parallel_invoke( [A00,A01] () { seq::hodlr::trsml(  *A00, A01->U_direct() ); },
                                [A00,A10] () { seq::hodlr::trsmuh( *A00, A10->V_direct() ); } );

        // TV = U(A_10) · ( V(A_10)^H · U(A_01) )
        auto  T  = blas::prod(  value_t(1), blas::adjoint( A10->V_direct() ), A01->U_direct() ); 
        auto  UT = blas::prod( value_t(-1), A10->U_direct(), T );

        addlr< value_t >( UT, A01->V_direct(), *A11, acc, approx );
        
        lu< value_t >( *A11, acc, approx );
    }// if
    else
    {
        auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
        
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
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, Hpro::to_string( "lu( %d )", A.id() ) );

    HLR_ASSERT( is_blocked( A ) );

    auto  BA  = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        {
            auto  dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *(BA->block( i, i )),
                                                                  128,
                                                                  tbb::dag::refine ) );

            hlr::tbb::dag::run( dag, acc );

            // Hpro::LU::factorise_rec( BA->block( i, i ), acc );
        }

        ::tbb::parallel_invoke(
            [BA,i,nbr,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbr,
                                     [BA,i,&acc] ( uint  j )
                                     {
                                         auto  dag = std::move( hlr::dag::gen_dag_solve_upper( *BA->block( i, i ),
                                                                                               *BA->block( j, i ),
                                                                                               128,
                                                                                               tbb::dag::refine ) );
                                                     
                                         hlr::tbb::dag::run( dag, acc );
                                         // Hpro::solve_upper_right( BA->block( j, i ),
                                         //                          BA->block( i, i ), nullptr, acc,
                                         //                          Hpro::solve_option_t( Hpro::block_wise, Hpro::general_diag, Hpro::store_inverse ) ); 
                                     } );
            },
                
            [BA,i,nbc,&acc] ()
            {
                ::tbb::parallel_for( i+1, nbc,
                                     [BA,i,&acc] ( uint  l )
                                     {
                                         auto  dag = std::move( hlr::dag::gen_dag_solve_lower( *BA->block( i, i ),
                                                                                               *BA->block( i, l ),
                                                                                               128,
                                                                                               tbb::dag::refine ) );
                                                     
                                         hlr::tbb::dag::run( dag, acc );
                                         // Hpro::solve_lower_left( apply_normal, BA->block( i, i ), nullptr,
                                         //                         BA->block( i, l ), acc,
                                         //                         Hpro::solve_option_t( Hpro::block_wise, Hpro::unit_diag, Hpro::store_inverse ) );
                                     } );
            } );

        ::tbb::parallel_for( ::tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                             [&,BA,i] ( const ::tbb::blocked_range2d< uint > & r )
                             {
                                 for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                 {
                                     for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                     {
                                         hlr::tbb::multiply( value_t(-1),
                                                             apply_normal, * BA->block( j, i ),
                                                             apply_normal, * BA->block( i, l ),
                                                             * BA->block( j, l ), acc, approx );
                                     }// for
                                 }// for
                             } );
    }// for
}

}// namespace tileh

//
// collection of arithmetic functions
//
struct tbb_arithmetic
{
    //
    // matrix vector multiplication
    //
    
    template < typename value_t >
    void
    mul_vec ( const value_t                             alpha,
              const Hpro::matop_t                       op_M,
              const Hpro::TMatrix< value_t > &          M,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y ) const
    {
        hlr::tbb::mul_vec( alpha, op_M, M, x, y );
    }

    template < typename value_t >
    void
    mul_vec ( const value_t                     alpha,
              const Hpro::matop_t               op_M,
              const Hpro::TMatrix< value_t > &  M,
              const blas::vector< value_t > &   x,
              blas::vector< value_t > &         y ) const
    {
        hlr::tbb::mul_vec( alpha, op_M, M, x, y );
    }

    // template < typename linop_t >
    // void
    // prod ( const typename linop_t::value_t                    alpha,
    //        const matop_t                                      op_M,
    //        const linop_t &                                    M,
    //        const blas::vector< typename linop_t::value_t > &  x,
    //        blas::vector< typename linop_t::value_t > &        y ) const
    // {
    //     if ( is_matrix( M ) )
    //         hlr::tbb::mul_vec( alpha, op_M, *cptrcast( &M, Hpro::TMatrix< value_t > ), x, y );
    //     else if constexpr ( supports_arithmetic< linop_t > )
    //         cptrcast( &M, arithmetic_support< linop_t > )->apply_add( *this, alpha, x, y, op_M );
    //     else
    //         M.apply_add( alpha, x, y, op_M );
    // }

    template < typename value_t >
    void
    prod ( const value_t                             alpha,
           const matop_t                             op_M,
           const Hpro::TLinearOperator< value_t > &  M,
           const blas::vector< value_t > &           x,
           blas::vector< value_t > &                 y ) const
    {
        if ( is_matrix( M ) )
            hlr::tbb::mul_vec( alpha, op_M, *cptrcast( &M, Hpro::TMatrix< value_t > ), x, y );
        else if ( dynamic_cast< const hlr::matrix::linop_sum< value_t > * >( &M ) != nullptr )
            cptrcast( &M, matrix::linop_sum< value_t > )->apply_add( *this, alpha, x, y, op_M );
        else if ( dynamic_cast< const hlr::matrix::linop_product< value_t > * >( &M ) != nullptr )
            cptrcast( &M, matrix::linop_product< value_t > )->apply_add( *this, alpha, x, y, op_M );
        else if ( dynamic_cast< const hlr::matrix::luinv_eval< value_t > * >( &M ) != nullptr )
            cptrcast( &M, matrix::luinv_eval< value_t > )->apply_add( *this, alpha, x, y, op_M );
        else
            M.apply_add( alpha, x, y, op_M );
    }

    //
    // vector solves
    //

    template < typename value_t >
    void
    solve_lower_tri ( const matop_t                       op_L,
                      const Hpro::TMatrix< value_t > &    L,
                      vector::scalar_vector< value_t > &  v,
                      const Hpro::diag_type_t             diag_mode ) const
    {
        hlr::tbb::solve_lower_tri( op_L, L, v, diag_mode );
    }

    template < typename value_t >
    void
    solve_upper_tri ( const matop_t                       op_U,
                      const Hpro::TMatrix< value_t > &    U,
                      vector::scalar_vector< value_t > &  v,
                      const Hpro::diag_type_t             diag_mode ) const
    {
        hlr::tbb::solve_upper_tri( op_U, U, v, diag_mode );
    }
};

constexpr tbb_arithmetic arithmetic{};

}// namespace tbb

template <> struct is_arithmetic<       tbb::tbb_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic< const tbb::tbb_arithmetic   > { static constexpr bool value = true; };
template <> struct is_arithmetic<       tbb::tbb_arithmetic & > { static constexpr bool value = true; };
template <> struct is_arithmetic< const tbb::tbb_arithmetic & > { static constexpr bool value = true; };

}// namespace hlr

#endif // __HLR_TBB_ARITH_HH
