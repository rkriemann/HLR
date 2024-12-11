#ifndef __HLR_ARITH_DETAIL_MULTIPLY_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#if defined(NDEBUG)
#  define HLR_MULT_PRINT   
#else
#  define HLR_MULT_PRINT   HLR_LOG( 4, Hpro::to_string( "multiply( %s %d, %s %d, %s %d )", \
                                                        A.typestr().c_str(), A.id(), \
                                                        B.typestr().c_str(), B.id(), \
                                                        C.typestr().c_str(), C.id() ) )
#endif

#include <hlr/arith/detail/multiply_blas.hh>
#include <hlr/arith/detail/multiply_uniform.hh>

namespace hlr {

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·B + C
//
/////////////////////////////////////////////////////////////////////////////////

//
// forward decl.(s)
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
           const accuracy &                  acc,
           const approx_t &                  approx );

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C );

//
// blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                    multiply< value_t >( alpha,
                                         op_A, *A.block( i, l, op_A ),
                                         op_B, *B.block( l, j, op_B ),
                                         *C.block( i, j ), acc, approx );
            }// if       
        }// for
    }// for
}

//
// blocked x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::lrmatrix< value_t > &          C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;
    
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( C.row_is(), C.col_is() );

    BC->set_block_struct( A.nblock_rows( op_A ), B.nblock_cols( op_B ) );
    
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new matrix::lrmatrix< value_t >( A_il->row_is( op_A ), B_lj->col_is( op_B ) ) );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // apply update
    hlr::add( value_t(1), *BC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;
    
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( C.row_is(), C.col_is() );

    BC->set_block_struct( A.nblock_rows( op_A ), B.nblock_cols( op_B ) );
    
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new matrix::lrsvmatrix< value_t >( A_il->row_is( op_A ), B_lj->col_is( op_B ) ) );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // apply update
    hlr::add( value_t(1), *BC, C, acc, approx );
}

//
// blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const Hpro::TBlockMatrix< value_t > &    A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const Hpro::TBlockMatrix< value_t > &    A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;

    auto  DB = B.mat();
    auto  DT = blas::matrix< value_t >( C.nrows(), C.ncols() );
    auto  T  = matrix::dense_matrix< value_t >( C.row_is(), C.col_is(), DT );
            
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  T_i = matrix::dense_matrix< value_t >( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  is_j = A_ij->col_is( op_A );

            if ( op_B == Hpro::apply_normal )
            {
                auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
                auto  B_j  = matrix::dense_matrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
                multiply( value_t(1), op_A, * A_ij, op_B, B_j, T_i );
            }// if
            else
            {
                auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
                auto  B_j  = matrix::dense_matrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
                multiply( value_t(1), op_A, * A_ij, op_B, B_j, T_i );
            }// else
        }// for

        auto  DT_i = blas::matrix< value_t >( DT, T_i.row_is() - T.row_ofs(), T_i.col_is() - T.col_ofs() );

        blas::add( value_t(1), T_i.mat(), DT_i );
    }// for

    hlr::add( alpha, T, C, acc );
}

//
// blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const Hpro::TBlockMatrix< value_t > &    A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::lrmatrix< value_t > &            C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrmatrix< value_t > &    B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), UC, VB );
    
    hlr::add( value_t(1), RC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    // ((A × U(B))·S(B))·V(B)
    auto  UB  = B.U( op_B );
    auto  VB  = B.V( op_B );
    auto  AxU = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, AxU );
    blas::prod_diag_ip( AxU, B.S() );

    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), AxU, VB );
    
    hlr::add( value_t(1), RC, C, acc, approx );
}

//
// blocked x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrmatrix< value_t > &    B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_MULT_PRINT;

    // (A × U)·V' = W·V'
    auto  UB = B.U( op_B );
    auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, W );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();

    // W·V' + C
    blas::prod( value_t(1), W, blas::adjoint( B.V( op_B ) ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrmatrix< value_t > &    B,
           matrix::dense_matrix< value_t > &      C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );

    // (A × U)·V' = W·V'
    auto  UB = B.U( op_B );
    auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, W );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    // W·V' + C
    blas::prod( value_t(1), W, blas::adjoint( B.V( op_B ) ), value_t(1), DC );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_MULT_PRINT;

    // (A × U)·S·V' = W·S·V'
    auto  UB  = B.U( op_B );
    auto  AxU = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, AxU );
    blas::prod_diag_ip( AxU, B.S() );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    // AxU·V' + C
    blas::prod( value_t(1), AxU, blas::adjoint( B.V( op_B ) ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

//
// blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrmatrix< value_t > &    B,
           matrix::lrmatrix< value_t > &          C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  lock    = std::scoped_lock( C.mutex() );
    auto [ U, V ] = approx( { UC, C.U() },
                            { VB, C.V() },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    // ((A × U(B))·S(B))·V(B)'
    auto  UB  = B.U( op_B );
    auto  VB  = B.V( op_B );
    auto  AxU = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, AxU );
    blas::prod_diag_ip( AxU, B.S() );

    auto  lock       = std::scoped_lock( C.mutex() );
    auto  UC         = blas::prod_diag( C.U(), C.S() );
    auto [ U, S, V ] = approx.approx_ortho( { AxU, UC },
                                            { VB,  C.V() },
                                            acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const Hpro::TBlockMatrix< value_t > &    B,
           Hpro::TBlockMatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const Hpro::TBlockMatrix< value_t > &    B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;

    auto  DA = A.mat();
    auto  DT = blas::matrix< value_t >( C.nrows(), C.ncols() );
    auto  T  = matrix::dense_matrix< value_t >( C.row_is(), C.col_is(), DT );
    
    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  T_j = matrix::dense_matrix< value_t >( C.row_is(), B.block( 0, j, op_B )->col_is( op_B ) );
        
        for ( uint  i = 0; i < B.nblock_rows( op_B ); ++i )
        {
            auto  B_ij = B.block( i, j, op_B );
            auto  is_i = B_ij->row_is( op_B );

            if ( op_A == Hpro::apply_normal )
            {
                auto  DA_i = blas::matrix< value_t >( DA, blas::range::all, is_i - A.col_ofs() );
                auto  A_i  = matrix::dense_matrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
                multiply( value_t(1), op_A, A_i, op_B, *B_ij, T_j );
            }// if
            else
            {
                auto  DA_i = blas::matrix< value_t >( DA, is_i - A.row_ofs(), blas::range::all );
                auto  A_i  = matrix::dense_matrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
                multiply( value_t(1), op_A, A_i, op_B, *B_ij, T_j );
            }// else
        }// for

        auto  DT_j = blas::matrix< value_t >( DT, T_j.row_is() - T.row_ofs(), T_j.col_is() - T.col_ofs() );

        blas::add( value_t(1), T_j.mat(), DT_j );
        // T.add_block( value_t(1), value_t(1), &T_j );
    }// for

    hlr::add( alpha, T, C, acc );
}

//
// dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const Hpro::TBlockMatrix< value_t > &    B,
           matrix::lrmatrix< value_t > &            C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;

    auto  DA = A.mat();
    auto  DB = B.mat();
    auto  DT = blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ) );
    auto  T  = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is(), std::move( DT ) );
        
    hlr::add< value_t >( value_t(1), T, C, acc, approx );
}

//
// dense x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;

    auto  DA = A.mat();
    auto  DB = B.mat();
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    // C = C + A B
    blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );

    auto  DA   = A.mat();
    auto  DB   = B.mat();

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();

    // C = C + A B
    blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ), value_t(1), DC );
}

//
// dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::lrmatrix< value_t > &            C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AxB = blas::prod( alpha,
                            blas::mat_view( op_A, A.mat() ),
                            blas::mat_view( op_B, B.mat() ) );
    
    auto  lock = std::scoped_lock( C.mutex() );

    blas::prod( value_t(1), C.U(), blas::adjoint( C.V() ), value_t(1), AxB );
    
    auto [ U, V ] = approx( AxB, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AxB = blas::prod( alpha,
                            blas::mat_view( op_A, A.mat() ),
                            blas::mat_view( op_B, B.mat() ) );
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  UC   = blas::prod_diag( C.U(), C.S() );

    blas::prod( value_t(1), UC, blas::adjoint( C.V() ), value_t(1), AxB );
    
    auto [ U, S, V ] = approx.approx_ortho( AxB, acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrmatrix< value_t > &      B,
           Hpro::TBlockMatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    // C = C + ( A U(B) ) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );
    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), AU, VB );

    hlr::add( alpha, RC, C, acc, approx );
}

//
// dense x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrmatrix< value_t > &      B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrmatrix< value_t > &      B,
           matrix::dense_matrix< value_t > &        C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );
    
    // C = C + ( A U(B) ) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), DC );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrsvmatrix< value_t > &    B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) S(B) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  SB = B.S();
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    blas::prod_diag_ip( AU, SB );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

//
// dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrmatrix< value_t > &      B,
           matrix::lrmatrix< value_t > &            C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( alpha, blas::mat_view( op_A, DA ), UB );

    auto  lock    = std::scoped_lock( C.mutex() );
    auto [ U, V ] = approx( { C.U(), AU },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrsvmatrix< value_t > &    B,
           matrix::lrsvmatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( alpha, blas::mat_view( op_A, DA ), UB );

    blas::prod_diag_ip( AU, B.S() );
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  UC   = blas::prod_diag( C.U(), C.S() );

    auto [ U, S, V ] = approx.approx_ortho( { UC,    AU },
                                            { C.V(), VB },
                                            acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// lowrank x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrmatrix< value_t > &    A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), UA, VC );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    // UA·SA·VA' × B = C computed as
    // - VC = B × VA
    // - C  = C + (UA·SA) × VC
    //
    // TODO: test if lrsv format is better also for add(...)
    //
    auto  UA = A.U( op_A );
    auto  US = blas::prod_diag( UA, A.S() );
    auto  VA = A.V( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), US, VC );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// lowrank x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrmatrix< value_t > &    A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_MULT_PRINT;

    // U·(V' × B) = U·X' with X = B'·V
    auto  VA = A.V( op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();

    // U·X' + C
    blas::prod( value_t(1), A.U( op_A ), blas::adjoint( X ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrmatrix< value_t > &    A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );

    // U·(V' × B) = U·X' with X = B'·V
    auto  VA = A.V( op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    // U·X' + C
    blas::prod( value_t(1), A.U( op_A ), blas::adjoint( X ), value_t(1), DC );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_MULT_PRINT;

    // U·S·(V' × B) = U·S·X' with X = B'·V
    auto  VA = A.V( op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    // S·X' = (X·S')' = (X·S)' sind S is diagonal and real
    blas::prod_diag_ip( X, A.S() );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    // U·X' + C
    blas::prod( value_t(1), A.U( op_A ), blas::adjoint( X ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

//
// lowrank x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrmatrix< value_t > &    A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::lrmatrix< value_t > &          C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  lock    = std::scoped_lock( C.mutex() );
    auto [ U, V ] = approx( { C.U(), UA },
                            { C.V(), VC },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    // U(A)·S(A)·V(A)' × B
    // computed as (B'×V(A))·S(A)
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );
    blas::prod_diag_ip( VC, A.S() );

    auto  lock       = std::scoped_lock( C.mutex() );
    auto  UC         = blas::prod_diag( C.U(), C.S() );
    auto [ U, S, V ] = approx.approx_ortho( { UC,    UA },
                                            { C.V(), VC },
                                            acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// lowrank x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrmatrix< value_t > &      A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    // C = C + U(A) ( B^H V(A) )^H
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( blas::adjoint( blas::mat_view( op_B, DB ) ), VA );
    auto  RC = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), UA, VB );

    hlr::add( alpha, RC, C, acc, approx );
}

//
// lowrank x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrmatrix< value_t > &      A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UA, VB, value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrmatrix< value_t > &      A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );
    
    // C = C + U(A) ( V(A)^H B )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UA, VB, value_t(1), DC );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrsvmatrix< value_t > &    A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::dense_matrix< value_t > &        C,
           const accuracy &                         acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) S(A) ( V(A)^H B )
    auto  UA = A.U( op_A );
    auto  SA = A.S();
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    blas::prod_diag_ip( SA, VB );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UA, VB, value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

//
// lowrank x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrmatrix< value_t > &      A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::lrmatrix< value_t > &            C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VB = blas::prod( alpha, blas::adjoint( blas::mat_view( op_B, B.mat() ) ), VA );

    auto  lock    = std::scoped_lock( C.mutex() );
    auto [ U, V ] = approx( { C.U(), UA },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::lrsvmatrix< value_t > &    A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &          C,
           const accuracy &                         acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A)·S(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  UA = A.U( op_A );
    auto  SA = A.S();
    auto  US = blas::prod_diag( UA, SA );
    auto  VA = A.V( op_A );
    auto  VB = blas::prod( alpha, blas::adjoint( blas::mat_view( op_B, B.mat() ) ), VA );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  UC   = blas::prod_diag( C.U(), C.S() );

    auto [ U, S, V ] = approx.approx_ortho( { UC,    US },
                                            { C.V(), VB },
                                            acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// lowrank x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const matrix::lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const matrix::lrmatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &      C,
           const accuracy &                     acc,
           const approx_t &                     approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod(      alpha, UA, T );
    auto  R  = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), UT, VB );
        
    hlr::add< value_t >( value_t(1), R, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) S(A) V(A)^H U(B) S(B) ] , [ V(C), V(B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );

    blas::prod_diag_ip( A.S(), T );
    blas::prod_diag_ip( T, B.S() );
    
    auto  UT = blas::prod( alpha, UA, T );
    auto  R  = matrix::lrmatrix< value_t >( C.row_is(), C.col_is(), UT, VB );
        
    hlr::add< value_t >( value_t(1), R, C, acc, approx );
}

//
// lowrank x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const matrix::lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const matrix::lrmatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &    C,
           const accuracy &                     acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod( value_t(1), UA, T );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const matrix::lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const matrix::lrmatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &    C )
{
    HLR_MULT_PRINT;
    HLR_ASSERT( ! C.is_compressed() );
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod( value_t(1), UA, T );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), DC );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           matrix::dense_matrix< value_t > &      C,
           const accuracy &                       acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) S ( V(A)^H U(B) ) T V(B)^H
    auto  UA = A.U( op_A );
    auto  SA = A.S();
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  SB = B.S();
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( blas::adjoint( VA ), UB );

    blas::prod_diag_ip( SA, T );
    blas::prod_diag_ip( T, SB );
    
    auto  UT = blas::prod( UA, T );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  DC   = C.mat();
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), DC );

    if ( C.is_compressed() )
        C.set_matrix( std::move( DC ), acc );
}

//
// lowrank x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const matrix::lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const matrix::lrmatrix< value_t > &  B,
           matrix::lrmatrix< value_t > &        C,
           const accuracy &                     acc,
           const approx_t &                     approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod(      alpha, UA, T );

    auto  lock    = std::scoped_lock( C.mutex() );
    auto [ U, V ] = approx( { C.U(), UT },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const matrix::lrsvmatrix< value_t > &  A,
           const Hpro::matop_t                    op_B,
           const matrix::lrsvmatrix< value_t > &  B,
           matrix::lrsvmatrix< value_t > &        C,
           const accuracy &                       acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );

    blas::prod_diag_ip( A.S(), T );
    blas::prod_diag_ip( T, B.S() );
    
    auto  UT = blas::prod( alpha, UA, T );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  UC   = blas::prod_diag( C.U(), C.S() );

    auto [ U, S, V ] = approx.approx_ortho( { UC,    UT },
                                            { C.V(), VB },
                                            acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

//
// blocked x dense = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   /* alpha */,
//            const Hpro::matop_t                             /* op_A */,
//            const Hpro::TBlockMatrix< value_t > &           /* A */,
//            const Hpro::matop_t                             /* op_B */,
//            const Hpro::TDenseMatrix< value_t > &           /* B */,
//            Hpro::TBlockMatrix< value_t > &                 /* C */,
//            const accuracy &                                /* acc */,
//            const approx_t &                                /* approx */ )
// {
//     HLR_ERROR( "todo" );
// }

//
// blocked x dense = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TBlockMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;

//     for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
//     {
//         HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
//         auto  C_i = Hpro::TDenseMatrix< value_t >( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
//         for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
//         {
//             auto  A_ij = A.block( i, j, op_A );
//             auto  DB   = blas::mat( B );
//             auto  is_j = A_ij->col_is( op_A );

//             if ( op_B == Hpro::apply_normal )
//             {
//                 auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
//                 auto  B_j  = Hpro::TDenseMatrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
//                 multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
//             }// if
//             else
//             {
//                 auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
//                 auto  B_j  = Hpro::TDenseMatrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
//                 multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
//             }// else
//         }// for

//         C.add_block( value_t(1), value_t(1), &C_i );
//     }// for
// }

//
// blocked x dense = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TBlockMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     // D = A×B
//     auto  D = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is() );

//     if ( op_B == apply_normal )
//     {
//         multiply( alpha, op_A, A, blas::mat( B ), blas::mat( D ) );

//         auto  lock = std::scoped_lock( C.mutex() );
        
//         // D = D + C
//         hlr::add< value_t >( value_t(1), C, D );

//         // approximate result and update C
//         auto [ U, V ] = approx( blas::mat( D ), acc );

//         C.set_lrmat( std::move( U ), std::move( V ) );
//     }// if
//     else
//     {
//         HLR_ERROR( "todo" );
//     }// else
// }

//
// blocked x lowrank = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TBlockMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  UB = blas::mat_U( B, op_B );
//     auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

//     multiply< value_t >( alpha, op_A, A, UB, UC );

//     auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), UC, blas::mat_V( B, op_B ) );
    
//     hlr::add< value_t >( value_t(1), RC, C, acc, approx );
// }

//
// blocked x lowrank = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TBlockMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;

//     // (A × U)·V' = W·V'
//     auto  UB = blas::mat_U( B, op_B );
//     auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

//     multiply< value_t >( alpha, op_A, A, UB, W );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     // W·V' + C
//     blas::prod( value_t(1), W, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
// }

//
// blocked x lowrank = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TBlockMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  UB = blas::mat_U( B, op_B );
//     auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

//     multiply< value_t >( alpha, op_A, A, UB, UC );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     auto [ U, V ] = approx( { blas::mat_U( C ), UC },
//                             { blas::mat_V( C ), blas::mat_V( B, op_B ) },
//                             acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

//
// dense x blocked = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   /* alpha */,
//            const Hpro::matop_t                             /* op_A */,
//            const Hpro::TDenseMatrix< value_t > &           /* A */,
//            const Hpro::matop_t                             /* op_B */,
//            const Hpro::TBlockMatrix< value_t > &           /* B */,
//            Hpro::TBlockMatrix< value_t > &                 /* C */,
//            const accuracy &                                /* acc */,
//            const approx_t &                                /* approx */ )
// {
//     HLR_ERROR( "todo" );
// }

//
// dense x blocked = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TBlockMatrix< value_t > &           B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;

//     for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
//     {
//         HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
//         auto  C_j = Hpro::TDenseMatrix< value_t >( C.row_is(), B.block( 0, j, op_B )->col_is( op_B ) );
        
//         for ( uint  i = 0; i < B.nblock_rows( op_B ); ++i )
//         {
//             auto  DA   = blas::mat( A );
//             auto  B_ij = B.block( i, j, op_B );
//             auto  is_i = B_ij->row_is( op_B );

//             if ( op_A == Hpro::apply_normal )
//             {
//                 auto  DA_i = blas::matrix< value_t >( DA, blas::range::all, is_i - A.col_ofs() );
//                 auto  A_i  = Hpro::TDenseMatrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
//                 multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
//             }// if
//             else
//             {
//                 auto  DA_i = blas::matrix< value_t >( DA, is_i - A.row_ofs(), blas::range::all );
//                 auto  A_i  = Hpro::TDenseMatrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
//                 multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
//             }// else
//         }// for

//         C.add_block( value_t(1), value_t(1), &C_j );
//     }// for
// }

//
// dense x blocked = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TBlockMatrix< value_t > &           B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  D = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is() );

//     if ( op_A == apply_normal )
//     {
//         // D = A×B
//         multiply( alpha, blas::mat( A ), op_B, B, blas::mat( D ) );

//         auto  lock = std::scoped_lock( C.mutex() );
    
//         // D = D + C
//         hlr::add< value_t >( value_t(1), C, D );

//         // approximate result and update C
//         auto [ U, V ] = approx( blas::mat( D ), acc );

//         C.set_lrmat( std::move( U ), std::move( V ) );
//     }// if
//     else
//     {
//         HLR_ERROR( "todo" );
//     }// else
// }

//
// dense x dense = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  DA = blas::mat( A );
//     auto  DB = blas::mat( B );
//     auto  DT = blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ) );
//     auto  T  = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is(), std::move( DT ) );
        
//     hlr::add< value_t >( value_t(1), T, C, acc, approx );
// }

//
// dense x dense = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;
    
//     auto  lock = std::scoped_lock( C.mutex() );
    
//     // C = C + A B
//     blas::prod( alpha,
//                 blas::mat_view( op_A, blas::mat( A ) ),
//                 blas::mat_view( op_B, blas::mat( B ) ),
//                 value_t(1), blas::mat( C ) );
// }

//
// dense x dense = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;
    
//     // [ U(C), V(C) ] = approx( C - A B )
//     auto  AB = blas::prod( alpha,
//                            blas::mat_view( op_A, blas::mat( A ) ),
//                            blas::mat_view( op_B, blas::mat( B ) ) );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     blas::prod( value_t(1), blas::mat_U( C ), blas::adjoint( blas::mat_V( C ) ), value_t(1), AB );

//     auto [ U, V ] = approx( AB, acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

//
// dense x lowrank = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     // C = C + ( A U(B) ) V(B)^H
//     auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, blas::mat( A ) ), blas::mat_U( B, op_B ) );
//     auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), AU, blas::mat_V( B, op_B ) );

//     hlr::add( alpha, RC, C, acc, approx );
// }

//
// dense x lowrank = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;
    
//     // C = C + ( A U(B) ) V(B)^H
//     auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, blas::mat( A ) ), blas::mat_U( B, op_B ) );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     blas::prod( alpha, AU, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
// }

//
// dense x lowrank = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TDenseMatrix< value_t > &           A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;
    
//     // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
//     auto  AU = blas::prod( alpha,
//                            blas::mat_view( op_A, blas::mat( A ) ),
//                            blas::mat_U( B, op_B ) );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     auto [ U, V ] = approx( { blas::mat_U( C ), AU },
//                             { blas::mat_V( C ), blas::mat_V( B, op_B ) },
//                             acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

//
// lowrank x blocked = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TBlockMatrix< value_t > &           B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  VA = blas::mat_V( A, op_A );
//     auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

//     multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

//     auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), blas::mat_U( A, op_A ), VC );
    
//     hlr::add< value_t >( value_t(1), RC, C, acc, approx );
// }

//
// lowrank x blocked = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TBlockMatrix< value_t > &           B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;

//     // U·(V' × B) = U·X' with X = B'·V
//     auto  VA = blas::mat_V( A, op_A );
//     auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

//     multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

//     auto  lock = std::scoped_lock( C.mutex() );

//     // U·X' + C
//     blas::prod( value_t(1), blas::mat_U( A, op_A ), blas::adjoint( X ), value_t(1), blas::mat( C ) );
// }

//
// lowrank x blocked = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TBlockMatrix< value_t > &           B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;

//     auto  VA = blas::mat_V( A, op_A );
//     auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

//     multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     auto [ U, V ] = approx( { blas::mat_U( C ), blas::mat_U( A, op_A ) },
//                             { blas::mat_V( C ), VC },
//                             acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

//
// lowrank x dense = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
//     auto  VB = blas::prod( blas::adjoint( blas::mat_view( op_B, blas::mat( B ) ) ),
//                            blas::mat_V( A, op_A ) );

//     auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), blas::mat_U( A, op_A ), VB );

//     hlr::add( alpha, RC, C, acc, approx );
// }

//
// lowrank x dense = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;
    
//     // C = C + U(A) ( V(A)^H B )
//     auto  VB = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_view( op_B, blas::mat( B ) ) );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     blas::prod( alpha, blas::mat_U( A, op_A ), VB, value_t(1), blas::mat( C ) );
// }

//
// lowrank x dense = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TDenseMatrix< value_t > &           B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;
    
//     // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
//     auto  VB = blas::prod( alpha,
//                            blas::adjoint( blas::mat_view( op_B, blas::mat( B ) ) ),
//                            blas::mat_V( A, op_A ) );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     auto [ U, V ] = approx( { blas::mat_U( C ), blas::mat_U( A, op_A ) },
//                             { blas::mat_V( C ), VB },
//                             acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

//
// lowrank x lowrank = blocked
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TBlockMatrix< value_t > &                 C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;
    
//     // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
//     auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
//     auto  UT = blas::prod(      alpha, blas::mat_U( A, op_A ), T );
//     auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( C.row_is(), C.col_is(), UT, blas::mat_V( B, op_B ) );
        
//     hlr::add< value_t >( value_t(1), *R, C, acc, approx );
// }

//
// lowrank x lowrank = dense
//
// template < typename value_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TDenseMatrix< value_t > &                 C )
// {
//     HLR_MULT_PRINT;
    
//     // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
//     auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
//     auto  UT = blas::prod( value_t(1), blas::mat_U( A, op_A ), T );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     blas::prod( alpha, UT, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
// }

//
// lowrank x lowrank = lowrank
//
// template < typename value_t,
//            typename approx_t >
// void
// multiply ( const value_t                                   alpha,
//            const Hpro::matop_t                             op_A,
//            const Hpro::TRkMatrix< value_t > &              A,
//            const Hpro::matop_t                             op_B,
//            const Hpro::TRkMatrix< value_t > &              B,
//            Hpro::TRkMatrix< value_t > &                    C,
//            const accuracy &                                acc,
//            const approx_t &                                approx )
// {
//     HLR_MULT_PRINT;
    
//     // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
//     auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
//     auto  UT = blas::prod(      alpha, blas::mat_U( A, op_A ), T );

//     auto  lock = std::scoped_lock( C.mutex() );
    
//     auto [ U, V ] = approx( { blas::mat_U( C ), UT },
//                             { blas::mat_V( C ), blas::mat_V( B, op_B ) },
//                             acc );
        
//     C.set_lrmat( std::move( U ), std::move( V ) );
// }

}// namespace hlr

/*
#!/usr/bin/env python3
#
# generates matrix multiplication function placeholders
#

# type abbreviations
types   = [ 'blocked', 'dense', 'lowrank', 'uniform' ]

# actual matrix types
classes = { 'blocked'   : 'Hpro::TBlockMatrix< value_t >',
            'lowrank'   : 'matrix::lrmatrix< value_t >',
            'dense'     : 'matrix::dense_matrix< value_t >',
            'uniform'   : 'matrix::uniform_lrmatrix< value_t >' }

# destination type needs approximation
approx  = { 'blocked' : True,
            'lowrank' : True,
            'dense'   : False,
            'uniform' : True }

# destination type is allowed (handled here)
as_dest = { 'blocked' : True,
            'lowrank' : True,
            'dense'   : True,
            'uniform' : False }

for A in types :
    for B in types :
        for C in types :
            if not as_dest[C] :
                continue
            
            print( '//' )
            print( '// %s x %s = %s' % ( A, B, C ) )
            print( '//' )
            if approx[C] :
                print( 'template < typename value_t,' )
                print( '           typename approx_t >' )
            else :
                print( 'template < typename value_t >' )
            print( 'void' )
            print( 'multiply ( const value_t                                   alpha,' )
            print( '           const Hpro::matop_t                             op_A,' )
            print( '           const {:<40}  A,'.format( classes[A] + ' &' ) )
            print( '           const Hpro::matop_t                             op_B,' )
            print( '           const {:<40}  B,'.format( classes[B] + ' &' ) )

            if approx[C] :
                print( '           {:<46}  C,'.format( classes[C] + ' &' ) )
                print( '           const accuracy &                                acc,' )
                print( '           const approx_t &                                approx )' )
            else :
                print( '           {:<46}  C )'.format( classes[C] + ' &' ) )

            print( '{' )
            print( '    HLR_ERROR( "todo" );' )
            print( '}' )
            print()

*/

#endif // __HLR_ARITH_DETAIL_MULTIPLY_HH
