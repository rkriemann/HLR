#ifndef __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions with compressed matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

namespace hlr
{

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
           const Hpro::TTruncAcc &           acc,
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
           const Hpro::TTruncAcc &                acc )
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
           const Hpro::TTruncAcc &                acc,
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc )
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
        // T.add_block( value_t(1), value_t(1), &T_i );
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                acc,
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
           const Hpro::TTruncAcc &                acc )
{
    HLR_MULT_PRINT;

    // (A × U)·V' = W·V'
    auto  UB = B.U( op_B );
    auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, W );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();
    
    C.decompress();
    
    // W·V' + C
    auto  DC = C.mat();

    blas::prod( value_t(1), W, blas::adjoint( B.V( op_B ) ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    std::scoped_lock  lock( C.mutex() );
    
    // W·V' + C
    auto  DC = C.mat();

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
           const Hpro::TTruncAcc &                acc )
{
    HLR_MULT_PRINT;

    // (A × U)·S·V' = W·S·V'
    auto  UB  = B.U( op_B );
    auto  AxU = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, AxU );
    blas::prod_diag( AxU, B.S() );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();
    
    C.decompress();
    
    // AxU·V' + C
    auto  DC = C.mat();
    
    blas::prod( value_t(1), AxU, blas::adjoint( B.V( op_B ) ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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
           const Hpro::TTruncAcc &                acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    std::scoped_lock  lock( C.mutex() );

    auto [ U, V ] = approx( { UC, C.U() },
                            { VB, C.V() },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc )
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc )
{
    HLR_MULT_PRINT;

    auto  DA = A.mat();
    auto  DB = B.mat();
    
    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    // C = C + A B
    auto  DC = C.mat();
    
    blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    auto  DA = A.mat();
    auto  DB = B.mat();
    
    std::scoped_lock  lock( C.mutex() );

    // C = C + A B
    auto  DC = C.mat();
    
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
           const Hpro::TTruncAcc &                  acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AxB = blas::prod( alpha,
                            blas::mat_view( op_A, A.mat() ),
                            blas::mat_view( op_B, B.mat() ) );
    
    std::scoped_lock  lock( C.mutex() );

    blas::prod( value_t(1), C.U(), blas::adjoint( C.V() ), value_t(1), AxB );
    
    auto [ U, V ] = approx( AxB, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    auto  DC = C.mat();
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    std::scoped_lock  lock( C.mutex() );
    
    auto  DC = C.mat();
    
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
           const Hpro::TTruncAcc &                  acc )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) S(B) V(B)^H
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  SB = B.S();
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, DA ), UB );

    blas::prod_diag( AU, SB );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();

    auto  DC = C.mat();
    
    blas::prod( alpha, AU, blas::adjoint( VB ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const Hpro::matop_t                      op_B,
           const matrix::lrmatrix< value_t > &      B,
           matrix::lrmatrix< value_t > &            C,
           const Hpro::TTruncAcc &                  acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  DA = A.mat();
    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    auto  AU = blas::prod( alpha, blas::mat_view( op_A, DA ), UB );

    std::scoped_lock  lock( C.mutex() );

    auto [ U, V ] = approx( { C.U(), AU },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
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
           const Hpro::TTruncAcc &                acc,
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
           const Hpro::TTruncAcc &                acc )
{
    HLR_MULT_PRINT;

    // U·(V' × B) = U·X' with X = B'·V
    auto  VA = A.V( op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    // U·X' + C
    auto  DC = C.mat();
    
    blas::prod( value_t(1), A.U( op_A ), blas::adjoint( X ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    std::scoped_lock  lock( C.mutex() );
    
    // U·X' + C
    auto  DC = C.mat();
    
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
           const Hpro::TTruncAcc &                acc )
{
    HLR_MULT_PRINT;

    // U·S·(V' × B) = U·S·X' with X = B'·V
    auto  VA = A.V( op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    // S·X' = (X·S')' = (X·S)' sind S is diagonal and real
    blas::prod_diag( X, A.S() );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    // U·X' + C
    auto  DC = C.mat();
    
    blas::prod( value_t(1), A.U( op_A ), blas::adjoint( X ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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
           const Hpro::TTruncAcc &                acc,
           const approx_t &                       approx )
{
    HLR_MULT_PRINT;

    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    std::scoped_lock  lock( C.mutex() );

    auto [ U, V ] = approx( { C.U(), UA },
                            { C.V(), VC },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
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
           const Hpro::TTruncAcc &                  acc,
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
           const Hpro::TTruncAcc &                  acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    auto  DC = C.mat();
    
    blas::prod( alpha, UA, VB, value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    std::scoped_lock  lock( C.mutex() );
    
    auto  DC = C.mat();
    
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
           const Hpro::TTruncAcc &                  acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) S(A) ( V(A)^H B )
    auto  UA = A.U( op_A );
    auto  SA = A.S();
    auto  VA = A.V( op_A );
    auto  DB = B.mat();
    auto  VB = blas::prod( value_t(1), blas::adjoint( VA ), blas::mat_view( op_B, DB ) );

    blas::prod_diag( SA, VB );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    auto  DC = C.mat();
    
    blas::prod( alpha, UA, VB, value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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
           const Hpro::TTruncAcc &                  acc,
           const approx_t &                         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );
    auto  VB = blas::prod( alpha, blas::adjoint( blas::mat_view( op_B, B.mat() ) ), VA );

    std::scoped_lock  lock( C.mutex() );

    auto [ U, V ] = approx( { C.U(), UA },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
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
           const Hpro::TTruncAcc &              acc,
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
           const Hpro::TTruncAcc &              acc )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  UA = A.U( op_A );
    auto  VA = A.V( op_A );

    auto  UB = B.U( op_B );
    auto  VB = B.V( op_B );
    
    auto  T  = blas::prod( value_t(1), blas::adjoint( VA ), UB );
    auto  UT = blas::prod( value_t(1), UA, T );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();
    
    auto  DC = C.mat();
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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

    std::scoped_lock  lock( C.mutex() );
    
    auto  DC = C.mat();
    
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
           const Hpro::TTruncAcc &                acc )
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

    blas::prod_diag( SA, T );
    blas::prod_diag( T, SB );
    
    auto  UT = blas::prod( UA, T );

    std::scoped_lock  lock( C.mutex() );
    const auto        was_compressed = C.is_compressed();

    C.decompress();

    auto  DC = C.mat();
    
    blas::prod( alpha, UT, blas::adjoint( VB ), value_t(1), DC );

    if ( was_compressed )
        C.compress( acc );
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
           const Hpro::TTruncAcc &              acc,
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

    std::scoped_lock  lock( C.mutex() );

    auto [ U, V ] = approx( { C.U(), UT },
                            { C.V(), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
