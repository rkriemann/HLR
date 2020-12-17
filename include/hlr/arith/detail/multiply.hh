#ifndef __HLR_ARITH_DETAIL_MULTIPLY_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

namespace hlr {

/////////////////////////////////////////////////////////////////////////////////
//
// multiplication with blas::matrix
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TMatrix &            A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C );

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TBlockMatrix &       A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = blas::matrix< value_t >( C, A.block( i, 0, op_A )->row_is( op_A ) - A.row_ofs( op_A ), blas::range::all );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  B_j  = blas::matrix< value_t >( B, A_ij->col_is( op_A ) - A.col_ofs( op_A ), blas::range::all );
            
            multiply( alpha, op_A, * A_ij, B_j, C_i );
        }// for
    }// for
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TRkMatrix &          A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    switch ( op_A )
    {
        case hpro::apply_normal :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_U< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_adjoint :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( blas::mat_U< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_V< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_conjugate :
        {
            HLR_ASSERT( ! hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( blas::mat_V< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_U< value_t >( A ), T, value_t(1), C );
        }
        break;

        case hpro::apply_transposed :
        {
            HLR_ASSERT( ! hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( blas::mat_U< value_t >( A ) ), B );

            blas::prod( alpha, blas::mat_V< value_t >( A ), T, value_t(1), C );
        }
        break;
    }// switch
}

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const blas::matrix< value_t > &              B,
           blas::matrix< value_t > &                    C )
{
    HLR_ASSERT(( op_A == apply_normal ) || ( op_A == apply_adjoint ));
    
    auto  VB  = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B );
    auto  SVB = blas::prod( blas::mat_view( op_A, A.coeff() ), VB );
    
    blas::prod( alpha, A.row_cb( op_A ).basis(), SVB, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TDenseMatrix &       A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    blas::prod( alpha, blas::mat_view( op_A, blas::mat< value_t >( A ) ), B, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                    alpha,
           const hpro::matop_t              op_A,
           const hpro::TMatrix &            A,
           const blas::matrix< value_t > &  B,
           blas::matrix< value_t > &        C )
{
    if ( is_blocked( A ) )
        multiply( alpha, op_A, * cptrcast( & A, hpro::TBlockMatrix ), B, C );
    else if ( is_lowrank( A ) )
        multiply( alpha, op_A, * cptrcast( & A, hpro::TRkMatrix ), B, C );
    else if ( matrix::is_uniform_lowrank( A ) )
        multiply( alpha, op_A, * cptrcast( & A, matrix::uniform_lrmatrix< value_t > ), B, C );
    else if ( is_dense( A ) )
        multiply( alpha, op_A, * cptrcast( & A, hpro::TDenseMatrix ), B, C );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·B + C
//
/////////////////////////////////////////////////////////////////////////////////

#define HLR_MULT_PRINT   HLR_LOG( 4, hpro::to_string( "multiply( %s %d, %s %d, %s %d )", \
                                                      A.typestr().c_str(), A.id(), \
                                                      B.typestr().c_str(), B.id(), \
                                                      C.typestr().c_str(), C.id() ) )
//
// forward decl.(s)
//
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
           const approx_t &         approx );

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
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

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    //
    // compute temporary block matrix BC and sub blocks
    // BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< hpro::TBlockMatrix >( C.row_is(), C.col_is() );

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
                        BC->set_block( i, j, new hpro::TRkMatrix( A_il->row_is( op_A ), B_lj->col_is( op_B ),
                                                                  hpro::value_type< value_t >::value ) );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    // apply update
    hlr::add< value_t >( value_t(1), *BC, C, acc, approx );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;
    
    //
    // perform block × block multiplication and compute local result
    // which is used to update C
    //

    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            std::unique_ptr< hpro::TDenseMatrix >  C_ij;
            
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( C_ij ) )
                        C_ij = std::make_unique< hpro::TDenseMatrix >( A_il->row_is( op_A ), B_lj->col_is( op_B ), hpro::value_type< value_t >::value );
                    
                    multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij );
                }// if
            }// if

            if ( ! is_null( C_ij ) )
                C.add_block( value_t(1), value_t(1), C_ij.get() );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), UC, blas::mat_V< value_t >( B, op_B ) );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UC },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V< value_t >( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), blas::mat_U< value_t >( A, op_A ), VC );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V< value_t >( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), VC },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;

    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = hpro::TDenseMatrix( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  DB   = blas::mat< value_t >( B );
            auto  is_j = A_ij->col_is( op_A );

            if ( op_B == hpro::apply_normal )
            {
                auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
            }// if
            else
            {
                auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
            }// else
        }// for

        C.add_block( hpro::real(1), hpro::real(1), &C_i );
    }// for
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;

    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  C_j = hpro::TDenseMatrix( C.row_is(), B.block( 0, j, op_B )->col_is( op_B ) );
        
        for ( uint  i = 0; i < B.nblock_rows( op_B ); ++i )
        {
            auto  DA   = blas::mat< value_t >( A );
            auto  B_ij = B.block( i, j, op_B );
            auto  is_i = B_ij->row_is( op_B );

            if ( op_A == hpro::apply_normal )
            {
                auto  DA_i = blas::matrix< value_t >( DA, blas::range::all, is_i - A.col_ofs() );
                auto  A_i  = hpro::TDenseMatrix( A.row_is( op_A ), is_i, DA_i );
            
                multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
            }// if
            else
            {
                auto  DA_i = blas::matrix< value_t >( DA, is_i - A.row_ofs(), blas::range::all );
                auto  A_i  = hpro::TDenseMatrix( A.row_is( op_A ), is_i, DA_i );
            
                multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
            }// else
        }// for

        C.add_block( hpro::real(1), hpro::real(1), &C_j );
    }// for
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TBlockMatrix &     C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), T );
    auto  R  = std::make_unique< hpro::TRkMatrix >( C.row_is(), C.col_is(), UT, blas::mat_V< value_t >( B, op_B ) );
        
    hlr::add< value_t >( value_t(1), *R, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;

    auto  DA = blas::mat< value_t >( A );
    auto  DB = blas::mat< value_t >( B );
    auto  DT = blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ) );
    auto  T  = hpro::TDenseMatrix( C.row_is(), C.col_is(), std::move( DT ) );
        
    hlr::add< value_t >( value_t(1), T, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TRkMatrix &        C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), T );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UT },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                          op_B,
           const matrix::uniform_lrmatrix< value_t > &  B,
           hpro::TRkMatrix &                            C,
           const hpro::TTruncAcc &                      acc,
           const approx_t &                             approx )
{
    HLR_MULT_PRINT;

    // C + A×B = C + (U·((S·(V' × W))·T))·X' = C + T·X'
    auto  T = blas::matrix< value_t >();

    {
        auto  VW    = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
        auto  SVW   = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );
        auto  SVWT  = blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) );

        T = std::move( blas::prod( alpha, A.row_cb( op_A ).basis(), SVWT ) );
    }

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), T },
                            { blas::mat_V< value_t >( C ), B.col_cb( op_B ).basis() },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( alpha,
                           blas::adjoint( blas::mat_view( op_B, blas::mat< value_t >( B ) ) ),
                           blas::mat_V< value_t >( A, op_A ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), VB },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                          op_B,
           const hpro::TDenseMatrix &                   B,
           hpro::TRkMatrix &                            C,
           const hpro::TTruncAcc &                      acc,
           const approx_t &                             approx )
{
    HLR_MULT_PRINT;

    // C + A × B = C + U·(S·(V' × B)) -> (B' × V)·S'
    auto  BV  = blas::prod( blas::mat_view( blas::adjoint( op_B ), blas::mat< value_t >( B ) ), A.col_cb( op_A ).basis() );
    auto  BVS = blas::prod( alpha, BV, blas::mat_view( blas::adjoint( op_A ), A.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), A.row_cb( op_A ).basis() },
                            { blas::mat_V< value_t >( C ), BVS },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = blas::prod( alpha,
                           blas::mat_view( op_A, blas::mat< value_t >( A ) ),
                           blas::mat_U< value_t >( B, op_B ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), AU },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}


template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const hpro::TDenseMatrix &                   A,
           const hpro::matop_t                          op_B,
           const matrix::uniform_lrmatrix< value_t > &  B,
           hpro::TRkMatrix &                            C,
           const hpro::TTruncAcc &                      acc,
           const approx_t &                             approx )
{
    HLR_MULT_PRINT;

    // C + A × B = C + ((A × U)·S)·V'
    auto  AU  = blas::prod( blas::mat_view( op_A, blas::mat< value_t >( A ) ), B.row_cb( op_B ).basis() );
    auto  AUS = blas::prod( alpha, AU, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), AUS },
                            { blas::mat_V< value_t >( C ), B.col_cb( op_B ).basis() },
                            acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TRkMatrix &           C,
           const hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = blas::prod( alpha,
                           blas::mat_view( op_A, blas::mat< value_t >( A ) ),
                           blas::mat_view( op_B, blas::mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( value_t(1), blas::mat_U< value_t >( C ), blas::adjoint( blas::mat_V< value_t >( C ) ), value_t(1), AB );

    auto [ U, V ] = approx( AB, acc );
        
    C.set_lrmat( U, V );
}

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TRkMatrix &  A,
           const hpro::matop_t      op_B,
           const hpro::TRkMatrix &  B,
           hpro::TDenseMatrix &     C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_U< value_t >( B, op_B ) );
    auto  UT = blas::prod( value_t(1), blas::mat_U< value_t >( A, op_A ), T );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UT, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                          op_B,
           const matrix::uniform_lrmatrix< value_t > &  B,
           hpro::TDenseMatrix &                         C )
{
    HLR_MULT_PRINT;
    
    // C = C + A×B = C + (U·((S·(V' × W))·T))·X'
    auto  VW    = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
    auto  SVW   = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );
    auto  SVWT  = blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) );
    auto  USVWT = blas::prod( A.row_cb( op_A ).basis(), SVWT );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, USVWT, blas::adjoint( B.col_cb( op_B ).basis() ), value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TRkMatrix &     B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, blas::mat< value_t >( A ) ), blas::mat_U< value_t >( B, op_B ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, AU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const hpro::TDenseMatrix &                   A,
           const hpro::matop_t                          op_B,
           const matrix::uniform_lrmatrix< value_t > &  B,
           hpro::TDenseMatrix &                         C )
{
    HLR_MULT_PRINT;
    
    // C = C + (( A U ) S) V'
    auto  AU  = blas::prod( blas::mat_view( op_A, blas::mat< value_t >( A ) ), B.row_cb( op_B ).basis() );
    auto  AUS = blas::prod( AU, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, AUS, blas::adjoint( B.col_cb( op_B ).basis() ), value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TRkMatrix &     A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_B, blas::mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), VB, value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                          op_B,
           const hpro::TDenseMatrix &                   B,
           hpro::TDenseMatrix &                         C )
{
    HLR_MULT_PRINT;
    
    // C = C + U·(S·(V'×B))
    auto  VB  = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), blas::mat_view( op_B, blas::mat< value_t >( B ) ) );
    auto  SVB = blas::prod( blas::mat_view( op_A, A.coeff() ), VB );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, A.row_cb( op_A ).basis(), SVB, value_t(1), blas::mat< value_t >( C ) );
}

template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TDenseMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TDenseMatrix &  B,
           hpro::TDenseMatrix &        C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    blas::prod( alpha,
                blas::mat_view( op_A, blas::mat< value_t >( A ) ),
                blas::mat_view( op_B, blas::mat< value_t >( B ) ),
                value_t(1), blas::mat< value_t >( C ) );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_HH
