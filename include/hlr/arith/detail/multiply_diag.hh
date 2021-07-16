#ifndef __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions with diagonal matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

namespace hlr { 

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication for blas::matrix B
//
//    C := α·D·B + C  
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C );

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_D,
                const hpro::TBlockMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    for ( uint  i = 0; i < std::min( D.nblock_rows(), D.nblock_cols() ); ++i )
    {
        auto  D_ii = D.block( i, i );

        HLR_ASSERT( ! is_null( D_ii ) );

        auto  B_i = blas::matrix< value_t >( B, D_ii->col_is( op_D ) - D.col_ofs(), blas::range::all );
        auto  C_i = blas::matrix< value_t >( C, D_ii->row_is( op_D ) - D.row_ofs(), blas::range::all );

        multiply_diag( alpha, op_D, *D_ii, B_i, C_i );
    }// for
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_D,
                const hpro::TDenseMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    blas::prod( alpha, blas::mat_view( op_D, blas::mat< value_t >( D ) ), B, value_t(1), C );
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    if ( is_blocked( D ) )
        multiply_diag( alpha, op_D, *cptrcast( &D, hpro::TBlockMatrix ), B, C );
    else if ( is_dense( D ) )
        multiply_diag( alpha, op_D, *cptrcast( &D, hpro::TDenseMatrix ), B, C );
    else
        HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
}
    
/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication for blas::matrix A and/or B
//
//    C := α·A·D·B + C  
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TMatrix &            A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C );

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TBlockMatrix &       A,
                const hpro::matop_t              op_D,
                const hpro::TBlockMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0, op_A ) ) );
        
        auto  C_i = blas::matrix< value_t >( C, A.block( i, 0, op_A )->row_is( op_A ) - A.row_ofs( op_A ), blas::range::all );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  D_jj = D.block( j, j, op_D );
            auto  B_j  = blas::matrix< value_t >( B, A_ij->col_is( op_A ) - A.col_ofs( op_A ), blas::range::all );
            
            multiply_diag( alpha, op_A, * A_ij, op_D, *D_jj, B_j, C_i );
        }// for
    }// for
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TRkMatrix &          A,
                const hpro::matop_t              op_D,
                const hpro::TBlockMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    //
    // C = C + A×D×B
    //   = C + U·V'×D×B
    //   = C + U·Y'
    //
    // with Y = B'×D'×V
    //

    auto  V  = blas::mat_V< value_t >( A, op_A );
    auto  DV = blas::matrix< value_t >( D.ncols(), A.rank() );

    multiply_diag( alpha, blas::adjoint( op_D ), D, V, DV );

    auto  Y  = blas::prod( blas::adjoint( B ), DV );

    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), blas::adjoint( Y ), value_t(1), C );
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TDenseMatrix &       A,
                const hpro::matop_t              op_D,
                const hpro::TDenseMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    auto  T = blas::prod( mat_view( op_D, blas::mat< value_t >( D ) ), B );

    blas::prod( alpha, mat_view( op_A, blas::mat< value_t >( A ) ), T, value_t(1), C );
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TRkMatrix &          A,
                const hpro::matop_t              op_D,
                const hpro::TDenseMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    // C = C + A×D×B
    //   = C + U·((V'×D)×B)
    auto  T1 = blas::prod( blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), mat_view( op_D, blas::mat< value_t >( D ) ) );
    auto  T2 = blas::prod( T1, B );

    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), T2, value_t(1), C );
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const hpro::matop_t              op_A,
                const hpro::TMatrix &            A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    if ( is_blocked( A ) )
    {
        if ( is_blocked( D ) )
            multiply_diag( alpha,
                           op_A, *cptrcast( &A, hpro::TBlockMatrix ),
                           op_D, *cptrcast( &D, hpro::TBlockMatrix ),
                           B, C );
        else if ( is_dense( D ) )
        { HLR_ERROR( "todo: blocked x dense" ); }
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( D ) )
            multiply_diag( alpha,
                           op_A, *cptrcast( &A, hpro::TRkMatrix ),
                           op_D, *cptrcast( &D, hpro::TBlockMatrix ),
                           B, C );
        else if ( is_dense( D ) )
            multiply_diag( alpha,
                           op_A, *cptrcast( &A, hpro::TRkMatrix ),
                           op_D, *cptrcast( &D, hpro::TDenseMatrix ),
                           B, C );
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( D ) )
        { HLR_ERROR( "todo: dense x blocked" ); }
        else if ( is_dense( D ) )
            multiply_diag( alpha,
                           op_A, *cptrcast( &A, hpro::TDenseMatrix ),
                           op_D, *cptrcast( &D, hpro::TDenseMatrix ),
                           B, C );
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for A : " + A.typestr() );
}



template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const hpro::matop_t              op_B,
                const hpro::TMatrix &            B,
                blas::matrix< value_t > &        C );

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TBlockMatrix &       D,
                const hpro::matop_t              op_B,
                const hpro::TBlockMatrix &       B,
                blas::matrix< value_t > &        C )
{
    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  C_0j = blas::matrix< value_t >( C, blas::range::all, B.block( 0, j, op_B )->col_is( op_B ) - B.col_ofs( op_B ) );
        
        for ( uint  l = 0; l < B.nblock_rows( op_B ); ++l )
        {
            auto  B_lj = B.block( l, j, op_B );
            auto  D_ll = D.block( l, l, op_D );
            auto  A_0l  = blas::matrix< value_t >( A, blas::range::all, B_lj->row_is( op_B ) - B.row_ofs( op_B ) );
            
            multiply_diag( alpha, A_0l, op_D, *D_ll, op_B, *B_lj, C_0j );
        }// for
    }// for
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const hpro::matop_t              op_B,
                const hpro::TMatrix &            B,
                blas::matrix< value_t > &        C )
{
    if ( is_blocked( B ) )
    {
        if ( is_blocked( D ) )
            multiply_diag( alpha,
                           A,
                           op_D, *cptrcast( &D, hpro::TBlockMatrix ),
                           op_B, *cptrcast( &B, hpro::TBlockMatrix ),
                           C );
        else if ( is_dense( D ) )
        { HLR_ERROR( "todo: blocked x dense" ); }
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( B ) )
    {
        if ( is_blocked( D ) )
        { HLR_ERROR( "todo: lowrank x blocked" ); }
        else if ( is_dense( D ) )
        { HLR_ERROR( "todo: lowrank x dense" ); }
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( B ) )
    {
        if ( is_blocked( D ) )
        { HLR_ERROR( "todo: dense x blocked" ); }
        else if ( is_dense( D ) )
        { HLR_ERROR( "todo: dense x dense" ); }
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
}



template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C );

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TBlockMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    HLR_ASSERT( D.nblock_rows() == D.nblock_cols() );
    
    for ( uint  i = 0; i < D.nblock_rows(); ++i )
    {
        auto  D_ii = D.block( i, i, op_D );

        HLR_ASSERT( ! is_null( D_ii ) );
        
        auto  A_i = blas::matrix< value_t >( A, blas::range::all, D_ii->row_is( op_D ) - D.row_ofs( op_D ) );
        auto  B_i = blas::matrix< value_t >( B, D_ii->col_is( op_D ) - D.col_ofs( op_D ), blas::range::all );
            
        multiply_diag( alpha, A_i, op_D, *D_ii, B_i, C );
    }// for
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TDenseMatrix &       D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    auto  T = blas::prod( mat_view( op_D, blas::mat< value_t >( D ) ), B );

    blas::prod( alpha, A, T, value_t(1), C );
}

template < typename value_t >
void
multiply_diag ( const value_t                    alpha,
                const blas::matrix< value_t > &  A,
                const hpro::matop_t              op_D,
                const hpro::TMatrix &            D,
                const blas::matrix< value_t > &  B,
                blas::matrix< value_t > &        C )
{
    if ( is_blocked( D ) )
        multiply_diag( alpha, A, op_D, *cptrcast( &D, hpro::TBlockMatrix ), B, C );
    else if ( is_dense( D ) )
        multiply_diag( alpha, A, op_D, *cptrcast( &D, hpro::TDenseMatrix ), B, C );
    else
        HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
}

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·D·B + C for general matrices
//
/////////////////////////////////////////////////////////////////////////////////

//
// forward decl. for general versions
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t            alpha,
                const hpro::matop_t      op_A,
                const hpro::TMatrix &    A,
                const hpro::matop_t      op_D,
                const hpro::TMatrix &    D,
                const hpro::matop_t      op_B,
                const hpro::TMatrix &    B,
                hpro::TMatrix &          C,
                const hpro::TTruncAcc &  acc,
                const approx_t &         approx );

template < typename value_t >
void
multiply_diag ( const value_t            alpha,
                const hpro::matop_t      op_A,
                const hpro::TMatrix &    A,
                const hpro::matop_t      op_D,
                const hpro::TMatrix &    D,
                const hpro::matop_t      op_B,
                const hpro::TMatrix &    B,
                hpro::TMatrix &          C );

//
// blocked x blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TBlockMatrix &                            C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), D.block( l, l, op_D ), B.block( l, j, op_B ) ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, *A.block( i, l, op_A ),
                                              op_D, *D.block( l, l, op_D ),
                                              op_B, *B.block( l, j, op_B ),
                                              *C.block( i, j ), acc, approx );
            }// if       
        }// for
    }// for
}

//
// blocked x blocked x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TDenseMatrix &                            C )
{
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
                    auto  D_ll = D.block( l, l, op_D );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( C_ij ) )
                        C_ij = std::make_unique< hpro::TDenseMatrix >( A_il->row_is( op_A ), B_lj->col_is( op_B ), hpro::value_type_v< value_t > );
                    
                    multiply_diag( alpha, op_A, *A_il, op_D, *D_ll, op_B, *B_lj, *C_ij );
                }// if
            }// if

            if ( ! is_null( C_ij ) )
                C.add_block( value_t(1), value_t(1), C_ij.get() );
        }// for
    }// for
}

//
// blocked x blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
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
                    auto  D_ll = D.block( l, l, op_D );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new hpro::TRkMatrix( A_il->row_is( op_A ), B_lj->col_is( op_B ),
                                                                  hpro::value_type_v< value_t > ) );
                    
                    multiply_diag< value_t >( alpha, op_A, *A_il, op_D, *D_ll, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    // apply update
    hlr::add< value_t >( value_t(1), *BC, C, acc, approx );
}

//
// blocked x dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TDenseMatrix &                            C )
{
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = hpro::TDenseMatrix( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  D_jj = D.block( j, j, op_D );
            auto  DB   = blas::mat< value_t >( B );
            auto  is_j = A_ij->col_is( op_A );

            if ( op_B == hpro::apply_normal )
            {
                auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply_diag( alpha, op_A, * A_ij, op_D, *D_jj, op_B, B_j, C_i );
            }// if
            else
            {
                auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
                auto  B_j  = hpro::TDenseMatrix( is_j, B.col_is( op_B ), DB_j );
            
                multiply_diag( alpha, op_A, * A_ij, op_D, *D_jj, op_B, B_j, C_i );
            }// else
        }// for

        C.add_block( hpro::real(1), hpro::real(1), &C_i );
    }// for
}

//
// blocked x blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    // DC = A×B
    auto  DC = hpro::TDenseMatrix( C.row_is(), C.col_is(), hpro::value_type_v< value_t > );

    if ( op_B == apply_normal )
    {
        multiply_diag( alpha, op_A, A, op_D, D, blas::mat< value_t >( B ), blas::mat< value_t >( DC ) );

        std::scoped_lock  lock( C.mutex() );
        
        // DC = DC + C
        hlr::add< value_t >( value_t(1), C, DC );

        // approximate result and update C
        auto [ U, V ] = approx( blas::mat< value_t >( DC ), acc );

        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

//
// blocked x dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TBlockMatrix &                            C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply_diag< value_t >( alpha, op_A, A, op_D, D, UB, UC );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), UC, blas::mat_V< value_t >( B, op_B ) );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// blocked x blocked x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TDenseMatrix &                            C )
{
    // (A × U)·V' = W·V'
    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply_diag< value_t >( alpha, op_A, A, op_D, D, UB, W );

    std::scoped_lock  lock( C.mutex() );
    
    // W·V' + C
    blas::prod( value_t(1), W, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), blas::mat< value_t >( C ) );
}

//
// blocked x blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TBlockMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    auto  UB = blas::mat_U< value_t >( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply_diag< value_t >( alpha, op_A, A, op_D, D, UB, UC );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UC },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// blocked x dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TBlockMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TDenseMatrix &                            C )
{
    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  C_j = hpro::TDenseMatrix( C.row_is(), B.block( 0, j, op_B )->col_is( op_B ) );
        
        for ( uint  i = 0; i < B.nblock_rows( op_B ); ++i )
        {
            auto  DA   = blas::mat< value_t >( A );
            auto  D_ii = D.block( i, i, op_D );
            auto  B_ij = B.block( i, j, op_B );
            auto  is_i = B_ij->row_is( op_B );

            if ( op_A == hpro::apply_normal )
            {
                auto  DA_i = blas::matrix< value_t >( DA, blas::range::all, is_i - A.col_ofs() );
                auto  A_i  = hpro::TDenseMatrix( A.row_is( op_A ), is_i, DA_i );
            
                multiply_diag( alpha, op_A, A_i, op_D, *D_ii, op_B, *B_ij, C_j );
            }// if
            else
            {
                auto  DA_i = blas::matrix< value_t >( DA, is_i - A.row_ofs(), blas::range::all );
                auto  A_i  = hpro::TDenseMatrix( A.row_is( op_A ), is_i, DA_i );
            
                multiply_diag( alpha, op_A, A_i, op_D, *D_ii, op_B, *B_ij, C_j );
            }// else
        }// for

        C.add_block( hpro::real(1), hpro::real(1), &C_j );
    }// for
}

//
// dense x blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    auto  DC = hpro::TDenseMatrix( C.row_is(), C.col_is(), hpro::value_type_v< value_t > );

    if ( op_A == apply_normal )
    {
        // D = A×B
        multiply_diag( alpha, blas::mat< value_t >( A ), op_D, D, op_B, B, blas::mat< value_t >( DC ) );

        std::scoped_lock  lock( C.mutex() );
    
        // DC = DC + C
        hlr::add< value_t >( value_t(1), C, DC );

        // approximate result and update C
        auto [ U, V ] = approx( blas::mat< value_t >( DC ), acc );

        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

//
// dense x dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TBlockMatrix &                            C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    auto  DA = blas::mat< value_t >( A );
    auto  DD = blas::mat< value_t >( D );
    auto  DB = blas::mat< value_t >( B );
    auto  T1 = blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_D, DD ) );
    auto  T2 = blas::prod( T1, blas::mat_view( op_B, DB ) );
    auto  T  = hpro::TDenseMatrix( C.row_is(), C.col_is(), std::move( T2 ) );
        
    hlr::add< value_t >( value_t(1), T, C, acc, approx );
}

//
// dense x dense x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TDenseMatrix &                            C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A D B
    auto  AD  = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );

    blas::prod( alpha,
                AD,
                blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ),
                value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// dense x dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AD  = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );
    auto  ADB = blas::prod( alpha,
                            AD,
                            blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( value_t(1), blas::mat_U< value_t >( C ), blas::adjoint( blas::mat_V< value_t >( C ) ), value_t(1), ADB );

    auto [ U, V ] = approx( ADB, acc );
        
    C.set_lrmat( U, V );
}

//
// dense x blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TDenseMatrix &                      /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TDenseMatrix &                            C )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A D U(B) ) V(B)^H
    auto  DU  = blas::prod( value_t(1),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ),
                            blas::mat_U< value_t >( B, op_B ) );
    auto  ADU = blas::prod( value_t(1),
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            DU );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, ADU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// dense x dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TDenseMatrix &                      A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A D U(B) ] , [ V(C), V(B) ] )
    auto  DU  = blas::prod( value_t(1),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ),
                            blas::mat_U< value_t >( B, op_B ) );
    auto  ADU = blas::prod( alpha,
                            blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                            DU );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), ADU },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

//
// lowrank x blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TBlockMatrix &                            C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    //
    // C + A × D × B = C + U · ( V' × D × B )
    //               = C + U·X'
    //
    // with X = B'×D'×V
    //
    
    auto  V = blas::mat_V< value_t >( A, op_A );
    auto  X = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply_diag( alpha, blas::adjoint( op_B ), B, blas::adjoint( op_D ), D, V, X );

    auto  RC = hpro::TRkMatrix( C.row_is(), C.col_is(), blas::mat_U< value_t >( A, op_A ), X );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// lowrank x blocked x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TBlockMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    //
    // C = truncate( C    + A×D×B )
    //   = truncate( U·V' + W·X'×D×B )
    //   = truncate( U·V' + W·Y' )
    //
    // with Y = B'×D'×X
    //

    auto  X = blas::mat_V< value_t >( A, op_A );
    auto  Y = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply_diag< value_t >( alpha, blas::adjoint( op_B ), B, blas::adjoint( op_D ), D, X, Y );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), Y },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// lowrank x dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense x blocked = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TBlockMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TRkMatrix &                               /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TDenseMatrix &                      /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense x dense = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TDenseMatrix &                            C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H D B )
    auto  VD  = blas::prod( value_t(1),
                            blas::adjoint( blas::mat_V< value_t >( A, op_A ) ),
                            blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) );
    auto  VDB = blas::prod( value_t(1),
                            VD,
                            blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, blas::mat_U< value_t >( A, op_A ), VDB, value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// lowrank x dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TDenseMatrix &                      B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H D B)^H ] )
    auto  DV  = blas::prod( value_t(1),
                            blas::adjoint( blas::mat_view( op_D, hpro::blas_mat< value_t >( D ) ) ),
                            blas::mat_V< value_t >( A, op_A ) );
    auto  BDV = blas::prod( alpha,
                            blas::adjoint( blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) ),
                            DV );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), blas::mat_U< value_t >( A, op_A ) },
                            { blas::mat_V< value_t >( C ), BDV },
                            acc );
        
    C.set_lrmat( U, V );
}

//
// lowrank x blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TBlockMatrix &                            C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    //
    // with A = U·V', B = W·X' compute
    //
    //   C = C + A × D × B
    //     = C + U·V' × D × W·X'
    //     = C + U·T·X'
    //
    //  with T = V'×D×W.
    //
    
    auto  T  = blas::matrix< value_t >( A.rank(), B.rank() );
    auto  U  = blas::mat_U< value_t >( A, op_A );

    // need to copy because V is partitioned in recursive call which
    // is only possible with real matrices and not matrix views
    auto  VH = blas::copy( blas::adjoint( blas::mat_V< value_t >( A, op_A ) ) );
    auto  W  = blas::mat_U< value_t >( B, op_B );
    auto  X  = blas::mat_V< value_t >( B, op_B );

    multiply_diag< value_t >( alpha, VH, op_D, D, W, T );
    
    auto  UT = blas::prod( value_t(1), U, T );
    auto  R  = hpro::TRkMatrix( C.row_is(), C.col_is(), std::move( UT ), std::move( X ) );
    
    hlr::add< value_t >( value_t(1), R, C, acc, approx );
}

//
// lowrank x blocked x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TBlockMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TDenseMatrix &                            /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TBlockMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    //
    // C = C + A×D×B
    //   = C + U·V' × D × W·X'
    //   = C + U·T·X'
    //
    //  with T = V'×D×W.
    //

    auto  T  = blas::matrix< value_t >( A.rank(), B.rank() );
    auto  U  = blas::mat_U< value_t >( A, op_A );

    // need to copy because V is partitioned in recursive call which
    // is only possible with real matrices and not matrix views
    auto  VH = blas::copy( blas::adjoint( blas::mat_V< value_t >( A, op_A ) ) );
    auto  W  = blas::mat_U< value_t >( B, op_B );
    auto  X  = blas::mat_V< value_t >( B, op_B );

    multiply_diag< value_t >( alpha, VH, op_D, D, W, T );
    
    auto  UT = blas::prod( value_t(1), U, T );
    auto  R  = hpro::TRkMatrix( C.row_is(), C.col_is(), std::move( UT ), std::move( X ) );
    
    hlr::add< value_t >( value_t(1), R, C, acc, approx );
}

//
// lowrank x dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   /* alpha */,
                const hpro::matop_t                             /* op_A */,
                const hpro::TRkMatrix &                         /* A */,
                const hpro::matop_t                             /* op_D */,
                const hpro::TDenseMatrix &                      /* D */,
                const hpro::matop_t                             /* op_B */,
                const hpro::TRkMatrix &                         /* B */,
                hpro::TBlockMatrix &                            /* C */,
                const hpro::TTruncAcc &                         /* acc */,
                const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense x lowrank = dense
//
template < typename value_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TDenseMatrix &                            C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) (( V(A)^H D) U(B) ) V(B)^H
    auto  VD   = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_D, blas::mat< value_t >( D ) ) );
    auto  VDU  = blas::prod( value_t(1), VD, blas::mat_U< value_t >( B, op_B ) );
    auto  UVDU = blas::prod( value_t(1), blas::mat_U< value_t >( A, op_A ), VDU );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UVDU, blas::adjoint( blas::mat_V< value_t >( B, op_B ) ), value_t(1), hpro::blas_mat< value_t >( C ) );
}

//
// lowrank x dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                                   alpha,
                const hpro::matop_t                             op_A,
                const hpro::TRkMatrix &                         A,
                const hpro::matop_t                             op_D,
                const hpro::TDenseMatrix &                      D,
                const hpro::matop_t                             op_B,
                const hpro::TRkMatrix &                         B,
                hpro::TRkMatrix &                               C,
                const hpro::TTruncAcc &                         acc,
                const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H D U(B) ] , [ V(C), V(B)^H ] )
    auto  VD  = blas::prod( value_t(1), blas::adjoint( blas::mat_V< value_t >( A, op_A ) ), blas::mat_view( op_D, blas::mat< value_t >( D ) ) );
    auto  VDU = blas::prod( value_t(1), VD, blas::mat_U< value_t >( B, op_B ) );
    auto  UT  = blas::prod(      alpha, blas::mat_U< value_t >( A, op_A ), VDU );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U< value_t >( C ), UT },
                            { blas::mat_V< value_t >( C ), blas::mat_V< value_t >( B, op_B ) },
                            acc );
        
    C.set_lrmat( U, V );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_DIAG_HH
