#ifndef __HLR_ARITH_DETAIL_MULTIPLY_BLAS_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_BLAS_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions for blas::matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

namespace hlr {

/////////////////////////////////////////////////////////////////////////////////
//
// multiplication with blas::matrix
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const blas::matrix< value_t > &   B,
           blas::matrix< value_t > &         C );

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const blas::matrix< value_t > &        B,
           blas::matrix< value_t > &              C )
{
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0, op_A ) ) );
        
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
multiply ( const value_t                       alpha,
           const Hpro::matop_t                 op_A,
           const blas::matrix< value_t > &     U,
           const blas::matrix< value_t > &     V,
           const blas::matrix< value_t > &     B,
           blas::matrix< value_t > &           C )
{
    switch ( op_A )
    {
        case Hpro::apply_normal :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( V ), B );

            blas::prod( alpha, U, T, value_t(1), C );
        }
        break;

        case Hpro::apply_adjoint :
        {
            auto  T = blas::prod( value_t(1), blas::adjoint( U ), B );

            blas::prod( alpha, V, T, value_t(1), C );
        }
        break;

        case Hpro::apply_conjugate :
        {
            HLR_ASSERT( ! Hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( V ), B );

            blas::prod( alpha, U, T, value_t(1), C );
        }
        break;

        case Hpro::apply_transposed :
        {
            HLR_ASSERT( ! Hpro::is_complex_type< value_t >::value );
                            
            auto  T = blas::prod( value_t(1), blas::transposed( U ), B );

            blas::prod( alpha, V, T, value_t(1), C );
        }
        break;
    }// switch
}

template < typename value_t >
void
multiply ( const value_t                       alpha,
           const Hpro::matop_t                 op_A,
           const Hpro::TRkMatrix< value_t > &  A,
           const blas::matrix< value_t > &     B,
           blas::matrix< value_t > &           C )
{
    if ( A.is_zero() )
        return;

    multiply( alpha, op_A, blas::mat_U( A ), blas::mat_V( A ), B, C );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const matrix::lrmatrix< value_t > &  A,
           const blas::matrix< value_t > &      B,
           blas::matrix< value_t > &            C )
{
    if ( A.is_zero() )
        return;

    const auto  U = A.U_decompressed();
    const auto  V = A.V_decompressed();
    
    multiply( alpha, op_A, U, V, B, C );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const Hpro::matop_t                    op_A,
           const Hpro::TDenseMatrix< value_t > &  A,
           const blas::matrix< value_t > &        B,
           blas::matrix< value_t > &              C )
{
    blas::prod( alpha, blas::mat_view( op_A, blas::mat( A ) ), B, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const Hpro::matop_t                      op_A,
           const matrix::dense_matrix< value_t > &  A,
           const blas::matrix< value_t > &          B,
           blas::matrix< value_t > &                C )
{
    const auto  D = A.mat_decompressed();
    
    blas::prod( alpha, blas::mat_view( op_A, D ), B, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const blas::matrix< value_t > &   B,
           blas::matrix< value_t > &         C )
{
    using namespace hlr::matrix;
    
    if      ( is_blocked( A ) )             multiply( alpha, op_A, * cptrcast( & A, Hpro::TBlockMatrix< value_t > ), B, C );
    else if ( is_compressible( A ) )
    {
        if      ( is_lowrank( A ) )         multiply( alpha, op_A, * cptrcast( & A, lrmatrix< value_t > ), B, C );
        else if ( is_uniform_lowrank( A ) ) multiply( alpha, op_A, * cptrcast( & A, uniform_lrmatrix< value_t > ), B, C );
        else if ( is_dense( A ) )           multiply( alpha, op_A, * cptrcast( & A, dense_matrix< value_t > ), B, C );
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// if
    else if ( is_lowrank( A ) )             multiply( alpha, op_A, * cptrcast( & A, Hpro::TRkMatrix< value_t > ), B, C );
    else if ( is_dense( A ) )               multiply( alpha, op_A, * cptrcast( & A, Hpro::TDenseMatrix< value_t > ), B, C );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}


template < typename value_t >
void
multiply ( const value_t                     alpha,
           const blas::matrix< value_t > &   A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           blas::matrix< value_t > &         C );

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const blas::matrix< value_t > &        A,
           const Hpro::matop_t                    op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           blas::matrix< value_t > &              C )
{
    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  C_0j = blas::matrix< value_t >( C, blas::range::all, B.block( 0, j, op_B )->col_is( op_B ) - B.col_ofs( op_B ) );
        
        for ( uint  l = 0; l < B.nblock_rows( op_B ); ++l )
        {
            auto  B_lj = B.block( l, j, op_B );
            auto  A_0l  = blas::matrix< value_t >( A, blas::range::all, B_lj->row_is( op_B ) - B.row_ofs( op_B ) );
            
            multiply( alpha, A_0l, op_B, * B_lj, C_0j );
        }// for
    }// for
}

template < typename value_t >
void
multiply ( const value_t                       alpha,
           const blas::matrix< value_t > &     A,
           const Hpro::matop_t                 op_B,
           const blas::matrix< value_t > &     U,
           const blas::matrix< value_t > &     V,
           blas::matrix< value_t > &           C )
{
    switch ( op_B )
    {
        case Hpro::apply_normal :
        {
            auto  T = blas::prod( value_t(1), A, U );

            blas::prod( alpha, T, blas::adjoint( V ), value_t(1), C );
        }
        break;

        case Hpro::apply_adjoint :
        {
            auto  T = blas::prod( value_t(1), A, V );

            blas::prod( alpha, T, blas::adjoint( U ), value_t(1), C );
        }
        break;

        case Hpro::apply_conjugate :
        {
            if constexpr( Hpro::is_complex_type_v< value_t > )
            {
                auto  Uc = blas::copy( U );

                blas::conj( Uc );
                
                auto  T = blas::prod( value_t(1), A, Uc );

                blas::prod( alpha, T, blas::transposed( V ), value_t(1), C );
            }// if
            else
            {
                auto  T = blas::prod( value_t(1), A, U );

                blas::prod( alpha, T, blas::transposed( V ), value_t(1), C );
            }// else
        }
        break;

        case Hpro::apply_transposed :
        {
            if constexpr( Hpro::is_complex_type_v< value_t > )
            {
                auto  Vc = blas::copy( V );

                blas::conj( Vc );
                
                auto  T = blas::prod( value_t(1), A, Vc );

                blas::prod( alpha, T, blas::transposed( U ), value_t(1), C );
            }// if
            else
            {
                auto  T = blas::prod( value_t(1), A, V );

                blas::prod( alpha, T, blas::transposed( U ), value_t(1), C );
            }// else
        }
        break;
    }// switch
}

template < typename value_t >
void
multiply ( const value_t                       alpha,
           const blas::matrix< value_t > &     A,
           const Hpro::matop_t                 op_B,
           const Hpro::TRkMatrix< value_t > &  B,
           blas::matrix< value_t > &           C )
{
    if ( B.is_zero() )
        return;

    multiply( alpha, A, op_B, blas::mat_U( B ), blas::mat_V( B ), C );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const blas::matrix< value_t > &      A,
           const Hpro::matop_t                  op_B,
           const matrix::lrmatrix< value_t > &  B,
           blas::matrix< value_t > &            C )
{
    if ( B.is_zero() )
        return;

    const auto  U = B.U_decompressed();
    const auto  V = B.V_decompressed();

    multiply( alpha, A, op_B, U, V, C );
}

template < typename value_t >
void
multiply ( const value_t                          alpha,
           const blas::matrix< value_t > &        A,
           const Hpro::matop_t                    op_B,
           const Hpro::TDenseMatrix< value_t > &  B,
           blas::matrix< value_t > &              C )
{
    blas::prod( alpha, A, blas::mat_view( op_B, blas::mat( B ) ), value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                            alpha,
           const blas::matrix< value_t > &          A,
           const Hpro::matop_t                      op_B,
           const matrix::dense_matrix< value_t > &  B,
           blas::matrix< value_t > &                C )
{
    const auto  D = B.mat_decompressed();
    
    blas::prod( alpha, A, blas::mat_view( op_B, D ), value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const blas::matrix< value_t > &   A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           blas::matrix< value_t > &         C )
{
    using namespace hlr::matrix;
    
    if      ( is_blocked( B ) )             multiply( alpha, A, op_B, * cptrcast( & B, Hpro::TBlockMatrix< value_t > ), C );
    else if ( is_compressible( B ) )
    {
        if      ( is_lowrank( B ) )         multiply( alpha, A, op_B, * cptrcast( & B, lrmatrix< value_t > ), C );
        else if ( is_uniform_lowrank( B ) ) multiply( alpha, A, op_B, * cptrcast( & B, uniform_lrmatrix< value_t > ), C );
        else if ( is_dense( B ) )           multiply( alpha, A, op_B, * cptrcast( & B, dense_matrix< value_t > ), C );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_lowrank( B ) )             multiply( alpha, A, op_B, * cptrcast( & B, Hpro::TRkMatrix< value_t > ), C );
    else if ( is_dense( B ) )               multiply( alpha, A, op_B, * cptrcast( & B, Hpro::TDenseMatrix< value_t > ), C );
    else
        HLR_ERROR( "unsupported matrix type : " + B.typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_BLAS_HH
