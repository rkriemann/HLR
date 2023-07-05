#ifndef __HLR_ARITH_DETAIL_MULTIPLY_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(NDEBUG)
#  define HLR_MULT_PRINT   HLR_LOG( 4, Hpro::to_string( "multiply( %s %d, %s %d, %s %d )", \
                                                        A.typestr().c_str(), A.id(), \
                                                        B.typestr().c_str(), B.id(), \
                                                        C.typestr().c_str(), C.id() ) )
#else
#  define HLR_MULT_PRINT   HLR_LOG( 4, Hpro::to_string( "multiply( %s %d, %s %d, %s %d )", \
                                                        A.typestr().c_str(), A.id(), \
                                                        B.typestr().c_str(), B.id(), \
                                                        C.typestr().c_str(), C.id() ) )
#endif

#include <hlr/arith/detail/multiply_blas.hh>
#include <hlr/arith/detail/multiply_compressed.hh>
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
// blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
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
// blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TBlockMatrix< value_t > &           /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TDenseMatrix< value_t > &           /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        HLR_ASSERT( ! is_null( A.block( i, 0 ) ) );
        
        auto  C_i = Hpro::TDenseMatrix< value_t >( A.block( i, 0, op_A )->row_is( op_A ), C.col_is() );
        
        for ( uint  j = 0; j < A.nblock_cols( op_A ); ++j )
        {
            auto  A_ij = A.block( i, j, op_A );
            auto  DB   = blas::mat( B );
            auto  is_j = A_ij->col_is( op_A );

            if ( op_B == Hpro::apply_normal )
            {
                auto  DB_j = blas::matrix< value_t >( DB, is_j - B.row_ofs(), blas::range::all );
                auto  B_j  = Hpro::TDenseMatrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
            }// if
            else
            {
                auto  DB_j = blas::matrix< value_t >( DB, blas::range::all, is_j - B.col_ofs() );
                auto  B_j  = Hpro::TDenseMatrix< value_t >( is_j, B.col_is( op_B ), DB_j );
            
                multiply( alpha, op_A, * A_ij, op_B, B_j, C_i );
            }// else
        }// for

        C.add_block( value_t(1), value_t(1), &C_i );
    }// for
}

//
// blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // D = A×B
    auto  D = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is() );

    if ( op_B == apply_normal )
    {
        multiply( alpha, op_A, A, blas::mat( B ), blas::mat( D ) );

        std::scoped_lock  lock( C.mutex() );
        
        // D = D + C
        hlr::add< value_t >( value_t(1), C, D );

        // approximate result and update C
        auto [ U, V ] = approx( blas::mat( D ), acc );

        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

//
// blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), UC, blas::mat_V( B, op_B ) );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// blocked x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    // (A × U)·V' = W·V'
    auto  UB = blas::mat_U( B, op_B );
    auto  W  = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, W );

    std::scoped_lock  lock( C.mutex() );
    
    // W·V' + C
    blas::prod( value_t(1), W, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
}

//
// blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  UB = blas::mat_U( B, op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.rank() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), UC },
                            { blas::mat_V( C ), blas::mat_V( B, op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TDenseMatrix< value_t > &           /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TBlockMatrix< value_t > &           /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
    {
        HLR_ASSERT( ! is_null( B.block( 0, j, op_B ) ) );
        
        auto  C_j = Hpro::TDenseMatrix< value_t >( C.row_is(), B.block( 0, j, op_B )->col_is( op_B ) );
        
        for ( uint  i = 0; i < B.nblock_rows( op_B ); ++i )
        {
            auto  DA   = blas::mat( A );
            auto  B_ij = B.block( i, j, op_B );
            auto  is_i = B_ij->row_is( op_B );

            if ( op_A == Hpro::apply_normal )
            {
                auto  DA_i = blas::matrix< value_t >( DA, blas::range::all, is_i - A.col_ofs() );
                auto  A_i  = Hpro::TDenseMatrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
                multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
            }// if
            else
            {
                auto  DA_i = blas::matrix< value_t >( DA, is_i - A.row_ofs(), blas::range::all );
                auto  A_i  = Hpro::TDenseMatrix< value_t >( A.row_is( op_A ), is_i, DA_i );
            
                multiply( alpha, op_A, A_i, op_B, *B_ij, C_j );
            }// else
        }// for

        C.add_block( value_t(1), value_t(1), &C_j );
    }// for
}

//
// dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  D = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is() );

    if ( op_A == apply_normal )
    {
        // D = A×B
        multiply( alpha, blas::mat( A ), op_B, B, blas::mat( D ) );

        std::scoped_lock  lock( C.mutex() );
    
        // D = D + C
        hlr::add< value_t >( value_t(1), C, D );

        // approximate result and update C
        auto [ U, V ] = approx( blas::mat( D ), acc );

        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

//
// dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  DA = blas::mat( A );
    auto  DB = blas::mat( B );
    auto  DT = blas::prod( alpha, blas::mat_view( op_A, DA ), blas::mat_view( op_B, DB ) );
    auto  T  = Hpro::TDenseMatrix< value_t >( C.row_is(), C.col_is(), std::move( DT ) );
        
    hlr::add< value_t >( value_t(1), T, C, acc, approx );
}

//
// dense x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    blas::prod( alpha,
                blas::mat_view( op_A, blas::mat( A ) ),
                blas::mat_view( op_B, blas::mat( B ) ),
                value_t(1), blas::mat( C ) );
}

//
// dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = approx( C - A B )
    auto  AB = blas::prod( alpha,
                           blas::mat_view( op_A, blas::mat( A ) ),
                           blas::mat_view( op_B, blas::mat( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( value_t(1), blas::mat_U( C ), blas::adjoint( blas::mat_V( C ) ), value_t(1), AB );

    auto [ U, V ] = approx( AB, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, blas::mat( A ) ), blas::mat_U( B, op_B ) );
    auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), AU, blas::mat_V( B, op_B ) );

    hlr::add( alpha, RC, C, acc, approx );
}

//
// dense x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + ( A U(B) ) V(B)^H
    auto  AU = blas::prod( value_t(1), blas::mat_view( op_A, blas::mat( A ) ), blas::mat_U( B, op_B ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, AU, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
}

//
// dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), A U(B) ] , [ V(C), V(B) ] )
    auto  AU = blas::prod( alpha,
                           blas::mat_view( op_A, blas::mat( A ) ),
                           blas::mat_U( B, op_B ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), AU },
                            { blas::mat_V( C ), blas::mat_V( B, op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// lowrank x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), blas::mat_U( A, op_A ), VC );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// lowrank x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    // U·(V' × B) = U·X' with X = B'·V
    auto  VA = blas::mat_V( A, op_A );
    auto  X  = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, X );

    std::scoped_lock  lock( C.mutex() );

    // U·X' + C
    blas::prod( value_t(1), blas::mat_U( A, op_A ), blas::adjoint( X ), value_t(1), blas::mat( C ) );
}

//
// lowrank x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    auto  VA = blas::mat_V( A, op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), A.rank() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), blas::mat_U( A, op_A ) },
                            { blas::mat_V( C ), VC },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// lowrank x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( blas::adjoint( blas::mat_view( op_B, blas::mat( B ) ) ),
                           blas::mat_V( A, op_A ) );

    auto  RC = Hpro::TRkMatrix< value_t >( C.row_is(), C.col_is(), blas::mat_U( A, op_A ), VB );

    hlr::add( alpha, RC, C, acc, approx );
}

//
// lowrank x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H B )
    auto  VB = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_view( op_B, blas::mat( B ) ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, blas::mat_U( A, op_A ), VB, value_t(1), blas::mat( C ) );
}

//
// lowrank x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) ] , [ V(C), (V(A)^H B)^H ] )
    auto  VB = blas::prod( alpha,
                           blas::adjoint( blas::mat_view( op_B, blas::mat( B ) ) ),
                           blas::mat_V( A, op_A ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), blas::mat_U( A, op_A ) },
                            { blas::mat_V( C ), VB },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// lowrank x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U( A, op_A ), T );
    auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( C.row_is(), C.col_is(), UT, blas::mat_V( B, op_B ) );
        
    hlr::add< value_t >( value_t(1), *R, C, acc, approx );
}

//
// lowrank x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + U(A) ( V(A)^H U(B) ) V(B)^H
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
    auto  UT = blas::prod( value_t(1), blas::mat_U( A, op_A ), T );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, UT, blas::adjoint( blas::mat_V( B, op_B ) ), value_t(1), blas::mat( C ) );
}

//
// lowrank x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;
    
    // [ U(C), V(C) ] = truncate( [ U(C), U(A) V(A)^H U(B) ] , [ V(C), V(B)^H ] )
    auto  T  = blas::prod( value_t(1), blas::adjoint( blas::mat_V( A, op_A ) ), blas::mat_U( B, op_B ) );
    auto  UT = blas::prod(      alpha, blas::mat_U( A, op_A ), T );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), UT },
                            { blas::mat_V( C ), blas::mat_V( B, op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

}// namespace hlr

/*
#!/usr/bin/env python3
#
# generates matrix multiplication function placeholders
#

# type abbreviations
types   = [ 'blocked', 'dense', 'lowrank', 'uniform' ]

# actual matrix types
classes = { 'blocked' : 'Hpro::TBlockMatrix< value_t >',
            'lowrank' : 'Hpro::TRkMatrix< value_t >',
            'dense'   : 'Hpro::TDenseMatrix< value_t >',
            'uniform' : 'matrix::uniform_lrmatrix< value_t >' }

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
                print( '           const Hpro::TTruncAcc &                         acc,' )
                print( '           const approx_t &                                approx )' )
            else :
                print( '           {:<46}  C )'.format( classes[C] + ' &' ) )

            print( '{' )
            print( '    HLR_ERROR( "todo" );' )
            print( '}' )
            print()

*/

#endif // __HLR_ARITH_DETAIL_MULTIPLY_HH
