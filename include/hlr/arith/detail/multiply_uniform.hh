#ifndef __HLR_ARITH_DETAIL_MULTIPLY_UNIFORM_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_UNIFORM_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions with uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

namespace hlr {

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const Hpro::matop_t                          op_A,
           const matrix::uniform_lrmatrix< value_t > &  A,
           const blas::matrix< value_t > &              B,
           blas::matrix< value_t > &                    C )
{
    HLR_ASSERT(( op_A == apply_normal ) || ( op_A == apply_adjoint ));
    
    auto  VB  = blas::prod( blas::adjoint( A.col_basis( op_A ) ), B );
    auto  SVB = blas::prod( blas::mat_view( op_A, A.coeff() ), VB );
    
    blas::prod( alpha, A.row_basis( op_A ), SVB, value_t(1), C );
}

template < typename value_t >
void
multiply ( const value_t                                alpha,
           const blas::matrix< value_t > &              A,
           const Hpro::matop_t                          op_B,
           const matrix::uniform_lrmatrix< value_t > &  B,
           blas::matrix< value_t > &                    C )
{
    HLR_ASSERT(( op_B == apply_normal ) || ( op_B == apply_adjoint ));

    // A × U·S·V'
    auto  AU  = blas::prod( A, B.row_basis( op_B ) );
    auto  AUS = blas::prod( AU, blas::mat_view( op_B, B.coeff() ) );
    
    blas::prod( alpha, AUS, B.col_basis( op_B ), value_t(1), C );
}

//
// blocked x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // (A × U)·S·V'
    auto  UB = B.row_basis( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), B.ncols() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  S  = blas::copy( blas::mat_view( op_B, B.coeff() ) );
    auto  RC = matrix::lrsmatrix< value_t >( C.row_is(), C.col_is(), UC, S, B.col_basis( op_B ) );
    
    hlr::add< value_t >( value_t(1), RC, C, acc, approx );
}

//
// blocked x uniform = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    // (A × U)·S·V'
    auto  UB = B.row_basis( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), UB.ncols() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  UxS = blas::prod( UC, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );

    blas::prod( value_t(1), UxS, blas::adjoint( B.col_basis( op_B ) ), value_t(1), blas::mat( C ) );
}

//
// blocked x uniform = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // (A × U)·S·V'
    auto  UB = B.row_basis( op_B );
    auto  UC = blas::matrix< value_t >( C.nrows(), UB.ncols() );

    multiply< value_t >( alpha, op_A, A, UB, UC );

    auto  US      = blas::prod( UC, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ W, X ] = approx( {                  US, blas::mat_U( C ) },
                            { B.col_basis( op_B ), blas::mat_V( C ) },
                            acc );

    C.set_lrmat( std::move( W ), std::move( X ) );
}

//
// dense x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TDenseMatrix< value_t > &           /* A */,
           const Hpro::matop_t                             /* op_B */,
           const matrix::uniform_lrmatrix< value_t > &     /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// dense x uniform = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + (( A U ) S) V'
    auto  AU  = blas::prod( blas::mat_view( op_A, blas::mat( A ) ), B.row_basis( op_B ) );
    auto  AUS = blas::prod( AU, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, AUS, blas::adjoint( B.col_basis( op_B ) ), value_t(1), blas::mat( C ) );
}

//
// dense x uniform = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // C + A × B = C + ((A × U)·S)·V'
    auto  AU  = blas::prod( blas::mat_view( op_A, blas::mat( A ) ), B.row_basis( op_B ) );
    auto  AUS = blas::prod( alpha, AU, blas::mat_view( op_B, B.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), AUS },
                            { blas::mat_V( C ), B.col_basis( op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// lowrank x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TRkMatrix< value_t > &              /* A */,
           const Hpro::matop_t                             /* op_B */,
           const matrix::uniform_lrmatrix< value_t > &     /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x uniform = dense
//
template < typename value_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TRkMatrix< value_t > &              /* A */,
           const Hpro::matop_t                             /* op_B */,
           const matrix::uniform_lrmatrix< value_t > &     /* B */,
           Hpro::TDenseMatrix< value_t > &                 /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x uniform = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const Hpro::TRkMatrix< value_t > &              /* A */,
           const Hpro::matop_t                             /* op_B */,
           const matrix::uniform_lrmatrix< value_t > &     /* B */,
           Hpro::TRkMatrix< value_t > &                    /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TBlockMatrix< value_t > &           /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;

    // U·S·(V' × B) = U·S·VC' as B' × V = VC
    auto  VA = A.col_basis( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), VA.ncols() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  UxS = blas::prod( A.row_basis( op_A ), blas::mat_view( op_A, A.coeff() ) );

    std::scoped_lock  lock( C.mutex() );

    blas::prod( value_t(1), UxS, blas::adjoint( VC ), value_t(1), blas::mat( C ) );
}

//
// uniform x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // U·S·(V' × B) as B' × V
    auto  VA = A.col_basis( op_A );
    auto  VC = blas::matrix< value_t >( C.ncols(), VA.ncols() );

    multiply< value_t >( alpha, blas::adjoint( op_B ), B, VA, VC );

    auto  VxS     = blas::prod( VC, blas::mat_view( blas::adjoint( op_A ), A.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ W, X ] = approx( { A.row_basis( op_A ), blas::mat_U( C ) },
                            {                 VxS, blas::mat_V( C ) },
                            acc );

    C.set_lrmat( std::move( W ), std::move( X ) );
}

//
// uniform x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TDenseMatrix< value_t > &           /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + U·(S·(V'×B))
    auto  VB  = blas::prod( blas::adjoint( A.col_basis( op_A ) ), blas::mat_view( op_B, blas::mat( B ) ) );
    auto  SVB = blas::prod( blas::mat_view( op_A, A.coeff() ), VB );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, A.row_basis( op_A ), SVB, value_t(1), blas::mat( C ) );
}

//
// uniform x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // C + A × B = C + U·(S·(V' × B)) -> (B' × V)·S'
    auto  BV  = blas::prod( blas::mat_view( blas::adjoint( op_B ), blas::mat( B ) ), A.col_basis( op_A ) );
    auto  BVS = blas::prod( alpha, BV, blas::mat_view( blas::adjoint( op_A ), A.coeff() ) );

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), A.row_basis( op_A ) },
                            { blas::mat_V( C ), BVS },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

//
// uniform x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TRkMatrix< value_t > &              /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TRkMatrix< value_t > &              /* B */,
           Hpro::TDenseMatrix< value_t > &                 /* C */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const Hpro::TRkMatrix< value_t > &              /* B */,
           Hpro::TRkMatrix< value_t > &                    /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   /* alpha */,
           const Hpro::matop_t                             /* op_A */,
           const matrix::uniform_lrmatrix< value_t > &     /* A */,
           const Hpro::matop_t                             /* op_B */,
           const matrix::uniform_lrmatrix< value_t > &     /* B */,
           Hpro::TBlockMatrix< value_t > &                 /* C */,
           const Hpro::TTruncAcc &                         /* acc */,
           const approx_t &                                /* approx */ )
{
    HLR_ERROR( "todo" );
}

//
// uniform x uniform = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    // C = C + A×B = C + (U·((S·(V' × W))·T))·X'
    auto  VW    = blas::prod( blas::adjoint( A.col_basis( op_A ) ), B.row_basis( op_B ) );
    auto  SVW   = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );
    auto  SVWT  = blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) );
    auto  USVWT = blas::prod( A.row_basis( op_A ), SVWT );

    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha, USVWT, blas::adjoint( B.col_basis( op_B ) ), value_t(1), blas::mat( C ) );
}

//
// uniform x uniform = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::uniform_lrmatrix< value_t > &     A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TRkMatrix< value_t > &                    C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_MULT_PRINT;

    // C + A×B = C + (U·((S·(V' × W))·T))·X' = C + T·X'
    auto  T = blas::matrix< value_t >();

    {
        auto  VW    = blas::prod( blas::adjoint( A.col_basis( op_A ) ), B.row_basis( op_B ) );
        auto  SVW   = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );
        auto  SVWT  = blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) );

        T = std::move( blas::prod( alpha, A.row_basis( op_A ), SVWT ) );
    }

    std::scoped_lock  lock( C.mutex() );
    
    auto [ U, V ] = approx( { blas::mat_U( C ), T },
                            { blas::mat_V( C ), B.col_basis( op_B ) },
                            acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_UNIFORM_HH
