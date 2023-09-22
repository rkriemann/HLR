#ifndef __HLR_ARITH_DETAIL_ADD_HH
#define __HLR_ARITH_DETAIL_ADD_HH
//
// Project     : HLR
// Module      : arith/add
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/matrix/lrmatrix.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/matrix/lrsvmatrix.hh"
#include "hlr/matrix/dense_matrix.hh"
#include "hlr/utils/log.hh"

namespace hlr
{

#if defined(NDEBUG)
#  define HLR_ADD_PRINT( msg )   
#else
#  define HLR_ADD_PRINT( msg )   HLR_LOG( 4, msg )
#endif


/////////////////////////////////////////////////////////////////////////////////
//
// compute C := C + α A with different types of A/C
//
/////////////////////////////////////////////////////////////////////////////////

//
// forward decl.
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C,
      const accuracy &                  acc,
      const approx_t &                  approx );

template < typename value_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C );

//
// blocked + blocked
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const Hpro::TBlockMatrix< value_t > &  A,
      Hpro::TBlockMatrix< value_t > &        C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    HLR_ASSERT(( A.block_rows() == C.nblock_rows() ) &&
               ( A.block_cols() == C.nblock_cols() ));

    if ( alpha == value_t(0) )
        return;

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            if ( is_null( A.block( i, j ) ) )
                continue;

            HLR_ASSERT( ! is_null( C.block( i, j ) ) );

            add< value_t >( alpha, * A.block( i, j ), * C.block( i, j ), acc, approx );
        }// for
    }// for
}

//
// blocked + lowrank
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const Hpro::TBlockMatrix< value_t > &  A,
      matrix::lrmatrix< value_t > &          C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // collect low-rank sub blocks of A and convert other blocks
    // into low-rank format
    //

    auto    lr_blocks = std::list< const matrix::lrmatrix< value_t > * >();
    auto    created   = std::list< std::unique_ptr< matrix::lrmatrix< value_t > > >();
    size_t  rank_A    = 0;

    for ( uint  i = 0; i < A.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < A.nblock_cols(); ++j )
        {
            const auto  A_ij = A.block( i, j );
            
            if ( ! is_null( A_ij ) )
            {
                if ( matrix::is_lowrank( A_ij ) )
                {
                    const auto  R_ij = cptrcast( A_ij, matrix::lrmatrix< value_t > );

                    if ( R_ij->rank() > 0 )
                    {
                        rank_A += R_ij->rank();
                        lr_blocks.push_back( R_ij );
                    }// if
                }// if
                else if ( matrix::is_dense( A_ij ) )
                {
                    auto  D_ij     = cptrcast( A_ij, matrix::dense_matrix< value_t > );
                    auto  M        = blas::copy( D_ij->mat() );
                    auto  [ U, V ] = approx( M, acc );

                    if ( U.ncols() > 0 )
                    {
                        auto  R_ij = std::make_unique< matrix::lrmatrix< value_t > >( A_ij->row_is(), A_ij->col_is(), std::move( U ), std::move( V ) );
                        
                        rank_A += R_ij->rank();
                        lr_blocks.push_back( R_ij.get() );
                        created.push_back( std::move( R_ij ) );
                    }// if
                }// if
                else
                {
                    HLR_ERROR( "not implemented" );
                    // auto  R_ij = to_rank( A_ij, acc, approx );

                    // if ( R_ij->rank() > 0 )
                    // {
                    //     rank_A += R_ij->rank();
                    //     lr_blocks.push_back( R_ij.get() );
                    //     created.push_back( std::move( R_ij ) );
                    // }// if
                }// else
            }// if
        }// for
    }// for

    if ( rank_A == 0 )
        return;
        
    //
    // combine C with all low-rank blocks into single low-rank matrix and truncate
    //

    auto  lock = std::scoped_lock( C.mutex() );

    if ( C.rank() > 0 )
        lr_blocks.push_back( &C );
        
    auto  U    = blas::matrix< value_t >( C.nrows(), rank_A + C.rank() );
    auto  V    = blas::matrix< value_t >( C.ncols(), rank_A + C.rank() );
    uint  pos  = 0;

    for ( auto  R_i : lr_blocks )
    {
        auto  U_i = blas::matrix< value_t >( U, R_i->row_is() - C.row_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );
        auto  V_i = blas::matrix< value_t >( V, R_i->col_is() - C.col_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );

        blas::copy( R_i->U(), U_i );
        blas::copy( R_i->V(), V_i );

        pos += R_i->rank();
    }// for

    auto [ W, X ] = approx( U, V, acc );

    C.set_lrmat( std::move( W ), std::move( X ), acc );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const Hpro::TBlockMatrix< value_t > &  A,
      matrix::lrsvmatrix< value_t > &        C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // collect low-rank sub blocks of A and convert other blocks
    // into low-rank format
    //

    auto    lr_blocks = std::list< const matrix::lrmatrix< value_t > * >();
    auto    created   = std::list< std::unique_ptr< matrix::lrmatrix< value_t > > >();
    size_t  rank_A    = 0;

    for ( uint  i = 0; i < A.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < A.nblock_cols(); ++j )
        {
            const auto  A_ij = A.block( i, j );
            
            if ( ! is_null( A_ij ) )
            {
                if ( matrix::is_lowrank( A_ij ) )
                {
                    const auto  R_ij = cptrcast( A_ij, matrix::lrmatrix< value_t > );

                    if ( R_ij->rank() > 0 )
                    {
                        rank_A += R_ij->rank();
                        lr_blocks.push_back( R_ij );
                    }// if
                }// if
                else if ( matrix::is_lowrank_sv( A_ij ) )
                {
                    const auto  R_ij = cptrcast( A_ij, matrix::lrsvmatrix< value_t > );

                    if ( R_ij->rank() > 0 )
                    {
                        auto  U = blas::copy( R_ij->U() );
                        auto  V = blas::copy( R_ij->V() );

                        blas::prod_diag_ip( U, R_ij->S() );
                        
                        auto  T_ij = std::make_unique< matrix::lrmatrix< value_t > >( A_ij->row_is(), A_ij->col_is(), std::move( U ), std::move( V ) );
                        
                        rank_A += T_ij->rank();
                        lr_blocks.push_back( T_ij.get() );
                        created.push_back( std::move( T_ij ) );
                    }// if
                }// if
                else if ( matrix::is_dense( A_ij ) )
                {
                    auto  D_ij     = cptrcast( A_ij, matrix::dense_matrix< value_t > );
                    auto  M        = blas::copy( D_ij->mat() );
                    auto  [ U, V ] = approx( M, acc );

                    if ( U.ncols() > 0 )
                    {
                        auto  T_ij = std::make_unique< matrix::lrmatrix< value_t > >( A_ij->row_is(), A_ij->col_is(), std::move( U ), std::move( V ) );
                        
                        rank_A += T_ij->rank();
                        lr_blocks.push_back( T_ij.get() );
                        created.push_back( std::move( T_ij ) );
                    }// if
                }// if
                else
                {
                    HLR_ERROR( "not implemented" );
                    // auto  R_ij = to_rank( A_ij, acc, approx );

                    // if ( R_ij->rank() > 0 )
                    // {
                    //     rank_A += R_ij->rank();
                    //     lr_blocks.push_back( R_ij.get() );
                    //     created.push_back( std::move( R_ij ) );
                    // }// if
                }// else
            }// if
        }// for
    }// for

    if ( rank_A == 0 )
        return;
        
    //
    // combine C with all low-rank blocks into single low-rank matrix and truncate
    //

    auto  lock = std::scoped_lock( C.mutex() );
    auto  U    = blas::matrix< value_t >( C.nrows(), rank_A + C.rank() );
    auto  V    = blas::matrix< value_t >( C.ncols(), rank_A + C.rank() );
    uint  pos  = 0;

    if ( C.rank() > 0 )
    {
        auto  U_i = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + C.rank() - 1 ) );
        auto  V_i = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + C.rank() - 1 ) );
        auto  CUS = blas::prod_diag( C.U(), C.S() );

        blas::copy( CUS,   U_i );
        blas::copy( C.V(), V_i );

        pos += C.rank();
    }// if
        
    for ( auto  R_i : lr_blocks )
    {
        auto  U_i = blas::matrix< value_t >( U, R_i->row_is() - C.row_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );
        auto  V_i = blas::matrix< value_t >( V, R_i->col_is() - C.col_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );

        blas::copy( R_i->U(), U_i );
        blas::copy( R_i->V(), V_i );

        pos += R_i->rank();
    }// for

    auto [ W, T, X ] = approx.approx_ortho( U, V, acc );

    C.set_lrmat( std::move( W ), std::move( T ), std::move( X ), acc );
}

//
// lowrank + blocked
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      Hpro::TBlockMatrix< value_t > &      C,
      const accuracy &                     acc,
      const approx_t &                     approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if (( alpha == value_t(0) ) || ( A.rank() == 0 ))
        return;
    
    //
    // restrict low-rank matrix to sub blocks and recurse
    //

    auto  UA = A.U();
    auto  VA = A.V();

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            const auto  C_ij = C.block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            auto  U_i  = blas::matrix< value_t >( UA, C_ij->row_is() - C.row_ofs(), blas::range::all );
            auto  V_j  = blas::matrix< value_t >( VA, C_ij->col_is() - C.col_ofs(), blas::range::all );
            auto  A_ij = matrix::lrmatrix< value_t >( C_ij->row_is(), C_ij->col_is(), U_i, V_j );

            add( alpha, A_ij, *C_ij, acc, approx );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const matrix::lrsvmatrix< value_t > &  A,
      Hpro::TBlockMatrix< value_t > &        C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if (( alpha == value_t(0) ) || ( A.rank() == 0 ))
        return;
    
    //
    // restrict low-rank matrix to sub blocks and recurse
    //

    auto  UA = A.U();
    auto  SA = A.S();
    auto  VA = A.V();

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            const auto  C_ij = C.block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            auto  U_i  = blas::matrix< value_t >( UA, C_ij->row_is() - C.row_ofs(), blas::range::all );
            auto  V_j  = blas::matrix< value_t >( VA, C_ij->col_is() - C.col_ofs(), blas::range::all );
            auto  A_ij = matrix::lrsvmatrix< value_t >( C_ij->row_is(), C_ij->col_is(), U_i, SA, V_j );

            add( alpha, A_ij, *C_ij, acc, approx );
        }// for
    }// for
}

//
// blocked + dense
//
template < typename value_t >
void
add ( const value_t                          alpha,
      const Hpro::TBlockMatrix< value_t > &  A,
      matrix::dense_matrix< value_t > &      C,
      const accuracy &                       acc )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // recurse for each block of A with corresponding virtual block of C
    //

    auto        lock           = std::scoped_lock( C.mutex() );
    auto        Cm             = C.mat();
    const auto  was_compressed = C.is_compressed();

    for ( uint  i = 0; i < A.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < A.nblock_cols(); ++j )
        {
            const auto  A_ij = A.block( i, j );

            if ( is_null( A_ij ) )
                continue;

            auto  D_ij = blas::matrix< value_t >( Cm,
                                                  A_ij->row_is() - C.row_ofs(), 
                                                  A_ij->col_is() - C.col_ofs() );
            auto  C_ij = matrix::dense_matrix( A_ij->row_is(), A_ij->col_is(), D_ij );

            add< value_t >( alpha, *A_ij, C_ij );
        }// for
    }// for

    if ( was_compressed )
        C.set_matrix( std::move( Cm ), acc );
}

//
// dense + blocked
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      Hpro::TBlockMatrix< value_t > &          C,
      const accuracy &                         acc,
      const approx_t &                         approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // recurse for each block of C with corresponding virtual block of A
    //

    auto  DA = A.mat();

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            const auto  C_ij = C.block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            auto  D_ij = blas::matrix< value_t >( DA,
                                                  C_ij->row_is() - C.row_ofs(), 
                                                  C_ij->col_is() - C.col_ofs() );
            auto  A_ij = matrix::dense_matrix( C_ij->row_is(), C_ij->col_is(), D_ij );

            add< value_t >( alpha, A_ij, *C_ij, acc, approx );
        }// for
    }// for
}

//
// lowrank + lowrank
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      matrix::lrmatrix< value_t > &        C,
      const accuracy &                     acc,
      const approx_t &                     approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  UA   = A.U();
    auto  VA   = A.V();
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  sUA = blas::copy( UA );

        blas::scale( alpha, sUA );

        auto [ U, V ] = approx( { sUA, C.U() },
                                { VA,  C.V() },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ), acc );
    }// if
    else
    {
        auto [ U, V ] = approx( { UA, C.U() },
                                { VA, C.V() },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ), acc );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      matrix::lrsvmatrix< value_t > &      C,
      const accuracy &                     acc,
      const approx_t &                     approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  CUS  = blas::prod_diag( C.U(), C.S() );
    auto  UA   = A.U();
    auto  VA   = A.V();
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  sUA = blas::copy( UA );

        blas::scale( alpha, sUA );

        auto [ U, S, V ] = approx.approx_ortho( { sUA, CUS },
                                                { VA,  C.V() },
                                                acc );
        
        C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
    }// if
    else
    {
        auto [ U, S, V ] = approx.approx_ortho( { UA, CUS },
                                                { VA, C.V() },
                                                acc );
        
        C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const matrix::lrsvmatrix< value_t > &  A,
      matrix::lrmatrix< value_t > &          C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  UA   = blas::prod_diag( A.U(), A.S() );
    auto  VA   = A.V();
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  sUA = blas::copy( UA );

        blas::scale( alpha, sUA );

        auto [ U, V ] = approx( { sUA, C.U() },
                                { VA,  C.V() },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ), acc );
    }// if
    else
    {
        auto [ U, V ] = approx( { UA, C.U() },
                                { VA, C.V() },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ), acc );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                          alpha,
      const matrix::lrsvmatrix< value_t > &  A,
      matrix::lrsvmatrix< value_t > &        C,
      const accuracy &                       acc,
      const approx_t &                       approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  CUS  = blas::prod_diag( C.U(), C.S() );
    auto  UA   = blas::prod_diag( A.U(), A.S() );
    auto  VA   = A.V();
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  sUA = blas::copy( UA );

        blas::scale( alpha, sUA );

        auto [ U, S, V ] = approx.approx_ortho( { sUA, CUS },
                                                { VA,  C.V() },
                                                acc );
        
        C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
    }// if
    else
    {
        auto [ U, S, V ] = approx.approx_ortho( { UA, CUS },
                                                { VA, C.V() },
                                                acc );
        
        C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                         alpha,
      const matrix::lrsmatrix< value_t > &  A,
      matrix::lrmatrix< value_t > &         C,
      const accuracy &                      acc,
      const approx_t &                      approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A)·S(A) ] , [ V(C), V(A) ] )
    auto  US = blas::prod( A.U(), A.S() );

    blas::scale( alpha, US );

    auto [ U, V ] = approx( {    US, C.U() },
                            { A.V(), C.V() },
                            acc );

    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      matrix::lrsmatrix< value_t > &       C,
      const accuracy &                     acc,
      const approx_t &                     approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    auto  WT = blas::prod( C.U(), C.S() );
    
    if ( alpha != value_t(1) )
    {
        auto  UA = blas::copy( blas::mat_U( A ) );

        blas::scale( alpha, UA );

        auto  [ U, V ] = approx( {                          UA, WT },
                                 { blas::mat_V( A ), C.V() },
                                 acc );
        auto  I        = blas::eye< value_t >( U.ncols(), V.ncols() );
        
        C.set_lrmat( std::move( U ), std::move( I ), std::move( V ) );
    }// if
    else
    {
        auto [ U, V ] = approx( { blas::mat_U( A ), WT },
                                { blas::mat_V( A ), C.V() },
                                acc );
        auto  I        = blas::eye< value_t >( U.ncols(), V.ncols() );
        
        C.set_lrmat( std::move( U ), std::move( I ), std::move( V ) );
    }// else
}

//
// dense + lowrank
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      matrix::lrmatrix< value_t > &            C,
      const accuracy &                         acc,
      const approx_t &                         approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  TA   = blas::copy( A.mat() );

    blas::prod( value_t(1), C.U(), blas::adjoint( C.V() ), alpha, TA );

    auto [ U, V ] = approx( TA, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      matrix::lrsvmatrix< value_t > &          C,
      const accuracy &                         acc,
      const approx_t &                         approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  TA   = blas::copy( A.mat() );
    auto  US   = blas::prod_diag( C.U(), C.S() );

    blas::prod( value_t(1), US, blas::adjoint( C.V() ), alpha, TA );

    auto [ U, S, V ] = approx.approx_ortho( TA, acc );
        
    C.set_lrmat( std::move( U ), std::move( S ), std::move( V ), acc );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      matrix::lrsmatrix< value_t > &           C,
      const accuracy &                         acc,
      const approx_t &                         approx )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  TA   = blas::copy( blas::mat( A ) );
    auto  US   = blas::prod( C.U(), C.S() );

    blas::prod( value_t(1), US, blas::adjoint( C.V() ), alpha, TA );

    auto  [ U, V ] = approx( TA, acc );
    auto  I        = blas::eye< value_t >( U.ncols(), V.ncols() );
        
    C.set_lrmat( std::move( U ), std::move( I ), std::move( V ) );
}

//
// lowrank + dense
//
template < typename value_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      matrix::dense_matrix< value_t > &    C )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto  lock = std::scoped_lock( C.mutex() );
    auto  UA   = A.U();
    auto  VA   = A.V();

    HLR_ASSERT( ! C.is_compressed() );

    auto  Cm = C.mat();
    
    blas::prod( alpha, UA, blas::adjoint( VA ), value_t(1), Cm );
}

template < typename value_t >
void
add ( const value_t                        alpha,
      const matrix::lrmatrix< value_t > &  A,
      matrix::dense_matrix< value_t > &    C,
      const accuracy &                     acc )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto        lock = std::scoped_lock( C.mutex() );
    auto        UA   = A.U();
    auto        VA   = A.V();
    auto        Cm   = C.mat();
    const auto  was_compressed = C.is_compressed();
    
    blas::prod( alpha, UA, blas::adjoint( VA ), value_t(1), Cm );

    if ( was_compressed )
        C.set_matrix( std::move( Cm ), acc );
}

template < typename value_t >
void
add ( const value_t                          alpha,
      const matrix::lrsvmatrix< value_t > &  A,
      matrix::dense_matrix< value_t > &      C )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto  UA   = blas::prod_diag( A.U(), A.S() );
    auto  VA   = A.V();

    auto  lock = std::scoped_lock( C.mutex() );

    HLR_ASSERT( ! C.is_compressed() );

    auto  Cm = C.mat();
    
    blas::prod( alpha, UA, blas::adjoint( VA ), value_t(1), Cm );
}

template < typename value_t >
void
add ( const value_t                          alpha,
      const matrix::lrsvmatrix< value_t > &  A,
      matrix::dense_matrix< value_t > &      C,
      const accuracy &                       acc )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto        UA   = blas::prod_diag( A.U(), A.S() );
    auto        VA   = A.V();

    auto        lock = std::scoped_lock( C.mutex() );
    const auto  was_compressed = C.is_compressed();
    
    auto  Cm = C.mat();
    
    blas::prod( alpha, UA, blas::adjoint( VA ), value_t(1), Cm );

    if ( was_compressed )
        C.set_matrix( std::move( Cm ), acc );
}

//
// dense + dense
//
template < typename value_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      matrix::dense_matrix< value_t > &        C,
      const accuracy &                         acc )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    auto        lock = std::scoped_lock( C.mutex() );
    auto        Cm   = C.mat();
    const auto  was_compressed = C.is_compressed();
    
    // C = C + α A
    
    blas::add( alpha, A.mat(), Cm );

    if ( was_compressed )
        C.set_matrix( std::move( Cm ), acc );
}

template < typename value_t >
void
add ( const value_t                            alpha,
      const matrix::dense_matrix< value_t > &  A,
      matrix::dense_matrix< value_t > &        C )
{
    HLR_ADD_PRINT( Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    auto  lock = std::scoped_lock( C.mutex() );
    auto  Cm   = C.mat();

    HLR_ASSERT( ! C.is_compressed() );
    
    // C = C + α A
    blas::add( alpha, A.mat(), Cm );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_ADD_HH
