#ifndef __HLR_ARITH_ADD_HH
#define __HLR_ARITH_ADD_HH
//
// Project     : HLib
// File        : add.hh
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/utils/log.hh"

namespace hlr
{

namespace hpro = HLIB;

//
// forward decl.
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TMatrix &    A,
      hpro::TMatrix &          C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx );

//
// compute C := C + α A with different types of A/C
//

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TBlockMatrix &  A,
      hpro::TBlockMatrix &        C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

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

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TBlockMatrix &  A,
      hpro::TRkMatrix &           C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // collect low-rank sub blocks of A and convert other blocks
    // into low-rank format
    //

    std::list< const hpro::TRkMatrix * >             lr_blocks;
    std::list< std::unique_ptr< hpro::TRkMatrix > >  created;
    size_t                                           rank_A = 0;

    for ( uint  i = 0; i < A.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < A.nblock_cols(); ++j )
        {
            const auto  A_ij = A.block( i, j );
            
            if ( ! is_null( A_ij ) )
            {
                if ( is_lowrank( A_ij ) )
                {
                    const auto  R_ij = cptrcast( A_ij, hpro::TRkMatrix );

                    if ( R_ij->rank() > 0 )
                    {
                        rank_A += R_ij->rank();
                        lr_blocks.push_back( R_ij );
                    }// if
                }// if
                else if ( is_dense( A_ij ) )
                {
                    auto  D_ij     = cptrcast( A_ij, hpro::TDenseMatrix );
                    auto  M        = blas::copy( hpro::blas_mat< value_t >( D_ij ) );
                    auto  [ U, V ] = approx( M, acc );

                    if ( U.ncols() > 0 )
                    {
                        auto  R_ij = std::make_unique< hpro::TRkMatrix >( A_ij->row_is(), A_ij->col_is(),
                                                                          std::move( U ), std::move( V ) );
                        
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

    std::scoped_lock  lock( C.mutex() );
    
    if ( C.rank() > 0 )
        lr_blocks.push_back( &C );
        
    blas::matrix< value_t >  U( C.nrows(), rank_A + C.rank() );
    blas::matrix< value_t >  V( C.ncols(), rank_A + C.rank() );
    uint                     pos = 0;

    for ( auto  R_i : lr_blocks )
    {
        auto  U_i = blas::matrix< value_t >( U, R_i->row_is() - C.row_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );
        auto  V_i = blas::matrix< value_t >( V, R_i->col_is() - C.col_ofs(), blas::range( pos, pos + R_i->rank() - 1 ) );

        blas::copy( blas::mat_U< value_t >( R_i ), U_i );
        blas::copy( blas::mat_V< value_t >( R_i ), V_i );

        pos += R_i->rank();
    }// for

    auto [ U_trunc, V_trunc ] = approx( U, V, acc );

    C.set_lrmat( std::move( U_trunc ), std::move( V_trunc ) );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TRkMatrix &  A,
      hpro::TBlockMatrix &     C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if (( alpha == value_t(0) ) || ( A.rank() == 0 ))
        return;
    
    //
    // restrict low-rank matrix to sub blocks and recurse
    //

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            const auto  C_ij = C.block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            const auto  U_i  = blas::matrix< value_t >( blas::mat_U< value_t >( A ), C_ij->row_is() - C.row_ofs(), blas::range::all );
            const auto  V_j  = blas::matrix< value_t >( blas::mat_V< value_t >( A ), C_ij->col_is() - C.col_ofs(), blas::range::all );
            const auto  A_ij = hpro::TRkMatrix( C_ij->row_is(), C_ij->col_is(), U_i, V_j );

            add( alpha, A_ij, *C_ij, acc, approx );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TBlockMatrix &  A,
      hpro::TDenseMatrix &        C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // recurse for each block of A with corresponding virtual block of C
    //

    std::scoped_lock  lock( C.mutex() );
    
    for ( uint  i = 0; i < A.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < A.nblock_cols(); ++j )
        {
            const auto  A_ij = A.block( i, j );

            if ( is_null( A_ij ) )
                continue;

            auto  D_ij = blas::matrix< value_t >( hpro::blas_mat< value_t >( C ),
                                                  A_ij->row_is() - C.row_ofs(), 
                                                  A_ij->col_is() - C.col_ofs() );
            auto  C_ij = hpro::TDenseMatrix( A_ij->row_is(), A_ij->col_is(), D_ij );

            add< value_t >( alpha, *A_ij, C_ij, acc, approx );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TDenseMatrix &  A,
      hpro::TBlockMatrix &        C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );

    if ( alpha == value_t(0) )
        return;

    //
    // recurse for each block of C with corresponding virtual block of A
    //

    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            const auto  C_ij = C.block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            auto  D_ij = blas::matrix< value_t >( hpro::blas_mat< value_t >( A ),
                                                  C_ij->row_is() - C.row_ofs(), 
                                                  C_ij->col_is() - C.col_ofs() );
            auto  A_ij = hpro::TDenseMatrix( C_ij->row_is(), C_ij->col_is(), D_ij );

            add< value_t >( alpha, A_ij, *C_ij, acc, approx );
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TRkMatrix &  A,
      hpro::TRkMatrix &        C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    std::scoped_lock  lock( C.mutex() );
    
    // [ U(C), V(C) ] = truncate( [ U(C), α U(A) ] , [ V(C), V(A) ] )
    if ( alpha != value_t(1) )
    {
        auto  UA = blas::copy( hpro::blas_mat_A< value_t >( A ) );

        blas::scale( alpha, UA );

        auto [ U, V ] = approx( {                               UA, hpro::blas_mat_A< value_t >( C ) },
                                { hpro::blas_mat_B< value_t >( A ), hpro::blas_mat_B< value_t >( C ) },
                                acc );
        C.set_lrmat( std::move( U ), std::move( V ) );
    }// if
    else
    {
        auto [ U, V ] = approx( { hpro::blas_mat_A< value_t >( A ), hpro::blas_mat_A< value_t >( C ) },
                                { hpro::blas_mat_B< value_t >( A ), hpro::blas_mat_B< value_t >( C ) },
                                acc );
        
        C.set_lrmat( std::move( U ), std::move( V ) );
    }// else
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TDenseMatrix &  A,
      hpro::TRkMatrix &           C,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    if ( alpha == value_t(0) )
        return;
    
    std::scoped_lock  lock( C.mutex() );
    
    auto  TA = blas::copy( hpro::blas_mat< value_t >( A ) );

    blas::prod( alpha,
                hpro::blas_mat_A< value_t >( C ),
                blas::adjoint( hpro::blas_mat_B< value_t >( C ) ),
                value_t( 1 ),
                TA );

    auto [ U, V ] = approx( TA, acc );
        
    C.set_lrmat( std::move( U ), std::move( V ) );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TRkMatrix &  A,
      hpro::TDenseMatrix &     C,
      const hpro::TTruncAcc &,
      const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    std::scoped_lock  lock( C.mutex() );
    
    blas::prod( alpha,
                hpro::blas_mat_A< value_t >( A ),
                blas::adjoint( hpro::blas_mat_B< value_t >( A ) ),
                value_t( 1 ),
                hpro::blas_mat< value_t >( C ) );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t               alpha,
      const hpro::TDenseMatrix &  A,
      hpro::TDenseMatrix &        C,
      const hpro::TTruncAcc &,
      const approx_t & )
{
    HLR_LOG( 4, hpro::to_string( "add( %d, %d )", A.id(), C.id() ) );
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + α A
    blas::add( alpha, hpro::blas_mat< value_t >( A ), hpro::blas_mat< value_t >( C ) );
}

//
// semi-automatic deduction of optimal "add" function
//

template < typename value_t,
           typename approx_t,
           typename matrixA_t >
void
add ( const value_t            alpha,
      const matrixA_t &        A,
      hpro::TMatrix &          C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    if      ( is_blocked( C ) ) add< value_t, approx_t >( alpha, A, *ptrcast( &C, hpro::TBlockMatrix ), acc, approx );
    else if ( is_dense(   C ) ) add< value_t, approx_t >( alpha, A, *ptrcast( &C, hpro::TDenseMatrix ), acc, approx );
    else if ( is_lowrank( C ) ) add< value_t, approx_t >( alpha, A, *ptrcast( &C, hpro::TRkMatrix ),    acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + C.typestr() );
}

template < typename value_t,
           typename approx_t >
void
add ( const value_t            alpha,
      const hpro::TMatrix &    A,
      hpro::TMatrix &          C,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    if      ( is_blocked( A ) ) add< value_t, approx_t, hpro::TBlockMatrix >( alpha, *cptrcast( &A, hpro::TBlockMatrix ), C, acc, approx );
    else if ( is_dense(   A ) ) add< value_t, approx_t, hpro::TDenseMatrix >( alpha, *cptrcast( &A, hpro::TDenseMatrix ), C, acc, approx );
    else if ( is_lowrank( A ) ) add< value_t, approx_t, hpro::TRkMatrix >(    alpha, *cptrcast( &A, hpro::TRkMatrix ),    C, acc, approx );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_ADD_HH
