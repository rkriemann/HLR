#ifndef __HLR_ARITH_ADD_HH
#define __HLR_ARITH_ADD_HH
//
// Project     : HLR
// Module      : add.hh
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

// #include <hpro/algebra/mat_add.hh> // DEBUG
// #include <hpro/matrix/convert.hh> // DEBUG

#include "hlr/arith/detail/add.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/utils/log.hh"

namespace hlr
{

// to enable accuracy tests
// #define HLR_ADD_TESTS

//
// compute C := C + α A with different types of A/C
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C,
      const Hpro::TTruncAcc &           acc,
      const approx_t &                  approx )
{
    #if defined(HLR_ADD_TESTS)
    
    auto  TA = matrix::convert_to_hpro( A );
    auto  TC = matrix::convert_to_hpro( C );

    Hpro::add( alpha, TA.get(), value_t(1), TC.get(), acc );

    #endif
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( C ) )            add< value_t, approx_t >( alpha, *BA, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(    C ) ) add< value_t, approx_t >( alpha, *BA, *ptrcast( &C, matrix::lrmatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( C ) ) add< value_t, approx_t >( alpha, *BA, *ptrcast( &C, matrix::lrsvmatrix< value_t > ), acc, approx );
        else if ( matrix::is_dense(      C ) ) add< value_t >(           alpha, *BA, *ptrcast( &C, matrix::dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        auto  RA = cptrcast( &A, matrix::lrmatrix< value_t > );
        
        if      ( is_blocked( C ) )            add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(    C ) ) add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, matrix::lrmatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( C ) ) add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, matrix::lrsvmatrix< value_t > ), acc, approx );
        else if ( matrix::is_dense(      C ) ) add< value_t >(           alpha, *RA, *ptrcast( &C, matrix::dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( matrix::is_lowrank_sv( A ) )
    {
        auto  RA = cptrcast( &A, matrix::lrsvmatrix< value_t > );
        
        if      ( is_blocked( C ) )            add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(    C ) ) add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, matrix::lrmatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( C ) ) add< value_t, approx_t >( alpha, *RA, *ptrcast( &C, matrix::lrsvmatrix< value_t > ), acc, approx );
        else if ( matrix::is_dense(      C ) ) add< value_t >(           alpha, *RA, *ptrcast( &C, matrix::dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  DA = cptrcast( &A, matrix::dense_matrix< value_t > );
        
        if      ( is_blocked( C ) )            add< value_t, approx_t >( alpha, *DA, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(    C ) ) add< value_t, approx_t >( alpha, *DA, *ptrcast( &C, matrix::lrmatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( C ) ) add< value_t, approx_t >( alpha, *DA, *ptrcast( &C, matrix::lrsvmatrix< value_t > ), acc, approx );
        else if ( matrix::is_dense(      C ) ) add< value_t >(           alpha, *DA, *ptrcast( &C, matrix::dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if defined(HLR_ADD_TESTS)

    auto  TX = matrix::convert_to_hpro( C );
    auto  DX1 = Hpro::to_dense( TC.get() );
    auto  DX2 = Hpro::to_dense( TX.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( *DX1, "X1" );
        io::matlab::write( *DX2, "X2" );
        std::cout << Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if

    #endif
}

//
// general version without approximation
//
template < typename value_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C )
{
    #if defined(HLR_ADD_TESTS)
    
    auto  TA = matrix::convert_to_hpro( A );
    auto  TC = matrix::convert_to_hpro( C );

    Hpro::add( alpha, TA.get(), value_t(1), TC.get(), Hpro::acc_exact );

    #endif
    
    HLR_ASSERT( matrix::is_dense( C ) );

    auto  DC = ptrcast( &C, matrix::dense_matrix< value_t > );
    
    if      ( matrix::is_dense(      A ) ) add< value_t >( alpha, *cptrcast( &A, matrix::dense_matrix< value_t > ), *DC );
    else if ( matrix::is_lowrank(    A ) ) add< value_t >( alpha, *cptrcast( &A, matrix::lrmatrix< value_t > ),     *DC );
    else if ( matrix::is_lowrank_sv( A ) ) add< value_t >( alpha, *cptrcast( &A, matrix::lrsvmatrix< value_t > ),   *DC );
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if defined(HLR_ADD_TESTS)
    
    auto  TX = matrix::convert_to_hpro( C );
    auto  DX1 = Hpro::to_dense( TC.get() );
    auto  DX2 = Hpro::to_dense( TX.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( *DX1, "X1" );
        io::matlab::write( *DX2, "X2" );
        std::cout << Hpro::to_string( "add( %d, %d )", A.id(), C.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if
    
    #endif
}

//
// compute M := M + λI
//
template < typename value_t >
void
add_identity ( Hpro::TMatrix< value_t > &  M,
               const value_t &             λ )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            auto  B_ii = B->block( i, i );
            
            if ( ! is_null( B_ii ) )
                add_identity( *B_ii, λ );
        }// for
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D  = ptrcast( &M, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        HLR_ASSERT( M.row_is() == M.col_is() );
        HLR_ASSERT( ! D->is_compressed() );

        for ( uint  i = 0; i < std::min( DD.nrows(), DD.ncols() ); ++i )
            DD(i,i) += λ;
    }// if
    else
        HLR_ERROR( "todo" );
}

//
// compute M := M + d with d representing entries of diagonal matrix
//
template < typename value_t >
void
add_diag ( Hpro::TMatrix< value_t > &       M,
           const blas::vector< value_t > &  d )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            auto  B_ii = B->block( i, i );
            
            HLR_ASSERT( ! is_null( B_ii ) );

            auto  d_i = blas::vector< value_t >( d, B_ii->row_is() - M.row_ofs() );
                
            add_diag( *B_ii, d_i );
        }// for
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D  = ptrcast( &M, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        HLR_ASSERT( ! D->is_compressed() );

        for ( uint  i = 0; i < std::min( DD.nrows(), DD.ncols() ); ++i )
            DD(i,i) += d(i);
    }// if
    else
        HLR_ERROR( "todo" );
}

}// namespace hlr

#endif // __HLR_ARITH_ADD_HH
