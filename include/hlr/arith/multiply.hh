#ifndef __HLR_ARITH_MULTIPLY_HH
#define __HLR_ARITH_MULTIPLY_HH
//
// Project     : HLR
// Module      : multiply
// Description : matrix multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

// #include <hpro/algebra/mat_mul.hh> // DEBUG
// #include <hpro/matrix/convert.hh> // DEBUG

#include <hlr/utils/log.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/add.hh>
#include <hlr/arith/detail/norm.hh>
#include <hlr/approx/svd.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

// #include <hlr/matrix/convert.hh> // DEBUG
#include <hlr/seq/matrix.hh> // DEBUG

#include <hlr/arith/detail/multiply.hh>
#include <hlr/arith/detail/multiply_diag.hh>

namespace hlr
{

// to enable accuracy tests
// #define HLR_MULT_TESTS

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·B + C
//
/////////////////////////////////////////////////////////////////////////////////

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
           const approx_t &                  approx )
{
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;

    using namespace hlr::matrix;

    // test initial data
    // A.check_data();
    // B.check_data();
    // C.check_data();
    
    #if defined(HLR_MULT_TESTS)

    auto  TA = matrix::convert_to_hpro( A );
    auto  TB = matrix::convert_to_hpro( B );
    auto  TC = matrix::convert_to_hpro( C );

    Hpro::multiply( alpha, op_A, *TA, op_B, *TB, value_t(1), *TC, acc );

    #endif

    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
            
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, * BA, op_B, *BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, * BA, op_B, *BB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank(    C ) ) multiply< value_t, approx_t >( alpha, op_A, * BA, op_B, *BB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, * BA, op_B, *BB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank( B ) )
        {
            auto  RB = cptrcast( &B, lrmatrix< value_t > );
                
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank_sv( B ) )
        {
            auto  RB = cptrcast( &B, lrsvmatrix< value_t > );
            
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *BA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_uniform_lowrank( B ) )
        {
            auto  UB = cptrcast( &B, uniform_lrmatrix< value_t > );
                
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *UB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *UB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *BA, op_B, *UB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, dense_matrix< value_t > );
                
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *BA, op_B, *DB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *BA, op_B, *DB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        auto  RA = cptrcast( &A, lrmatrix< value_t > );
            
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );

            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank( B ) )
        {
            auto  RB = cptrcast( &B, lrmatrix< value_t > );
                    
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, dense_matrix< value_t > );
                    
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_lowrank_sv( A ) )
    {
        auto  RA = cptrcast( &A, lrsvmatrix< value_t > );
            
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );

            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *BB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank_sv( B ) )
        {
            auto  RB = cptrcast( &B, lrsvmatrix< value_t > );
                    
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, dense_matrix< value_t > );
                    
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *RA, op_B, *DB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_uniform_lowrank( A ) )
    {
        auto  UA = cptrcast( &A, uniform_lrmatrix< value_t > );
        
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
                
            if      ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *UA, op_B, *BB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *UA, op_B, *BB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_uniform_lowrank( B ) )
        {
            auto  UB = cptrcast( &B, uniform_lrmatrix< value_t > );
            
            if      ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *UA, op_B, *UB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *UA, op_B, *UB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, dense_matrix< value_t > );
                
            if      ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *UA, op_B, *DB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *UA, op_B, *DB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  DA = cptrcast( &A, dense_matrix< value_t > );
            
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
            
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *BB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank(    C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *BB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *DA, op_B, *BB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank( B ) )
        {
            auto  RB = cptrcast( &B, lrmatrix< value_t > );
                    
            if      ( is_blocked( C ) )         multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_lowrank_sv( B ) )
        {
            auto  RB = cptrcast( &B, lrsvmatrix< value_t > );
                    
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *DA, op_B, *RB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_uniform_lowrank( B ) )
        {
            auto  UB =  cptrcast( &B, matrix::uniform_lrmatrix< value_t > );
            
            if      ( matrix::is_lowrank( C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *UB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(   C ) ) multiply< value_t >(           alpha, op_A, *DA, op_B, *UB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, dense_matrix< value_t > );
            
            if      ( is_blocked( C ) )            multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank_sv( C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *DB, * ptrcast( &C, lrsvmatrix< value_t > ), acc, approx );
            else if ( matrix::is_lowrank(    C ) ) multiply< value_t, approx_t >( alpha, op_A, *DA, op_B, *DB, * ptrcast( &C, lrmatrix< value_t > ), acc, approx );
            else if ( matrix::is_dense(      C ) ) multiply< value_t >(           alpha, op_A, *DA, op_B, *DB, * ptrcast( &C, dense_matrix< value_t > ), acc );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    // test data in result
    // C.check_data();
    
    #if defined(HLR_MULT_TESTS)
    
    auto  TX = matrix::convert_to_hpro( C );
    auto  DX1 = Hpro::to_dense( TC.get() );
    auto  DX2 = Hpro::to_dense( TX.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( *DX1, "X1" );
        io::matlab::write( *DX2, "X2" );
        std::cout << Hpro::to_string( "multiply( %d, %d, %d )", A.id(), B.id(), C.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if

    #endif

    // // recompress after update
    // if ( compress::is_compressible( C ) )
    // {
    //     auto  lock = std::scoped_lock( C.mutex() );
        
    //     dynamic_cast< compressible * >( &C )->compress( acc );
    // }// if
}

template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C,
           const Hpro::TTruncAcc &           acc )
{
    auto  apx = approx::SVD< value_t >();

    multiply( alpha, op_A, A, op_B, B, C, acc, apx );
}

//
// version for matrix types without (!) approximation
//
template < typename value_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C )
{
    if (( alpha == value_t(0) ) || A.is_zero() || B.is_zero() )
        return;
    
    using namespace hlr::matrix;
    
    if ( ! matrix::is_dense( C ) )
        HLR_ERROR( "unsupported matrix type : " + C.typestr() );

    #if defined(HLR_MULT_TESTS)

    auto  TA = matrix::convert_to_hpro( A );
    auto  TB = matrix::convert_to_hpro( B );
    auto  TC = matrix::convert_to_hpro( C );

    Hpro::multiply( alpha, op_A, *TA, op_B, *TB, value_t(1), *TC, Hpro::acc_exact );

    #endif

    auto  DC = ptrcast( &C, matrix::dense_matrix< value_t > );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( B ) )                 multiply< value_t >( alpha, op_A, *BA, op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank_sv(      B ) ) multiply< value_t >( alpha, op_A, *BA, op_B, * cptrcast( &B, lrsvmatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank(         B ) ) multiply< value_t >( alpha, op_A, *BA, op_B, * cptrcast( &B, lrmatrix< value_t > ), *DC );
        else if ( matrix::is_uniform_lowrank( B ) ) multiply< value_t >( alpha, op_A, *BA, op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ), *DC );
        else if ( matrix::is_dense(           B ) ) multiply< value_t >( alpha, op_A, *BA, op_B, * cptrcast( &B, dense_matrix< value_t > ), *DC );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        auto  RA = cptrcast( &A, lrmatrix< value_t > );
        
        if      ( is_blocked( B ) )         multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank( B ) ) multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, lrmatrix< value_t > ), *DC );
        else if ( matrix::is_dense(   B ) ) multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, dense_matrix< value_t > ), *DC );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_lowrank_sv( A ) )
    {
        auto  RA = cptrcast( &A, lrsvmatrix< value_t > );
            
        if      ( is_blocked( B ) )            multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank_sv( B ) ) multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, lrsvmatrix< value_t > ), *DC );
        else if ( matrix::is_dense(      B ) ) multiply< value_t >( alpha, op_A, *RA, op_B, * cptrcast( &B, dense_matrix< value_t > ), *DC );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_uniform_lowrank( A ) )
    {
        auto  UA = cptrcast( &A, uniform_lrmatrix< value_t > );
        
        if      ( is_blocked( B ) )                 multiply< value_t >( alpha, op_A, *UA, op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ), *DC );
        else if ( matrix::is_uniform_lowrank( B ) ) multiply< value_t >( alpha, op_A, *UA, op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ), *DC );
        else if ( matrix::is_dense(           B ) ) multiply< value_t >( alpha, op_A, *UA, op_B, * cptrcast( &B, matrix::dense_matrix< value_t > ), *DC );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  DA = cptrcast( &A, dense_matrix< value_t > );
            
        if      ( is_blocked( B ) )            multiply< value_t >( alpha, op_A, *DA, op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank_sv( B ) ) multiply< value_t >( alpha, op_A, *DA, op_B, * cptrcast( &B, lrsvmatrix< value_t > ), *DC );
        else if ( matrix::is_lowrank(    B ) ) multiply< value_t >( alpha, op_A, *DA, op_B, * cptrcast( &B, lrmatrix< value_t > ), *DC );
        else if ( matrix::is_dense(      B ) ) multiply< value_t >( alpha, op_A, *DA, op_B, * cptrcast( &B, dense_matrix< value_t > ), *DC );
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    #if defined(HLR_MULT_TESTS)
    
    auto  TX = matrix::convert_to_hpro( C );
    auto  DX1 = Hpro::to_dense( TC.get() );
    auto  DX2 = Hpro::to_dense( TX.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( *DX1, "X1" );
        io::matlab::write( *DX2, "X2" );
        std::cout << Hpro::to_string( "multiply( %d, %d, %d )", A.id(), B.id(), C.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if

    #endif
}

//
// matrix multiplication with generation of result matrix
// - only supported for non-structured results
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B )
{
    std::unique_ptr< Hpro::TMatrix< value_t > >  C;

    if ( matrix::is_lowrank( A ) )
    {
        //
        // U·V' × B = W·X'
        //

        auto  RA = cptrcast( &A, matrix::lrmatrix< value_t > );
        auto  V  = std::move( RA->V( op_A ) );
        auto  W  = std::move( blas::copy( RA->U( op_A ) ) );
        auto  k  = RA->rank();
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), k );

        multiply( alpha, blas::adjoint( op_B ), B, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if
        
        return std::make_unique< matrix::lrmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_lowrank( B ) )
    {
        //
        // A × U·V' = W·X'
        //

        auto  RB = cptrcast( &B, matrix::lrmatrix< value_t > );
        auto  U  = std::move( RB->U( op_B ) );
        auto  X  = std::move( blas::copy( RB->V( op_B ) ) );
        auto  k  = RB->rank();
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), k );
        
        multiply( alpha, op_A, A, U, W );

        return std::make_unique< matrix::lrmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_lowrank_sv( A ) )
    {
        //
        // (U·S)·(V' × B) = W·T·X'
        //

        auto  RA = cptrcast( &A, matrix::lrsvmatrix< value_t > );
        auto  U  = RA->U( op_A );
        auto  S  = RA->S();
        auto  V  = RA->V( op_A );
        auto  k  = RA->rank();
        
        auto  W  = blas::prod_diag( U, S );
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), k );

        multiply( alpha, blas::adjoint( op_B ), B, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if
        
        return std::make_unique< matrix::lrsvmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_lowrank_sv( B ) )
    {
        //
        // (A × U)·(S·V') = W·T·X'
        //

        auto  RB = cptrcast( &B, matrix::lrsvmatrix< value_t > );
        auto  U  = RB->U( op_B );
        auto  S  = RB->S();
        auto  V  = RB->V( op_B );
        auto  k  = RB->rank();
        
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), k );
        auto  X  = blas::prod_diag( V, S );
        
        multiply( alpha, op_A, A, U, W );

        return std::make_unique< matrix::lrsvmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_dense_any( A, B ) )
    {
        C = std::make_unique< matrix::dense_matrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ) );

        multiply( alpha, op_A, A, op_B, B, *C );
    }// if
    else
        HLR_ERROR( "unsupported matrix types : " + A.typestr() + " × " + B.typestr() );

    return C;
}

//
// compute C = C + α op( A ) op( B ) with additional approximation
// by omitting sub products based on Frobenius norm of factors
//
template < typename value_t,
           typename approx_t >
void
multiply_apx ( const value_t                     alpha,
               const matop_t                     op_A,
               const Hpro::TMatrix< value_t > &  A,
               const matop_t                     op_B,
               const Hpro::TMatrix< value_t > &  B,
               Hpro::TMatrix< value_t > &        C,
               const Hpro::TTruncAcc &           acc,
               const approx_t &                  approx,
               Hpro::real_type_t< value_t >      tol )
{
    if ( is_blocked_all( A, B, C ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                auto  C_ij = BC->block(i,j);
            
                for ( uint  l = 0; l < BA->nblock_rows( op_A ); ++l )
                {
                    auto  A_il = BA->block( i, l, op_A );
                    auto  B_lj = BB->block( l, j, op_B );
                
                    if ( is_null_any( A_il, B_lj ) )
                        continue;
                    
                    HLR_ASSERT( ! is_null( C_ij ) );
            
                    multiply_apx< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *C_ij, acc, approx, tol );
                }// for
            }// for
        }// for
    }// if
    else
    {
        if ( is_lowrank( C ) )
        {
            //
            // look for Frobenius norm of factors and return if too small
            //

            const auto  norm_A = norm::frobenius( A );
            const auto  norm_B = norm::frobenius( B );

            if ( norm_A * norm_B < tol )
                return;
        }// if
        
        multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx );
    }// else
}

/////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication C := α·A·D·B + C
//
// with (block-) diagonal D
//
/////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B,
                Hpro::TMatrix< value_t > &        C,
                const Hpro::TTruncAcc &           acc,
                const approx_t &                  approx )
{
    #if defined(HLR_MULT_TESTS)

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  DD = hlr::seq::matrix::copy_diag( D );
    auto  TD = hlr::matrix::convert_to_dense< value_t >( *DD );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    io::matlab::write( *TC, "C1" );
    
    multiply_diag( alpha, op_A, *TA, op_D, *TD, op_B, *TB, *TC );

    // io::matlab::write( *TA, "A" );
    // io::matlab::write( *TD, "D" );
    // io::matlab::write( *TB, "B" );
    // io::matlab::write( *TC, "C" );
    
    #endif

    if ( is_blocked( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_lowrank( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else if ( is_dense( B ) )
            {
                if ( is_blocked( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TBlockMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_lowrank( C ) )
                    multiply_diag< value_t, approx_t >( alpha,
                                                        op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                                        op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                                        op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                                        * ptrcast( &C, Hpro::TRkMatrix< value_t > ),
                                                        acc, approx );
                else if ( is_dense( C ) )
                    multiply_diag< value_t >( alpha,
                                              op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                              op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                              op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                              * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
                else
                    HLR_ERROR( "unsupported matrix type for C : " + C.typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for A : " + A.typestr() );

    
    #if defined(HLR_MULT_TESTS)
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    // io::matlab::write( *TT, "T" );
    
    blas::add( value_t(-1), blas::mat( *TC ), blas::mat( *TT ) );

    if ( blas::norm_F( blas::mat( *TT ) ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( blas::mat( *TT ) ) << std::endl;
        hlr::breakpoint();
    }// if
    
    #endif
}

template < typename value_t >
void
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B,
                Hpro::TMatrix< value_t > &        C )
{
    #if defined(HLR_MULT_TESTS)

    auto  TA = hlr::matrix::convert_to_dense< value_t >( A );
    auto  DD = hlr::seq::matrix::copy_diag( D );
    auto  TD = hlr::matrix::convert_to_dense< value_t >( *DD );
    auto  TB = hlr::matrix::convert_to_dense< value_t >( B );
    auto  TC = hlr::matrix::convert_to_dense< value_t >( C );

    multiply_diag( alpha, op_A, *TA, op_D, *TD, op_B, *TB, *TC );

    #endif
    
    HLR_ASSERT( is_dense( D ) || is_blocked( D ) );
    HLR_ASSERT( is_dense( C ) );

    if ( is_blocked( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TBlockMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_lowrank( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TRkMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TBlockMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else if ( is_dense( D ) )
        {
            if ( is_blocked( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TBlockMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_lowrank( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TRkMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else if ( is_dense( B ) )
                multiply_diag< value_t >( alpha,
                                          op_A, * cptrcast( &A, Hpro::TDenseMatrix< value_t > ),
                                          op_D, * cptrcast( &D, Hpro::TDenseMatrix< value_t > ),
                                          op_B, * cptrcast( &B, Hpro::TDenseMatrix< value_t > ),
                                          * ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type for B : " + B.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for A : " + A.typestr() );

    
    #if defined(HLR_MULT_TESTS)
    
    auto  TT = hlr::matrix::convert_to_dense< value_t >( C );

    blas::add( value_t(-1), TC->mat(), TT->mat_dbg() );

    if ( blas::norm_F( TT->mat_dbg() ) > 1e-4 )
    {
        std::cout << A.id() << " × " << D.id() << " × " << B.id() << " -> " << C.id() << " : " << blas::norm_F( TT->mat_dbg() ) << std::endl;
        hlr::breakpoint();
    }// if

    #endif
}

//
// generate and return destination matrix
// - only supported fr non-structured results
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
multiply_diag ( const value_t                     alpha,
                const Hpro::matop_t               op_A,
                const Hpro::TMatrix< value_t > &  A,
                const Hpro::matop_t               op_D,
                const Hpro::TMatrix< value_t > &  D,
                const Hpro::matop_t               op_B,
                const Hpro::TMatrix< value_t > &  B )
{
    if ( matrix::is_lowrank( A ) )
    {
        // U·V' × D × B = W·X' ⇒ B' × D' × V = X
        auto  RA = cptrcast( &A, matrix::lrmatrix< value_t > );
        auto  V  = RA->V( op_A );
        auto  W  = blas::copy( RA->U( op_A ) );
        auto  X  = blas::matrix< value_t >( B.ncols( op_B ), RA->rank() );

        multiply_diag( alpha, blas::adjoint( op_B ), B, blas::adjoint( op_D ), D, V, X );

        if ( op_A == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< matrix::lrmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_lowrank( B ) )
    {
        // A × D × U·V' = W·X' ⇒ A × D × U = W
        auto  RB = cptrcast( &B, matrix::lrmatrix< value_t > );
        auto  U  = RB->U( op_B );
        auto  W  = blas::matrix< value_t >( A.nrows( op_A ), RB->rank() );
        auto  X  = blas::copy( RB->V( op_B ) );

        multiply_diag( alpha, op_A, A, op_D, D, U, W );

        if ( op_B == apply_transposed )
        {
            blas::conj( W );
            blas::conj( X );
        }// if

        return std::make_unique< matrix::lrmatrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_dense_any( A, B ) )
    {
        auto  C = std::make_unique< matrix::dense_matrix< value_t > >( A.row_is( op_A ), B.col_is( op_B ) );

        multiply_diag( alpha, op_A, A, op_D, D, op_B, B, *C );

        return C;
    }// if
    else
        HLR_ERROR( "unsupported matrix types : " + A.typestr() + " × " + B.typestr() );

    return std::unique_ptr< Hpro::TMatrix< value_t > >();
}

////////////////////////////////////////////////////////////////////////////////
//
// compute Hadamard product A = α A ⊗ B 
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
multiply_hadamard ( const value_t                     alpha,
                    Hpro::TMatrix< value_t > &        A,
                    const Hpro::TMatrix< value_t > &  B,
                    const Hpro::TTruncAcc &           acc,
                    const approx_t &                  approx )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = ptrcast( &A,  Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                auto  A_ij = BA->block( i, j );
                auto  B_ij = BB->block( i, j );
                
                HLR_ASSERT( ! is_null_any( A_ij, B_ij ) );
            
                multiply_hadamard< value_t >( alpha, *A_ij, *B_ij, acc, approx );
            }// for
        }// for
    }// if
    else if ( matrix::is_dense_all( A, B ) )
    {
        auto        DA     = ptrcast( &A,  matrix::dense_matrix< value_t > );
        auto        DB     = cptrcast( &B, matrix::dense_matrix< value_t > );
        auto        blas_A = blas::copy( DA->mat() );
        auto        blas_B = DB->mat();
        const auto  nrows  = DA->nrows();
        const auto  ncols  = DA->ncols();

        for ( size_t  i = 0; i < nrows*ncols; ++i )
            blas_A.data()[i] *= alpha * blas_B.data()[i];

        DA->set_matrix( blas_A, acc );
    }// if
    else if ( matrix::is_lowrank_all( A, B ) )
    {
        auto  RA = ptrcast( &A,  matrix::lrmatrix< value_t > );
        auto  RB = cptrcast( &B, matrix::lrmatrix< value_t > );

        //
        // construct product with rank rank(A)·rank(B) and fill
        // new low-rank vectors based on hadamard product
        //
        //  a_ij · b_ij = ( Σ_l u^l_i · v^l_j ) ( Σ_k w^k_i · x^k_j )
        //              = Σ_l Σ_k ( u^l_i · w^k_i ) ( v^l_j · x^k_j )
        //
        //  i.e., C = Y·Z' with y^p_i = u^l_i · w^k_i and
        //                      z^p_j = v^l_j · x^k_j
        //
        //  with p = l·rank(B)+k
        //

        auto  rank_A = RA->rank();
        auto  rank_B = RB->rank();
        auto  rank   = rank_A * rank_B;

        auto  nrows = RA->nrows();
        auto  ncols = RA->ncols();

        auto  U = RA->U();
        auto  V = RA->V();
        auto  W = RB->U();
        auto  X = RB->V();
        auto  Y = blas::matrix< value_t >( nrows, rank );
        auto  Z = blas::matrix< value_t >( ncols, rank );

        uint  p = 0;
        
        for ( uint  l = 0; l < rank_A; ++l )
        {
            auto  u_l = U.column( l );
            auto  v_l = V.column( l );
                
            for ( uint  k = 0; k < rank_B; ++k, ++p )
            {
                auto  w_k = W.column( k );
                auto  x_k = X.column( k );
                auto  y_p = Y.column( p );
                auto  z_p = Z.column( p );

                for ( size_t  i = 0; i < nrows; ++i )
                    y_p(i) = alpha * u_l(i) * w_k(i);

                for ( size_t  j = 0; j < ncols; ++j )
                    z_p(j) = v_l(j) * x_k(j);
            }// for
        }// for

        //
        // truncate Y·Z and copy back to A
        //

        auto [ Y_acc, Z_acc ] = approx( Y, Z, acc );
        
        RA->set_lrmat( std::move( Y_acc ), std::move( Z_acc ), acc );
    }// if
}

////////////////////////////////////////////////////////////////////////////////
//
// compute with diagonal matrix
//
////////////////////////////////////////////////////////////////////////////////

//
// compute from left: A = D·A
//
template < typename value_t,
           typename approx_t >
void
multiply_diag ( const blas::vector< value_t > &  D,
                Hpro::TMatrix< value_t > &       A )
{
    HLR_ERROR( "todo" );
}

//
// compute from right: A = A·D
//
template < typename value_t >
void
multiply_diag ( Hpro::TMatrix< value_t > &       A,
                const blas::vector< value_t > &  diag )
{
    if ( is_blocked( A ) )
    {
        auto  B = ptrcast( &A,  Hpro::TBlockMatrix< value_t > );
        
        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  diag_j = blas::vector< value_t >();
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  A_ij = B->block( i, j );
                
                if ( is_null( A_ij ) )
                    continue;

                if ( diag_j.length() == 0 )
                {
                    //
                    // initialize sub diagonal
                    //

                    diag_j = std::move( blas::vector< value_t >( diag, A_ij->col_is() - A.col_ofs() ) );
                }// if
                
                multiply_diag( *A_ij, diag_j );
            }// for
        }// for
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A,  matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        HLR_ASSERT( ! D->is_compressed() );
        
        for ( size_t  j = 0; j < DD.ncols(); ++j )
        {
            auto  A_j = DD.column( j );

            blas::scale( diag(j), A_j );
        }// for
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        //
        // A·D = U (V' · D), so scale rows of V
        //
        
        auto  R = ptrcast( &A,  matrix::lrmatrix< value_t > );
        auto  V = R->V();

        HLR_ASSERT( ! R->is_compressed() );
        
        for ( size_t  i = 0; i < V.nrows(); ++i )
        {
            auto  V_i = V.row( i );

            blas::scale( diag(i), V_i );
        }// for
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_MULTIPLY_HH
