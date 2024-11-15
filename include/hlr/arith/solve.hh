#ifndef __HLR_ARITH_SOLVE_HH
#define __HLR_ARITH_SOLVE_HH
//
// Project     : HLR
// Module      : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
// #include <hpro/matrix/convert.hh>    // DEBUG
// #include <hpro/algebra/solve_tri.hh> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/utils/log.hh>

#include <hlr/arith/detail/solve.hh>
#include <hlr/arith/detail/solve_vec.hh>

namespace hlr
{

// to enable accuracy tests
// #define HLR_SOLVE_TESTS

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with diagonal matrix D
//
// solve D·X = M (from_left) or X·D = M (from_right)
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
solve_diag ( const eval_side_t                 side,
             const diag_type_t                 diag,
             const matop_t                     op_D,
             const Hpro::TMatrix< value_t > &  D,
             Hpro::TMatrix< value_t > &        M,
             const accuracy &                  acc,
             const approx_t &                  approx )
{
    if ( M.is_zero() )
        return;
    
    if ( is_blocked( D ) )
    {
        auto  BD = cptrcast( & D, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( M ) )            solve_diag( side, diag, op_D, * BD, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else if ( matrix::is_dense( D ) )
    {
        auto  DD = cptrcast( & D, matrix::dense_matrix< value_t > );
        
        if      ( is_blocked( M ) )            solve_diag( side, diag, op_D, * DD, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
}

template < typename value_t >
void
solve_diag ( const eval_side_t                 side,
             const diag_type_t                 diag,
             const matop_t                     op_D,
             const Hpro::TMatrix< value_t > &  D,
             Hpro::TMatrix< value_t > &        M )
{
    if ( M.is_zero() )
        return;

    HLR_ASSERT( (( side == from_left  ) && ( D.col_is() == M.row_is() )) ||
                (( side == from_right ) && ( M.col_is() == D.row_is() )) );
                
    if ( is_blocked( D ) )
    {
        auto  BD = cptrcast( & D, Hpro::TBlockMatrix< value_t > );
        
        if      ( matrix::is_lowrank_sv( M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_diag( side, diag, op_D, * BD, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else if ( is_dense( D ) )
    {
        auto  DD = cptrcast( & D, matrix::dense_matrix< value_t > );

        if      ( matrix::is_lowrank_sv( M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_diag( side, diag, op_D, * DD, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower triangular matrix D
//
// solve op(D) x = v 
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_diag ( const Hpro::matop_t               op_D,
             const Hpro::TMatrix< value_t > &  D,
             Hpro::TScalarVector< value_t > &  v,
             const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_diag( %d )", D.id() ) );
        
    if ( is_blocked( D ) )
    {
        auto        BD  = cptrcast( & D, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BD->nblock_rows();
        const auto  nbc = BD->nblock_cols();
            
        for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            auto  D_ii = BD->block( i, i );
            
            if ( ! is_null( D_ii ) )
            {
                auto  v_i = v.sub_vector( D_ii->col_is() );
                
                solve_diag( op_D, *D_ii, v_i, diag_mode );
            }// if
        }// for
    }// if
    else if ( is_dense( D ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming D contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_D, D, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for D : " + D.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with lower triangular matrix L
//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    using namespace hlr::matrix;
    
    if ( M.is_zero() )
        return;

    #if defined(HLR_SOLVE_TESTS)
    auto  TL = convert_to_hpro( L );
    auto  TM = convert_to_hpro( M );

    if ( side == from_left )
        Hpro::solve_lower_left< value_t >( apply_normal, TL.get(), TM.get(), acc, { Hpro::block_wise, diag } );
    else
        Hpro::solve_lower_right< value_t >( TM.get(), apply_normal, TL.get(), acc, { Hpro::block_wise, diag } );
    #endif
            
    if ( is_blocked( L ) )
    {
        auto  BL = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( M ) )            solve_lower_tri( side, diag, *BL, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(    M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrmatrix< value_t > ), acc );
        else if ( matrix::is_lowrank_sv( M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrsvmatrix< value_t > ), acc );
        else if ( matrix::is_lowrankS(   M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }// if
    else if ( matrix::is_dense( L ) )
    {
        auto  DL = cptrcast( & L, dense_matrix< value_t > );
        
        if      ( is_blocked( M ) )            solve_lower_tri( side, diag, *DL, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrsvmatrix< value_t > ), acc );
        else if ( matrix::is_lowrank(    M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrmatrix< value_t > ), acc );
        else if ( matrix::is_lowrankS(   M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }// if
    else if ( matrix::is_sparse_eigen( L ) )
    {
        auto  SL = cptrcast( & L, matrix::sparse_matrix< value_t > );
        
        if      ( is_blocked( M ) )              solve_lower_tri( side, diag, *SL, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(      M ) ) solve_lower_tri( side, diag, *SL, * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_dense(        M ) ) solve_lower_tri( side, diag, *SL, * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else if ( matrix::is_sparse_eigen( M ) ) solve_lower_tri( side, diag, *SL, * ptrcast( & M, matrix::sparse_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    // test data in result
    // M.check_data();
    
    // if ( compress::is_compressible( M ) )
    //     dynamic_cast< compressible * >( &M )->compress( acc );

    #if defined(HLR_SOLVE_TESTS)
    auto  TC  = convert_to_hpro( M );
    auto  DX1 = Hpro::to_dense( TM.get() );
    auto  DX2 = Hpro::to_dense( TC.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( blas::mat( DX1 ), "X1" );
        io::matlab::write( blas::mat( DX2 ), "X2" );
        std::cout << Hpro::to_string( "solve_lower_tri( %d, %d )", L.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if
    #endif
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M )
{
    using namespace hlr::matrix;
    
    if ( M.is_zero() )
        return;

    HLR_ASSERT( (( side == from_left  ) && ( L.col_is() == M.row_is() )) ||
                (( side == from_right ) && ( M.col_is() == L.row_is() )) );

    #if defined(HLR_SOLVE_TESTS)
    auto  TL = convert_to_hpro( L );
    auto  TM = convert_to_hpro( M );

    if ( side == from_left )
        Hpro::solve_lower_left< value_t >( apply_normal, TL.get(), TM.get(), Hpro::acc_exact, { Hpro::block_wise, diag } );
    else
        Hpro::solve_lower_right< value_t >( TM.get(), apply_normal, TL.get(), Hpro::acc_exact, { Hpro::block_wise, diag } );
    #endif
            
    if ( is_blocked( L ) )
    {
        auto  BL = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
            
        if      ( matrix::is_lowrank_sv( M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_lower_tri( side, diag, *BL, * ptrcast( & M, dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }// if
    else if ( matrix::is_dense( L ) )
    {
        auto  DL = cptrcast( & L, dense_matrix< value_t > );
        
        if      ( matrix::is_lowrank_sv( M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_lower_tri( side, diag, *DL, * ptrcast( & M, dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    #if defined(HLR_SOLVE_TESTS)
    auto  TC  = convert_to_hpro( M );
    auto  DX1 = Hpro::to_dense( TM.get() );
    auto  DX2 = Hpro::to_dense( TC.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
        std::cout << Hpro::to_string( "solve_lower_tri( %d, %d )", L.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    #endif
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower triangular matrix L
//
// solve op(L) x = v 
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_lower_tri ( const Hpro::matop_t               op_L,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode );

template < typename value_t >
void
trsvl ( const Hpro::matop_t               op_L,
        const Hpro::TMatrix< value_t > &  L,
        Hpro::TScalarVector< value_t > &  v,
        const Hpro::diag_type_t           diag_mode )
{
    solve_lower_tri( op_L, L, v, diag_mode );
}

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with upper triangular matrix U
//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    using namespace hlr::matrix;
    
    if ( M.is_zero() )
        return;

    #if defined(HLR_SOLVE_TESTS)
    auto  TU = convert_to_hpro( U );
    auto  TM = convert_to_hpro( M );

    if ( side == from_left )
        Hpro::solve_upper_left< value_t >( apply_adjoint, TU.get(), TM.get(), acc, { Hpro::block_wise, diag } );
    else
        Hpro::solve_upper_right< value_t >( TM.get(), TU.get(), acc, { Hpro::block_wise, diag } );
    #endif
            
    if ( is_blocked( U ) )
    {
        auto  BU = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked(            M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrsvmatrix< value_t > ), acc );
        else if ( matrix::is_lowrank(    M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrmatrix< value_t > ), acc );
        else if ( matrix::is_lowrankS(   M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else if ( matrix::is_dense( U ) )
    {
        auto  DU = cptrcast( & U, dense_matrix< value_t > );
        
        if      ( is_blocked(            M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank_sv( M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrsvmatrix< value_t > ), acc );
        else if ( matrix::is_lowrank(    M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrmatrix< value_t > ), acc );
        else if ( matrix::is_lowrankS(   M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else if ( matrix::is_sparse_eigen( U ) )
    {
        auto  SU = cptrcast( & U, matrix::sparse_matrix< value_t > );
        
        if      ( is_blocked(              M ) ) solve_upper_tri( side, diag, *SU, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( matrix::is_lowrank(      M ) ) solve_upper_tri( side, diag, *SU, * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_dense(        M ) ) solve_upper_tri( side, diag, *SU, * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else if ( matrix::is_sparse_eigen( M ) ) solve_upper_tri( side, diag, *SU, * ptrcast( & M, matrix::sparse_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type : " + U.typestr() );

    // test data in result
    // M.check_data();
    
    #if defined(HLR_SOLVE_TESTS)
    auto  TC  = convert_to_hpro( M );
    auto  DX1 = Hpro::to_dense( TM.get() );
    auto  DX2 = Hpro::to_dense( TC.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
        std::cout << Hpro::to_string( "solve_upper_tri( %d, %d )", U.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    #endif
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M )
{
    using namespace hlr::matrix;
    
    if ( M.is_zero() )
        return;
    
    #if defined(HLR_SOLVE_TESTS)
    auto  TU  = convert_to_hpro( U );
    auto  TM  = convert_to_hpro( M );
    auto  DX0 = Hpro::to_dense( TM.get() );

    io::matlab::write( *DX0, "X0" );
    
    if ( side == from_left )
        Hpro::solve_upper_left< value_t >( apply_adjoint, TU.get(), TM.get(), Hpro::acc_exact, { Hpro::block_wise, diag } );
    else
        Hpro::solve_upper_right< value_t >( TM.get(), TU.get(), Hpro::acc_exact, { Hpro::block_wise, diag } );
    #endif

    if ( is_blocked( U ) )
    {
        auto  BU = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        
        if      ( matrix::is_lowrank_sv( M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_upper_tri( side, diag, *BU, * ptrcast( & M, dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else if ( matrix::is_dense( U ) )
    {
        auto  DU = cptrcast( & U, dense_matrix< value_t > );
        
        if      ( matrix::is_lowrank_sv( M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrsvmatrix< value_t > ) );
        else if ( matrix::is_lowrank(    M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrmatrix< value_t > ) );
        else if ( matrix::is_lowrankS(   M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, lrsmatrix< value_t > ) );
        else if ( matrix::is_dense(      M ) ) solve_upper_tri( side, diag, *DU, * ptrcast( & M, dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type : " + U.typestr() );

    #if defined(HLR_SOLVE_TESTS)
    auto  TC  = convert_to_hpro( M );
    auto  DX1 = Hpro::to_dense( TM.get() );
    auto  DX2 = Hpro::to_dense( TC.get() );

    blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    {
        io::matlab::write( *DX1, "X1" );
        io::matlab::write( *DX2, "X2" );
        std::cout << Hpro::to_string( "solve_upper_tri( %d, %d )", U.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    }// if
    #endif
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with upper triangular matrix U
//
// solve op(U) x = v
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_upper_tri ( const Hpro::matop_t               op_U,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode );

template < typename value_t >
void
trsvu ( const Hpro::matop_t               op_U,
        const Hpro::TMatrix< value_t > &  U,
        Hpro::TScalarVector< value_t > &  v,
        const Hpro::diag_type_t           diag_mode )
{
    solve_upper_tri( op_U, U, v, diag_mode );
}

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const Hpro::TDenseMatrix< value_t > &  U,
         Hpro::TMatrix< value_t > &             X )
{
    HLR_LOG( 4, Hpro::to_string( "trsmuh( %d, %d )", U.id(), X.id() ) );
    
    if ( is_lowrank( X ) )
    {
        auto  RX = ptrcast( &X, Hpro::TRkMatrix< value_t > );
        auto  Y  = blas::copy( blas::mat_V( RX ) );

        blas::prod( value_t(1), blas::adjoint( blas::mat( U ) ), Y, value_t(0), blas::mat_V( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( &X, Hpro::TDenseMatrix< value_t > );
        auto  Y  = copy( blas::mat( DX ) );
    
        blas::prod( value_t(1), Y, blas::mat( U ), value_t(0), blas::mat( DX ) );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_SOLVE_HH
