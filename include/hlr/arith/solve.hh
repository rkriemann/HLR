#ifndef __HLR_ARITH_SOLVE_HH
#define __HLR_ARITH_SOLVE_HH
//
// Project     : HLib
// File        : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
// #include <hpro/algebra/solve_tri.hh> // DEBUG
// #include <hpro/algebra/mat_conv.hh> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/utils/log.hh>

#include <hlr/arith/detail/solve.hh>

namespace hlr
{

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
             const Hpro::TTruncAcc &           acc,
             const approx_t &                  approx )
{
    if ( M.is_zero() )
        return;
    
    if ( is_blocked( D ) )
    {
        if ( is_blocked( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else if ( is_dense( D ) )
    {
        if ( is_blocked( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
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
        if ( is_lowrank( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + D.typestr() );
    }// if
    else if ( is_dense( D ) )
    {
        if ( is_lowrank( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_diag( side, diag, op_D, * cptrcast( & D, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
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
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( M.is_zero() )
        return;
    
    // auto  Mc = M.copy();

    // if ( side == from_left )
    //     Hpro::solve_lower_left( apply_normal, &L, nullptr, Mc.get(), acc, { Hpro::block_wise, diag } );
    // else
    //     Hpro::solve_lower_right( Mc.get(), apply_normal, &L, nullptr, acc, { Hpro::block_wise, diag } );
            
    if ( is_blocked( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    // auto  DX1 = Hpro::to_dense( &M );
    // auto  DX2 = Hpro::to_dense( Mc.get() );

    // blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    // if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    // {
    //     Hpro::DBG::write(  M,  "M.mat", "M" );
    //     Hpro::DBG::write( *Mc, "Mc.mat", "Mc" );
    //     std::cout << Hpro::to_string( "svltr( %d, %d )", L.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
    // }// if
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M )
{
    if ( M.is_zero() )
        return;

    HLR_ASSERT( (( side == from_left  ) && ( L.col_is() == M.row_is() )) ||
                (( side == from_right ) && ( M.col_is() == L.row_is() )) );
                
    if ( is_blocked( L ) )
    {
        if ( is_lowrank( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_lowrank( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_lower_tri( side, diag, * cptrcast( & L, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
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
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_lower_tri( %d )", L.id() ) );
        
    if ( is_blocked( L ) )
    {
        auto        BL  = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();
            
        if ( op_L == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->col_is() );
                
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if
            
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );

                    if ( ! is_null( L_ji ) )
                    {
                        auto  v_j = v.sub_vector( L_ji->row_is() );
                        auto  v_i = v.sub_vector( L_ji->col_is() );
                    
                        mul_vec< value_t >( value_t(-1), op_L, *L_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min( nbr, nbc )-1; i >= 0; --i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->row_is() );
                    
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if

                for ( int  j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        auto  v_i = v.sub_vector( L_ij->col_is() );
                        auto  v_j = v.sub_vector( L_ij->row_is() );
                    
                        mul_vec( value_t(-1), op_L, *L_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( L ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming L contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_L, L, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

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
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( M.is_zero() )
        return;
    
    // auto  Mc = M.copy();

    // if ( side == from_left )
    //     Hpro::solve_upper_left( apply_adjoint, &U, nullptr, Mc.get(), acc, { Hpro::block_wise, diag } );
    // else
    //     Hpro::solve_upper_right( Mc.get(), &U, nullptr, acc, { Hpro::block_wise, diag } );
            
    if ( is_blocked( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type : " + U.typestr() );

    // auto  DX1 = Hpro::to_dense( &M );
    // auto  DX2 = Hpro::to_dense( Mc.get() );

    // blas::add( value_t(-1), blas::mat( DX1 ), blas::mat( DX2 ) );
    // if ( blas::norm_F( blas::mat( DX2 ) ) > 1e-14 )
    //     std::cout << Hpro::to_string( "svutr( %d, %d )", U.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat( DX2 ) ) << std::endl;
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M )
{
    if ( M.is_zero() )
        return;
    
    if ( is_blocked( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TBlockMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TRkMatrix< value_t > ) );
        else if ( matrix::is_lowrankS( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, matrix::lrsmatrix< value_t > ) );
        else if ( is_dense( M ) )
            solve_upper_tri( side, diag, * cptrcast( & U, Hpro::TDenseMatrix< value_t > ), * ptrcast( & M, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type : " + U.typestr() );
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
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_upper_tri( %d )", U.id() ) );
        
    if ( is_blocked( U ) )
    {
        auto        BU  = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();
            
        if ( op_U == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min<int>( nbr, nbc )-1; i >= 0; --i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->col_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if
            
                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );

                    if ( ! is_null( U_ji ) )
                    {
                        auto  v_j = v.sub_vector( U_ji->row_is() );
                        auto  v_i = v.sub_vector( U_ji->col_is() );
                    
                        mul_vec( value_t(-1), op_U, *U_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->row_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if

                for ( uint  j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        auto  v_i = v.sub_vector( U_ij->col_is() );
                        auto  v_j = v.sub_vector( U_ij->row_is() );
                    
                        HLR_LOG( 4, Hpro::to_string( "update( %d, ", U_ij->id() ) + v_j.is().to_string() + ", " + v_i.is().to_string() );
                        mul_vec( value_t(-1), op_U, *U_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( U ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming U contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_U, U, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

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
