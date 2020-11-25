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
#include <hlr/utils/log.hh>

#include <hlr/arith/detail/solve.hh>

namespace hlr
{

namespace hpro = HLIB;

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
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    // auto  Mc = M.copy();

    // if ( side == from_left )
    //     hpro::solve_lower_left( apply_normal, &L, nullptr, Mc.get(), acc, { hpro::block_wise, diag } );
    // else
    //     hpro::solve_lower_right( Mc.get(), apply_normal, &L, nullptr, acc, { hpro::block_wise, diag } );
            
    if ( is_blocked( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    // auto  DX1 = hpro::to_dense( &M );
    // auto  DX2 = hpro::to_dense( Mc.get() );

    // blas::add( value_t(-1), blas::mat< value_t >( DX1 ), blas::mat< value_t >( DX2 ) );
    // if ( blas::norm_F( blas::mat< value_t >( DX2 ) ) > 1e-14 )
    // {
    //     hpro::DBG::write(  M,  "M.mat", "M" );
    //     hpro::DBG::write( *Mc, "Mc.mat", "Mc" );
    //     std::cout << hpro::to_string( "svltr( %d, %d )", L.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat< value_t >( DX2 ) ) << std::endl;
    // }// if
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M )
{
    if ( is_blocked( L ) )
    {
        if ( is_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
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

inline
void
trsvl ( const hpro::matop_t      op_L,
        const hpro::TMatrix &    L,
        hpro::TScalarVector &    v,
        const hpro::diag_type_t  diag_mode )
{
    HLR_LOG( 4, HLIB::to_string( "trsvl( %d )", L.id() ) );
        
    if ( is_blocked( L ) )
    {
        auto        BL  = cptrcast( & L, hpro::TBlockMatrix );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();
            
        if ( op_L == hpro::apply_normal )
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
                
                    trsvl( op_L, *L_ii, v_i, diag_mode );
                }// if
            
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );

                    if ( ! is_null( L_ji ) )
                    {
                        auto  v_j = v.sub_vector( L_ji->row_is() );
                        auto  v_i = v.sub_vector( L_ji->col_is() );
                    
                        mul_vec< real >( real(-1), op_L, *L_ji, v_i, v_j );
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
                
                    trsvl( op_L, *L_ii, v_i, diag_mode );
                }// if

                for ( int  j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        auto  v_i = v.sub_vector( L_ij->col_is() );
                        auto  v_j = v.sub_vector( L_ij->row_is() );
                    
                        mul_vec( real(-1), op_L, *L_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( L ) )
    {
        if ( diag_mode == hpro::general_diag )
        {
            //
            // assuming L contains inverse (store_inverse!)
            //

            auto  vc = hpro::TScalarVector( v );

            v.scale( 0 );
            mul_vec( real(1), op_L, L, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
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
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    // auto  Mc = M.copy();

    // if ( side == from_left )
    //     hpro::solve_upper_left( apply_adjoint, &U, nullptr, Mc.get(), acc, { hpro::block_wise, diag } );
    // else
    //     hpro::solve_upper_right( Mc.get(), &U, nullptr, acc, { hpro::block_wise, diag } );
            
    if ( is_blocked( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );

    // auto  DX1 = hpro::to_dense( &M );
    // auto  DX2 = hpro::to_dense( Mc.get() );

    // blas::add( value_t(-1), blas::mat< value_t >( DX1 ), blas::mat< value_t >( DX2 ) );
    // if ( blas::norm_F( blas::mat< value_t >( DX2 ) ) > 1e-14 )
    //     std::cout << hpro::to_string( "svutr( %d, %d )", U.id(), M.id() ) << ", error = " << blas::norm_F( blas::mat< value_t >( DX2 ) ) << std::endl;
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M )
{
    if ( is_blocked( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ) );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with upper triangular matrix U
//
// solve op(U) x = v
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

inline
void
trsvu ( const hpro::matop_t      op_U,
        const hpro::TMatrix &    U,
        hpro::TScalarVector &    v,
        const hpro::diag_type_t  diag_mode )
{
    HLR_LOG( 4, HLIB::to_string( "trsvu( %d )", U.id() ) );
        
    if ( is_blocked( U ) )
    {
        auto        BU  = cptrcast( & U, hpro::TBlockMatrix );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();
            
        if ( op_U == hpro::apply_normal )
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
                
                    trsvu( op_U, *U_ii, v_i, diag_mode );
                }// if
            
                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );

                    if ( ! is_null( U_ji ) )
                    {
                        auto  v_j = v.sub_vector( U_ji->row_is() );
                        auto  v_i = v.sub_vector( U_ji->col_is() );
                    
                        mul_vec( real(-1), op_U, *U_ji, v_i, v_j );
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
                
                    trsvu( op_U, *U_ii, v_i, diag_mode );
                }// if

                for ( uint  j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        auto  v_i = v.sub_vector( U_ij->col_is() );
                        auto  v_j = v.sub_vector( U_ij->row_is() );
                    
                        HLR_LOG( 4, HLIB::to_string( "update( %d, ", U_ij->id() ) + v_j.is().to_string() + ", " + v_i.is().to_string() );
                        mul_vec( real(-1), op_U, *U_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( is_dense( U ) )
    {
        if ( diag_mode == hpro::general_diag )
        {
            //
            // assuming U contains inverse (store_inverse!)
            //

            auto  vc = hpro::TScalarVector( v );

            v.scale( 0 );
            mul_vec( real(1), op_U, U, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const hpro::TDenseMatrix *  U,
         hpro::TMatrix *             X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d, %d )", U->id(), X->id() ) );
    
    if ( is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, hpro::TRkMatrix );
        auto  Y  = blas::copy( hpro::blas_mat_B< value_t >( RX ) );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( U ) ), Y,
                    value_t(0), hpro::blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, hpro::TDenseMatrix );
        auto  Y  = copy( hpro::blas_mat< value_t >( DX ) );
    
        blas::prod( value_t(1), Y, hpro::blas_mat< value_t >( U ),
                    value_t(0), hpro::blas_mat< value_t >( DX ) );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_SOLVE_HH
