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

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( B %d, B %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint i = 0; i < M.nblock_rows(); ++i )
        {
            const auto  L_ii = L.block( i, i );

            HLR_ASSERT( ! is_null( L_ii ) );
            
            for ( uint j = 0; j < M.nblock_cols(); ++j )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, approx );
            }// for

            for ( uint  k = i+1; k < M.nblock_rows(); ++k )
            {
                for ( uint  j = 0; j < M.nblock_cols(); ++j )
                {
                    if ( ! is_null_any( L.block(k,i), M.block(i,j) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(k,j) ) );
                        
                        multiply< value_t >( value_t(-1),
                                             apply_normal, *L.block(k,i),
                                             apply_normal, *M.block(i,j),
                                             *M.block(k,j), acc, approx );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  L,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( B %d, R %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = hpro::TDenseMatrix( M.row_is(), hpro::is( 0, M.rank()-1 ), blas::mat_U< value_t >( M ) );

        solve_lower_tri< value_t >( side, diag, L, U, acc, approx );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  L,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( B %d, D %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint j = 0; j < L.nblock_cols(); ++j )
        {
            const auto  L_jj = L.block( j, j );

            HLR_ASSERT( ! is_null( L_jj ) );

            auto  M_j = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                 L_jj->col_is() - L.col_ofs(),
                                                 blas::range::all );
            auto  D_j = hpro::TDenseMatrix( L_jj->col_is(), M.col_is(), M_j );
                
            solve_lower_tri< value_t >( side, diag, *L_jj, D_j, acc, approx );

            for ( uint  k = j+1; k < L.nblock_rows(); ++k )
            {
                auto  L_kj = L.block( k, j );
                
                if ( ! is_null( L_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                         L_kj->row_is() - L.row_ofs(),
                                                         blas::range::all );
                    auto  D_k = hpro::TDenseMatrix( L_kj->row_is(), M.col_is(), M_k );
                    
                    multiply< value_t >( value_t(-1),
                                         apply_normal, *L_kj,
                                         apply_normal, D_j,
                                         D_k, acc, approx );
                }// for
            }// for
        }// for
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( D %d, B %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ERROR( "todo" );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  L,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = hpro::TDenseMatrix( M.row_is(), hpro::is( 0, M.rank()-1 ), blas::mat_U< value_t >( M ) );

        solve_lower_tri< value_t >( side, diag, L, U, acc, approx );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  L,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svltr( D %d, D %d )", L.id(), M.id() ) );
    
    // hpro::DBG::write( L, "L.mat", "L" );
    // hpro::DBG::write( M, "M.mat", "M" );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        auto  Mc = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), hpro::blas_mat< value_t >( L ), Mc, value_t(0), hpro::blas_mat< value_t >( M ) );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else

    // hpro::DBG::write( M, "X.mat", "X" );
}

//
// matrix type based deduction of special versions
//
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
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
        else if ( is_dense( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
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

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower triangular matrix L
//
// solve op(L) x = v 
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

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

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( B %d, B %d )", U.id(), M.id() ) );
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        for ( uint j = 0; j < M.nblock_cols(); ++j )
        {
            const auto  U_jj = U.block( j, j );

            HLR_ASSERT( ! is_null( U_jj ) );
            
            for ( uint i = 0; i < M.nblock_rows(); ++i )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, approx );
            }// for
            
            for ( uint  k = j+1; k < M.nblock_cols(); ++k )
            {
                for ( uint  i = 0; i < M.nblock_rows(); ++i )
                {
                    if ( ! is_null_any( M.block(i,j), U.block(j,k) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(i,k) ) );
                        
                        multiply< value_t >( value_t(-1),
                                             apply_normal, *M.block(i,j),
                                             apply_normal, *U.block(j,k),
                                             *M.block(i,k), acc, approx );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  U,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( B %d, R %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = hpro::TDenseMatrix( M.col_is(), hpro::is( 0, M.rank()-1 ), blas::mat_V< value_t >( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  U,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( B %d, D %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // assumption: U' is used
        //

        const matop_t  op_U = apply_adjoint;
        
        for ( uint j = 0; j < U.nblock_cols( op_U ); ++j )
        {
            const auto  U_jj = U.block( j, j, op_U );

            HLR_ASSERT( ! is_null( U_jj ) );

            auto  M_j = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                 U_jj->col_is( op_U ) - U.col_ofs( op_U ),
                                                 blas::range::all );
            auto  D_j = hpro::TDenseMatrix( U_jj->col_is( op_U ), M.col_is(), M_j );
            
            solve_upper_tri< value_t >( side, diag, *U_jj, D_j, acc, approx );
            
            for ( uint  k = j+1; k < U.nblock_rows( op_U ); ++k )
            {
                auto  U_kj = U.block( k, j, op_U );
                    
                if ( ! is_null( U_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                         U_kj->row_is( op_U ) - U.row_ofs( op_U ),
                                                         blas::range::all );
                    auto  D_k = hpro::TDenseMatrix( U_kj->row_is( op_U ), M.col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), op_U, *U_kj, apply_normal, D_j, D_k, acc, approx );
                }// if
            }// for
        }// for
    }// if
    else
    {
        for ( uint i = 0; i < U.nblock_rows(); ++i )
        {
            const auto  U_ii = U.block( i, i );

            HLR_ASSERT( ! is_null( U_ii ) );

            auto  M_i = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                 blas::range::all,
                                                 U_ii->row_is() - U.row_ofs() );
            auto  D_i = hpro::TDenseMatrix( M.row_is(), U_ii->row_is(), M_i );
            
            solve_upper_tri< value_t >( side, diag, *U_ii, D_i, acc, approx );
            
            for ( uint  k = i+1; k < U.nblock_cols(); ++k )
            {
                auto  U_ik = U.block(i,k);
                    
                if ( ! is_null( U_ik ) )
                {
                    auto  M_k = blas::matrix< value_t >( hpro::blas_mat< value_t >( M ),
                                                         blas::range::all,
                                                         U_ik->col_is() - U.col_ofs() );
                    auto  D_k = hpro::TDenseMatrix( M.row_is(), U_ik->col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), apply_normal, D_i, apply_normal, *U_ik, D_k, acc, approx );
                }// if
            }// for
        }// for
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( D %d, B %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  U,
                  hpro::TRkMatrix &           M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = hpro::TDenseMatrix( M.col_is(), hpro::is( 0, M.rank()-1 ), blas::mat_V< value_t >( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TDenseMatrix &  U,
                  hpro::TDenseMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "svutr( D %d, D %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    // hpro::DBG::write( U, "U.mat", "U" );
    // hpro::DBG::write( M, "M.mat", "M" );
    
    if ( side == from_left )
    {
        // assuming U' X = M
        const matop_t  op_U = apply_adjoint;
        auto           Mc   = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), blas::mat_view( op_U, hpro::blas_mat< value_t >( U ) ), Mc, value_t(0), hpro::blas_mat< value_t >( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( hpro::blas_mat< value_t >( M ) );

        blas::prod( value_t(1), Mc, hpro::blas_mat< value_t >( U ), value_t(0), hpro::blas_mat< value_t >( M ) );
    }// else

    // hpro::DBG::write( M, "X.mat", "X" );
}

//
// matrix type based deduction of special versions
//
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
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, approx );
        else if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TRkMatrix ), acc, approx );
        else if ( is_dense( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ), acc, approx );
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

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with upper triangular matrix U
//
// solve op(U) x = v
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

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
