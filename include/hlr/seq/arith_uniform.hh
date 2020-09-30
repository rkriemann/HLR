#ifndef __HLR_SEQ_ARITH_UNIFORM_HH
#define __HLR_SEQ_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : seq/arith_uniform.hh
// Description : sequential arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/seq/matrix.hh>

namespace hlr { namespace seq { namespace uniform {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x
//
namespace detail
{

using matrix::cluster_basis;
using vector::scalar_vector;
using vector::uniform_vector;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec ( const value_t                                       alpha,
          const hpro::matop_t                                 op_M,
          const hpro::TMatrix &                               M,
          const uniform_vector< cluster_basis< value_t > > &  x,
          uniform_vector< cluster_basis< value_t > > &        y,
          const scalar_vector< value_t > &                    sx,
          scalar_vector< value_t > &                          sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            auto  y_i = y.block( i );
            
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j, op_M );
                auto  x_j  = x.block( j );
            
                if ( ! is_null( B_ij ) )
                {
                    mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D   = cptrcast( &M, hpro::TDenseMatrix );
        auto  x_i = blas::vector< value_t >( hpro::blas_vec< value_t >( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( hpro::blas_vec< value_t >( sy ), M.row_is( op_M ) - sy.ofs() );
        
        blas::mulvec( alpha, blas::mat_view( op_M, hpro::blas_mat< value_t >( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        
        if ( op_M == hpro::apply_normal )
        {
            blas::mulvec( value_t(1), R->coeff(), x.coeffs(), value_t(1), y.coeffs() );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            HLR_ASSERT( false );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            blas::mulvec( value_t(1), blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
scalar_to_uniform ( const cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( hpro::blas_vec< value_t >( v ), cb.cluster() - v.ofs() );
        auto  s    = cb.transform_forward( v_cb );

        u->set_coeffs( std::move( s ) );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
    }// if

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
make_uniform ( const cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, make_uniform( *cb.son(i) ).release() );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                          v )
{
    if ( u.basis().rank() > 0 )
    {
        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( hpro::blas_vec< value_t >( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        for ( uint  i = 0; i < u.nblocks(); ++i )
            add_uniform_to_scalar( *u.block(i), v );
    }// if
}
    
}// namespace detail

template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const hpro::matop_t                       op_M,
          const hpro::TMatrix &                     M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          matrix::cluster_basis< value_t > &        rowcb,
          matrix::cluster_basis< value_t > &        colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb );

    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y );
}

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// locally add low-rank update W·X to block M_ij
//
template < typename value_t >
void
addlr_local ( hpro::TBlockMatrix &                   M,
              matrix::uniform_lrmatrix< value_t > &  M_ij,
              const uint                             i,
              const uint                             j,
              const blas::matrix< value_t > &        W,
              const blas::matrix< value_t > &        X,
              const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // current bases and coupling
    const auto  U_i   = M_ij.row_cb().basis();
    const auto  S_ij  = M_ij.coeff();
    const auto  V_j   = M_ij.col_cb().basis();

    // extended bases and coupling
    const auto  rank  = W.ncols();
    const auto  Ue_i  = blas::join_row< value_t >( { U_i, W } );
    const auto  I     = blas::identity< value_t >( rank );
    const auto  Se_ij = blas::diag< value_t >( { S_ij, I } );
    const auto  Ve_j  = blas::join_row< value_t >( { V_j, X } );

    if ( true )
    {
        io::matlab::write( U_i,   "U" );
        io::matlab::write( S_ij,  "S" );
        io::matlab::write( V_j,   "V" );
        io::matlab::write( Ue_i,  "Ue" );
        io::matlab::write( Se_ij, "Se" );
        io::matlab::write( Ve_j,  "Ve" );
        io::matlab::write( W,     "W" );
        io::matlab::write( X,     "X" );
    }// if
    
    //
    // new row basis is computed as the left singular vectors of Ue_i · Se_ij · Ve_j'
    // which can be simplified to Ue_i · Se_ij · R_j' Q_j' with QR decomposition
    // Ve_j = Q_j R_j
    //

    auto  Un_i = blas::matrix< value_t >();
                
    {
        auto  R_j = blas::matrix< value_t >();
        auto  Q_j = blas::copy( Ve_j ); // need to copy since modified during QR
                
        blas::qr_wrapper( Q_j, R_j, false );
                
        const auto  SR_ij = blas::prod( value_t(1), Se_ij, blas::adjoint( R_j ) );
        auto        Us    = blas::prod( value_t(1), Ue_i, SR_ij );
        auto        Ss    = blas::vector< real_t >();
                
        blas::svd( Us, Ss );
                    
        const auto  rank_U = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

        Un_i = std::move( blas::copy( U_rank ) );
    }

    //
    // new column basis is computed as left singular vectors of Ve_j · Se_ij' · Ue_i'
    // simplified to Ve_j · Se_ij' · R_i' · Q_i' with Ue_i = Q_i R_i
    //

    auto  Vn_j = blas::matrix< value_t >();
                
    {
        auto  R_i = blas::matrix< value_t >();
        auto  Q_i = blas::copy( Ue_i ); // need to copy since modified during QR
                
        blas::qr_wrapper( Q_i, R_i, false );
                    
        const auto  RS_ij = blas::prod( value_t(1), R_i, Se_ij );
        auto        Us    = blas::prod( value_t(1), Ve_j, blas::adjoint( RS_ij ) );
        auto        Ss    = blas::vector< real_t >();
                    
        blas::svd( Us, Ss );
                    
        const auto  rank_V = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_V-1 ) );

        Vn_j = std::move( blas::copy( V_rank ) );
    }

    //
    // new B_ij is computed as
    //
    //     Un_i ( Un_i' Ue_i Se_ij Ve_j' Vn_j ) Vn_j'
    //
    // therefore new coupling matrix is
    //
    //     Un_i' Ue_i Se_ij Ve_j' Vn_j = TU_i Se_ij TVj'
    //
    // with
    //
    //     TU_i = Un_i' Ue_i  and  TV_j = Vn_j' Ve_j
    //
                
    const auto  TU_i  = blas::prod( value_t(1), blas::adjoint( Un_i ), Ue_i );
    const auto  TV_j  = blas::prod( value_t(1), blas::adjoint( Vn_j ), Ve_j );
    auto        T_ij  = blas::prod( value_t(1), TU_i, Se_ij );
    auto        Sn_ij = blas::prod( value_t(1), T_ij, blas::adjoint( TV_j ) );

    M_ij.set_coeff_unsafe( std::move( Sn_ij ) );
                
    //
    // transform coupling matrix for blocks in current block row as
    //
    //   TU_i · ⎛S_ik⎞
    //          ⎝  0 ⎠
    //

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if (( k == j ) || ! matrix::is_uniform_lowrank( B_ik ))
            continue;
                    
        auto        R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        const auto  S_ik  = R_ik->coeff();
        const auto  Se_ik = blas::extend( S_ik, rank, 0 );  // [ S_ik ; 0 ]
        auto        Sn_ik = blas::prod( value_t(1), TU_i, Se_ik );

        R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
    }// for
                
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   (S_kj 0) TV_j
    //

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if (( k == i ) || ! matrix::is_uniform_lowrank( B_kj ))
            continue;
                    
        auto        R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        const auto  S_kj  = R_kj->coeff();
        const auto  Se_kj = blas::extend( S_kj, 0, rank ); // [ S_kj, 0 ]
        auto        Sn_kj = blas::prod( value_t(1), Se_kj, blas::adjoint( TV_j ) );

        R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un_i ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn_j ) );
}

template < typename value_t >
void
addlr_local2 ( hpro::TBlockMatrix &                   M,
               matrix::uniform_lrmatrix< value_t > &  M_ij,
               const uint                             i,
               const uint                             j,
               const blas::matrix< value_t > &        W,
               const blas::matrix< value_t > &        X,
               const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // current bases and coupling
    const auto  U_i   = M_ij.row_cb().basis();
    const auto  S_ij  = M_ij.coeff();
    const auto  V_j   = M_ij.col_cb().basis();

    // extended bases and coupling
    const auto  rank  = W.ncols();
    const auto  Ue_i  = blas::join_row< value_t >( { U_i, W } );
    const auto  I     = blas::identity< value_t >( M_ij.rank() );
    const auto  Se_ij = blas::diag< value_t >( { I, S_ij } );
    const auto  Ve_j  = blas::join_row< value_t >( { V_j, X } );

    if ( true )
    {
        io::matlab::write( U_i,   "U" );
        io::matlab::write( S_ij,  "S" );
        io::matlab::write( V_j,   "V" );
        io::matlab::write( Ue_i,  "Ue" );
        io::matlab::write( Se_ij, "Se" );
        io::matlab::write( Ve_j,  "Ve" );
        io::matlab::write( W,     "W" );
        io::matlab::write( X,     "X" );
    }// if
    
    //
    // new row basis is computed as the left singular vectors of Ue_i · Se_ij · Ve_j'
    // which can be simplified to Ue_i · Se_ij · R_j' Q_j' with QR decomposition
    // Ve_j = Q_j R_j
    //

    auto  Un_i = blas::matrix< value_t >();
                
    {
        auto  R_j = blas::matrix< value_t >();
        auto  Q_j = blas::copy( Ve_j ); // need to copy since modified during QR
                
        blas::qr_wrapper( Q_j, R_j, false );
                
        const auto  SR_ij = blas::prod( value_t(1), Se_ij, blas::adjoint( R_j ) );
        auto        Us    = blas::prod( value_t(1), Ue_i, SR_ij );
        auto        Ss    = blas::vector< real_t >();
                
        blas::svd( Us, Ss );
                    
        const auto  rank_U = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

        Un_i = std::move( blas::copy( U_rank ) );
    }

    //
    // new column basis is computed as left singular vectors of Ve_j · Se_ij' · Ue_i'
    // simplified to Ve_j · Se_ij' · R_i' · Q_i' with Ue_i = Q_i R_i
    //

    auto  Vn_j = blas::matrix< value_t >();
                
    {
        auto  R_i = blas::matrix< value_t >();
        auto  Q_i = blas::copy( Ue_i ); // need to copy since modified during QR
                
        blas::qr_wrapper( Q_i, R_i, false );
                    
        const auto  RS_ij = blas::prod( value_t(1), R_i, Se_ij );
        auto        Us    = blas::prod( value_t(1), Ve_j, blas::adjoint( RS_ij ) );
        auto        Ss    = blas::vector< real_t >();
                    
        blas::svd( Us, Ss );
                    
        const auto  rank_V = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_V-1 ) );

        Vn_j = std::move( blas::copy( V_rank ) );
    }

    //
    // new B_ij is computed as
    //
    //     Un_i ( Un_i' Ue_i Se_ij Ve_j' Vn_j ) Vn_j'
    //
    // therefore new coupling matrix is
    //
    //     Un_i' Ue_i Se_ij Ve_j' Vn_j = TU_i Se_ij TVj'
    //
    // with
    //
    //     TU_i = Un_i' Ue_i  and  TV_j = Vn_j' Ve_j
    //
                
    const auto  TU_i  = blas::prod( value_t(1), blas::adjoint( Un_i ), Ue_i );
    const auto  TV_j  = blas::prod( value_t(1), blas::adjoint( Vn_j ), Ve_j );
    auto        T_ij  = blas::prod( value_t(1), TU_i, Se_ij );
    auto        Sn_ij = blas::prod( value_t(1), T_ij, blas::adjoint( TV_j ) );

    if ( true )
    {
        io::matlab::write( Un_i,   "Un" );
        io::matlab::write( Sn_ij,  "Sn" );
        io::matlab::write( Vn_j,   "Vn" );
    }// if

    M_ij.set_coeff_unsafe( std::move( Sn_ij ) );
                
    //
    // transform coupling matrix for blocks in current block row as
    //
    //   TU_i · ⎛S_ik⎞
    //          ⎝  0 ⎠
    //

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if (( k == j ) || ! matrix::is_uniform_lowrank( B_ik ))
            continue;
                    
        auto        R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        const auto  S_ik  = R_ik->coeff();
        const auto  Se_ik = blas::extend( S_ik, rank, 0 );  // [ S_ik ; 0 ]
        auto        Sn_ik = blas::prod( value_t(1), TU_i, Se_ik );

        R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
    }// for
                
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   (S_kj 0) TV_j
    //

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if (( k == i ) || ! matrix::is_uniform_lowrank( B_kj ))
            continue;
                    
        auto        R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        const auto  S_kj  = R_kj->coeff();
        const auto  Se_kj = blas::extend( S_kj, 0, rank ); // [ S_kj, 0 ]
        auto        Sn_kj = blas::prod( value_t(1), Se_kj, blas::adjoint( TV_j ) );

        R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un_i ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn_j ) );
}

//
// add global low-rank matrix W·X' to H²-matrix M
//
template < typename value_t >
void
addlr ( hpro::TMatrix &                  M,
        const blas::matrix< value_t > &  W,
        const blas::matrix< value_t > &  X,
        const hpro::TTruncAcc &          acc )
{
    HLR_ASSERT( is_blocked( M ) );

    auto  B = ptrcast( &M, hpro::TBlockMatrix );
    
    //
    // use inefficient method adding only local updates
    //

    for ( uint  i = 0; i < B->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );
            auto  W_i  = blas::matrix( W, B_ij->row_is() - B->row_ofs(), blas::range::all );
            auto  X_j  = blas::matrix( X, B_ij->col_is() - B->col_ofs(), blas::range::all );
            auto  I    = blas::identity< value_t >( X_j.ncols() );
                        
            if ( matrix::is_uniform_lowrank( B_ij ) )
            {
                auto  R_ij = ptrcast( B_ij, matrix::uniform_lrmatrix< value_t > );

                addlr_local( *B, *R_ij, i, j, W_i, X_j, acc );
            }// if
            else if ( is_dense( B_ij ) )
            {
                auto  D_ij = ptrcast( B_ij, hpro::TDenseMatrix );

                blas::prod( value_t(1), W_i, blas::adjoint( X_j ), value_t(1), blas::mat< value_t >( D_ij ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_ij->typestr() );
        }// for
    }// for
}

//
// matrix multiplication
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    aA,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    aB,
           hpro::TMatrix &          aC,
           const hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( is_blocked_all( aA, aB, aC ) );

    auto  A = cptrcast( &aA, hpro::TBlockMatrix );
    auto  B = cptrcast( &aB, hpro::TBlockMatrix );
    auto  C = ptrcast(  &aC, hpro::TBlockMatrix );

    HLR_ASSERT( C->nblock_rows()       == A->nblock_rows( op_A ) );
    HLR_ASSERT( C->nblock_cols()       == B->nblock_cols( op_B ) );
    HLR_ASSERT( A->nblock_cols( op_A ) == B->nblock_rows( op_B ) );

    for ( uint  i = 0; i < C->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C->nblock_cols(); ++j )
        {
            auto  C_ij = C->block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            for ( uint  k = 0; k < A->nblock_cols( op_A ); ++k )
            {
                auto  A_ik = A->block( i, k, op_A );
                auto  B_kj = B->block( k, j, op_B );

                auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
                auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
                auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

                blas::prod( value_t(1), blas::mat< value_t >( DA ), blas::mat< value_t >( DB ),
                            value_t(1), blas::mat< value_t >( DC ) );

                HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

                std::cout << A_ik->id() << " × " << B_kj->id() << " -> " << C_ij->id() << std::endl;
                
                //
                // due to TLR format, C_ij, A_ik and B_kj can only be dense or uniform-lowrank
                // hence, handle all combinations
                //

                if ( matrix::is_uniform_lowrank( C_ij ) )
                {
                    auto  RC_ij = ptrcast( C_ij, matrix::uniform_lrmatrix< value_t > );
                    
                    if ( matrix::is_uniform_lowrank( A_ik ) )
                    {
                        auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            //
                            // U S_C V' + U S_A W' · W S_B V' =
                            // U S_C V' + U S_A · S_B V'      =
                            // U ( S_C + S_A · S_B ) V'
                            //
                            
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );

                            blas::prod( alpha,
                                        blas::mat_view( op_A, RA_ik->coeff() ),
                                        blas::mat_view( op_B, RB_kj->coeff() ),
                                        value_t(1),
                                        RC_ij->coeff() );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // U S_C V' + U S_A W' · B
                            //
                            // add low-rank update ( U · S_A ) ( W' · B ) to C and update bases
                            //
                            
                            auto        DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                            const auto  US    = blas::prod( value_t(1), RA_ik->row_cb().basis(), RA_ik->coeff() );
                            const auto  BW    = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( DB_kj ) ), RA_ik->col_cb().basis() );

                            addlr_local< value_t >( *C, *RC_ij, i, j, US, BW, acc );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else if ( is_dense( A_ik ) )
                    {
                        auto  DA_ik = cptrcast( A_ik, hpro::TDenseMatrix );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            //
                            // U S_C V' + A · W S_B V'
                            //
                            // add low-rank update ( A W ) ( V S_B' )' to C and update bases
                            //
                            
                            auto        RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            const auto  AW    = blas::prod( value_t(1), blas::mat< value_t >( DA_ik ), RB_kj->row_cb().basis() );
                            const auto  VS    = blas::prod( value_t(1), RB_kj->col_cb().basis(), blas::adjoint( RB_kj->coeff() ) );

                            addlr_local< value_t >( *C, *RC_ij, i, j, AW, VS, acc );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // U S_C V' + A · B
                            //
                            // compute A·B, convert to low-rank, add to C and update bases
                            //
                            
                            auto        DB_kj    = cptrcast( B_kj, hpro::TDenseMatrix );
                            auto        AB       = blas::prod( value_t(1), blas::mat< value_t >( DA_ik ), blas::mat< value_t >( DB_kj ) );
                            const auto  [ W, X ] = approx::svd( AB, acc );

                            addlr_local< value_t >( *C, *RC_ij, i, j, W, X, acc );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
                }// if
                else if ( is_dense( C_ij ) )
                {
                    auto  DC_ij = ptrcast( C_ij, hpro::TDenseMatrix );
                    
                    if ( matrix::is_uniform_lowrank( A_ik ) )
                    {
                        auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            //
                            // C = C + U_A S_A ( V_A' · U_B ) S_B V_B'
                            //   = C + U_A ( S_A S_B ) V_B'
                            //

                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            auto  VU    = blas::prod( value_t(1), blas::adjoint( RA_ik->col_cb().basis() ), RB_kj->row_cb().basis() );
                            auto  S1    = blas::prod( value_t(1), RA_ik->coeff(), VU );
                            auto  S     = blas::prod( alpha, S1, RB_kj->coeff() );
                            auto  UT    = blas::prod( value_t(1), RA_ik->row_cb( op_A ).basis(), S );

                            blas::prod( value_t(1), UT, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // C = C + U_A ( S_A ( V_A' · B ) )
                            //
                            
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                            auto  VB    = blas::prod( value_t(1),
                                                      blas::adjoint( RA_ik->col_cb( op_A ).basis() ),
                                                      blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ) );
                            auto  SVB   = blas::prod( alpha,
                                                      blas::mat_view( op_A, RA_ik->coeff() ),
                                                      VB );

                            blas::prod( value_t(1), RA_ik->row_cb( op_A ).basis(), SVB,
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else if ( is_dense( A_ik ) )
                    {
                        auto  DA_ik = cptrcast( A_ik, hpro::TDenseMatrix );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            //
                            // C = C + ( ( A · U_B ) S_B ) V_B'
                            //

                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            auto  AU    = blas::prod( value_t(1),
                                                      blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                                      RB_kj->row_cb( op_B ).basis() );
                            auto  AUS   = blas::prod( alpha,
                                                      AU,
                                                      blas::mat_view( op_B, RB_kj->coeff() ) );

                            blas::prod( value_t(1), AUS, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // C = C + A · B
                            //
                            
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );

                            blas::prod( alpha,
                                        blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                        blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ),
                                        value_t(1),
                                        hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C_ij->typestr() );

                auto  DD  = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

                blas::add( value_t(-1), blas::mat< value_t >( DC ), blas::mat< value_t >( DD ) );
                
                std::cout << A_ik->id() << " × " << B_kj->id() << " -> " << C_ij->id() << " : "
                          << blas::norm_F( blas::mat< value_t >( DD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) << std::endl;
            }// for
        }// for
    }// for
}

//
// matrix multiplication
//
template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
            
        blas::invert( blas::mat< value_t >( A_ii ) );

        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal so just solve with U, e.g. X_ji U_ii = M_ji
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, blas::mat< value_t >( A_ii ), value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji U_ii = Ũ_j Ŝ_ji Ṽ_i' U_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' U_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii'^-1 V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  U_j  = R_ji->row_cb().basis();
                auto  S_ji = R_ji->coeff();
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( A_ii ) ), V_i );

                addlr_local2< value_t >( *BA, *R_ji, j, i, U_j, MV_i, acc );
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                multiply< value_t >( value_t(-1),
                                     apply_normal, *BA->block( j, i ),
                                     apply_normal, *BA->block( i, l ),
                                     *BA->block( j, l ), acc );
            }// for
        }// for
    }// for
}

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
