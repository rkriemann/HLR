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

namespace detail
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
                
        blas::qr( Q_j, R_j, false );
                
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
                
        blas::qr( Q_i, R_i, false );
                    
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

    {
        auto  US1 = blas::prod( value_t(1), Ue_i, Se_ij );
        auto  M1  = blas::prod( value_t(1), US1, blas::adjoint( Ve_j ) );
            
        auto  US2 = blas::prod( value_t(1), Un_i, Sn_ij );
        auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( Vn_j ) );

        blas::add( value_t(-1), M1, M2 );
        std::cout << "addlr     : " << M_ij.id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    }
    
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

        {
            auto  US1 = blas::prod( value_t(1), R_ik->row_cb().basis(), S_ik );
            auto  M1  = blas::prod( value_t(1), US1, blas::adjoint( R_ik->col_cb().basis() ) );
            
            auto  US2 = blas::prod( value_t(1), Un_i, Sn_ik );
            auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( R_ik->col_cb().basis() ) );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "addlr row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
        }
        
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

        {
            auto  US1 = blas::prod( value_t(1), R_kj->row_cb().basis(), S_kj );
            auto  M1  = blas::prod( value_t(1), US1, blas::adjoint( R_kj->col_cb().basis() ) );
            
            auto  US2 = blas::prod( value_t(1), R_kj->row_cb().basis(), Sn_kj );
            auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( Vn_j ) );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "addlr col : " << R_kj->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
        }
        
        R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un_i ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn_j ) );
}

//
// perform α A_ik · B_kj + C_ij
//
template < typename value_t >
void
multiply ( const value_t               alpha,
           const matop_t               op_A,
           const hpro::TBlockMatrix &  A,
           const matop_t               op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const uint                  i,
           const uint                  k,
           const uint                  j,
           const hpro::TTruncAcc &     acc )
{
    auto  A_ik = A.block( i, k );
    auto  B_kj = B.block( k, j );
    auto  C_ij = C.block( i, j );
    
    auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
    auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
    auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    blas::prod( alpha, blas::mat< value_t >( DA ), blas::mat< value_t >( DB ),
                value_t(1), blas::mat< value_t >( DC ) );

    HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

    // if (( A_ik->id() == 17 ) && ( B_kj->id() == 3 ) && ( C_ij->id() == 19 ))
    //     std::cout << std::endl;
    
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
                // U S_C V' + α U S_A W' · X S_B V' =
                // U ( S_C + α S_A W' · X S_B ) V'
                //
                            
                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WX    = blas::prod( value_t(1), blas::adjoint( RA_ik->col_cb().basis() ), RB_kj->row_cb().basis() );
                auto  SWX   = blas::prod( value_t(1), RA_ik->coeff(), WX );

                blas::prod( alpha, SWX, RB_kj->coeff(), value_t(1), RC_ij->coeff() );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α U S_A W' · B
                //
                // add low-rank update α ( U · S_A ) ( W' · B ) to C and update bases
                //
                            
                auto        DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                const auto  US    = blas::prod( alpha,      RA_ik->row_cb().basis(), RA_ik->coeff() );
                const auto  BW    = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( DB_kj ) ), RA_ik->col_cb().basis() );

                addlr_local< value_t >( C, *RC_ij, i, j, US, BW, acc );
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
                // U S_C V' + α A · W S_B V'
                //
                // add low-rank update ( A W ) ( V S_B' )' to C and update bases
                //
                            
                auto        RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                const auto  AW    = blas::prod( alpha,      blas::mat< value_t >( DA_ik ), RB_kj->row_cb().basis() );
                const auto  VS    = blas::prod( value_t(1), RB_kj->col_cb().basis(), blas::adjoint( RB_kj->coeff() ) );

                addlr_local< value_t >( C, *RC_ij, i, j, AW, VS, acc );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α A · B
                //
                // compute A·B, convert to low-rank, add to C and update bases
                //
                            
                auto        DB_kj    = cptrcast( B_kj, hpro::TDenseMatrix );
                auto        AB       = blas::prod( alpha, blas::mat< value_t >( DA_ik ), blas::mat< value_t >( DB_kj ) );
                const auto  [ W, X ] = approx::svd( AB, acc );

                addlr_local< value_t >( C, *RC_ij, i, j, W, X, acc );
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
                // C = C + α U S_A ( W' · X ) S_B V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WX    = blas::prod( value_t(1), blas::adjoint( RA_ik->col_cb().basis() ), RB_kj->row_cb().basis() );
                auto  SWX   = blas::prod( value_t(1), RA_ik->coeff(), WX );
                auto  SWXS  = blas::prod( value_t(1), SWX, RB_kj->coeff() );
                auto  US    = blas::prod( value_t(1), RA_ik->row_cb( op_A ).basis(), SWXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                auto  VB    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_cb( op_A ).basis() ),
                                          blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ) );
                auto  SVB   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VB );

                blas::prod( alpha,      RA_ik->row_cb( op_A ).basis(), SVB,
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
                // C = C + ( ( A · U ) S_B ) V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  AU    = blas::prod( value_t(1),
                                          blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                          RB_kj->row_cb( op_B ).basis() );
                auto  AUS   = blas::prod( value_t(1),
                                          AU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      AUS, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
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
}

//
// perform α A_ik · D_kk · B_kj + C_ij
//
template < typename value_t >
void
multiply ( const value_t               alpha,
           const matop_t               op_A,
           const hpro::TBlockMatrix &  A,
           const matop_t               op_D,
           const hpro::TBlockMatrix &  D,
           const matop_t               op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const uint                  i,
           const uint                  k,
           const uint                  j,
           const hpro::TTruncAcc &     acc )
{
    auto  A_ik = A.block( i, k );
    auto  D_kk = D.block( k, k );
    auto  B_kj = B.block( k, j );
    auto  C_ij = C.block( i, j );
    
    auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
    auto  DD = hlr::seq::matrix::convert_to_dense< value_t >( * D_kk );
    auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
    auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    auto  AxD = blas::prod( value_t(1), blas::mat< value_t >( DA ), blas::mat< value_t >( DD ) );
    
    blas::prod( alpha, AxD, blas::mat< value_t >( DB ), value_t(1), blas::mat< value_t >( DC ) );

    HLR_ASSERT( ! is_null_any( A_ik, D_kk, B_kj ) );
    HLR_ASSERT(   is_dense( D_kk ) );

    //
    // due to TLR format, C_ij, A_ik and B_kj can only be dense or uniform-lowrank
    // hence, handle all combinations
    //

    auto  DD_kk = cptrcast( D_kk, hpro::TDenseMatrix );

    if ( matrix::is_uniform_lowrank( C_ij ) )
    {
        auto  RC_ij = ptrcast( C_ij, matrix::uniform_lrmatrix< value_t > );
                    
        if ( matrix::is_uniform_lowrank( A_ik ) )
        {
            auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // U S_C V' + α U S_A W' · D · X S_B V' =
                // U ( S_C + α S_A W' · D · X S_B ) V'
                //
                            
                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WD    = blas::prod( value_t(1), blas::adjoint( RA_ik->col_cb().basis() ), blas::mat< value_t >( DD_kk ) );
                auto  WDX   = blas::prod( value_t(1), WD, RB_kj->row_cb().basis() );
                auto  SWDX  = blas::prod( value_t(1), RA_ik->coeff(), WDX );

                blas::prod( alpha, SWDX, RB_kj->coeff(), value_t(1), RC_ij->coeff() );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α U S_A W' · D · B
                //
                // add low-rank update α ( U · S_A ) ( B' · D' · W )' to C and update bases
                //
                            
                auto        DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                const auto  US    = blas::prod( alpha,      RA_ik->row_cb().basis(), RA_ik->coeff() );
                const auto  DW    = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( DD_kk ) ), RA_ik->col_cb().basis() );
                const auto  BDW   = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( DB_kj ) ), DW );

                addlr_local< value_t >( C, *RC_ij, i, j, US, BDW, acc );
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
                // U S_C V' + α A · D · W S_B V'
                //
                // add low-rank update ( A D W ) ( V S_B' )' to C and update bases
                //
                            
                auto        RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                const auto  DW    = blas::prod( alpha,      blas::mat< value_t >( DD_kk ), RB_kj->row_cb().basis() );
                const auto  ADW   = blas::prod( value_t(1), blas::mat< value_t >( DA_ik ), DW );
                const auto  VS    = blas::prod( value_t(1), RB_kj->col_cb().basis(), blas::adjoint( RB_kj->coeff() ) );

                addlr_local< value_t >( C, *RC_ij, i, j, ADW, VS, acc );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α A · D · B
                //
                // compute A·B, convert to low-rank, add to C and update bases
                //
                            
                auto        DB_kj    = cptrcast( B_kj, hpro::TDenseMatrix );
                auto        AD       = blas::prod( alpha, blas::mat< value_t >( DA_ik ), blas::mat< value_t >( DD_kk ) );
                auto        ADB      = blas::prod( value_t(1), AD, blas::mat< value_t >( DB_kj ) );
                const auto  [ W, X ] = approx::svd( ADB, acc );

                addlr_local< value_t >( C, *RC_ij, i, j, W, X, acc );
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
                // C = C + α U S_A ( W' · D · X ) S_B V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WD    = blas::prod( value_t(1), blas::adjoint( RA_ik->col_cb().basis() ), blas::mat< value_t >( DD_kk ) );
                auto  WDX   = blas::prod( value_t(1), WD, RB_kj->row_cb().basis() );
                auto  SWDX  = blas::prod( value_t(1), RA_ik->coeff(), WDX );
                auto  SWDXS = blas::prod( value_t(1), SWDX, RB_kj->coeff() );
                auto  US    = blas::prod( value_t(1), RA_ik->row_cb( op_A ).basis(), SWDXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · D · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                auto  VD    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_cb( op_A ).basis() ),
                                          blas::mat_view( op_D, hpro::blas_mat< value_t >( DD_kk ) ) );
                auto  VDB   = blas::prod( value_t(1),
                                          VD,
                                          blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ) );
                auto  SVDB  = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VDB );

                blas::prod( alpha,      RA_ik->row_cb( op_A ).basis(), SVDB,
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
                // C = C + ( ( A · D · U ) S_B ) V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  DU    = blas::prod( value_t(1),
                                          blas::mat_view( op_D, hpro::blas_mat< value_t >( DD_kk ) ),
                                          RB_kj->row_cb( op_B ).basis() );
                auto  ADU   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                          DU );
                auto  ADUS  = blas::prod( value_t(1),
                                          ADU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      ADUS, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + A · D · B
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                auto  AD    = blas::prod( value_t(1),
                                          blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                          blas::mat_view( op_D, hpro::blas_mat< value_t >( DD_kk ) ) );


                blas::prod( alpha,
                            AD,
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

    auto  TD  = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    blas::add( value_t(-1), blas::mat< value_t >( DC ), blas::mat< value_t >( TD ) );
                
    std::cout << A_ik->id() << " × " << D_kk->id() << " × " << B_kj->id() << " -> " << C_ij->id() << " : "
              << blas::norm_F( blas::mat< value_t >( TD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) << std::endl;
}

//
// extend column basis by X
//
template < typename value_t >
void
extend_col_basis ( hpro::TBlockMatrix &                   M,
                   matrix::uniform_lrmatrix< value_t > &  M_ij,
                   const uint                             i,
                   const uint                             j,
                   const blas::matrix< value_t > &        X,
                   const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // determine number of rows of matrix R below (sum of row ranks)
    size_t  nrows_Q    = 0;
    bool    have_other = false;
    
    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) )
        {
            if ( k != i )
                have_other = true;
            
            nrows_Q += cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > )->row_rank();
        }// if
    }// for

    if ( ! have_other )
    {
        // since there is no other low-rank block, just replace basis by X
        const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( blas::copy( X ) ) );
        return;
    }// if
    
    // extended column basis
    auto  V  = M_ij.col_cb().basis();
    auto  Ve = blas::join_row< value_t >( { V, X } );

    io::matlab::write( V, "V" );
    io::matlab::write( Ve, "Ve" );
    io::matlab::write( X, "X" );
    // compute QR of row basis for each block in column and assemble
    // all results into common matrix Q
    auto    Qe  = blas::matrix< value_t >( nrows_Q, Ve.ncols() );
    size_t  pos = 0;

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) )
        {
            const auto  R_kj     = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
            const auto  rank_k   = R_kj->row_rank();
            auto        U_k      = blas::copy( R_kj->row_cb().basis() );
            auto        R        = blas::matrix< value_t >( rank_k, rank_k );

            io::matlab::write( U_k, "Uk" );
            
            blas::qr( U_k, R, false );

            // std::cout << blas::norm_F( R ) << std::endl;
            
            const auto  S_kj     = R_kj->coeff();
            auto        T        = blas::prod( value_t(1), R, S_kj );

            blas::scale( value_t(1) / blas::norm_2( T ), T );
            
            io::matlab::write( S_kj, "Skj" );
            
            if ( k == i )
            {
                auto  Qe_k = blas::matrix( Qe,
                                           blas::range( pos, pos + rank_k-1 ),
                                           blas::range( V.ncols(), Ve.ncols() - 1 ) );

                blas::copy( T, Qe_k );
            }// if
            else
            {
                auto  Qe_k = blas::matrix( Qe,
                                           blas::range( pos, pos + rank_k-1 ),
                                           blas::range( 0, V.ncols() - 1 ) );

                blas::copy( T, Qe_k );
            }// else

            pos += rank_k;
        }// if
    }// for

    io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    io::matlab::write( R, "R" );
    
    auto  VeR = blas::prod( value_t(1), Ve, blas::adjoint( R ) );
    auto  Ss  = blas::vector< real_t >();

    blas::svd( VeR, Ss );

    io::matlab::write( VeR, "VeR" );
    io::matlab::write( Ss, "Ss" );
    
    const auto  rank   = acc.trunc_rank( Ss );
    const auto  V_rank = blas::matrix( VeR, blas::range::all, blas::range( 0, rank-1 ) );
    auto        Vn     = blas::copy( V_rank );
    // auto        Vn     = blas::copy( Ve );

    // blas::qr( Vn, R );
    
    io::matlab::write( Vn, "Vn" );

    const auto  TV     = blas::prod( value_t(1), blas::adjoint( Vn ), Ve );

    io::matlab::write( TV, "TV" );

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   (S_kj 0) TV  or  ( 0 S_ij ) TV
    //

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if ( ! matrix::is_uniform_lowrank( B_kj ) )
            continue;
                    
        auto  R_kj = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        auto  S_kj = R_kj->coeff();
        auto  Z    = blas::zeros< value_t >( S_kj.nrows(), X.ncols() );

        if ( k == i )
        {
            auto  Se_kj = blas::join_row< value_t >( { Z, S_kj } );
            auto  Sn_kj = blas::prod( value_t(1), Se_kj, blas::adjoint( TV ) );

            auto  US1   = blas::prod( value_t(1), R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( value_t(1), US1, blas::adjoint( X ) );

            auto  US2   = blas::prod( value_t(1), R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( value_t(1), US2, blas::adjoint( Vn ) );

            io::matlab::write( M1, "M1" );
            io::matlab::write( M2, "M2" );
            io::matlab::write( Se_kj, "Se" );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "extend col : " << R_kj->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// if
        else
        {
            auto  Se_kj = blas::join_row< value_t >( { S_kj, Z } );
            auto  Sn_kj = blas::prod( value_t(1), Se_kj, blas::adjoint( TV ) );

            auto  US1   = blas::prod( value_t(1), R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( value_t(1), US1, blas::adjoint( R_kj->col_cb().basis() ) );

            auto  US2   = blas::prod( value_t(1), R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( value_t(1), US2, blas::adjoint( Vn ) );

            io::matlab::write( M1, "M1" );
            io::matlab::write( M2, "M2" );
            io::matlab::write( Se_kj, "Se" );
            
            blas::add( value_t(-1), M1, M2 );
            std::cout << "extend col : " << R_kj->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// else

        io::matlab::write( R_kj->row_cb().basis(), "U" );
        io::matlab::write( R_kj->coeff(), "Sn" );
        io::matlab::write( S_kj, "S" );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

//
// extend row basis by W
//
template < typename value_t >
void
extend_row_basis ( hpro::TBlockMatrix &                   M,
                   matrix::uniform_lrmatrix< value_t > &  M_ij,
                   const uint                             i,
                   const uint                             j,
                   const blas::matrix< value_t > &        W,
                   const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // determine number of rows of matrix R below (sum of column ranks)
    size_t  nrows_Q    = 0;
    bool    have_other = false;
    
    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( matrix::is_uniform_lowrank( M_ik ) )
        {
            if ( k != j )
                have_other = true;
            
            nrows_Q += cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > )->col_rank();
        }// if
    }// for

    if ( ! have_other )
    {
        // since there is no other low-rank block, just replace basis by X and return
        const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( blas::copy( W ) ) );
        return;
    }// if
    
    // extended row basis
    auto  U  = M_ij.row_cb().basis();
    auto  Ue = blas::join_row< value_t >( { U, W } );

    io::matlab::write( U, "U" );
    io::matlab::write( Ue, "Ue" );
    io::matlab::write( W, "W" );
    
    // compute QR of column basis for each block in row and assemble
    // all results into common matrix Q
    auto    Qe  = blas::matrix< value_t >( nrows_Q, Ue.ncols() );
    size_t  pos = 0;

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( ! matrix::is_uniform_lowrank( M_ik ) )
            continue;
        
        const auto  R_ik   = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
        const auto  rank_k = R_ik->col_rank();
        auto        V_k    = blas::copy( R_ik->col_cb().basis() );
        auto        R      = blas::matrix< value_t >( rank_k, rank_k );

        io::matlab::write( V_k, "Vk" );
            
        blas::qr( V_k, R, false );
        
        // std::cout << blas::norm_F( R ) << std::endl;
        
        const auto  S_ik  = R_ik->coeff();
        auto        RS_ik = blas::prod( value_t(1), R, blas::adjoint( S_ik ) );

        // scale each matrix by norm (|R·S'| = |R_ik| assuming orthogonal bases)
        // to givt each block equal weight in computed row basis
        blas::scale( value_t(1) / blas::norm_2( RS_ik ), RS_ik );
            
        io::matlab::write( S_ik, "Sik" );
            
        if ( k == j )
        {
            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( U.ncols(), Ue.ncols() - 1 ) );

            blas::copy( RS_ik, Qe_k );
        }// if
        else
        {
            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( 0, U.ncols() - 1 ) );

            blas::copy( RS_ik, Qe_k );
        }// else

        io::matlab::write( Qe, "Qe" );
        
        pos += rank_k;
    }// for

    io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    io::matlab::write( R, "R" );
    
    auto  UeR = blas::prod( value_t(1), Ue, blas::adjoint( R ) );
    auto  Ss  = blas::vector< real_t >();

    blas::svd( UeR, Ss );

    io::matlab::write( UeR, "UeR" );
    io::matlab::write( Ss, "Ss" );
    
    const auto  rank   = acc.trunc_rank( Ss );
    const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank-1 ) );
    auto        Un     = blas::copy( U_rank );
    
    io::matlab::write( Un, "Un" );

    const auto  TU     = blas::prod( value_t(1), blas::adjoint( Un ), Ue );

    io::matlab::write( TU, "TU" );

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   TU ⎛S_kj⎞  or  TU ⎛  0 ⎞
    //      ⎝ 0  ⎠         ⎝S_kj⎠
    //

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if ( ! matrix::is_uniform_lowrank( B_ik ) )
            continue;
                    
        auto  R_ik = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        auto  S_ik = R_ik->coeff();
        auto  Z    = blas::zeros< value_t >( W.ncols(), S_ik.ncols() );

        if ( k == j )
        {
            auto  Se_ik = blas::join_col< value_t >( { Z, S_ik } );
            auto  Sn_ik = blas::prod( value_t(1), TU, Se_ik );

            auto  US1   = blas::prod( value_t(1), W, S_ik );
            auto  M1    = blas::prod( value_t(1), US1, blas::adjoint( R_ik->col_cb().basis() ) );

            auto  US2   = blas::prod( value_t(1), Un, Sn_ik );
            auto  M2    = blas::prod( value_t(1), US2, blas::adjoint( R_ik->col_cb().basis() ) );

            io::matlab::write( M1, "M1" );
            io::matlab::write( M2, "M2" );
            io::matlab::write( Se_ik, "Se" );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
        else
        {
            auto  Se_ik = blas::join_col< value_t >( { S_ik, Z } );
            auto  Sn_ik = blas::prod( value_t(1), TU, Se_ik );

            auto  US1   = blas::prod( value_t(1), R_ik->row_cb().basis(), S_ik );
            auto  M1    = blas::prod( value_t(1), US1, blas::adjoint( R_ik->col_cb().basis() ) );

            auto  US2   = blas::prod( value_t(1), Un, Sn_ik );
            auto  M2    = blas::prod( value_t(1), US2, blas::adjoint( R_ik->col_cb().basis() ) );

            io::matlab::write( M1, "M1" );
            io::matlab::write( M2, "M2" );
            io::matlab::write( Se_ik, "Se" );
            
            blas::add( value_t(-1), M1, M2 );
            std::cout << "extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// else

        io::matlab::write( R_ik->col_cb().basis(), "V" );
        io::matlab::write( R_ik->coeff(), "Sn" );
        io::matlab::write( S_ik, "S" );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
}

}// namespace detail

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

                detail::addlr_local( *B, *R_ij, i, j, W_i, X_j, acc );
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
            HLR_ASSERT( ! is_null( C->block( i, j ) ) );

            for ( uint  k = 0; k < A->nblock_cols( op_A ); ++k )
            {
                detail::multiply( alpha, op_A, *A, op_B, *B, *C, i, k, j, acc );
            }// for
        }// for
    }// for
}

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
//
template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const hpro::TMatrix &    REF )
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

        {
            auto  BREF   = cptrcast( &REF, hpro::TBlockMatrix );
            auto  REF_ii = cptrcast( BREF->block( i, i ), hpro::TDenseMatrix );
            auto  DA     = blas::copy( blas::mat< value_t >( A_ii ) );
            auto  DREF   = blas::copy( blas::mat< value_t >( REF_ii ) );

            blas::add( value_t(-1), DREF, DA );

            std::cout << "REF : " << A_ii->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
        }

        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is unit diagonal so just solve with U, e.g. X_ji U_ii = M_ji
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, blas::mat< value_t >( A_ii ), value_t(0), blas::mat< value_t >( D_ji ) );

                {
                    auto  BREF   = cptrcast( &REF, hpro::TBlockMatrix );
                    auto  REF_ji = cptrcast( BREF->block( j, i ), hpro::TDenseMatrix );
                    auto  DA     = blas::copy( blas::mat< value_t >( D_ji ) );
                    auto  DREF   = blas::copy( blas::mat< value_t >( REF_ji ) );

                    blas::add( value_t(-1), DREF, DA );

                    std::cout << "REF : " << A_ji->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
                }
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji U_ii = Ũ_j Ŝ_ji Ṽ_i' U_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' U_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  U_j  = R_ji->row_cb().basis();
                auto  S_ji = R_ji->coeff();
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( value_t(1), blas::adjoint( blas::mat< value_t >( A_ii ) ), V_i );
                auto  R    = blas::matrix< value_t >();

                blas::qr( MV_i, R );

                auto  Sn_ji = blas::prod( value_t(1), S_ji, blas::adjoint( R ) );

                auto  US  = blas::prod( value_t(1), U_j, S_ji );
                auto  USV = blas::prod( value_t(1), US,  blas::adjoint( V_i ) );
                auto  M1  = blas::prod( value_t(1), USV, blas::mat< value_t >( A_ii ) );

                R_ji->set_coeff_unsafe( Sn_ji );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc );
                
                auto  US2 = blas::prod( value_t(1), R_ji->row_cb().basis(), R_ji->coeff() );
                auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( R_ji->col_cb().basis() ) );

                io::matlab::write( M2, "M2" );
                
                blas::add( value_t(-1), M1, M2 );

                io::matlab::write( M1, "M1" );
                std::cout << "solve " << R_ji->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

                {
                    auto  BREF    = cptrcast( &REF, hpro::TBlockMatrix );
                    auto  REF_ji  = cptrcast( BREF->block( j, i ), hpro::TRkMatrix );
                    auto  DA_ji   = matrix::convert_to_dense< value_t >( *R_ji );
                    auto  DREF_ji = matrix::convert_to_dense< value_t >( *REF_ji );
                    auto  DA      = blas::copy( blas::mat< value_t >( DA_ji ) );
                    auto  DREF    = blas::copy( blas::mat< value_t >( DREF_ji ) );

                    io::matlab::write( *DA_ji, "A" );
                    io::matlab::write( *DREF_ji, "REF" );
                    
                    blas::add( value_t(-1), DREF, DA );

                    std::cout << "REF : " << A_ji->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
                }
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, j, i, l, acc );
            }// for
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
//
template < typename value_t >
void
ldu ( hpro::TMatrix &          A,
      const hpro::TTruncAcc &  acc,
      const hpro::TMatrix &    REF )
{
    HLR_LOG( 4, hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );

        HLR_ASSERT( is_dense( A_ii ) );
        
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat< value_t >( ptrcast( A_ii, hpro::TDenseMatrix ) );
            
        blas::invert( D_ii );

        {
            auto  BREF   = cptrcast( &REF, hpro::TBlockMatrix );
            auto  REF_ii = cptrcast( BREF->block( i, i ), hpro::TDenseMatrix );
            auto  DA     = blas::copy( D_ii );
            auto  DREF   = blas::copy( blas::mat< value_t >( REF_ii ) );

            blas::add( value_t(-1), DREF, DA );

            std::cout << "REF : " << A_ii->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
        }

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );

                {
                    auto  BREF   = cptrcast( &REF, hpro::TBlockMatrix );
                    auto  REF_ji = cptrcast( BREF->block( j, i ), hpro::TDenseMatrix );
                    auto  DA     = blas::copy( blas::mat< value_t >( D_ji ) );
                    auto  DREF   = blas::copy( blas::mat< value_t >( REF_ji ) );

                    blas::add( value_t(-1), DREF, DA );

                    std::cout << "REF : " << A_ji->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
                }
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji D_ii = Ũ_j Ŝ_ji Ṽ_i' D_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' D_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( D_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  U_j  = R_ji->row_cb().basis();
                auto  S_ji = R_ji->coeff();
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( value_t(1), blas::adjoint( D_ii ), V_i );
                auto  R    = blas::matrix< value_t >();

                blas::qr( MV_i, R );

                auto  Sn_ji = blas::prod( value_t(1), S_ji, blas::adjoint( R ) );

                auto  US  = blas::prod( value_t(1), U_j, S_ji );
                auto  USV = blas::prod( value_t(1), US,  blas::adjoint( V_i ) );
                auto  M1  = blas::prod( value_t(1), USV, D_ii );

                R_ji->set_coeff_unsafe( Sn_ji );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc );
                
                auto  US2 = blas::prod( value_t(1), R_ji->row_cb().basis(), R_ji->coeff() );
                auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( R_ji->col_cb().basis() ) );

                io::matlab::write( M2, "M2" );
                
                blas::add( value_t(-1), M1, M2 );

                io::matlab::write( M1, "M1" );
                std::cout << "solve upper " << R_ji->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

                {
                    auto  BREF    = cptrcast( &REF, hpro::TBlockMatrix );
                    auto  REF_ji  = cptrcast( BREF->block( j, i ), hpro::TRkMatrix );
                    auto  DA_ji   = matrix::convert_to_dense< value_t >( *R_ji );
                    auto  DREF_ji = matrix::convert_to_dense< value_t >( *REF_ji );
                    auto  DA      = blas::copy( blas::mat< value_t >( DA_ji ) );
                    auto  DREF    = blas::copy( blas::mat< value_t >( DREF_ji ) );

                    io::matlab::write( *DA_ji, "A" );
                    io::matlab::write( *DREF_ji, "REF" );
                    
                    blas::add( value_t(-1), DREF, DA );

                    std::cout << "REF : " << A_ji->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
                }
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  U_ij = BA->block( i, j );


            // auto  DT_ij = hlr::seq::matrix::convert_to_dense< value_t >( *U_ij );
            // auto  DC_ij = blas::copy( blas::mat< value_t >( DT_ij ) );

            // blas::prod( value_t(1), D_ii, DC_ij, value_t(0), blas::mat< value_t >( DT_ij ) );

            
            
            if ( is_dense( U_ij ) )
            {
                auto  D_ij = ptrcast( U_ij, hpro::TDenseMatrix );
                auto  T_ij = blas::copy( blas::mat< value_t >( D_ij ) );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), blas::mat< value_t >( D_ij ) );
            }// else
            else if ( matrix::is_uniform_lowrank( U_ij ) )
            {
                // U_ij = W·T·X' = D_ii^-1·U·S·V' = D_ii^-1·A_ij
                // ⟶ W = D_ii^-1·U, T=S, X = V
                auto  R_ij = ptrcast( U_ij, matrix::uniform_lrmatrix< value_t > );
                auto  U_i  = R_ij->row_cb().basis();
                auto  S_ij = R_ij->coeff();
                auto  V_j  = R_ij->col_cb().basis();
                auto  MU_i = blas::prod( value_t(1), D_ii, U_i );
                auto  R    = blas::matrix< value_t >();

                // ensure orthogonal bases (and update coefficients)
                blas::qr( MU_i, R );

                auto  Sn_ij = blas::prod( value_t(1), R, S_ij );


                auto  US  = blas::prod( value_t(1), U_i, S_ij );
                auto  USV = blas::prod( value_t(1), US,  blas::adjoint( V_j ) );
                auto  M1  = blas::prod( value_t(1), D_ii, USV );

                R_ij->set_coeff_unsafe( Sn_ij );

                detail::extend_row_basis< value_t >( *BA, *R_ij, i, j, MU_i, acc );
                

                auto  US2 = blas::prod( value_t(1), R_ij->row_cb().basis(), R_ij->coeff() );
                auto  M2  = blas::prod( value_t(1), US2, blas::adjoint( R_ij->col_cb().basis() ) );

                blas::add( value_t(-1), M1, M2 );

                std::cout << "solve lower " << R_ij->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

                {
                    auto  BREF    = cptrcast( &REF, hpro::TBlockMatrix );
                    auto  REF_ij  = cptrcast( BREF->block( i, j ), hpro::TRkMatrix );
                    auto  DA_ij   = matrix::convert_to_dense< value_t >( *R_ij );
                    auto  DREF_ij = matrix::convert_to_dense< value_t >( *REF_ij );
                    auto  DA      = blas::copy( blas::mat< value_t >( DA_ij ) );
                    auto  DREF    = blas::copy( blas::mat< value_t >( DREF_ij ) );

                    // io::matlab::write( *DA_ji, "A" );
                    // io::matlab::write( *DREF_ji, "REF" );
                    
                    blas::add( value_t(-1), DREF, DA );

                    std::cout << "REF : " << U_ij->id() << " : " << blas::norm_F( DA ) / blas::norm_F( DREF ) << std::endl;
                }
            }// if


            // auto  TT_ij = hlr::seq::matrix::convert_to_dense< value_t >( *U_ij );

            // blas::add( value_t(-1), blas::mat< value_t >( *DT_ij ), blas::mat< value_t >( *TT_ij ) );

            // std::cout << U_ij->id() << " : " << blas::norm_F( blas::mat< value_t >( *TT_ij ) ) << std::endl;
        }// for

        //
        // update trailing sub matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                detail::multiply( value_t(-1),
                                  apply_normal, *BA,
                                  apply_normal, *BA,
                                  apply_normal, *BA,
                                  *BA, j, i, l, acc );
            }// for
        }// for
    }// for
}

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
