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

    using  real_t = typename hpro::real_type< value_t >::type_t;
    
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

                //////////////////////////////////////////////////////
                //
                // compute block row bases for Z = [ Z_0, ..., Z_q-1 ] for
                // all q low-rank blocks Z_k in i'th block row
                //
                //////////////////////////////////////////////////////

                auto  M1 = blas::matrix< value_t >();
                
                {
                    auto  V_j  = R_ij->col_cb().basis();
                    auto  U_i  = R_ij->row_cb().basis();
                    auto  S_ij = R_ij->coeff();

                    auto  US   = blas::prod( value_t(1), U_i, S_ij );
                    auto  USV  = blas::prod( value_t(1), US, blas::adjoint( V_j ) );

                    io::matlab::write(  V_j, "V" );
                    io::matlab::write(  U_i, "U" );
                    io::matlab::write( S_ij, "S" );
                    io::matlab::write(  USV, "M" );
                    
                    blas::prod( value_t(1), W_i, blas::adjoint( X_j ), value_t(1), USV );

                    io::matlab::write( USV, "M1" );
                    io::matlab::write( W_i, "W" );
                    io::matlab::write( X_j, "X" );

                    M1 = std::move( USV );
                }
                
                //
                // compute QR decomposition for Q R_j = [ V_j, X_j ] with [ V_j, X_j ]
                // being the extended column cluster basis
                //
                
                auto  U_i   = R_ij->row_cb().basis();
                auto  V_j   = R_ij->col_cb().basis();
                auto  S_ij  = R_ij->coeff();
                auto  Ue_i  = blas::join_row< value_t >( { U_i, W_i } );
                auto  Se_ij = blas::diag< value_t >( { S_ij, I } );
                auto  Ve_j  = blas::join_row< value_t >( { V_j, X_j } );

                io::matlab::write( Ue_i, "Uei" );
                io::matlab::write( Ve_j, "Vej" );
                io::matlab::write( Se_ij, "Seij" );
                
                auto  R_j = blas::matrix< value_t >();

                {
                    auto  Q_j = blas::copy( Ve_j ); // need to copy since modified during QR
                
                    blas::qr_wrapper( Q_j, R_j, false );
                }

                io::matlab::write( R_j, "Rj" );
                
                auto  SR_ij = blas::prod( value_t(1), Se_ij, blas::adjoint( R_j ) );
                io::matlab::write( SR_ij, "SRij" );
                auto  UeR   = blas::prod( value_t(1), Ue_i, SR_ij );
                auto  Usv   = blas::vector< real_t >();

                io::matlab::write( UeR, "UeR" );
                
                blas::svd( UeR, Usv );

                const auto  rank_U = acc.trunc_rank( Usv );
                const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank_U-1 ) );
                const auto  Un_i   = blas::copy( U_rank );

                io::matlab::write( Un_i, "Uni" );


                auto  R_i = blas::matrix< value_t >();

                {
                    auto  Q_i = blas::copy( Ue_i ); // need to copy since modified during QR
                
                    blas::qr_wrapper( Q_i, R_i, false );
                }

                io::matlab::write( R_i, "Ri" );
                
                auto  RS_ij = blas::prod( value_t(1), R_i, Se_ij );
                auto  VeR   = blas::prod( value_t(1), Ve_j, blas::adjoint( RS_ij ) );
                auto  Vsv   = blas::vector< real_t >();

                blas::svd( VeR, Vsv );

                const auto  rank_V = acc.trunc_rank( Vsv );
                const auto  V_rank = blas::matrix( VeR, blas::range::all, blas::range( 0, rank_V-1 ) );
                const auto  Vn_j   = blas::copy( V_rank );

                io::matlab::write( Vn_j, "Vnj" );



                
                const auto  TU_i  = blas::prod( value_t(1), blas::adjoint( Un_i ), Ue_i );
                const auto  TV_j  = blas::prod( value_t(1), blas::adjoint( Vn_j ), Ve_j );
                auto        T_ij  = blas::prod( value_t(1), TU_i, Se_ij );
                auto        Sn_ij = blas::prod( value_t(1), T_ij, blas::adjoint( TV_j ) );

                io::matlab::write( TU_i, "TUi" );
                io::matlab::write( TV_j, "TVj" );
                io::matlab::write( Sn_ij, "Snij" );
                
                {
                    auto  US = blas::prod( value_t(1), Un_i, Sn_ij );
                    auto  M2 = blas::prod( value_t(1), US, blas::adjoint( Vn_j ) );

                    // io::matlab::write( M2, "M2" );
                    
                    const auto nM1 = blas::norm_F( M1 );
                    blas::add( value_t(-1), M2, M1 );
                    std::cout << R_ij->id() << " : error = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) / nM1 ) << std::endl;
                }

                R_ij->set_coeff_unsafe( std::move( Sn_ij ) );
                
                //
                // transform coefficients for blocks in current block row/column
                //

                for ( uint  k = 0; k < B->nblock_cols(); ++k )
                {
                    if ( k == j )
                        continue;
                    
                    auto  B_ik = B->block( i, k );
                    
                    if ( ! matrix::is_uniform_lowrank( B_ik ) )
                        continue;
                    
                    auto        R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
                    const auto  S_ik  = R_ik->coeff();

                    io::matlab::write( S_ik, "Sik" );
                    io::matlab::write( R_ik->row_cb().basis(), "Uik" );
                    io::matlab::write( R_ik->col_cb().basis(), "Vik" );

                    const auto  V_k   = R_ik->col_cb().basis();
                    auto        US    = blas::prod( value_t(1), U_i, S_ik );
                    auto        M1    = blas::prod( value_t(1), US, blas::adjoint( V_k ) );
                        
                    const auto  Se_ik = blas::extend( S_ik, X_j.ncols(), 0 );  // [ S_ik ; 0 ]
                    auto        Sn_ik = blas::prod( value_t(1), TU_i, Se_ik );

                    auto        TS    = blas::prod( value_t(1), Un_i, Sn_ik );
                    auto        M2    = blas::prod( value_t(1), TS, blas::adjoint( V_k ) );

                    const auto nM1 = blas::norm_F( M1 );
                    blas::add( value_t(-1), M2, M1 );
                    std::cout << R_ik->id() << " : U error = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) / nM1 ) << std::endl;

                    R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
                }// for
                
                for ( uint  k = 0; k < B->nblock_rows(); ++k )
                {
                    if ( k == i )
                        continue;
                    
                    auto  B_kj = B->block( k, j );
                    
                    if ( ! matrix::is_uniform_lowrank( B_kj ) )
                        continue;
                    
                    auto        R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                    const auto  S_kj  = R_kj->coeff();

                    auto        US    = blas::prod( value_t(1), R_kj->row_cb().basis(), S_kj );
                    auto        M1    = blas::prod( value_t(1), US, blas::adjoint( V_j ) );
                        
                    const auto  Se_kj = blas::extend( S_kj, 0, X_j.ncols() ); // [ S_kj, 0 ]
                    auto        Sn_kj = blas::prod( value_t(1), Se_kj, blas::adjoint( TV_j ) );

                    auto        TS    = blas::prod( value_t(1), R_kj->row_cb().basis(), Sn_kj );
                    auto        M2    = blas::prod( value_t(1), TS, blas::adjoint( Vn_j ) );

                    const auto nM1 = blas::norm_F( M1 );
                    blas::add( value_t(-1), M2, M1 );
                    std::cout << R_kj->id() << " : V error = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) / nM1 ) << std::endl;

                    R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
                }// for

                //
                // finally adjust cluster bases
                //

                const_cast< matrix::cluster_basis< value_t > * >( & R_ij->row_cb() )->set_basis( std::move( Un_i ) );
                const_cast< matrix::cluster_basis< value_t > * >( & R_ij->col_cb() )->set_basis( std::move( Vn_j ) );
                
                // //
                // // compute QR for joined matrices R_k
                // //
                
                // auto  Vs = blas::join_col( RSs );
                // auto  R  = blas::matrix< value_t >();

                // io::matlab::write( Vs, "Vs" );
                // blas::qr_wrapper( Vs, R, false );

                //
                // compute SVD from extended basis [ U_i, W_i ] and S_ij · R^T, e.g.,
                // for [ U_i, W_i ] · S_ij · R'
                // - only need singular values and left singular vectors
                //

                // auto  UeR = blas::prod( value_t(1), Ue_i, blas::adjoint( RS_ij ) );
                // auto  Sv  = blas::vector< real_t >();

                // io::matlab::write( UeR, "UeR" );
                
                // blas::svd( UeR, Sv );

                // io::matlab::write( UeR, "U" );
                // io::matlab::write( Sv,  "Sv" );

                // const auto  rank   = acc.trunc_rank( Sv );
                // const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank-1 ) );
                // const auto  Un_i   = blas::copy( U_rank );
                
                //
                // transform coefficients into new block row basis
                //
                
                // const auto  T_i  = blas::prod( value_t(1), blas::adjoint( Un_i ), Ue_i );
                // auto        S_ij = blas::prod( value_t(1), T_i, Se_k );

                // for ( uint  k = 0; k < B->nblock_cols(); ++k )
                // {
                //     auto  B_ik = B->block( i, k );
                    
                //     if ( ! matrix::is_uniform_lowrank( B_ik ) )
                //         continue;
                    
                //     auto  R_ik = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
                    
                //     if ( k == j )
                //     {
                //         auto  Se_k = blas::diag< value_t >( { R_ik->coeff(), I } );
                //         auto  S_k  = blas::prod( value_t(1), T_i, Se_k );
                        
                //         R_ik->set_coeff( S_k );
                //     }// if
                //     else
                //     {
                //         auto  Se_k = blas::diag< value_t >( { R_ik->coeff(), Z } );
                //         auto  S_k  = blas::prod( value_t(1), T_i, Se_k );
                        
                //         R_ik->set_coeff( S_k );
                //     }// if
                // }// for
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
    auto  C = cptrcast( &aC, hpro::TBlockMatrix );

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

                HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

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
                            // add low-rank update U ( S_A W' · B ) to C and update bases
                            //
                            
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
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
                            // add low-rank update ( A W S_B ) V' to C and update bases
                            //
                            
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // U S_C V' + A · B
                            //
                            // compute A·B, convert to low-rank, add to C and update bases
                            //
                            
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
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
                            // C = C + U_A S_A V_A' · U_B S_B V_B'
                            //   = C + U_A ( S_A S_B ) V_B'
                            //

                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            auto  S     = blas::prod( alpha,
                                                      blas::mat_view( op_A, RA_ik->coeff() ),
                                                      blas::mat_view( op_B, RB_kj->coeff() ) );
                            auto  UT    = blas::prod( value_t(1),
                                                      RA_ik->row_cb( op_A ).basis(),
                                                      S );

                            blas::prod( value_t(1), UT, RB_kj->col_cb( op_B ).basis(),
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            //
                            // C = C + U_A ( S_A ( V_A' · B ) )
                            //
                            
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                            auto  VB    = blas::prod( value_t(1),
                                                      RA_ik->col_cb( op_A ).basis(),
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

                            blas::prod( value_t(1), AUS, RB_kj->col_cb( op_B ).basis(),
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
            }// for
        }// for
    }// for
}

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
