#ifndef __HLR_ARITH_DETAIL_UNIFORM_TLR_HH
#define __HLR_ARITH_DETAIL_UNIFORM_TLR_HH
//
// Project     : HLR
// Module      : arith/uniform
// Description : arithmetic functions for uniform TLR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/approx/svd.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace uniform { namespace tlr { namespace detail {

//
// locally add low-rank update W·X to block M_ij
//
// template < typename value_t >
// void
// addlr_local ( Hpro::TBlockMatrix< value_t > &                   M,
//               matrix::uniform_lrmatrix< value_t > &  M_ij,
//               const uint                             i,
//               const uint                             j,
//               const blas::matrix< value_t > &        W,
//               const blas::matrix< value_t > &        X,
//               const Hpro::TTruncAcc &                acc )
// {
//     using  real_t = Hpro::real_type_t< value_t >;

//     // std::cout << "addlr : " << M_ij.id() << std::endl;

//     // current bases and coupling
//     const auto  U_i   = M_ij.row_basis();
//     const auto  S_ij  = M_ij.coeff();
//     const auto  V_j   = M_ij.col_basis();

//     // extended bases and coupling
//     const auto  rank  = W.ncols();
//     const auto  Ue_i  = blas::join_row< value_t >( { U_i, W } );
//     const auto  I     = blas::identity< value_t >( rank );
//     auto        Se_ij = blas::diag< value_t >( { S_ij, I } );
//     const auto  Ve_j  = blas::join_row< value_t >( { V_j, X } );

//     // const auto  norm_Mij = blas::norm_F( S_ij ); blas::norm_2( S_ij );
//     // const auto  norm_WX  = blas::norm_F( W, X ); hlr::norm::spectral( lowrank_operator{ W, X } );
        
//     // std::cout << norm_Mij << " / " << norm_WX << std::endl;

//     // const auto  scale = norm_WX / norm_Mij;
    
//     // for ( uint  l = 0; l < rank; ++l )
//     //     Se_ij( l + S_ij.nrows(), l + S_ij.ncols() ) *= scale;
              
//     // if ( true )
//     // {
//     //     io::matlab::write( U_i,   "U" );
//     //     io::matlab::write( S_ij,  "S" );
//     //     io::matlab::write( V_j,   "V" );
//     //     io::matlab::write( Ue_i,  "Ue" );
//     //     io::matlab::write( Se_ij, "Se" );
//     //     io::matlab::write( Ve_j,  "Ve" );
//     //     io::matlab::write( W,     "W" );
//     //     io::matlab::write( X,     "X" );
//     // }// if
    
//     //
//     // new row basis is computed as the left singular vectors of Ue_i · Se_ij · Ve_j'
//     // which can be simplified to Ue_i · Se_ij · R_j' Q_j' with QR decomposition
//     // Ve_j = Q_j R_j
//     //

//     auto  Un_i = blas::matrix< value_t >();
                
//     {
//         auto  R_j = blas::matrix< value_t >();
//         auto  Q_j = blas::copy( Ve_j ); // need to copy since modified during QR
                
//         blas::qr( Q_j, R_j, false );
                
//         const auto  SR_ij = blas::prod( Se_ij, blas::adjoint( R_j ) );
//         auto        Us    = blas::prod( Ue_i, SR_ij );
//         auto        Ss    = blas::vector< real_t >();

//         blas::svd( Us, Ss );
                    
//         const auto  rank_U = acc.trunc_rank( Ss );
//         const auto  U_rank = blas::matrix< value_t >( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

//         Un_i = std::move( blas::copy( U_rank ) );
//     }

//     //
//     // new column basis is computed as left singular vectors of Ve_j · Se_ij' · Ue_i'
//     // simplified to Ve_j · Se_ij' · R_i' · Q_i' with Ue_i = Q_i R_i
//     //

//     auto  Vn_j = blas::matrix< value_t >();
                
//     {
//         auto  R_i = blas::matrix< value_t >();
//         auto  Q_i = blas::copy( Ue_i ); // need to copy since modified during QR
                
//         blas::qr( Q_i, R_i, false );
                    
//         const auto  RS_ij = blas::prod( R_i, Se_ij );
//         auto        Us    = blas::prod( Ve_j, blas::adjoint( RS_ij ) );
//         auto        Ss    = blas::vector< real_t >();
                    
//         blas::svd( Us, Ss );
                    
//         const auto  rank_V = acc.trunc_rank( Ss );
//         const auto  V_rank = blas::matrix< value_t >( Us, blas::range::all, blas::range( 0, rank_V-1 ) );

//         Vn_j = std::move( blas::copy( V_rank ) );
//     }

//     // io::matlab::write( Un_i, "Un" );
//     // io::matlab::write( Vn_j, "Vn" );
    
//     //
//     // new B_ij is computed as
//     //
//     //     Un_i ( Un_i' Ue_i Se_ij Ve_j' Vn_j ) Vn_j'
//     //
//     // therefore new coupling matrix is
//     //
//     //     Un_i' Ue_i Se_ij Ve_j' Vn_j = TU_i Se_ij TVj'
//     //
//     // with
//     //
//     //     TU_i = Un_i' Ue_i  and  TV_j = Vn_j' Ve_j
//     //
                
//     for ( uint  l = 0; l < rank; ++l )
//         Se_ij( l + S_ij.nrows(), l + S_ij.ncols() ) = value_t(1);
    
//     const auto  TU_i  = blas::prod( blas::adjoint( Un_i ), Ue_i );
//     const auto  TV_j  = blas::prod( blas::adjoint( Vn_j ), Ve_j );
//     auto        T_ij  = blas::prod( TU_i, Se_ij );
//     auto        Sn_ij = blas::prod( T_ij, blas::adjoint( TV_j ) );

//     // {
//     //     auto  US1 = blas::prod( Ue_i, Se_ij );
//     //     auto  M1  = blas::prod( US1, blas::adjoint( Ve_j ) );
            
//     //     auto  US2 = blas::prod( Un_i, Sn_ij );
//     //     auto  M2  = blas::prod( US2, blas::adjoint( Vn_j ) );

//     //     blas::add( value_t(-1), M1, M2 );
//     //     std::cout << "addlr     : " << M_ij.id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
//     // }
    
//     M_ij.set_coeff_unsafe( std::move( Sn_ij ) );
                
//     //
//     // transform coupling matrix for blocks in current block row as
//     //
//     //   TU_i · ⎛S_ik⎞
//     //          ⎝  0 ⎠
//     //

//     for ( uint  k = 0; k < M.nblock_cols(); ++k )
//     {
//         auto  B_ik = M.block( i, k );
                    
//         if (( k == j ) || ! matrix::is_uniform_lowrank( B_ik ))
//             continue;
                    
//         auto        R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
//         const auto  S_ik  = R_ik->coeff();
//         const auto  Se_ik = blas::extend( S_ik, rank, 0 );  // [ S_ik ; 0 ]
//         auto        Sn_ik = blas::prod( TU_i, Se_ik );

//         // {
//         //     auto  US1 = blas::prod( R_ik->row_basis(), S_ik );
//         //     auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_basis() ) );
            
//         //     auto  US2 = blas::prod( Un_i, Sn_ik );
//         //     auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_basis() ) );

//         //     blas::add( value_t(-1), M1, M2 );
//         //     std::cout << "addlr row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
//         // }
        
//         R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
//     }// for
                
//     //
//     // transform coupling matrix for blocks in current block column as
//     //
//     //   (S_kj 0) TV_j
//     //

//     for ( uint  k = 0; k < M.nblock_rows(); ++k )
//     {
//         auto  B_kj = M.block( k, j );
                    
//         if (( k == i ) || ! matrix::is_uniform_lowrank( B_kj ))
//             continue;
                    
//         auto        R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
//         const auto  S_kj  = R_kj->coeff();
//         const auto  Se_kj = blas::extend( S_kj, 0, rank ); // [ S_kj, 0 ]
//         auto        Sn_kj = blas::prod( Se_kj, blas::adjoint( TV_j ) );

//         // {
//         //     auto  US1 = blas::prod( R_kj->row_basis(), S_kj );
//         //     auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_basis() ) );
            
//         //     auto  US2 = blas::prod( R_kj->row_basis(), Sn_kj );
//         //     auto  M2  = blas::prod( US2, blas::adjoint( Vn_j ) );

//         //     blas::add( value_t(-1), M1, M2 );
//         //     std::cout << "addlr col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
//         // }
        
//         R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
//     }// for

//     //
//     // finally adjust cluster bases
//     //

//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un_i ) );
//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn_j ) );
// }

// //
// // add block-local low-rank update W·X' to M_ij but adjust bases based
// // on global data (full block-row/column)
// //
// template < typename value_t >
// void
// addlr_global ( Hpro::TBlockMatrix< value_t > &                   M,
//                matrix::uniform_lrmatrix< value_t > &  M_ij,
//                const uint                             i,
//                const uint                             j,
//                const blas::matrix< value_t > &        W,
//                const blas::matrix< value_t > &        X,
//                const Hpro::TTruncAcc &                acc )
// {
//     using  real_t = Hpro::real_type_t< value_t >;

//     // std::cout << "addlr : " << M_ij.id() << std::endl;

//     // io::matlab::write( M_ij.row_basis(), "U" );
//     // io::matlab::write( M_ij.coeff(), "S" );
//     // io::matlab::write( M_ij.col_basis(), "V" );
//     // io::matlab::write( W, "W" );
//     // io::matlab::write( X, "X" );

//     const auto  U       = M_ij.row_basis();
//     const auto  V       = M_ij.col_basis();
//     const auto  Ue      = blas::join_row< value_t >( { U, W } );
//     const auto  Ve      = blas::join_row< value_t >( { V, X } );
//     const auto  rank_WX = W.ncols();

//     //
//     // compute QR of extended basis for correct scaling below
//     //

//     auto        RU      = blas::matrix< value_t >();
//     auto        RV      = blas::matrix< value_t >();
//     const auto  I       = blas::identity< value_t >( rank_WX );
//     const auto  Se_ij   = blas::diag< value_t >( { M_ij.coeff(), I } );
//     real_t      norm_ij = real_t(0);
    
//     {
//         auto  QU = blas::copy( Ue );
//         auto  QV = blas::copy( Ve );

//         blas::qr( QU, RU );
//         blas::qr( QV, RV );

//         const auto  T1 = blas::prod( RU, Se_ij );
//         const auto  T2 = blas::prod( T1, blas::adjoint( RV ) );

//         norm_ij = blas::norm_2( T2 );
//     }

//     if ( norm_ij == real_t(0) )
//         HLR_ERROR( "todo: zero norm" );
    
//     //
//     // compute new row basis
//     //

//     auto  Un = blas::matrix< value_t >();

//     {
//         //
//         // collect all low-rank blocks
//         //
        
//         auto    Qi      = std::list< blas::matrix< value_t > >();
//         size_t  nrows_Q = 0;
//         size_t  ncols_Q = 0;
        
//         for ( uint  k = 0; k < M.nblock_cols(); ++k )
//         {
//             auto  M_ik = M.block( i, k );
        
//             if ( ! matrix::is_uniform_lowrank( M_ik ) )
//                 continue;

//             auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );

//             if ( k == j )
//             {
//                 //
//                 // (V_j X) ⎛S_ij'  0⎞ = QV RV ⎛S_ij'  0⎞ = QV RV Se_ij'
//                 //         ⎝ 0     I⎠         ⎝ 0     I⎠
//                 //
                
//                 auto  RS_ik = blas::prod( RV, blas::adjoint( Se_ij ) );

//                 // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( norm_ij ) << std::endl;
//                 blas::scale( value_t(1) / norm_ij, RS_ik );
                
//                 nrows_Q += RS_ik.nrows();
//                 ncols_Q  = RS_ik.ncols();

//                 Qi.push_back( std::move( RS_ik ) );
//             }// if
//             else
//             {
//                 //
//                 // V_k S_kj' with orthogonal V_k
//                 //
                
//                 auto  S_ik    = blas::copy( blas::adjoint( R_ik->coeff() ) );
//                 auto  norm_ik = blas::norm_2( S_ik );

//                 if ( norm_ik > real_t(0) )
//                 {
//                     // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_ik ) ) << std::endl;
//                     blas::scale( value_t(1) / norm_ik, S_ik );

//                     nrows_Q += S_ik.nrows();
                
//                     Qi.push_back( std::move( S_ik ) );
//                 }// if
//             }// else
//         }// for

//         //
//         // assemble Q
//         //
        
//         auto    Q   = blas::matrix< value_t >( nrows_Q, ncols_Q );
//         size_t  pos = 0;

//         for ( auto  RS : Qi )
//         {
//             auto  Q_k = blas::matrix< value_t >( Q,
//                                                  blas::range( pos, pos + RS.nrows()-1 ),
//                                                  blas::range( 0, RS.ncols()-1 ) );

//             blas::copy( RS, Q_k );
//             pos += RS.nrows();
//         }// for

//         // io::matlab::write( Q, "Q" );

//         auto  R = blas::matrix< value_t >();
        
//         blas::qr( Q, R, false );

//         //
//         // compute column basis of ( U W )·R
//         //
        
//         auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
//         auto  Ss  = blas::vector< real_t >();

//         // io::matlab::write( UeR, "Us" );
        
//         blas::svd( UeR, Ss );

//         // io::matlab::write( UeR, "Ul" );
        
//         const auto  rank   = acc.trunc_rank( Ss );
//         const auto  U_rank = blas::matrix< value_t >( UeR, blas::range::all, blas::range( 0, rank-1 ) );

//         Un = std::move( blas::copy( U_rank ) );
    
//         // io::matlab::write( Un, "Un" );
//     }

//     //
//     // compute new column basis
//     //

//     auto  Vn = blas::matrix< value_t >();

//     {
//         //
//         // collect all low-rank blocks
//         //
        
//         auto    Qi      = std::list< blas::matrix< value_t > >();
//         size_t  nrows_Q = 0;
//         size_t  ncols_Q = 0;
        
//         for ( uint  k = 0; k < M.nblock_rows(); ++k )
//         {
//             auto  M_kj = M.block( k, j );
        
//             if ( ! matrix::is_uniform_lowrank( M_kj ) )
//                 continue;

//             auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );

//             if ( i == k )
//             {
//                 //
//                 // (U_i W) ⎛S_ij  0⎞ = QU RU ⎛S_ij  0⎞ = QU RU Se_ij
//                 //         ⎝ 0    I⎠         ⎝ 0    I⎠
//                 //

//                 auto  RS_kj = blas::prod( RU, Se_ij );
                
//                 // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( norm_ij ) << std::endl;
//                 blas::scale( value_t(1) / norm_ij, RS_kj );
                
//                 nrows_Q += RS_kj.nrows();
//                 ncols_Q  = RS_kj.ncols();
                
//                 Qi.push_back( std::move( RS_kj ) );
//             }// if
//             else
//             {
//                 //
//                 // U_k ( S_kj 0 ), U_k is assumed to be orthogonal
//                 //
                
//                 auto  S_kj    = R_kj->coeff();
//                 auto  RS_kj   = blas::copy( S_kj );
//                 auto  norm_kj = blas::norm_2( RS_kj );

//                 if ( norm_kj > real_t(0) )
//                 {
//                     // scale each matrix by norm to give each block equal weight in computed row basis
//                     // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( RS_kj ) ) << std::endl;
//                     blas::scale( value_t(1) / norm_kj, RS_kj );

//                     nrows_Q += RS_kj.nrows();
                
//                     Qi.push_back( std::move( RS_kj ) );
//                 }// if
//             }// else
//         }// for

//         //
//         // assemble Q
//         //
        
//         auto    Q   = blas::matrix< value_t >( nrows_Q, ncols_Q );
//         size_t  pos = 0;

//         for ( auto  RS : Qi )
//         {
//             auto  Q_k = blas::matrix< value_t >( Q,
//                                                  blas::range( pos, pos + RS.nrows()-1 ),
//                                                  blas::range( 0, RS.ncols()-1 ) );

//             blas::copy( RS, Q_k );
//             pos += RS.nrows();
//         }// for

//         // io::matlab::write( Q, "Q" );
        
//         auto  R = blas::matrix< value_t >();
        
//         blas::qr( Q, R, false );

//         //
//         // compute column basis of ( V X )·R'
//         //
        
//         auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
//         auto  Ss  = blas::vector< real_t >();

//         // io::matlab::write( VeR, "Vs" );
        
//         blas::svd( VeR, Ss );

//         // io::matlab::write( VeR, "Vl" );
        
//         const auto  rank   = acc.trunc_rank( Ss );
//         const auto  V_rank = blas::matrix< value_t >( VeR, blas::range::all, blas::range( 0, rank-1 ) );

//         Vn = std::move( blas::copy( V_rank ) );
    
//         // io::matlab::write( Vn, "Vn" );
//     }
    
//     //
//     // update coupling matrices
//     //

//     const auto  TU = blas::prod( blas::adjoint( Un ), U );
//     const auto  TV = blas::prod( blas::adjoint( Vn ), V );

//     {
//         const auto  TUe = blas::prod( blas::adjoint( Un ), Ue );
//         const auto  TVe = blas::prod( blas::adjoint( Vn ), Ve );
//         auto        T  = blas::prod( TUe, Se_ij );
//         auto        Sn = blas::prod( T, blas::adjoint( TVe ) );

//         // {
//         //     auto  US1 = blas::prod( M_ij.row_basis(), M_ij.coeff() );
//         //     auto  M1  = blas::prod( US1, blas::adjoint( M_ij.col_basis() ) );

//         //     blas::prod( W, blas::adjoint( X ), value_t(1), M1 );

//         //     auto  US2 = blas::prod( Un, Sn );
//         //     auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

//         //     blas::add( value_t(-1), M1, M2 );

//         //     std::cout << M_ij.id() << " error =    " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//         // }

//         M_ij.set_coeff_unsafe( std::move( Sn ) );
//     }

//     for ( uint  k = 0; k < M.nblock_cols(); ++k )
//     {
//         auto  M_ik = M.block( i, k );
        
//         if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
//         {
//             auto  R_ik = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
//             auto  S    = R_ik->coeff();
//             // auto  Se   = blas::extend( S, rank_WX, 0 ); // [ S ; 0 ]
//             auto  Sn   = blas::prod( TU, S );

//             // {
//             //     auto  US1 = blas::prod( R_ik->row_basis(), R_ik->coeff() );
//             //     auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_basis() ) );
//             //     auto  US2 = blas::prod( Un, Sn );
//             //     auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_basis() ) );

//             //     blas::add( value_t(-1), M1, M2 );

//             //     std::cout << R_ik->id() << " row error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//             // }

//             R_ik->set_coeff_unsafe( std::move( Sn ) );
//         }// if
//     }// for

//     for ( uint  k = 0; k < M.nblock_rows(); ++k )
//     {
//         auto  M_kj = M.block( k, j );
        
//         if ( matrix::is_uniform_lowrank( M_kj ) && ( i != k ))
//         {
//             auto  R_kj = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
//             auto  S    = R_kj->coeff();
//             // auto  Se   = blas::extend( S, 0, rank_WX ); // [ S , 0 ]
//             auto  Sn   = blas::prod( S, blas::adjoint( TV ) );
            
//             // {
//             //     auto  US1 = blas::prod( R_kj->row_basis(), R_kj->coeff() );
//             //     auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_basis() ) );
//             //     auto  US2 = blas::prod( R_kj->row_basis(), Sn );
//             //     auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

//             //     blas::add( value_t(-1), M1, M2 );

//             //     std::cout << R_kj->id() << " col error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//             // }

//             R_kj->set_coeff_unsafe( std::move( Sn ) );
//         }// if
//     }// for

//     //
//     // finally adjust cluster bases
//     //

//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
// }

// template < typename value_t >
// void
// addlr_global_ref ( Hpro::TBlockMatrix< value_t > &                   M,
//                    matrix::uniform_lrmatrix< value_t > &  M_ij,
//                    const uint                             i,
//                    const uint                             j,
//                    const blas::matrix< value_t > &        W,
//                    const blas::matrix< value_t > &        X,
//                    const Hpro::TTruncAcc &                acc )
// {
//     using  real_t = Hpro::real_type_t< value_t >;

//     // std::cout << "addlr : " << M_ij.id() << std::endl;

//     //
//     // compute new row basis
//     //

//     const auto  rank_WX = W.ncols();

//     // io::matlab::write( M_ij.row_basis(), "U" );
//     // io::matlab::write( M_ij.coeff(), "S" );
//     // io::matlab::write( M_ij.col_basis(), "V" );
//     // io::matlab::write( W, "W" );
//     // io::matlab::write( X, "X" );

//     //
//     // set up full block row for total cluster basis and
//     // compute exact basis with respect to given accuracy
//     //

//     auto  Un = blas::matrix< value_t >();
//     auto  Vn = blas::matrix< value_t >();
    
//     {
//         // determine dimensions of glock row
//         size_t  nrows = 0;
//         size_t  ncols = 0;

//         for ( uint  k = 0; k < M.nblock_cols(); ++k )
//         {
//             auto  M_ik = M.block( i, k );
        
//             if ( matrix::is_uniform_lowrank( M_ik ) )
//             {
//                 nrows  = M_ik->nrows();
//                 ncols += M_ik->ncols();
//             }// if
//         }// for

//         auto    Xt = blas::matrix< value_t >( nrows, ncols );
//         size_t  pos = 0;

//         for ( uint  k = 0; k < M.nblock_cols(); ++k )
//         {
//             auto  M_ik = M.block( i, k );
        
//             if ( matrix::is_uniform_lowrank( M_ik ) )
//             {
//                 auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
//                 auto  U    = R_ik->row_basis();
//                 auto  S    = R_ik->coeff();
//                 auto  V    = R_ik->col_basis();

//                 auto  US   = blas::prod( U, S );
//                 auto  D_ik = blas::prod( US, blas::adjoint( V ) );

//                 if ( k == j )
//                     blas::prod( W, blas::adjoint(X), value_t(1), D_ik );

//                 auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

//                 std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_ik ) ) << std::endl;
//                 blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
//                 blas::copy( D_ik, Xt_k );

//                 pos += D_ik.ncols();
//             }// if
//         }// for

//         // io::matlab::write( Xt, "Xt" );
        
//         auto  Ss = blas::vector< real_t >();

//         blas::svd( Xt, Ss );

//         const auto  rank   = acc.trunc_rank( Ss );
//         const auto  U_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

//         Un = std::move( blas::copy( U_rank ) );

//         // io::matlab::write( Un, "Un" );
//     }

//     {
//         // determine dimensions of glock row
//         size_t  nrows = 0;
//         size_t  ncols = 0;

//         for ( uint  k = 0; k < M.nblock_rows(); ++k )
//         {
//             auto  M_kj = M.block( k, j );
        
//             if ( matrix::is_uniform_lowrank( M_kj ) )
//             {
//                 nrows += M_kj->nrows();
//                 ncols  = M_kj->ncols();
//             }// if
//         }// for

//         auto    Xt = blas::matrix< value_t >( ncols, nrows );  // adjoint (!)
//         size_t  pos = 0;

//         for ( uint  k = 0; k < M.nblock_rows(); ++k )
//         {
//             auto  M_kj = M.block( k, j );
        
//             if ( matrix::is_uniform_lowrank( M_kj ) )
//             {
//                 auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
//                 auto  U    = R_kj->row_basis();
//                 auto  S    = R_kj->coeff();
//                 auto  V    = R_kj->col_basis();

//                 auto  VS   = blas::prod( V, blas::adjoint( S ) );
//                 auto  D_kj = blas::prod( VS, blas::adjoint( U ) );

//                 if ( i == k )
//                     blas::prod( X, blas::adjoint(W), value_t(1), D_kj );

//                 auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

//                 std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_kj ) ) << std::endl;
//                 blas::scale( value_t(1) / blas::norm_2( D_kj ), D_kj );
//                 blas::copy( D_kj, Xt_k );

//                 pos += D_kj.ncols();
//             }// if
//         }// for

//         // io::matlab::write( Xt, "Xt" );
        
//         auto  Ss = blas::vector< real_t >();

//         blas::svd( Xt, Ss );

//         const auto  rank   = acc.trunc_rank( Ss );
//         const auto  V_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

//         Vn = std::move( blas::copy( V_rank ) );

//         // io::matlab::write( Vn, "Vn" );
//     }

//     //
//     // update coupling matrices
//     //

//     const auto  U  = M_ij.row_basis();
//     const auto  V  = M_ij.col_basis();
//     const auto  Ue = blas::join_row< value_t >( { U, W } );
//     const auto  Ve = blas::join_row< value_t >( { V, X } );
//     const auto  TU = blas::prod( blas::adjoint( Un ), Ue );
//     const auto  TV = blas::prod( blas::adjoint( Vn ), Ve );

//     {
//         auto  I  = blas::identity< value_t >( rank_WX );
//         auto  S  = M_ij.coeff();
//         auto  Se = blas::diag< value_t >( { S, I } );
//         auto  T  = blas::prod( TU, Se );
//         auto  Sn = blas::prod( T, blas::adjoint( TV ) );

//         {
//             auto  US1 = blas::prod( M_ij.row_basis(), M_ij.coeff() );
//             auto  M1  = blas::prod( US1, blas::adjoint( M_ij.col_basis() ) );

//             blas::prod( W, blas::adjoint( X ), value_t(1), M1 );

//             auto  US2 = blas::prod( Un, Sn );
//             auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

//             blas::add( value_t(-1), M1, M2 );

//             std::cout << M_ij.id() << " error =    " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//         }

//         M_ij.set_coeff_unsafe( std::move( Sn ) );
//     }

//     for ( uint  k = 0; k < M.nblock_cols(); ++k )
//     {
//         auto  M_ik = M.block( i, k );
        
//         if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
//         {
//             auto  R_ik = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
//             auto  S    = R_ik->coeff();
//             auto  Se   = blas::extend( S, rank_WX, 0 ); // [ S ; 0 ]
//             auto  Sn   = blas::prod( TU, Se );

//             {
//                 auto  US1 = blas::prod( R_ik->row_basis(), R_ik->coeff() );
//                 auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_basis() ) );
//                 auto  US2 = blas::prod( Un, Sn );
//                 auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_basis() ) );

//                 blas::add( value_t(-1), M1, M2 );

//                 std::cout << R_ik->id() << " row error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//             }

//             R_ik->set_coeff_unsafe( std::move( Sn ) );
//         }// if
//     }// for

//     for ( uint  k = 0; k < M.nblock_rows(); ++k )
//     {
//         auto  M_kj = M.block( k, j );
        
//         if ( matrix::is_uniform_lowrank( M_kj ) && ( i != k ))
//         {
//             auto  R_kj = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
//             auto  S    = R_kj->coeff();
//             auto  Se   = blas::extend( S, 0, rank_WX ); // [ S , 0 ]
//             auto  Sn   = blas::prod( Se, blas::adjoint( TV ) );
            
//             {
//                 auto  US1 = blas::prod( R_kj->row_basis(), R_kj->coeff() );
//                 auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_basis() ) );
//                 auto  US2 = blas::prod( R_kj->row_basis(), Sn );
//                 auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

//                 blas::add( value_t(-1), M1, M2 );

//                 std::cout << R_kj->id() << " col error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
//             }

//             R_kj->set_coeff_unsafe( std::move( Sn ) );
//         }// if
//     }// for

//     //
//     // finally adjust cluster bases
//     //

//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
//     const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
// }

//
// compute new row basis for block row of M with M being replaced by W·T·X'
// assuming all involved bases are orthogonal (X is not needed for computation)
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_updated_row_basis ( const Hpro::TBlockMatrix< value_t > &  M,
                            const uint                             i,
                            const uint                             j,
                            const blas::matrix< value_t > &        W,
                            const blas::matrix< value_t > &        T,
                            const Hpro::TTruncAcc &                acc,
                            const approx_t &                       approx )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute new row basis of
    //
    //   (U·S_1· V_1'  U·S_2·V_2'  ...  U·S_j·V_j'  W·T·X')
    //
    //    = (U W) ⎛S_1·V_1'  S_2·V_2' ... S_j·V_j'  0  ⎞
    //            ⎝   0         0            0     T·X'⎠
    //
    //    = (U W) ⎛V_1·S_1'  0  ⎞'
    //            ⎜V_2·S_2'  0  ⎟
    //            ⎜      ...    ⎟
    //            ⎜V_j·S_j'  0  ⎟
    //            ⎝   0     X·T'⎠
    //
    //    = (U W) ⎛⎛V_1              ⎞ ⎛S_1'  0 ⎞⎞'
    //            ⎜⎜    V_2          ⎟ ⎜S_2'  0 ⎟⎟
    //            ⎜⎜        ...      ⎟·⎜   ...  ⎟⎟
    //            ⎜⎜            V_j  ⎟ ⎜S_j'  0 ⎟⎟
    //            ⎝⎝                X⎠ ⎝ 0    T'⎠⎠
    //
    // Since V_i and X are orthogonal, one can skip those for bases computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S_1' 0 ⎞ = S
    //         ⎜S_2' 0 ⎟
    //         ⎜  ...  ⎟
    //         ⎜S_j' 0 ⎟
    //         ⎝ 0   T'⎠
    //
    // of which also Q is omitted, which leaves to compute the column basis of
    //
    //   (U W) R' = U_e R'
    //
    // The S_i and T are scaled by the (spectral) norm of the corresponding block
    // U_i·S_i·V' and W·T·X' to achieve the relative precision for all blocks.
    //

    auto  M_ij = cptrcast( M.block( i, j ), matrix::uniform_lrmatrix< value_t > );
        
    // determine number of rows of matrix S below (sum of column ranks)
    size_t  nrows_S = T.ncols();  // known apriori
    
    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( matrix::is_uniform_lowrank( M_ik ) && ( M_ik != M_ij ))
            nrows_S += cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > )->col_rank();
    }// for

    if ( nrows_S == T.ncols() )
    {
        //
        // since there is no other low-rank block, new row basis is W
        //

        return std::move( blas::copy( W ) );
    }// if
    else
    {
        // extended row basis
        auto  U  = M_ij->row_basis();
        auto  Ue = blas::join_row< value_t >( { U, W } );

        // compute QR of column basis for each block in row and assemble
        // all results into common matrix Q
        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( ! matrix::is_uniform_lowrank( M_ik ) )
                continue;
        
            if ( M_ik == M_ij )
            {
                // R_ik = W T X' with W/X being orthogonal, hence |R_ik| = |T|
                const auto  rank = T.ncols();
                auto        S_ik = blas::copy( T );

                blas::scale( value_t(1) / norm::spectral( T ), S_ik );

                auto  S_k = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + rank-1 ),
                                                     blas::range( U.ncols(), Ue.ncols() - 1 ) );

                blas::copy( blas::adjoint( S_ik ), S_k );
                pos += rank;
            }// if
            else
            {
                // R_ik = U_i S_ik V_k' with U_i/V_k being orthogonal, hence |R_ik| = |S_ik|
                const auto  R_ik    = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                const auto  rank    = R_ik->col_rank();
                auto        S_ik    = blas::copy( R_ik->coeff() );
                auto        norm_ik = norm::spectral( S_ik );

                if ( norm_ik != real_t(0) )
                    blas::scale( value_t(1) / norm_ik, S_ik );
            
                auto  S_k = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + rank-1 ),
                                                     blas::range( 0, U.ncols() - 1 ) );

                blas::copy( blas::adjoint( S_ik ), S_k );
                pos += rank;
            }// else
        }// for

        // compute QR of assembled matrix, and compute SVD of
        // product with extended column basis
        auto  R = blas::matrix< value_t >();
        
        blas::qr( S, R, false );

        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
        auto  Un  = approx.column_basis( UeR, acc );

        return  Un;
    }// else
}

//
// compute new column basis for block column of M with M being replaced by W·T·X'
// assuming all involved bases are orthogonal (W is not needed for computation)
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_updated_col_basis ( const Hpro::TBlockMatrix< value_t > &  M,
                            const uint                             i,
                            const uint                             j,
                            const blas::matrix< value_t > &        T,
                            const blas::matrix< value_t > &        X,
                            const Hpro::TTruncAcc &                acc,
                            const approx_t &                       approx )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute new column basis
    //
    //   ⎛U_1 S_1 V'⎞' 
    //   ⎜U_2 S_2 V'⎟
    //   ⎜  ...     ⎟ = (V X) ⎛S_1'·U_1' S_2'·U_2' ... S_j'·U_j'   0  ⎞
    //   ⎜U_j S_j V'⎟         ⎝    0         0             0     T'·W'⎠
    //   ⎝  W T X'  ⎠
    //
    //                = (V X) ⎛U_1·S_1⎞'   (V X) ⎛⎛U_1           ⎞⎛S_1⎞⎞'
    //                        ⎜U_2·S_2⎟          ⎜⎜   U_2        ⎟⎜S_2⎟⎟
    //                        ⎜  ...  ⎟  =       ⎜⎜      ...     ⎟⎜...⎟⎟
    //                        ⎜U_j·S_j⎟          ⎜⎜         U_j  ⎟⎜S_j⎟⎟
    //                        ⎝  W·T  ⎠          ⎝⎝             W⎠⎝ T ⎠⎠
    //
    // Since U_* and W are orthogonal, one can skip those for bases computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S_1  0⎞ = S
    //         ⎜S_2  0⎟
    //         ⎜ ...  ⎟
    //         ⎜S_j  0⎟
    //         ⎝ 0   T⎠
    //
    // and finally column basis of
    //
    //   (V X) R' = V_e R'
    //
    // Please note, that the S_i and T are scaled by the (spectral) norm of the
    // corresponding block U_i·S_i·V' and W·T·X'
    //

    auto  M_ij = cptrcast( M.block( i, j ), matrix::uniform_lrmatrix< value_t > );
    
    // determine number of rows of matrix S below (sum of row ranks)
    size_t  nrows_S = T.nrows(); // known apriori
    
    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) && ( M_kj != M_ij ))
            nrows_S += cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > )->row_rank();
    }// for

    if ( nrows_S == T.nrows() )
    {
        //
        // since there is no other low-rank block, new basis is X
        //

        return std::move( blas::copy( X ) );
    }// if
    else
    {
        //
        // otherwise compute new basis
        //
            
        auto  V  = M_ij->col_basis();
        auto  Ve = blas::join_row< value_t >( { V, X } );
    
        // assemble normalized coefficient matrices into common matrix S
        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;

            if ( M_kj == M_ij )
            {
                // R_kj = W T X' with W/X being orthogonal, hence |R_kj| = |T|
                const auto  rank = T.nrows();
                auto        S_kj = blas::copy( T );
                    
                blas::scale( value_t(1) / norm::spectral( T ), S_kj );
                
                auto  S_k = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + rank-1 ),
                                                     blas::range( V.ncols(), Ve.ncols() - 1 ) );

                blas::copy( S_kj, S_k );
                pos += rank;
            }// if
            else
            {
                // R_kj = U_k S_kj V_j' and U_k/V_j are orthogonal, hence |R_kj| = |S_kj|
                const auto  R_kj    = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                const auto  rank    = R_kj->row_rank();
                auto        S_kj    = blas::copy( R_kj->coeff() );
                auto        norm_kj = norm::spectral( S_kj );

                if ( norm_kj != real_t(0) )
                    blas::scale( value_t(1) / norm_kj, S_kj );

                auto  S_k = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + rank-1 ),
                                                     blas::range( 0, V.ncols() - 1 ) );

                blas::copy( S_kj, S_k );
                pos += rank;
            }// else
        }// for

        // compute QR of assembled matrix, and compute SVD of
        // product with extended column basis
        auto  R = blas::matrix< value_t >();

        blas::qr( S, R, false );

        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
        auto  Vn  = approx.column_basis( VeR, acc );
        
        return  Vn;
    }// else
}

//
// replace M_ij = U·S·V' by W·T·V' and update row bases of
// all other blocks in block row
//
// - ASSUMPTION: W is orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_row_basis ( Hpro::TBlockMatrix< value_t > &  M,
                   const uint                       i,
                   const uint                       j,
                   const blas::matrix< value_t > &  W,
                   const blas::matrix< value_t > &  T,
                   const Hpro::TTruncAcc &          acc,
                   const approx_t &                 approx )
{
    auto  M_ij = ptrcast( M.block( i, j ), matrix::uniform_lrmatrix< value_t > );
    auto  Un   = compute_updated_row_basis( M, i, j, W, T, acc, approx );
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   TU ⎛S_kj⎞  or  TU ⎛  0 ⎞
    //      ⎝ 0  ⎠         ⎝S_kj⎠
    //

    auto  U  = M_ij->row_basis();
    auto  TU = blas::prod( blas::adjoint( Un ), U );

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( ! matrix::is_uniform_lowrank( M_ik ) )
            continue;
                    
        if ( M_ik != M_ij )
        {
            auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
    }// for

    //
    // compute coupling of M_ij as Un' W T
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  Sn = blas::prod( TW, T );

    M_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    M_ij->row_cb().set_basis( std::move( Un ) );
}

//
// replace M_ij = U·S·V' by U·T·X' and update column bases of
// all other blocks in block column
// - ASSUMPTION: X is orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_col_basis ( Hpro::TBlockMatrix< value_t > &  M,
                   const uint                       i,
                   const uint                       j,
                   const blas::matrix< value_t > &  T,
                   const blas::matrix< value_t > &  X,
                   const Hpro::TTruncAcc &          acc,
                   const approx_t &                 approx )
{
    auto  M_ij = ptrcast( M.block( i, j ), matrix::uniform_lrmatrix< value_t > );
    auto  Vn   = compute_updated_col_basis( M, i, j, T, X, acc, approx );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        auto  V  = M_ij->col_basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
            
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;
                    
            if ( M_kj != M_ij )
            {
                auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

                R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
            }// if
        }// for
    }

    //
    // compute coupling of M as T X' Vn
    //

    auto  TX = blas::prod( T,  blas::adjoint( X ) );
    auto  Sn = blas::prod( TX, Vn );

    M_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    M_ij->col_cb().set_basis( std::move( Vn ) );
}

//
// replace U·S·V' of M by W·T·X' and update row/column bases
// - ASSUMPTION: W and X are orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_row_col_basis ( Hpro::TBlockMatrix< value_t > &  M,
                       const uint                       i,
                       const uint                       j,
                       const blas::matrix< value_t > &  W,
                       const blas::matrix< value_t > &  T,
                       const blas::matrix< value_t > &  X,
                       const Hpro::TTruncAcc &          acc,
                       const approx_t &                 approx )
{
    auto  M_ij = ptrcast( M.block( i, j ), matrix::uniform_lrmatrix< value_t > );
    auto  Vn   = compute_updated_col_basis( M, i, j, T, X, acc, approx );
    auto  Un   = compute_updated_row_basis( M, i, j, W, T, acc, approx );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        auto  V  = M_ij->col_basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
            
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;
                    
            if ( M_kj != M_ij )
            {
                auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

                R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
            }// if
        }// for
    }

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   Un'·U·S_i = TU·S_i  with TU = Un'·U
        //

        auto  U  = M_ij->row_basis();
        auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
            
            if ( ! matrix::is_uniform_lowrank( M_ik ) )
                continue;
                    
            if ( M_ik != M_ij )
            {
                auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

                R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
            }// if
        }// for
    }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    M_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    M_ij->col_cb().set_basis( std::move( Vn ) );
    M_ij->row_cb().set_basis( std::move( Un ) );
}

//
// perform α A_ik · B_kj + C_ij
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                          alpha,
           const matop_t                          op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const matop_t                          op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const uint                             i,
           const uint                             k,
           const uint                             j,
           const Hpro::TTruncAcc &                acc,
           const approx_t &                       approx )
{
    auto  A_ik = A.block( i, k );
    auto  B_kj = B.block( k, j );
    auto  C_ij = C.block( i, j );

    // // DEBUG {
    // auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
    // auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
    // auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    // blas::prod( alpha, blas::mat< value_t >( DA ), blas::mat< value_t >( DB ),
    //             value_t(1), blas::mat< value_t >( DC ) );
    // // DEBUG }

    // std::cout << A_ik->id() << " × " << B_kj->id() << " -> " << C_ij->id() << std::endl;
        
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
                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  XW    = blas::prod( blas::adjoint( RA_ik->col_basis( op_A ) ), RB_kj->row_basis( op_B ) );
                auto  TXW   = blas::prod( RA_ik->coeff(), XW );
                auto  TXWR  = blas::prod( alpha, TXW, RB_kj->coeff() );

                if ( & RC_ij->row_cb() == & RA_ik->row_cb( op_A ) )
                {
                    if ( & RC_ij->col_cb() == & RB_kj->col_cb( op_B ) )
                    {
                        //
                        // C + A×B = U·S·V' + α·U·T·X' × W·R·V' = U ( S + α· T·X'×W·R ) V'
                        //
                            
                        blas::add( value_t(1), TXWR, RC_ij->coeff() );
                    }// if
                    else
                    {
                        //
                        // C + A×B = U·S·V' + α·U·T·X' × W·R·Z' = U ( S·V' + α·T·X'×W·R·Z' )
                        //
                            
                        auto  [ Sn, Vn ] = hlr::uniform::detail::add_col( *RC_ij, TXWR, RB_kj->col_basis() );

                        update_col_basis< value_t >( C, i, j, Sn, Vn, acc, approx );
                    }// else
                }// if
                else
                {
                    if ( & RC_ij->col_cb() == & RB_kj->col_cb( op_B ) )
                    {
                        //
                        // C + A×B = U·S·V' + α·Y·T·X' × W·R·V' = ( U·S + α·Y·T·X'×W·R) · V'
                        //
                            
                        auto  [ Un, Sn ] = hlr::uniform::detail::add_row( *RC_ij, RA_ik->row_basis(), TXWR );

                        update_row_basis< value_t >( C, i, j, Un, Sn, acc, approx );
                    }// if
                    else
                    {
                        //
                        // C + A×B = U·S·V' + α·Y·T·X' × W·R·Z'
                        //
                            
                        auto  [ Un, Sn, Vn ] = hlr::uniform::detail::add_row_col( *RC_ij, RA_ik->row_basis(), TXWR, RB_kj->col_basis() );

                        update_row_col_basis< value_t >( C, i, j, Un, Sn, Vn, acc, approx );
                    }// else
                }// else
            }// if
            else if ( is_dense( B_kj ) )
            {
                auto  DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto  BX    = blas::prod( alpha, blas::adjoint( blas::mat< value_t >( DB_kj ) ), RA_ik->col_basis() );

                if ( & RC_ij->row_cb() == & RA_ik->row_cb( op_A ) )
                {
                    //
                    // U·S·V' + α·U·T·X' × B = U·S·V' + α·U·T·(B'·X)'
                    //                       = U·( S·V' + α·T·(B'·X)' )
                    //                       = U·(S α·T) (V (B'·X))'
                    //
                    // modify column basis of C_ij to (V B'X)
                    //
                            
                    auto  [ Sn, Vn ] = hlr::uniform::detail::add_col( *RC_ij, RA_ik->coeff(), BX );

                    update_col_basis< value_t >( C, i, j, Sn, Vn, acc, approx );
                }// if
                else
                {
                    //
                    // U·S·V' + α·W·T·X' × B = U·S·V' + α·W·T·(B'·X)'
                    //
                    // modify row and column basis of C_ij to (U W) and (V (B'·X)), resp.
                    //
                            
                    auto  [ Un, Sn, Vn ] = hlr::uniform::detail::add_row_col( *RC_ij, RA_ik->row_basis(), RA_ik->coeff(), BX );

                    update_row_col_basis< value_t >( C, i, j, Un, Sn, Vn, acc, approx );
                }// else
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else if ( is_dense( A_ik ) )
        {
            auto  DA_ik = cptrcast( A_ik, Hpro::TDenseMatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  AW    = blas::prod( alpha, blas::mat< value_t >( DA_ik ), RB_kj->row_basis() );

                if ( & RC_ij->col_cb() == & RB_kj->col_cb( op_B ) )
                {
                    //
                    // U·S·V' + α·A × W·T·V' = ( U·S + α·(A×W)·T ) V'
                    //                       = ⎛ (U α·A·W) ⎛S⎞⎞ V'
                    //                         ⎝           ⎝T⎠⎠
                    //
                    // modify row basis of C_ij to (U A·W)
                    //
                    
                    auto  [ Un, Sn ] = hlr::uniform::detail::add_row( *RC_ij, AW, RB_kj->coeff() );
                    
                    update_row_basis( C, i, j, Un, Sn, acc, approx );
                }// if
                else
                {
                    //
                    // U·S·V' + α·A × W·T·X' = U·S·V' + α·(A·W)·T·X'
                    //
                    // modify row and column basis of C_ij to (U A·W) and (V X), resp.
                    //
                    
                    auto  [ Un, Sn, Vn ] = hlr::uniform::detail::add_row_col( *RC_ij, AW, RB_kj->coeff(), RB_kj->col_basis() );
                    
                    update_row_col_basis( C, i, j, Un, Sn, Vn, acc, approx );
                }// else
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S V' + α A × B
                //
                // compute A·B ≈ W·T·X' and replace block by
                //
                //  ( U W ) ⎛S  ⎞ (V X)'
                //          ⎝  T⎠
                //
                            
                auto  DB_kj          = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto  AB             = blas::prod( alpha, blas::mat< value_t >( DA_ik ), blas::mat< value_t >( DB_kj ) );
                auto  [ W, X ]       = approx::svd( AB, acc );
                auto  I              = blas::identity< value_t >( W.ncols() );
                auto  [ Un, Sn, Vn ] = hlr::uniform::detail::add_row_col( *RC_ij, W, I, X );

                update_row_col_basis( C, i, j, Un, Sn, Vn, acc, approx );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
    }// if
    else if ( is_dense( C_ij ) )
    {
        auto  DC_ij = ptrcast( C_ij, Hpro::TDenseMatrix< value_t > );
                    
        if ( matrix::is_uniform_lowrank( A_ik ) )
        {
            auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // C = C + α U S_A ( W' · X ) S_B V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WX    = blas::prod( blas::adjoint( RA_ik->col_basis() ), RB_kj->row_basis() );
                auto  SWX   = blas::prod( RA_ik->coeff(), WX );
                auto  SWXS  = blas::prod( SWX, RB_kj->coeff() );
                auto  US    = blas::prod( RA_ik->row_basis( op_A ), SWXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_basis( op_B ) ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto  VB    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_basis( op_A ) ),
                                          blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ) );
                auto  SVB   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VB );

                blas::prod( alpha,      RA_ik->row_basis( op_A ), SVB,
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else if ( is_dense( A_ik ) )
        {
            auto  DA_ik = cptrcast( A_ik, Hpro::TDenseMatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // C = C + ( ( A · U ) S_B ) V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  AU    = blas::prod( value_t(1),
                                          blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                                          RB_kj->row_basis( op_B ) );
                auto  AUS   = blas::prod( value_t(1),
                                          AU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      AUS, blas::adjoint( RB_kj->col_basis( op_B ) ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + A · B
                //
                            
                auto  DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );

                blas::prod( alpha,
                            blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                            blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ),
                            value_t(1),
                            blas::mat< value_t >( DC_ij ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + C_ij->typestr() );

    // // DEBUG {
    // auto  DD  = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    // blas::add( value_t(-1), blas::mat< value_t >( DC ), blas::mat< value_t >( DD ) );

    // if ( blas::norm_F( blas::mat< value_t >( DD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) > 1e-10 )
    // {
    //     std::cout << "    multiply : "
    //               << A_ik->id() << " × " << B_kj->id() << " -> " << C_ij->id() << " : "
    //               << blas::norm_F( blas::mat< value_t >( DD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) << std::endl;
    // }// if
    // // DEBUG }
}

//
// perform α A_ik · D · B_kj + C_ij
//
template < typename value_t >
void
multiply ( const value_t                          alpha,
           const matop_t                          op_A,
           const Hpro::TBlockMatrix< value_t > &  A,
           const matop_t                          op_D,
           const Hpro::TDenseMatrix< value_t > &  D,
           const matop_t                          op_B,
           const Hpro::TBlockMatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &        C,
           const uint                             i,
           const uint                             k,
           const uint                             j,
           const Hpro::TTruncAcc &                acc )
{
    auto  A_ik = A.block( i, k );
    auto  B_kj = B.block( k, j );
    auto  C_ij = C.block( i, j );

    // // DEBUG {
    // auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
    // auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
    // auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    // auto  AxD = blas::prod( blas::mat< value_t >( DA ), blas::mat< value_t >( D ) );
    
    // blas::prod( alpha, AxD, blas::mat< value_t >( DB ), value_t(1), blas::mat< value_t >( DC ) );
    // // DEBUG }

    HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

    //
    // due to TLR format, C_ij, A_ik and B_kj can only be dense or uniform-lowrank
    // hence, handle all combinations
    //

    auto  DD = blas::mat< value_t >( D );

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
                auto  WD    = blas::prod( blas::adjoint( RA_ik->col_basis() ), DD );
                auto  WDX   = blas::prod( WD, RB_kj->row_basis() );
                auto  SWDX  = blas::prod( RA_ik->coeff(), WDX );

                blas::prod( alpha, SWDX, RB_kj->coeff(), value_t(1), RC_ij->coeff() );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α U S_A W' · D · B
                //
                // add low-rank update α ( U · S_A ) ( B' · D' · W )' to C and update bases
                //
                            
                auto        DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                const auto  US    = blas::prod( alpha, RA_ik->row_basis(), RA_ik->coeff() );
                const auto  DW    = blas::prod( blas::adjoint( DD ), RA_ik->col_basis() );
                const auto  BDW   = blas::prod( blas::adjoint( blas::mat< value_t >( DB_kj ) ), DW );

                HLR_ERROR( "TODO" );
                // addlr_global< value_t >( C, *RC_ij, i, j, US, BDW, acc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else if ( is_dense( A_ik ) )
        {
            auto  DA_ik = cptrcast( A_ik, Hpro::TDenseMatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // U S_C V' + α A · D · W S_B V'
                //
                // add low-rank update ( A D W ) ( V S_B' )' to C and update bases
                //
                            
                auto        RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                const auto  DW    = blas::prod( alpha, DD, RB_kj->row_basis() );
                const auto  ADW   = blas::prod( blas::mat< value_t >( DA_ik ), DW );
                const auto  VS    = blas::prod( RB_kj->col_basis(), blas::adjoint( RB_kj->coeff() ) );

                HLR_ERROR( "TODO" );
                // addlr_global< value_t >( C, *RC_ij, i, j, ADW, VS, acc );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α A · D · B
                //
                // compute A·B, convert to low-rank, add to C and update bases
                //
                            
                auto        DB_kj    = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto        AD       = blas::prod( alpha, blas::mat< value_t >( DA_ik ), DD );
                auto        ADB      = blas::prod( AD, blas::mat< value_t >( DB_kj ) );
                const auto  [ W, X ] = approx::svd( ADB, acc );

                HLR_ERROR( "TODO" );
                // addlr_global< value_t >( C, *RC_ij, i, j, W, X, acc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
    }// if
    else if ( is_dense( C_ij ) )
    {
        auto  DC_ij = ptrcast( C_ij, Hpro::TDenseMatrix< value_t > );
                    
        if ( matrix::is_uniform_lowrank( A_ik ) )
        {
            auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // C = C + α U S_A ( W' · D · X ) S_B V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  WD    = blas::prod( blas::adjoint( RA_ik->col_basis() ), DD );
                auto  WDX   = blas::prod( WD, RB_kj->row_basis() );
                auto  SWDX  = blas::prod( RA_ik->coeff(), WDX );
                auto  SWDXS = blas::prod( SWDX, RB_kj->coeff() );
                auto  US    = blas::prod( RA_ik->row_basis( op_A ), SWDXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_basis( op_B ) ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · D · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto  VD    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_basis( op_A ) ),
                                          blas::mat_view( op_D, DD ) );
                auto  VDB   = blas::prod( value_t(1),
                                          VD,
                                          blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ) );
                auto  SVDB  = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VDB );

                blas::prod( alpha,      RA_ik->row_basis( op_A ), SVDB,
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else if ( is_dense( A_ik ) )
        {
            auto  DA_ik = cptrcast( A_ik, Hpro::TDenseMatrix< value_t > );
                        
            if ( matrix::is_uniform_lowrank( B_kj ) )
            {
                //
                // C = C + ( ( A · D · U ) S_B ) V'
                //

                auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                auto  DU    = blas::prod( value_t(1),
                                          blas::mat_view( op_D, DD ),
                                          RB_kj->row_basis( op_B ) );
                auto  ADU   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                                          DU );
                auto  ADUS  = blas::prod( value_t(1),
                                          ADU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      ADUS, blas::adjoint( RB_kj->col_basis( op_B ) ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + A · D · B
                //
                            
                auto  DB_kj = cptrcast( B_kj, Hpro::TDenseMatrix< value_t > );
                auto  AD    = blas::prod( value_t(1),
                                          blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                                          blas::mat_view( op_D, DD ) );


                blas::prod( alpha,
                            AD,
                            blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ),
                            value_t(1),
                            blas::mat< value_t >( DC_ij ) );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + C_ij->typestr() );

    // // DEBUG {
    // auto  TD  = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    // blas::add( value_t(-1), blas::mat< value_t >( DC ), blas::mat< value_t >( TD ) );
                
    // std::cout << "    multiply : "
    //           << A_ik->id() << " × " << D.id() << " × " << B_kj->id() << " -> " << C_ij->id() << " : "
    //           << blas::norm_F( blas::mat< value_t >( TD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) << std::endl;
    // // DEBUG }
}

//
// adjust bases U/V to hold U·S·V' + W·X'
//
template < typename value_t >
void
addlr_local ( blas::matrix< value_t > &        U,
              blas::matrix< value_t > &        S,
              blas::matrix< value_t > &        V,
              const blas::matrix< value_t > &  W,
              const blas::matrix< value_t > &  X,
              const Hpro::TTruncAcc &          acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // Compute new row basis Un as column basis of
    //
    //   U·S·V' + W·X' = ( U W ) ⎛S 0⎞ (V X)' = Ue Se Ve'
    //                           ⎝0 I⎠
    //                 = Ue Se Rv' Qv'
    //
    // for which column basis for Ue Se Rv' is sufficient.
    // New coefficients are then computed as Un' Ue Se.
    //
    // The new column basis is computed for the transposed matrix.
    //

    //
    // first compute QR of W/X for efficient norm computation of W·X'
    //

    auto  QW = blas::copy( W );
    auto  QX = blas::copy( X );
    auto  RW = blas::matrix< value_t >();
    auto  RX = blas::matrix< value_t >();

    blas::qr( QW, RW );
    blas::qr( QX, RX );

    auto        T        = blas::prod( RW, blas::adjoint( RX ) );
    const auto  norm_USV = blas::norm2( S );
    const auto  norm_WX  = blas::norm2( T );
    auto        S1       = blas::copy( S );
    auto        T1       = blas::copy( T );

    blas::scale( value_t(1) / norm_USV, S1 );
    blas::scale( value_t(1) / norm_WX,  T1 );
    
    // extended bases and coupling
    const auto  Ue  = blas::join_row< value_t >( { U, QW } );
    auto        Se1 = blas::diag< value_t >( { S1, T1 } );
    const auto  Ve  = blas::join_row< value_t >( { V, QX } );

    //
    // new row basis is computed as the left singular vectors of
    //
    //   (U W) ⎛S·V'  0  ⎞ = (U W) ⎛V·S'  0  ⎞' = (U W) ⎛⎛V  ⎞⎛S'   ⎞⎞'
    //         ⎝ 0   T·X'⎠         ⎝ 0   X·T'⎠          ⎝⎝  X⎠⎝   T'⎠⎠
    //
    // of which ⎛V  ⎞ is orthogonal and can be omitted. 
    //          ⎝  X⎠
    //
    // With QR decomposition Q·R = ⎛S'   ⎞
    //                             ⎝   T'⎠
    //
    // one ends up with the left singular vectors of (U W) R'.
    //

    auto  Un = blas::matrix< value_t >();
                
    {
        auto  R  = blas::matrix< value_t >();
        auto  Q  = blas::copy( blas::adjoint( Se1 ) ); // need to copy since modified during QR
                
        blas::qr( Q, R, false );
                
        auto  Us = blas::prod( Ue, blas::adjoint( R ) );
        auto  Ss = blas::vector< real_t >();

        blas::svd( Us, Ss );
                    
        const auto  rank_U = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

        Un = std::move( blas::copy( U_rank ) );
    }

    //
    // new column basis is computed as the left singular vectors of
    //
    //   (V X) ⎛S'·U'   0  ⎞ = (V X) ⎛U·S  0 ⎞' = (V X) ⎛⎛U  ⎞⎛S  ⎞⎞'
    //         ⎝  0   T'·W'⎠         ⎝ 0  T·W⎠          ⎝⎝  W⎠⎝  T⎠⎠
    //
    // of which ⎛U  ⎞ is orthogonal and can be omitted. 
    //          ⎝  W⎠
    //
    // With QR decomposition Q·R = ⎛S  ⎞
    //                             ⎝  T⎠
    //
    // one ends up with the left singular vectors of (V X) R'.
    //

    auto  Vn = blas::matrix< value_t >();
                
    {
        auto  R  = blas::matrix< value_t >();
        auto  Q  = blas::copy( Se1 ); // need to copy since modified during QR
                
        blas::qr( Q, R, false );
                    
        auto  Vs = blas::prod( Ve, blas::adjoint( R ) );
        auto  Ss = blas::vector< real_t >();
                    
        blas::svd( Vs, Ss );
                    
        const auto  rank_V = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( Vs, blas::range::all, blas::range( 0, rank_V-1 ) );

        Vn = std::move( blas::copy( V_rank ) );
    }
    
    //
    // new coupling matrix is
    //
    //   Un' Ue Se Ve' Vn = TU Se TVj'
    //
    // with TU = Un' Ue and TV = Vn' Ve
    //
                
    auto        Se = blas::diag< value_t >( { S, T } );
    const auto  TU = blas::prod( blas::adjoint( Un ), Ue );
    const auto  TV = blas::prod( blas::adjoint( Vn ), Ve );
    auto        TS = blas::prod( TU, Se );
    auto        Sn = blas::prod( TS, blas::adjoint( TV ) );

    // // DEBUG {
    // {
    //     auto  US1 = blas::prod( Ue, Se );
    //     auto  M1  = blas::prod( US1, blas::adjoint( Ve ) );

    //     {
    //         auto  UM  = blas::prod( blas::adjoint( Un ), M1 );
    //         auto  UUM = blas::prod( Un, UM );
            
    //         blas::add( value_t(-1), M1, UUM );
    //         std::cout << "          addlr Un   : " << boost::format( "%.4e" ) % ( blas::norm_F( UUM ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  MV  = blas::prod( M1, Vn );
    //         auto  MVV = blas::prod( MV, blas::adjoint( Vn ) );
            
    //         blas::add( value_t(-1), M1, MVV );
    //         std::cout << "          addlr Vn   : " << boost::format( "%.4e" ) % ( blas::norm_F( MVV ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  US2 = blas::prod( Un, Sn );
    //         auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

    //         blas::add( value_t(-1), M1, M2 );
    //         std::cout << "          addlr     : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    //     }
    // }
    // // DEBUG }

    U = std::move( Un );
    S = std::move( Sn );
    V = std::move( Vn );
}

//
// apply updates to block M_ij, e.g., M_ij = M_ij - Σ_k=0..min(i,j)-1 M_ik M_kj
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
apply_updates ( Hpro::TBlockMatrix< value_t > &  M,
                const uint                       i,
                const uint                       j,
                const Hpro::TTruncAcc &          acc )
{
    auto       C     = M.block( i, j );          
    const int  minij = std::min( i, j );


    // DEBUG {
    // auto  D1  = matrix::convert_to_dense< value_t >( *C );
    // auto  apx = approx::SVD< value_t >();

    // if ( is_dense( C ) )
    // {
    //     //
    //     // if M_ij is dense, directly apply all updates
    //     //
        
    //     for ( int  k = 0; k < minij; ++k )
    //     {
    //         multiply( value_t(-1), apply_normal, M, apply_normal, M, M, i, k, j, acc );
    //     }// for

    //     // DEBUG {
    //     auto  D2  = matrix::convert_to_dense< value_t >( *C );

    //     blas::add( value_t(-1), blas::mat< value_t >( D1 ), blas::mat< value_t >( D2 ) );
    //     std::cout << "    update " << C->id() << " : " << boost::format( "%.4e" ) % ( blas::norm2( blas::mat< value_t >( D2 ) ) / blas::norm2( blas::mat< value_t >( D1 ) ) ) << std::endl;
    //     // DEBUG }
        
        
    //     return { blas::matrix< value_t >(),
    //              blas::matrix< value_t >(),
    //              blas::matrix< value_t >() };
    // }// if

    if ( matrix::is_uniform_lowrank( C ) )
    {
        //
        // apply all updates but update row/column bases only locally
        //

        auto  R = ptrcast( C, matrix::uniform_lrmatrix< value_t > );
        auto  U = blas::copy( R->row_basis() );
        auto  S = blas::copy( R->coeff() );
        auto  V = blas::copy( R->col_basis() );

        // signals modification of U/V basis in C and therefore no longer having
        // same row/column basis as other blocks in block row/column
        bool  changed_basis = false;
        
        for ( int  k = 0; k < minij; ++k )
        {
            const auto  A = M.block( i, k );
            const auto  B = M.block( k, j );

            // // DEBUG {
            // {
            //     const auto  DA = matrix::convert_to_dense< value_t >( * M.block( i, k ) );
            //     const auto  DB = matrix::convert_to_dense< value_t >( * M.block( k, j ) );
            
            //     hlr::multiply< value_t >( value_t(-1), apply_normal, *DA, apply_normal, *DB, *D1, acc, apx );
            // }
            // // DEBUG }
            
            if ( matrix::is_uniform_lowrank( A ) )
            {
                auto  RA = cptrcast( A, matrix::uniform_lrmatrix< value_t > );
                        
                if ( matrix::is_uniform_lowrank( B ) )
                {
                    //
                    // U S_C V' - U S_A W' · X S_B V' =
                    // U ( S_C - S_A W' · X S_B ) V'
                    //
                            
                    const auto  RB  = cptrcast( B, matrix::uniform_lrmatrix< value_t > );
                    const auto  WX  = blas::prod( blas::adjoint( RA->col_basis() ), RB->row_basis() );
                    const auto  SWX = blas::prod( RA->coeff(), WX );

                    if ( changed_basis )
                    {
                        auto  SWXS = blas::prod( value_t(-1), SWX, RB->coeff() );
                        auto  USWXS = blas::prod( RA->row_basis(), SWXS );

                        addlr_local< value_t >( U, S, V, USWXS, RB->col_basis(), acc );
                    }// if
                    else
                    {
                        blas::prod( value_t(-1), SWX, RB->coeff(), value_t(1), S );
                    }// else
                }// if
                else if ( is_dense( B ) )
                {
                    //
                    // U S_C V' - U S_A W' · B
                    //
                    // add low-rank update α ( U · S_A ) ( B' · W )' to C and update bases
                    //
                            
                    auto        DB = cptrcast( B, Hpro::TDenseMatrix< value_t > );
                    const auto  US = blas::prod( value_t(-1), RA->row_basis(), RA->coeff() );
                    const auto  BW = blas::prod( blas::adjoint( blas::mat< value_t >( DB ) ), RA->col_basis() );

                    addlr_local< value_t >( U, S, V, US, BW, acc );
                    changed_basis = true;
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + B->typestr() );
            }// if
            else if ( is_dense( A ) )
            {
                auto  DA = cptrcast( A, Hpro::TDenseMatrix< value_t > );
                        
                if ( matrix::is_uniform_lowrank( B ) )
                {
                    //
                    // U S_C V' - A · W S_B V'
                    //
                    // add low-rank update α ( A W ) ( V S_B' )' to C and update bases
                    //
                            
                    auto        RB = cptrcast( B, matrix::uniform_lrmatrix< value_t > );
                    const auto  AW = blas::prod( value_t(-1), blas::mat< value_t >( DA ), RB->row_basis() );
                    const auto  VS = blas::prod( RB->col_basis(), blas::adjoint( RB->coeff() ) );

                    addlr_local< value_t >( U, S, V, AW, VS, acc );
                    changed_basis = true;
                }// if
                else if ( is_dense( B ) )
                {
                    //
                    // U S_C V' - A · B
                    //
                    // compute A·B, convert to low-rank, add to C and update bases
                    //
                            
                    auto        DB       = cptrcast( B, Hpro::TDenseMatrix< value_t > );
                    auto        AB       = blas::prod( value_t(-1), blas::mat< value_t >( DA ), blas::mat< value_t >( DB ) );
                    const auto  [ W, X ] = approx::svd( AB, acc );

                    addlr_local< value_t >( U, S, V, W, X, acc );
                    changed_basis = true;
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + B->typestr() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + A->typestr() );

            // // DEBUG {
            // auto  T1  = blas::prod( U, S );
            // auto  D2  = blas::prod( T1, blas::adjoint( V ) );

            // blas::add( value_t(-1), blas::mat< value_t >( D1 ), D2 );
            // std::cout << "      update " << C->id() << " : " << boost::format( "%.4e" ) % ( blas::norm2( D2 ) / blas::norm2( blas::mat< value_t >( D1 ) ) ) << std::endl;
            // // DEBUG }
        }// for

        return { std::move( U ),
                 std::move( S ),
                 std::move( V ) };
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + C->typestr() );
}

}}}}// namespace hlr::uniform::tlr::detail

#endif // __HLR_ARITH_DETAIL_UNIFORM_HH
