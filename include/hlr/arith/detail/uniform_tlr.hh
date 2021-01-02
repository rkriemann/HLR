#ifndef __HLR_ARITH_DETAIL_UNIFORM_TLR_HH
#define __HLR_ARITH_DETAIL_UNIFORM_TLR_HH
//
// Project     : HLib
// Module      : arith/uniform
// Description : arithmetic functions for uniform TLR matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace uniform { namespace tlr { namespace detail {

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

    std::cout << "addlr : " << M_ij.id() << std::endl;

    // current bases and coupling
    const auto  U_i   = M_ij.row_cb().basis();
    const auto  S_ij  = M_ij.coeff();
    const auto  V_j   = M_ij.col_cb().basis();

    // extended bases and coupling
    const auto  rank  = W.ncols();
    const auto  Ue_i  = blas::join_row< value_t >( { U_i, W } );
    const auto  I     = blas::identity< value_t >( rank );
    auto        Se_ij = blas::diag< value_t >( { S_ij, I } );
    const auto  Ve_j  = blas::join_row< value_t >( { V_j, X } );

    // const auto  norm_Mij = blas::norm_F( S_ij ); blas::norm_2( S_ij );
    // const auto  norm_WX  = blas::norm_F( W, X ); hlr::norm::spectral( lowrank_operator{ W, X } );
        
    // std::cout << norm_Mij << " / " << norm_WX << std::endl;

    // const auto  scale = norm_WX / norm_Mij;
    
    // for ( uint  l = 0; l < rank; ++l )
    //     Se_ij( l + S_ij.nrows(), l + S_ij.ncols() ) *= scale;
              
    // if ( true )
    // {
    //     io::matlab::write( U_i,   "U" );
    //     io::matlab::write( S_ij,  "S" );
    //     io::matlab::write( V_j,   "V" );
    //     io::matlab::write( Ue_i,  "Ue" );
    //     io::matlab::write( Se_ij, "Se" );
    //     io::matlab::write( Ve_j,  "Ve" );
    //     io::matlab::write( W,     "W" );
    //     io::matlab::write( X,     "X" );
    // }// if
    
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
                
        const auto  SR_ij = blas::prod( Se_ij, blas::adjoint( R_j ) );
        auto        Us    = blas::prod( Ue_i, SR_ij );
        auto        Ss    = blas::vector< real_t >();

        blas::svd( Us, Ss );
                    
        const auto  rank_U = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

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
                    
        const auto  RS_ij = blas::prod( R_i, Se_ij );
        auto        Us    = blas::prod( Ve_j, blas::adjoint( RS_ij ) );
        auto        Ss    = blas::vector< real_t >();
                    
        blas::svd( Us, Ss );
                    
        const auto  rank_V = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( Us, blas::range::all, blas::range( 0, rank_V-1 ) );

        Vn_j = std::move( blas::copy( V_rank ) );
    }

    // io::matlab::write( Un_i, "Un" );
    // io::matlab::write( Vn_j, "Vn" );
    
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
                
    for ( uint  l = 0; l < rank; ++l )
        Se_ij( l + S_ij.nrows(), l + S_ij.ncols() ) = value_t(1);
    
    const auto  TU_i  = blas::prod( blas::adjoint( Un_i ), Ue_i );
    const auto  TV_j  = blas::prod( blas::adjoint( Vn_j ), Ve_j );
    auto        T_ij  = blas::prod( TU_i, Se_ij );
    auto        Sn_ij = blas::prod( T_ij, blas::adjoint( TV_j ) );

    // {
    //     auto  US1 = blas::prod( Ue_i, Se_ij );
    //     auto  M1  = blas::prod( US1, blas::adjoint( Ve_j ) );
            
    //     auto  US2 = blas::prod( Un_i, Sn_ij );
    //     auto  M2  = blas::prod( US2, blas::adjoint( Vn_j ) );

    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "addlr     : " << M_ij.id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    // }
    
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
        auto        Sn_ik = blas::prod( TU_i, Se_ik );

        // {
        //     auto  US1 = blas::prod( R_ik->row_cb().basis(), S_ik );
        //     auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            
        //     auto  US2 = blas::prod( Un_i, Sn_ik );
        //     auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

        //     blas::add( value_t(-1), M1, M2 );
        //     std::cout << "addlr row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
        // }
        
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
        auto        Sn_kj = blas::prod( Se_kj, blas::adjoint( TV_j ) );

        // {
        //     auto  US1 = blas::prod( R_kj->row_cb().basis(), S_kj );
        //     auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            
        //     auto  US2 = blas::prod( R_kj->row_cb().basis(), Sn_kj );
        //     auto  M2  = blas::prod( US2, blas::adjoint( Vn_j ) );

        //     blas::add( value_t(-1), M1, M2 );
        //     std::cout << "addlr col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
        // }
        
        R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un_i ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn_j ) );
}

//
// add block-local low-rank update W·X' to M_ij but adjust bases based
// on global data (full block-row/column)
//
template < typename value_t >
void
addlr_global ( hpro::TBlockMatrix &                   M,
               matrix::uniform_lrmatrix< value_t > &  M_ij,
               const uint                             i,
               const uint                             j,
               const blas::matrix< value_t > &        W,
               const blas::matrix< value_t > &        X,
               const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // std::cout << "addlr : " << M_ij.id() << std::endl;

    // io::matlab::write( M_ij.row_cb().basis(), "U" );
    // io::matlab::write( M_ij.coeff(), "S" );
    // io::matlab::write( M_ij.col_cb().basis(), "V" );
    // io::matlab::write( W, "W" );
    // io::matlab::write( X, "X" );

    const auto  U       = M_ij.row_cb().basis();
    const auto  V       = M_ij.col_cb().basis();
    const auto  Ue      = blas::join_row< value_t >( { U, W } );
    const auto  Ve      = blas::join_row< value_t >( { V, X } );
    const auto  rank_WX = W.ncols();

    //
    // compute QR of extended basis for correct scaling below
    //

    auto        RU      = blas::matrix< value_t >();
    auto        RV      = blas::matrix< value_t >();
    const auto  I       = blas::identity< value_t >( rank_WX );
    const auto  Se_ij   = blas::diag< value_t >( { M_ij.coeff(), I } );
    real_t      norm_ij = real_t(0);

    {
        auto  QU = blas::copy( Ue );
        auto  QV = blas::copy( Ve );

        blas::qr( QU, RU );
        blas::qr( QV, RV );

        const auto  T1 = blas::prod( RU, Se_ij );
        const auto  T2 = blas::prod( T1, blas::adjoint( RV ) );

        norm_ij = blas::norm_2( T2 );
    }
    
    //
    // compute new row basis
    //

    auto  Un = blas::matrix< value_t >();

    {
        //
        // collect all low-rank blocks
        //
        
        auto    Qi      = std::list< blas::matrix< value_t > >();
        size_t  nrows_Q = 0;
        size_t  ncols_Q = 0;
        
        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( ! matrix::is_uniform_lowrank( M_ik ) )
                continue;

            auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );

            if ( k == j )
            {
                //
                // (V_j X) ⎛S_ij'  0⎞ = QV RV ⎛S_ij'  0⎞ = QV RV Se_ij'
                //         ⎝ 0     I⎠         ⎝ 0     I⎠
                //
                
                auto  RS_ik = blas::prod( RV, blas::adjoint( Se_ij ) );

                // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( norm_ij ) << std::endl;
                blas::scale( value_t(1) / norm_ij, RS_ik );
                
                nrows_Q += RS_ik.nrows();
                ncols_Q  = RS_ik.ncols();
                
                Qi.push_back( std::move( RS_ik ) );
            }// if
            else
            {
                //
                // V_k S_kj' with orthogonal V_k
                //
                
                auto  S_ik = blas::copy( blas::adjoint( R_ik->coeff() ) );

                // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_ik ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( S_ik ), S_ik );

                nrows_Q += S_ik.nrows();
                
                Qi.push_back( std::move( S_ik ) );
            }// else
        }// for

        //
        // assemble Q
        //
        
        auto    Q   = blas::matrix< value_t >( nrows_Q, ncols_Q );
        size_t  pos = 0;

        for ( auto  RS : Qi )
        {
            auto  Q_k = blas::matrix< value_t >( Q,
                                                 blas::range( pos, pos + RS.nrows()-1 ),
                                                 blas::range( 0, RS.ncols()-1 ) );

            blas::copy( RS, Q_k );
            pos += RS.nrows();
        }// for

        // io::matlab::write( Q, "Q" );

        auto  R = blas::matrix< value_t >();
        
        blas::qr( Q, R, false );

        //
        // compute column basis of ( U W )·R
        //
        
        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
        auto  Ss  = blas::vector< real_t >();

        // io::matlab::write( UeR, "Us" );
        
        blas::svd( UeR, Ss );

        // io::matlab::write( UeR, "Ul" );
        
        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( UeR, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );
    
        // io::matlab::write( Un, "Un" );
    }

    //
    // compute new column basis
    //

    auto  Vn = blas::matrix< value_t >();

    {
        //
        // collect all low-rank blocks
        //
        
        auto    Qi      = std::list< blas::matrix< value_t > >();
        size_t  nrows_Q = 0;
        size_t  ncols_Q = 0;
        
        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;

            auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );

            if ( i == k )
            {
                //
                // (U_i W) ⎛S_ij  0⎞ = QU RU ⎛S_ij  0⎞ = QU RU Se_ij
                //         ⎝ 0    I⎠         ⎝ 0    I⎠
                //

                auto  RS_kj = blas::prod( RU, Se_ij );
                
                // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( norm_ij ) << std::endl;
                blas::scale( value_t(1) / norm_ij, RS_kj );
                
                nrows_Q += RS_kj.nrows();
                ncols_Q  = RS_kj.ncols();
                
                Qi.push_back( std::move( RS_kj ) );
            }// if
            else
            {
                //
                // U_k ( S_kj 0 ), U_k is assumed to be orthogonal
                //
                
                auto  S_kj  = R_kj->coeff();
                auto  RS_kj = blas::copy( S_kj );

                // scale each matrix by norm to give each block equal weight in computed row basis
                // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( RS_kj ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( RS_kj ), RS_kj );

                nrows_Q += RS_kj.nrows();
                
                Qi.push_back( std::move( RS_kj ) );
            }// else
        }// for

        //
        // assemble Q
        //
        
        auto    Q   = blas::matrix< value_t >( nrows_Q, ncols_Q );
        size_t  pos = 0;

        for ( auto  RS : Qi )
        {
            auto  Q_k = blas::matrix< value_t >( Q,
                                                 blas::range( pos, pos + RS.nrows()-1 ),
                                                 blas::range( 0, RS.ncols()-1 ) );

            blas::copy( RS, Q_k );
            pos += RS.nrows();
        }// for

        // io::matlab::write( Q, "Q" );
        
        auto  R = blas::matrix< value_t >();
        
        blas::qr( Q, R, false );

        //
        // compute column basis of ( V X )·R'
        //
        
        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
        auto  Ss  = blas::vector< real_t >();

        // io::matlab::write( VeR, "Vs" );
        
        blas::svd( VeR, Ss );

        // io::matlab::write( VeR, "Vl" );
        
        const auto  rank   = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( VeR, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );
    
        // io::matlab::write( Vn, "Vn" );
    }
    
    //
    // update coupling matrices
    //

    const auto  TU = blas::prod( blas::adjoint( Un ), U );
    const auto  TV = blas::prod( blas::adjoint( Vn ), V );

    {
        const auto  TUe = blas::prod( blas::adjoint( Un ), Ue );
        const auto  TVe = blas::prod( blas::adjoint( Vn ), Ve );
        auto        T  = blas::prod( TUe, Se_ij );
        auto        Sn = blas::prod( T, blas::adjoint( TVe ) );

        // {
        //     auto  US1 = blas::prod( M_ij.row_cb().basis(), M_ij.coeff() );
        //     auto  M1  = blas::prod( US1, blas::adjoint( M_ij.col_cb().basis() ) );

        //     blas::prod( W, blas::adjoint( X ), value_t(1), M1 );

        //     auto  US2 = blas::prod( Un, Sn );
        //     auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

        //     blas::add( value_t(-1), M1, M2 );

        //     std::cout << M_ij.id() << " error =    " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
        // }

        M_ij.set_coeff_unsafe( std::move( Sn ) );
    }

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
        {
            auto  R_ik = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
            auto  S    = R_ik->coeff();
            // auto  Se   = blas::extend( S, rank_WX, 0 ); // [ S ; 0 ]
            auto  Sn   = blas::prod( TU, S );

            // {
            //     auto  US1 = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
            //     auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            //     auto  US2 = blas::prod( Un, Sn );
            //     auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

            //     blas::add( value_t(-1), M1, M2 );

            //     std::cout << R_ik->id() << " row error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
            // }

            R_ik->set_coeff_unsafe( std::move( Sn ) );
        }// if
    }// for

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) && ( i != k ))
        {
            auto  R_kj = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
            auto  S    = R_kj->coeff();
            // auto  Se   = blas::extend( S, 0, rank_WX ); // [ S , 0 ]
            auto  Sn   = blas::prod( S, blas::adjoint( TV ) );
            
            // {
            //     auto  US1 = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
            //     auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            //     auto  US2 = blas::prod( R_kj->row_cb().basis(), Sn );
            //     auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

            //     blas::add( value_t(-1), M1, M2 );

            //     std::cout << R_kj->id() << " col error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
            // }

            R_kj->set_coeff_unsafe( std::move( Sn ) );
        }// if
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

template < typename value_t >
void
addlr_global_ref ( hpro::TBlockMatrix &                   M,
                   matrix::uniform_lrmatrix< value_t > &  M_ij,
                   const uint                             i,
                   const uint                             j,
                   const blas::matrix< value_t > &        W,
                   const blas::matrix< value_t > &        X,
                   const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    std::cout << "addlr : " << M_ij.id() << std::endl;

    //
    // compute new row basis
    //

    const auto  rank_WX = W.ncols();

    // io::matlab::write( M_ij.row_cb().basis(), "U" );
    // io::matlab::write( M_ij.coeff(), "S" );
    // io::matlab::write( M_ij.col_cb().basis(), "V" );
    // io::matlab::write( W, "W" );
    // io::matlab::write( X, "X" );

    //
    // set up full block row for total cluster basis and
    // compute exact basis with respect to given accuracy
    //

    auto  Un = blas::matrix< value_t >();
    auto  Vn = blas::matrix< value_t >();
    
    {
        // determine dimensions of glock row
        size_t  nrows = 0;
        size_t  ncols = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                nrows  = M_ik->nrows();
                ncols += M_ik->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( nrows, ncols );
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                auto  U    = R_ik->row_cb().basis();
                auto  S    = R_ik->coeff();
                auto  V    = R_ik->col_cb().basis();

                auto  US   = blas::prod( U, S );
                auto  D_ik = blas::prod( US, blas::adjoint( V ) );

                if ( k == j )
                    blas::prod( W, blas::adjoint(X), value_t(1), D_ik );

                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

                std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_ik ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
                blas::copy( D_ik, Xt_k );

                pos += D_ik.ncols();
            }// if
        }// for

        // io::matlab::write( Xt, "Xt" );
        
        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );

        // io::matlab::write( Un, "Un" );
    }

    {
        // determine dimensions of glock row
        size_t  nrows = 0;
        size_t  ncols = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) )
            {
                nrows += M_kj->nrows();
                ncols  = M_kj->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( ncols, nrows );  // adjoint (!)
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) )
            {
                auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                auto  U    = R_kj->row_cb().basis();
                auto  S    = R_kj->coeff();
                auto  V    = R_kj->col_cb().basis();

                auto  VS   = blas::prod( V, blas::adjoint( S ) );
                auto  D_kj = blas::prod( VS, blas::adjoint( U ) );

                if ( i == k )
                    blas::prod( X, blas::adjoint(W), value_t(1), D_kj );

                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

                std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_kj ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( D_kj ), D_kj );
                blas::copy( D_kj, Xt_k );

                pos += D_kj.ncols();
            }// if
        }// for

        // io::matlab::write( Xt, "Xt" );
        
        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );

        // io::matlab::write( Vn, "Vn" );
    }

    //
    // update coupling matrices
    //

    const auto  U  = M_ij.row_cb().basis();
    const auto  V  = M_ij.col_cb().basis();
    const auto  Ue = blas::join_row< value_t >( { U, W } );
    const auto  Ve = blas::join_row< value_t >( { V, X } );
    const auto  TU = blas::prod( blas::adjoint( Un ), Ue );
    const auto  TV = blas::prod( blas::adjoint( Vn ), Ve );

    {
        auto  I  = blas::identity< value_t >( rank_WX );
        auto  S  = M_ij.coeff();
        auto  Se = blas::diag< value_t >( { S, I } );
        auto  T  = blas::prod( TU, Se );
        auto  Sn = blas::prod( T, blas::adjoint( TV ) );

        {
            auto  US1 = blas::prod( M_ij.row_cb().basis(), M_ij.coeff() );
            auto  M1  = blas::prod( US1, blas::adjoint( M_ij.col_cb().basis() ) );

            blas::prod( W, blas::adjoint( X ), value_t(1), M1 );

            auto  US2 = blas::prod( Un, Sn );
            auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

            blas::add( value_t(-1), M1, M2 );

            std::cout << M_ij.id() << " error =    " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
        }

        M_ij.set_coeff_unsafe( std::move( Sn ) );
    }

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
        {
            auto  R_ik = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
            auto  S    = R_ik->coeff();
            auto  Se   = blas::extend( S, rank_WX, 0 ); // [ S ; 0 ]
            auto  Sn   = blas::prod( TU, Se );

            {
                auto  US1 = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
                auto  M1  = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
                auto  US2 = blas::prod( Un, Sn );
                auto  M2  = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

                blas::add( value_t(-1), M1, M2 );

                std::cout << R_ik->id() << " row error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
            }

            R_ik->set_coeff_unsafe( std::move( Sn ) );
        }// if
    }// for

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) && ( i != k ))
        {
            auto  R_kj = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
            auto  S    = R_kj->coeff();
            auto  Se   = blas::extend( S, 0, rank_WX ); // [ S , 0 ]
            auto  Sn   = blas::prod( Se, blas::adjoint( TV ) );
            
            {
                auto  US1 = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
                auto  M1  = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
                auto  US2 = blas::prod( R_kj->row_cb().basis(), Sn );
                auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

                blas::add( value_t(-1), M1, M2 );

                std::cout << R_kj->id() << " col error = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2( M1 ) ) << std::endl;
            }

            R_kj->set_coeff_unsafe( std::move( Sn ) );
        }// if
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
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

    // // DEBUG {
    // auto  DA = hlr::seq::matrix::convert_to_dense< value_t >( * A_ik );
    // auto  DB = hlr::seq::matrix::convert_to_dense< value_t >( * B_kj );
    // auto  DC = hlr::seq::matrix::convert_to_dense< value_t >( * C_ij );

    // blas::prod( alpha, blas::mat< value_t >( DA ), blas::mat< value_t >( DB ),
    //             value_t(1), blas::mat< value_t >( DC ) );
    // // DEBUG }

    HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

    // std::cout << A_ik->id() << " × " << B_kj->id() << " -> " << C_ij->id() << std::endl;
        
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
                auto  WX    = blas::prod( blas::adjoint( RA_ik->col_cb().basis() ), RB_kj->row_cb().basis() );
                auto  SWX   = blas::prod( RA_ik->coeff(), WX );

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
                const auto  US    = blas::prod( alpha, RA_ik->row_cb().basis(), RA_ik->coeff() );
                const auto  BW    = blas::prod( blas::adjoint( blas::mat< value_t >( DB_kj ) ), RA_ik->col_cb().basis() );

                addlr_global< value_t >( C, *RC_ij, i, j, US, BW, acc );
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
                const auto  VS    = blas::prod( RB_kj->col_cb().basis(), blas::adjoint( RB_kj->coeff() ) );

                addlr_global< value_t >( C, *RC_ij, i, j, AW, VS, acc );
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

                addlr_global< value_t >( C, *RC_ij, i, j, W, X, acc );
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
                auto  WX    = blas::prod( blas::adjoint( RA_ik->col_cb().basis() ), RB_kj->row_cb().basis() );
                auto  SWX   = blas::prod( RA_ik->coeff(), WX );
                auto  SWXS  = blas::prod( SWX, RB_kj->coeff() );
                auto  US    = blas::prod( RA_ik->row_cb( op_A ).basis(), SWXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                auto  VB    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_cb( op_A ).basis() ),
                                          blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ) );
                auto  SVB   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VB );

                blas::prod( alpha,      RA_ik->row_cb( op_A ).basis(), SVB,
                            value_t(1), blas::mat< value_t >( DC_ij ) );
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
                                          blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                                          RB_kj->row_cb( op_B ).basis() );
                auto  AUS   = blas::prod( value_t(1),
                                          AU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      AUS, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + A · B
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );

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

    // // if ( blas::norm_F( blas::mat< value_t >( DD ) ) / blas::norm_F( blas::mat< value_t >( DC ) ) > 1e-10 )
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
multiply ( const value_t               alpha,
           const matop_t               op_A,
           const hpro::TBlockMatrix &  A,
           const matop_t               op_D,
           const hpro::TDenseMatrix &  D,
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
                auto  WD    = blas::prod( blas::adjoint( RA_ik->col_cb().basis() ), DD );
                auto  WDX   = blas::prod( WD, RB_kj->row_cb().basis() );
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
                            
                auto        DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                const auto  US    = blas::prod( alpha,      RA_ik->row_cb().basis(), RA_ik->coeff() );
                const auto  DW    = blas::prod( blas::adjoint( DD ), RA_ik->col_cb().basis() );
                const auto  BDW   = blas::prod( blas::adjoint( blas::mat< value_t >( DB_kj ) ), DW );

                addlr_global< value_t >( C, *RC_ij, i, j, US, BDW, acc );
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
                const auto  DW    = blas::prod( alpha,      DD, RB_kj->row_cb().basis() );
                const auto  ADW   = blas::prod( blas::mat< value_t >( DA_ik ), DW );
                const auto  VS    = blas::prod( RB_kj->col_cb().basis(), blas::adjoint( RB_kj->coeff() ) );

                addlr_global< value_t >( C, *RC_ij, i, j, ADW, VS, acc );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // U S_C V' + α A · D · B
                //
                // compute A·B, convert to low-rank, add to C and update bases
                //
                            
                auto        DB_kj    = cptrcast( B_kj, hpro::TDenseMatrix );
                auto        AD       = blas::prod( alpha, blas::mat< value_t >( DA_ik ), DD );
                auto        ADB      = blas::prod( AD, blas::mat< value_t >( DB_kj ) );
                const auto  [ W, X ] = approx::svd( ADB, acc );

                addlr_global< value_t >( C, *RC_ij, i, j, W, X, acc );
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
                auto  WD    = blas::prod( blas::adjoint( RA_ik->col_cb().basis() ), DD );
                auto  WDX   = blas::prod( WD, RB_kj->row_cb().basis() );
                auto  SWDX  = blas::prod( RA_ik->coeff(), WDX );
                auto  SWDXS = blas::prod( SWDX, RB_kj->coeff() );
                auto  US    = blas::prod( RA_ik->row_cb( op_A ).basis(), SWDXS );

                blas::prod( alpha,      US, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + α U ( S_A ( V' · D · B ) )
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                auto  VD    = blas::prod( value_t(1),
                                          blas::adjoint( RA_ik->col_cb( op_A ).basis() ),
                                          blas::mat_view( op_D, DD ) );
                auto  VDB   = blas::prod( value_t(1),
                                          VD,
                                          blas::mat_view( op_B, blas::mat< value_t >( DB_kj ) ) );
                auto  SVDB  = blas::prod( value_t(1),
                                          blas::mat_view( op_A, RA_ik->coeff() ),
                                          VDB );

                blas::prod( alpha,      RA_ik->row_cb( op_A ).basis(), SVDB,
                            value_t(1), blas::mat< value_t >( DC_ij ) );
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
                                          blas::mat_view( op_D, DD ),
                                          RB_kj->row_cb( op_B ).basis() );
                auto  ADU   = blas::prod( value_t(1),
                                          blas::mat_view( op_A, blas::mat< value_t >( DA_ik ) ),
                                          DU );
                auto  ADUS  = blas::prod( value_t(1),
                                          ADU,
                                          blas::mat_view( op_B, RB_kj->coeff() ) );

                blas::prod( alpha,      ADUS, blas::adjoint( RB_kj->col_cb( op_B ).basis() ),
                            value_t(1), blas::mat< value_t >( DC_ij ) );
            }// if
            else if ( is_dense( B_kj ) )
            {
                //
                // C = C + A · D · B
                //
                            
                auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
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
              const hpro::TTruncAcc &          acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

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
apply_updates ( hpro::TBlockMatrix &     M,
                const uint               i,
                const uint               j,
                const hpro::TTruncAcc &  acc )
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
        auto  U = blas::copy( R->row_cb().basis() );
        auto  S = blas::copy( R->coeff() );
        auto  V = blas::copy( R->col_cb().basis() );

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
                    const auto  WX  = blas::prod( blas::adjoint( RA->col_cb().basis() ), RB->row_cb().basis() );
                    const auto  SWX = blas::prod( RA->coeff(), WX );

                    if ( changed_basis )
                    {
                        auto  SWXS = blas::prod( value_t(-1), SWX, RB->coeff() );
                        auto  USWXS = blas::prod( RA->row_cb().basis(), SWXS );

                        addlr_local< value_t >( U, S, V, USWXS, RB->col_cb().basis(), acc );
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
                            
                    auto        DB = cptrcast( B, hpro::TDenseMatrix );
                    const auto  US = blas::prod( value_t(-1), RA->row_cb().basis(), RA->coeff() );
                    const auto  BW = blas::prod( blas::adjoint( blas::mat< value_t >( DB ) ), RA->col_cb().basis() );

                    addlr_local< value_t >( U, S, V, US, BW, acc );
                    changed_basis = true;
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + B->typestr() );
            }// if
            else if ( is_dense( A ) )
            {
                auto  DA = cptrcast( A, hpro::TDenseMatrix );
                        
                if ( matrix::is_uniform_lowrank( B ) )
                {
                    //
                    // U S_C V' - A · W S_B V'
                    //
                    // add low-rank update α ( A W ) ( V S_B' )' to C and update bases
                    //
                            
                    auto        RB = cptrcast( B, matrix::uniform_lrmatrix< value_t > );
                    const auto  AW = blas::prod( value_t(-1), blas::mat< value_t >( DA ), RB->row_cb().basis() );
                    const auto  VS = blas::prod( RB->col_cb().basis(), blas::adjoint( RB->coeff() ) );

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
                            
                    auto        DB       = cptrcast( B, hpro::TDenseMatrix );
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
