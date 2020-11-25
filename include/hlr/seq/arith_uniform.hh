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
#include <hlr/arith/multiply.hh>
#include <hlr/arith/solve.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/seq/matrix.hh>
#include <hlr/utils/hash.hh>

#include <hlr/seq/norm.hh> // DEBUG
#include <hlr/arith/operator_wrapper.hh> // DEBUG

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
        auto  x_i = blas::vector< value_t >( blas::vec< value_t >( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec< value_t >( sy ), M.row_is( op_M ) - sy.ofs() );
        
        blas::mulvec( alpha, blas::mat_view( op_M, blas::mat< value_t >( D ) ), x_i, value_t(1), y_j );
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
        auto  v_cb = blas::vector< value_t >( blas::vec< value_t >( v ), cb.cluster() - v.ofs() );
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
        auto  v_u = blas::vector< value_t >( blas::vec< value_t >( v ), u.is() - v.ofs() );
            
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
    // const auto  norm_WX  = blas::norm_F( W, X ); hlr::seq::norm::spectral( lowrank_operator{ W, X } );
        
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
                    
        const auto  RS_ij = blas::prod( R_i, Se_ij );
        auto        Us    = blas::prod( Ve_j, blas::adjoint( RS_ij ) );
        auto        Ss    = blas::vector< real_t >();
                    
        blas::svd( Us, Ss );
                    
        const auto  rank_V = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_V-1 ) );

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
            auto  Q_k = blas::matrix( Q,
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
        const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank-1 ) );

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
            auto  Q_k = blas::matrix( Q,
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
        const auto  V_rank = blas::matrix( VeR, blas::range::all, blas::range( 0, rank-1 ) );

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

                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

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
        const auto  U_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

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

                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

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
        const auto  V_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

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
                const auto  US    = blas::prod( alpha,      RA_ik->row_cb().basis(), RA_ik->coeff() );
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
// replace column basis of block M_ij by X and update basis
// of block row to [ V, X ]
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

    //
    // compute QR of X for norm computation later
    //

    auto  QX = blas::copy( X );
    auto  RX = blas::matrix< value_t >();

    blas::qr( QX, RX );
    
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
        //
        // since there is no other low-rank block, just replace basis by (orthogonalized) X
        //
        
        auto  Sn = blas::prod( M_ij.coeff(), blas::adjoint( RX ) );

        M_ij.set_coeff_unsafe( std::move( Sn ) );
        const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( blas::copy( QX ) ) );
        return;
    }// if
    
    // extended column basis
    auto  V  = M_ij.col_cb().basis();
    auto  Ve = blas::join_row< value_t >( { V, X } );

    // io::matlab::write( V, "V" );
    // io::matlab::write( Ve, "Ve" );
    // io::matlab::write( X, "X" );
    
    // compute QR of row basis for each block in column and assemble
    // all results into common matrix Q
    auto    Qe  = blas::matrix< value_t >( nrows_Q, Ve.ncols() );
    size_t  pos = 0;

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( ! matrix::is_uniform_lowrank( M_kj ) )
            continue;
        
        const auto  R_kj   = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
        const auto  rank_k = R_kj->row_rank();
        auto        S_kj   = blas::copy( R_kj->coeff() );

        if ( k == i )
        {
            // R_kj = U_k S_kj X' and U_k is orthogonal,
            // therefore |R_kj| = |S_kj X'| = |S_kj RX' QX'| = |S_kj RX'|
            auto  SR_kj = blas::prod( S_kj, blas::adjoint( RX ) );

            // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( SR_kj ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( SR_kj ), S_kj );
                
            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( V.ncols(), Ve.ncols() - 1 ) );

            blas::copy( S_kj, Qe_k );
        }// if
        else
        {
            // R_kj = U_k S_kj V_j' and U_k/V_j are orthogonal,
            // therefore |R_kj| = |S_kj|
            // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_kj ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( S_kj ), S_kj );

            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( 0, V.ncols() - 1 ) );

            blas::copy( S_kj, Qe_k );
        }// else

        pos += rank_k;
    }// for

    // io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    // io::matlab::write( R, "R" );
    
    auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
    auto  Ss  = blas::vector< real_t >();

    blas::svd( VeR, Ss );

    // io::matlab::write( VeR, "VeR" );
    // io::matlab::write( Ss, "Ss" );
    
    const auto  rank   = acc.trunc_rank( Ss );
    const auto  V_rank = blas::matrix( VeR, blas::range::all, blas::range( 0, rank-1 ) );
    auto        Vn     = blas::copy( V_rank );
    
    // io::matlab::write( Vn, "Vn" );

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   (S_kj 0) TV  or  ( 0 S_ij ) TV
    //

    const auto  TV = blas::prod( blas::adjoint( Vn ), V );

    // io::matlab::write( TV, "TV" );

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if ( ! matrix::is_uniform_lowrank( B_kj ) )
            continue;
                    
        auto  R_kj = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        auto  S_kj = R_kj->coeff();

        if ( k == i )
        {
            auto  TX    = blas::prod( blas::adjoint( Vn ), X );
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TX ) );

            // auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            // auto  M1    = blas::prod( US1, blas::adjoint( X ) );
            // auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            // auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// if
        else
        {
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TV ) );

            // auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            // auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            // auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// else
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

//
// replace column basis of block M_ij by X and update basis
// of block row to [ V, X ]
// - use SVD of full block column to compute new basis
//
template < typename value_t >
void
extend_col_basis_ref ( hpro::TBlockMatrix &                   M,
                       matrix::uniform_lrmatrix< value_t > &  M_ij,
                       const uint                             i,
                       const uint                             j,
                       const blas::matrix< value_t > &        X,
                       const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    //
    // construct full block column Xt, perform SVD and
    // use singular vectors for new basis (exact approach)
    //

    auto  Vn = blas::matrix< value_t >();

    {
        // determine dimensions of block column
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

                auto  D_kj = blas::matrix< value_t >();

                if ( i == k )
                {
                    // replace V by X
                    auto  XS = blas::prod( X, blas::adjoint( S ) );

                    D_kj = std::move( blas::prod( XS, blas::adjoint( U ) ) );
                }// if
                else
                {
                    auto  VS = blas::prod( V, blas::adjoint( S ) );

                    D_kj = std::move( blas::prod( VS, blas::adjoint( U ) ) );
                }// else
                
                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

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
        const auto  V_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );

        // io::matlab::write( Vn, "Vn" );
    }

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   Vn'·V·S_kj' or Vn'·X·S_ij'
    //

    const auto  V  = M_ij.col_cb().basis();
    const auto  TV = blas::prod( blas::adjoint( Vn ), V );

    // io::matlab::write( TV, "TV" );

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if ( ! matrix::is_uniform_lowrank( B_kj ) )
            continue;
                    
        auto  R_kj = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        auto  S_kj = R_kj->coeff();

        if ( k == i )
        {
            // was: U_i S_ij X' -> U_i ( S_ij ( X' Vn ) ) Vn'
            auto  TX    = blas::prod( blas::adjoint( Vn ), X );
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TX ) );

            auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( US1, blas::adjoint( X ) );
            auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// if
        else
        {
            // was: U_k S_kj V_j' -> U_i ( S_ij ( V_j' Vn ) ) Vn' = U_i ( S_ij TV' ) Vn'
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TV ) );

            auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            blas::add( value_t(-1), M1, M2 );
            std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// else
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

//
// replace row basis for M_ij by W and update cluster basis of
// block row to [ U, W ]
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

    //
    // compute QR of W for norm computation later
    //

    auto  QW = blas::copy( W );
    auto  RW = blas::matrix< value_t >();

    blas::qr( QW, RW );
    
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
        //
        // since there is no other low-rank block, just replace basis by (orthogonalized) W and return
        //
        
        auto  Sn = blas::prod( RW, M_ij.coeff() );
        
        M_ij.set_coeff_unsafe( std::move( Sn ) );
        const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( blas::copy( QW ) ) );
        return;
    }// if
    
    // extended row basis
    auto  U  = M_ij.row_cb().basis();
    auto  Ue = blas::join_row< value_t >( { U, W } );

    // io::matlab::write( U, "U" );
    // io::matlab::write( Ue, "Ue" );
    // io::matlab::write( W, "W" );
    
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
        auto        S_ik   = blas::copy( R_ik->coeff() );

        // io::matlab::write( R_ik->row_cb().basis(), "Ui" );
        // io::matlab::write( R_ik->coeff(), "Sik" );
        // io::matlab::write( R_ik->col_cb().basis(), "Vk" );
        
        if ( k == j )
        {
            // R_ik = W S_ik V_k' and V_k is orthogonal,
            // therefore |R_ik| = |W S_ik| = |QW RW S_ik| = |RW S_ik|
            auto  RS_ik = blas::prod( RW, S_ik );

            // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( RS_ik ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( RS_ik ), S_ik );

            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( U.ncols(), Ue.ncols() - 1 ) );

            blas::copy( blas::adjoint( S_ik ), Qe_k );
        }// if
        else
        {
            // R_ik = U_i S_ik V_k' and U_i/V_k are orthogonal,
            // therefore |R_ik| = |S_ik|

            // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_ik ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( S_ik ), S_ik );
            
            auto  Qe_k = blas::matrix( Qe,
                                       blas::range( pos, pos + rank_k-1 ),
                                       blas::range( 0, U.ncols() - 1 ) );

            blas::copy( blas::adjoint( S_ik ), Qe_k );
        }// else

        // io::matlab::write( Qe, "Qe" );
        
        pos += rank_k;
    }// for

    // io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    // io::matlab::write( R, "R" );
    
    auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
    auto  Ss  = blas::vector< real_t >();

    blas::svd( UeR, Ss );

    // io::matlab::write( UeR, "UeR" );
    // io::matlab::write( Ss, "Ss" );
    
    const auto  rank   = acc.trunc_rank( Ss );
    const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank-1 ) );
    auto        Un     = blas::copy( U_rank );
    const auto  TU     = blas::prod( blas::adjoint( Un ), U );

    // io::matlab::write( Un, "Un" );
    
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

        if ( k == j )
        {
            auto  TW    = blas::prod( blas::adjoint( Un ), W );
            auto  Sn_ik = blas::prod( TW, S_ik );

            // auto  US1   = blas::prod( W, S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
        else
        {
            auto  Sn_ik = blas::prod( TU, S_ik );

            // auto  US1   = blas::prod( R_ik->row_cb().basis(), S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// else
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace row basis for M_ij by W and update cluster basis of
// block row to [ U, W ]
// - use SVD of full block row matrix to compute new basis
//   (reference implementation)
//
template < typename value_t >
void
extend_row_basis_ref ( hpro::TBlockMatrix &                   M,
                       matrix::uniform_lrmatrix< value_t > &  M_ij,
                       const uint                             i,
                       const uint                             j,
                       const blas::matrix< value_t > &        W,
                       const hpro::TTruncAcc &                acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    //
    // compute full block row, perform SVD and use
    // singular vectors as new row basis
    //

    const auto  U  = M_ij.row_cb().basis();
    auto        Un = blas::matrix< value_t >();

    {
        // determine dimensions of block row
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
                auto  S    = R_ik->coeff();
                auto  V    = R_ik->col_cb().basis();
                auto  D_ik = blas::matrix< value_t >();

                if ( k == j )
                {
                    auto  WS = blas::prod( W, S );

                    D_ik = std::move( blas::prod( WS, blas::adjoint( V ) ) );
                }// if
                else
                {
                    auto  US = blas::prod( U, S );

                    D_ik = std::move( blas::prod( US, blas::adjoint( V ) ) );
                }// else

                // io::matlab::write( D_ik, "Dik" );
                
                // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_ik ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
                
                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

                blas::copy( D_ik, Xt_k );

                pos += D_ik.ncols();
            }// if
        }// for

        // io::matlab::write( Xt, "Xt" );
        
        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );

        // io::matlab::write( Un, "Un" );
    }
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   Un' U S_kj  or  Un' W S_ij
    //

    const auto  TU = blas::prod( blas::adjoint( Un ), U );
        
    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if ( ! matrix::is_uniform_lowrank( B_ik ) )
            continue;
                    
        auto  R_ik = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        auto  S_ik = R_ik->coeff();

        if ( k == j )
        {
            auto  TW    = blas::prod( blas::adjoint( Un ), W );
            auto  Sn_ik = blas::prod( TW, S_ik );

            // auto  US1   = blas::prod( W, S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
        else
        {
            auto  Sn_ik = blas::prod( TU, S_ik );

            // auto  US1   = blas::prod( R_ik->row_cb().basis(), S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// else
    }// for

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
}

//
// recompute i'th row basis and j'th column basis while replacing
// block M_ij by W·T·X', e.g., extend row basis to cover [ U, W ]
// and column basis for [ V, X ] with U,V being the current row/
// column bases
//
// ASSUMPTIONs
//  - W and X are orthogonal
//  - M_ij is a uniform lowrank matrix
//
template < typename value_t >
void
extend_row_col_basis ( hpro::TBlockMatrix &       M,
                       const uint                 i,
                       const uint                 j,
                       blas::matrix< value_t > &  W,
                       blas::matrix< value_t > &  T,
                       blas::matrix< value_t > &  X,
                       const hpro::TTruncAcc &    acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // io::matlab::write( W, "W" );
    // io::matlab::write( T, "T" );
    // io::matlab::write( X, "X" );

    auto  M_ij = M.block( i, j );
    auto  R_ij = ptrcast( M_ij, matrix::uniform_lrmatrix< value_t > );

    //
    // compute new column basis
    //
    //   ⎛U_1 S_1 V'⎞' 
    //   ⎜U_2 S_2 V'⎟
    //   ⎜  ...     ⎟  = (V X) ⎛S_1'·U_1' S_2'·U_2' ... S_j'·U_j'   0  ⎞
    //   ⎜U_j S_j V'⎟          ⎝    0         0             0     T'·W'⎠
    //   ⎝W   T   X'⎠
    //                 = (V X) ⎛U_1·S_1  0 ⎞'
    //                         ⎜U_2·S_2  0 ⎟
    //                         ⎜     ...   ⎟
    //                         ⎜U_j·S_j  0 ⎟
    //                         ⎝   0    W·T⎠
    //
    // Since U_i and W are orthogonal, one can skip those for bases computation.
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
                                  
    auto  Vn = blas::matrix< value_t >();

    {
        // determine number of rows of matrix S below (sum of row ranks)
        size_t  nrows_S = T.nrows(); // known apriori
    
        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) && ( k != i ))
                nrows_S += cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > )->row_rank();
        }// for

        if ( nrows_S == T.nrows() )
        {
            //
            // since there is no other low-rank block, new basis is X
            //

            Vn = std::move( blas::copy( X ) );
        }// if
        else
        {
            //
            // otherwise compute new basis
            //
            
            auto  V  = R_ij->col_cb().basis();
            auto  Ve = blas::join_row< value_t >( { V, X } );
    
            // assemble normalized coefficient matrices into common matrix S
            auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
            size_t  pos = 0;

            for ( uint  k = 0; k < M.nblock_rows(); ++k )
            {
                auto  M_kj = M.block( k, j );
        
                if ( ! matrix::is_uniform_lowrank( M_kj ) )
                    continue;

                if ( k == i )
                {
                    // R_kj = W T X' with W/X being orthogonal, hence |R_kj| = |T|
                    const auto  rank = T.nrows();
                    auto        S_kj = blas::copy( T );
                    
                    blas::scale( value_t(1) / blas::norm_2( T ), S_kj );
                
                    auto  S_k = blas::matrix( S,
                                              blas::range( pos, pos + rank-1 ),
                                              blas::range( V.ncols(), Ve.ncols() - 1 ) );

                    blas::copy( S_kj, S_k );
                    pos += rank;
                }// if
                else
                {
                    // R_kj = U_k S_kj V_j' and U_k/V_j are orthogonal, hence |R_kj| = |S_kj|
                    const auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                    const auto  rank = R_kj->row_rank();
                    auto        S_kj = blas::copy( R_kj->coeff() );
                    
                    blas::scale( value_t(1) / blas::norm_2( S_kj ), S_kj );

                    auto  S_k = blas::matrix( S,
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
            auto  Ss  = blas::vector< real_t >();

            blas::svd( VeR, Ss );

            const auto  rank   = acc.trunc_rank( Ss );
            const auto  V_rank = blas::matrix( VeR, blas::range::all, blas::range( 0, rank-1 ) );

            Vn = std::move( blas::copy( V_rank ) );
    
            //
            // transform coupling matrix for blocks in current block column as
            //
            //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
            //

            const auto  TV = blas::prod( blas::adjoint( Vn ), V );

            for ( uint  k = 0; k < M.nblock_rows(); ++k )
            {
                auto  B_kj = M.block( k, j );
                    
                if ( ! matrix::is_uniform_lowrank( B_kj ) )
                    continue;
                    
                if ( k != i )
                {
                    auto  R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                    auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

                    // // DEBUG {
                    // {
                    //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
                    //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
                    //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
                    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
                    //     blas::add( value_t(-1), M1, M2 );
                    //     std::cout << "    ext col/row : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
                    // }
                    // // DEBUG }

                    R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
                }// if
            }// for
        }// else
    }

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

    auto  Un = blas::matrix< value_t >();

    {
        // determine number of rows of matrix S below (sum of column ranks)
        size_t  nrows_S = T.ncols();  // known apriori
    
        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
                nrows_S += cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > )->col_rank();
        }// for

        if ( nrows_S == T.ncols() )
        {
            //
            // since there is no other low-rank block, new row basis is W
            //

            Un = std::move( blas::copy( W ) );
        }// if
        else
        {
            // extended row basis
            auto  U  = R_ij->row_cb().basis();
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
        
                if ( k == j )
                {
                    // R_ik = W T X' with W/X being orthogonal, hence |R_ik| = |T|
                    const auto  rank = T.ncols();
                    auto        S_ik = blas::copy( T );

                    blas::scale( value_t(1) / blas::norm_2( T ), S_ik );

                    auto  S_k = blas::matrix( S,
                                              blas::range( pos, pos + rank-1 ),
                                              blas::range( U.ncols(), Ue.ncols() - 1 ) );

                    blas::copy( blas::adjoint( S_ik ), S_k );
                    pos += rank;
                }// if
                else
                {
                    // R_ik = U_i S_ik V_k' with U_i/V_k being orthogonal, hence |R_ik| = |S_ik|
                    const auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                    const auto  rank = R_ik->col_rank();
                    auto        S_ik = blas::copy( R_ik->coeff() );
                    
                    blas::scale( value_t(1) / blas::norm_2( S_ik ), S_ik );
            
                    auto  S_k = blas::matrix( S,
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
            auto  Ss  = blas::vector< real_t >();

            blas::svd( UeR, Ss );

            const auto  rank   = acc.trunc_rank( Ss );
            const auto  U_rank = blas::matrix( UeR, blas::range::all, blas::range( 0, rank-1 ) );

            Un = std::move( blas::copy( U_rank ) );

            //
            // transform coupling matrix for blocks in current block column as
            //
            //   Un'·U·S_i = TU·S_i  with TU = Un'·U
            //

            const auto  TU = blas::prod( blas::adjoint( Un ), U );

            for ( uint  k = 0; k < M.nblock_cols(); ++k )
            {
                auto  B_ik = M.block( i, k );
                    
                if ( ! matrix::is_uniform_lowrank( B_ik ) )
                    continue;
                    
                if ( k != j )
                {
                    auto  R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
                    auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

                    // // DEBUG {
                    // {
                    //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
                    //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
                    //     auto  US2   = blas::prod( Un, Sn_ik );
                    //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
                        
                    //     blas::add( value_t(-1), M1, M2 );
                    //     std::cout << "    ext row/col : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
                    // }
                    // // DEBUG }

                    R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
                }// if
            }// for
        }// else
    }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    // // DEBUG {
    // io::matlab::write( Un, "Un" );
    // io::matlab::write( Sn, "Sn" );
    // io::matlab::write( Vn, "Vn" );
    
    // {
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext    /    : " << R_ij->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    R_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & R_ij->col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::cluster_basis< value_t > * >( & R_ij->row_cb() )->set_basis( std::move( Un ) );
}

template < typename value_t >
void
extend_row_col_basis_ref ( hpro::TBlockMatrix &       M,
                           const uint                 i,
                           const uint                 j,
                           blas::matrix< value_t > &  W,
                           blas::matrix< value_t > &  T,
                           blas::matrix< value_t > &  X,
                           const hpro::TTruncAcc &    acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    auto  M_ij = M.block( i, j );
    auto  R_ij = ptrcast( M_ij, matrix::uniform_lrmatrix< value_t > );

    //
    // compute new column basis of block column
    //
    //   ⎛U_1 S_1 V'⎞
    //   ⎜U_2 S_2 V'⎟
    //   ⎜  ...     ⎟
    //   ⎜U_j S_j V'⎟
    //   ⎝W   T   X'⎠
    //
                                  
    auto  Vn = blas::matrix< value_t >();

    {
        // determine dimensions of block column
        size_t  nrows = W.nrows();
        size_t  ncols = X.nrows();

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) && ( k != i ))
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
                auto  D_kj = blas::matrix< value_t >();

                if ( i == k )
                {
                    auto  XT = blas::prod( X, blas::adjoint( T ) );

                    D_kj = std::move( blas::prod( XT, blas::adjoint( W ) ) );
                }// if
                else
                {
                    auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                    auto  VS   = blas::prod( R_kj->col_cb().basis(), blas::adjoint( R_kj->coeff() ) );

                    D_kj = std::move( blas::prod( VS, blas::adjoint( R_kj->row_cb().basis() ) ) );
                }// else
                
                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

                blas::scale( value_t(1) / blas::norm_2( D_kj ), D_kj );
                blas::copy( D_kj, Xt_k );

                pos += D_kj.ncols();
            }// if
        }// for

        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );

        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        const auto  V  = R_ij->col_cb().basis();
        const auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  B_kj = M.block( k, j );
            
            if ( ! matrix::is_uniform_lowrank( B_kj ) || ( k == i ))
                continue;
            
            auto  R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

            // // DEBUG {
            // {
            //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
            //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
            
            //     blas::add( value_t(-1), M1, M2 );
            //     std::cout << "    ext col/row : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            // }
            // // DEBUG }
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// for
    }

    //
    // compute new row basis of block row
    //
    //   (U·S_1· V_1'  U·S_2·V_2'  ...  U·S_j·V_j'  W·T·X')
    //

    auto  Un = blas::matrix< value_t >();

    {
        // determine dimensions of block row
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
                auto  D_ik = blas::matrix< value_t >();

                if ( k == j )
                {
                    auto  WT = blas::prod( W, T );

                    D_ik = std::move( blas::prod( WT, blas::adjoint( X ) ) );
                }// if
                else
                {
                    auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                    auto  US   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );

                    D_ik = std::move( blas::prod( US, blas::adjoint( R_ik->col_cb().basis() ) ) );
                }// else

                blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
                
                auto  Xt_k = blas::matrix( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

                blas::copy( D_ik, Xt_k );

                pos += D_ik.ncols();
            }// if
        }// for

        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );

        //
        // transform coupling matrix for blocks in current block column as
        //
        //   Un'·U·S_i = TU·S_i  with TU = Un'·U
        //
        
        const auto  U  = R_ij->row_cb().basis();
        const auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  B_ik = M.block( i, k );
                    
            if ( ! matrix::is_uniform_lowrank( B_ik ) || ( k == j ))
                continue;
                    
            auto  R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

            // // DEBUG {
            // {
            //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
            //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            //     auto  US2   = blas::prod( Un, Sn_ik );
            //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            //     blas::add( value_t(-1), M1, M2 );
            //     std::cout << "    ext row/col : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            // }
            // // DEBUG }
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// for
    }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    // // DEBUG {
    // {
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext    /    : " << R_ij->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    R_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & R_ij->col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::cluster_basis< value_t > * >( & R_ij->row_cb() )->set_basis( std::move( Un ) );
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

                detail::addlr_global( *B, *R_ij, i, j, W_i, X_j, acc );
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
     const hpro::TTruncAcc &  acc )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        std::cout << i << std::endl;
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  D_ii = blas::mat< value_t >( A_ii );
            
        blas::invert( D_ii );

        //
        // L is unit diagonal so just solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
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
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( blas::adjoint( blas::mat< value_t >( A_ii ) ), V_i );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc );
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

namespace detail
{

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
        const auto  U_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_U-1 ) );

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
        const auto  V_rank = blas::matrix( Vs, blas::range::all, blas::range( 0, rank_V-1 ) );

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

}// namespace detail

template < typename value_t >
void
lu_lazy ( hpro::TMatrix &          A,
          const hpro::TTruncAcc &  acc,
          hpro::TMatrix &          REF )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( & A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    // DEBUG {
    // auto  approx     = approx::SVD< value_t >();
    // auto  comp_error = [BA,&REF] ( const int  i,
    //                                const int  j )
    //                    {
    //                        auto  BREF   = cptrcast( & REF, hpro::TBlockMatrix );
    //                        auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( i, j ) );
    //                        auto  LOC_ij = matrix::convert_to_dense< value_t >( * BA->block( i, j ) );
    //                        auto  M1     = blas::copy( blas::mat< value_t >( REF_ij ) );
    //                        auto  M2     = blas::copy( blas::mat< value_t >( LOC_ij ) );
                           
    //                        blas::add( value_t(-1), M1, M2 );

    //                        const auto  err = blas::norm_2( M2 ) / blas::norm_2( M1 );

    //                        std::cout << "  error " << BA->block( i, j )->id() << " : " << boost::format( "%.4e" ) % err << std::endl;
    //                    };

    // auto  BREF = ptrcast( & REF, hpro::TBlockMatrix );
    // DEBUG }
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        std::cout << i << std::endl;
        
        //
        // invert diagonal block
        //
        
        for ( int  k = 0; k < int(i); ++k )
            detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, i, k, i, acc );

        // // DEBUG {
        // for ( int  k = 0; k < int(i); k++ )
        //     hlr::seq::multiply< value_t >( value_t(-1),
        //                                    apply_normal, *BREF->block( i, k ),
        //                                    apply_normal, *BREF->block( k, i ),
        //                                    *BREF->block( i, i ), acc, approx );
        
        // auto  REFA_ii = ptrcast( BREF->block( i, i ), hpro::TDenseMatrix );
        // auto  REFD_ii = blas::mat< value_t >( REFA_ii );

        // // comp_error( i, i );
        
        // blas::invert( REFD_ii );
        // // DEBUG }
        
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  D_ii = blas::mat< value_t >( A_ii );
            
        blas::invert( D_ii );

        // comp_error( i, i );

        //
        // solve with L, e.g. L_ii X_ij = M_ij
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            // comp_error( i, j );
            
            auto  A_ij = BA->block( i, j );

            // // DEBUG {
            // // only update block as L = I
            // for ( int  k = 0; k < int(i); k++ )
            //     hlr::seq::multiply< value_t >( value_t(-1),
            //                                    hpro::apply_normal, *BREF->block( i, k ),
            //                                    hpro::apply_normal, *BREF->block( k, j ),
            //                                    *BREF->block( i, j ), acc, approx );
            // // DEBUG }
                
            if ( is_dense( A_ij ) )
            {
                for ( int  k = 0; k < int(i); ++k )
                    detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, i, k, j, acc );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ij ) )
            {
                auto  [ Uu, Su, Vu ] = detail::apply_updates< value_t >( *BA, i, j, acc );

                // even without solving, still need to update bases
                detail::extend_row_col_basis< value_t >( *BA, i, j, Uu, Su, Vu, acc );
                
                // comp_error( i, j );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + A_ij->typestr() );
        }// for
        
        //
        // solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            // // DEBUG {
            // for ( int  k = 0; k < int(i); k++ )
            //     hlr::seq::multiply< value_t >( value_t(-1),
            //                                    hpro::apply_normal, *BREF->block( j, k ),
            //                                    hpro::apply_normal, *BREF->block( k, i ),
            //                                    *BREF->block( j, i ), acc, approx );
            // // DEBUG }
            
            if ( is_dense( A_ji ) )
            {
                for ( int  k = 0; k < int(i); ++k )
                    detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, j, k, i, acc );
                
                // // DEBUG {
                // {
                //     auto  D_ji = ptrcast( BREF->block( j, i ), hpro::TDenseMatrix );
                //     auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                //     blas::prod( value_t(1), T_ji, REFD_ii, value_t(0), blas::mat< value_t >( D_ji ) );
                // }
                // // DEBUG }
                
                // X_ji = M_ji U_ii^-1
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                auto  [ Uu, Su, Vu ] = detail::apply_updates< value_t >( *BA, j, i, acc );

                // {
                //     auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( j, i ) );
                //     auto  M1     = blas::copy( blas::mat< value_t >( REF_ij ) );
                //     auto  T1     = blas::prod( Uu, Su );
                //     auto  M2     = blas::prod( T1, blas::adjoint( Vu ) );
                           
                //     blas::add( value_t(-1), M2, M1 );
                //     std::cout << "  error " << BA->block( j, i )->id() << " : " << boost::format( "%.4e" ) % blas::norm_2( M1 ) << std::endl;
                // }

                // // DEBUG {
                // {
                //     auto  REFR_ji = ptrcast( BREF->block( j, i ), hpro::TRkMatrix );
                //     auto  V       = blas::copy( blas::mat_V< value_t >( REFR_ji ) );

                //     blas::prod( value_t(1), blas::adjoint( REFD_ii ), V, value_t(0), blas::mat_V< value_t >( REFR_ji ) );
                // }
                // // DEBUG }
                
                //
                // X_ji U_ii = Ũ_j Ŝ_ji Ṽ_i' U_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' U_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii^-T V_i )
                //
                
                auto  MV_i = blas::prod( blas::adjoint( D_ii ), Vu );
                auto  RV   = blas::matrix< value_t >();

                // ensure orthogonality in new basis
                blas::qr( MV_i, RV );
                
                auto  T = blas::prod( Su, blas::adjoint( RV ) );
                    
                detail::extend_row_col_basis< value_t >( *BA, j, i, Uu, T, MV_i, acc );

                // comp_error( j, i );
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
//
template < typename value_t >
void
ldu ( hpro::TMatrix &          A,
      const hpro::TTruncAcc &  acc )
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
        
        HLR_ASSERT( is_dense( BA->block( i, i ) ) );

        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat< value_t >( ptrcast( A_ii, hpro::TDenseMatrix ) );
        
        blas::invert( D_ii );

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
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji D_ii = Ũ_j Ŝ_ji Ṽ_i' D_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' D_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( D_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( blas::adjoint( D_ii ), V_i );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc );
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
                auto  MU_i = blas::prod( D_ii, U_i );

                detail::extend_row_basis< value_t >( *BA, *R_ij, i, j, MU_i, acc );
            }// if
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
                                  apply_normal, *cptrcast( T_ii.get(), hpro::TDenseMatrix ),
                                  apply_normal, *BA,
                                  *BA, j, i, l, acc );
            }// for
        }// for
    }// for
}

}// namespace tlr


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// arithmetic functions for general matrices
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

using  indexset = hpro::TIndexSet;

////////////////////////////////////////////////////////////
//
// LU factorization
//
////////////////////////////////////////////////////////////

namespace detail
{

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

// //
// // matrix multiplication C := α·A·B + C
// //
// template < typename value_t >
// void
// multiply ( const value_t               alpha,
//            const hpro::matop_t         op_A,
//            const hpro::TBlockMatrix &  A,
//            const hpro::matop_t         op_B,
//            const hpro::TBlockMatrix &  B,
//            hpro::TBlockMatrix &        C,
//            const hpro::TTruncAcc &     acc,
//            const uniform_map_t &       rowmap,
//            const uniform_map_t &       colmap )
// {
//     for ( uint  i = 0; i < C.nblock_rows(); ++i )
//     {
//         for ( uint  j = 0; j < C.nblock_cols(); ++j )
//         {
//             HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
//             for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
//             {
//                 if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
//                     multiply< value_t >( alpha,
//                                          op_A, *A.block( i, l, op_A ),
//                                          op_B, *B.block( l, j, op_B ),
//                                          *C.block( i, j ), acc, rowmap, colmap );
//             }// if       
//         }// for
//     }// for
// }

// template < typename value_t >
// void
// multiply ( const value_t                          alpha,
//            const hpro::matop_t                    op_A,
//            const hpro::TBlockMatrix &             A,
//            const hpro::matop_t                    op_B,
//            const hpro::TBlockMatrix &             B,
//            matrix::uniform_lrmatrix< value_t > &  C,
//            const hpro::TTruncAcc &                acc,
//            const uniform_map_t &                  rowmap,
//            const uniform_map_t &                  colmap )
// {
// }

template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const uniform_map_t &    rowmap,
           const uniform_map_t &    colmap )
{
    // if ( is_blocked( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha, 
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TBlockMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else if ( matrix::is_uniform_lowrank( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, matrix::uniform_lrmatrix< value_t > ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else if ( is_dense( A ) )
    // {
    //     if ( is_blocked( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TBlockMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( matrix::is_uniform_lowrank( B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, matrix::uniform_lrmatrix< value_t > ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else if ( is_dense(   B ) )
    //     {
    //         if ( is_blocked( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, hpro::TBlockMatrix ),
    //                                  acc, rowmap, colmap );
    //         else if ( matrix::is_uniform_lowrank( C ) )
    //             multiply< value_t >( alpha,
    //                                  op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                  op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                  * ptrcast( &C, matrix::uniform_lrmatrix< value_t > ),
    //                                  acc, rowmap, colmap );
    //         else if ( is_dense(   C ) )
    //             hlr::multiply< value_t >( alpha,
    //                                       op_A, * cptrcast( &A, hpro::TDenseMatrix ),
    //                                       op_B, * cptrcast( &B, hpro::TDenseMatrix ),
    //                                       * ptrcast( &C, hpro::TDenseMatrix ) );
    //         else
    //             HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    //     }// if
    //     else
    //         HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    // }// if
    // else
    //     HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
template < typename value_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap );

template < typename value_t >
void
solve_lower_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  L,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const uniform_map_t &       rowmap,
                  const uniform_map_t &       colmap )
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
                    solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, rowmap, colmap );
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
                                             *M.block(k,j),
                                             acc, rowmap, colmap );
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

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const hpro::TMatrix &                  L,
                  matrix::uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &                acc,
                  const uniform_map_t &                  rowmap,
                  const uniform_map_t &                  colmap )
{
    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L W T X' = U S V'
        // as L W = U
        //

        auto  U = blas::copy( M.row_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.row_is(), hpro::is( 0, U.ncols()-1 ), U );

        hlr::solve_lower_tri< value_t >( side, diag, L, D );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap )
{
    if ( is_blocked( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( matrix::is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( matrix::is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
template < typename value_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap );

template < typename value_t >
void
solve_upper_tri ( const eval_side_t           side,
                  const diag_type_t           diag,
                  const hpro::TBlockMatrix &  U,
                  hpro::TBlockMatrix &        M,
                  const hpro::TTruncAcc &     acc,
                  const uniform_map_t &       rowmap,
                  const uniform_map_t &       colmap )
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
                    solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, rowmap, colmap );
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
                                             *M.block(i,k),
                                             acc, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const hpro::TMatrix &                  U,
                  matrix::uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &                acc,
                  const uniform_map_t &                  rowmap,
                  const uniform_map_t &                  colmap )
{
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve W T X' R = U S V', e.g., X' R = V', as R' X = V
        //

        auto  V = blas::copy( M.col_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.col_is(), hpro::is( 0, V.ncols()-1 ), V );

        hlr::solve_upper_tri< value_t >( from_left, diag, U, D );
    }// else
}

    template < typename value_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  const hpro::TTruncAcc &  acc,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap )
{
    if ( is_blocked( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( matrix::is_uniform_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, matrix::uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + U.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

//
// recursive LU factorization
//
template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const uniform_map_t &    rowmap,
     const uniform_map_t &    colmap )
{
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, hpro::TBlockMatrix );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), acc, rowmap, colmap );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag,
                                                *BA->block( i, i ), *BA->block( j, i ),
                                                acc, rowmap, colmap );
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag,
                                                *BA->block( i, i ), *BA->block( i, j ),
                                                acc, rowmap, colmap );
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                {
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                    {
                        HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                    
                        multiply( value_t(-1),
                                  apply_normal, *BA->block( j, i ),
                                  apply_normal, *BA->block( i, l ),
                                  *BA->block( j, l ),
                                  acc, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else if ( is_dense( A ) )
    {
        auto  D = ptrcast( &A, hpro::TDenseMatrix );

        invert< value_t >( *D );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace detail

template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc )
{
    //
    // construct mapping of A_{t × s} to set of uniform leaves per t/s
    //

    auto  rowmap = detail::uniform_map_t();
    auto  colmap = detail::uniform_map_t();

    auto  blocks = std::list< hpro::TMatrix *>{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix *>();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  B = ptrcast( M, hpro::TBlockMatrix );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            subblocks.push_back( B->block( i, j ) );
            }// if
            else if ( matrix::is_uniform_lowrank( M ) )
            {
                rowmap[ M->row_is() ].push_back( M );
                colmap[ M->col_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while

    //
    // perform actual LU factorization
    //

    detail::lu< value_t >( A, acc, rowmap, colmap );
}

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
