#ifndef __HLR_ARITH_DETAIL_UNIFORM_HH
#define __HLR_ARITH_DETAIL_UNIFORM_HH
//
// Project     : HLib
// Module      : arith/uniform
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/norm.hh> // DEBUG
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/convert.hh> // DEBUG
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>

namespace hlr { namespace uniform { namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// mat-vec : y = y + α op( M ) x
//
////////////////////////////////////////////////////////////////////////////////

using matrix::cluster_basis;
using matrix::uniform_lrmatrix;
using matrix::is_uniform_lowrank;
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
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );
        
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


////////////////////////////////////////////////////////////
//
// LU factorization
//
////////////////////////////////////////////////////////////

using  indexset = hpro::TIndexSet;

namespace detail
{

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

//
// compute M=U·S·V' + W·T·X'
// - ASSUMPTION: W and X are orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
add ( const uniform_lrmatrix< value_t > &  M,
      const blas::matrix< value_t > &      W,
      const blas::matrix< value_t > &      T,
      const blas::matrix< value_t > &      X,
      const hpro::TTruncAcc &              acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
    //
    // scale coupling matrices w.r.t. norm
    //
    
    const auto  norm_USV = blas::norm2( S );
    const auto  norm_WTX = blas::norm2( T );
    auto        S1       = blas::copy( S );
    auto        T1       = blas::copy( T );

    blas::scale( value_t(1) / norm_USV, S1 );
    blas::scale( value_t(1) / norm_WTX, T1 );
    
    // extended bases and coupling
    const auto  Ue  = blas::join_row< value_t >( { U, W } );
    auto        Se1 = blas::diag< value_t >( { S1, T1 } );
    const auto  Ve  = blas::join_row< value_t >( { V, X } );

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

    return { std::move( Un ), std::move( Sn ), std::move( Vn ) };
}
      
//
// compute M=U·S·V' + W·T·V'
// - ASSUMPTION: W is orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::matrix< value_t > >
add_row ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      W,
          const blas::matrix< value_t > &      T,
          const hpro::TTruncAcc &              acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
    //
    // scale coupling matrices w.r.t. norm
    //
    
    const auto  norm_USV = blas::norm2( S );
    const auto  norm_WTV = blas::norm2( T );
    auto        S1       = blas::copy( blas::adjoint( S ) );
    auto        T1       = blas::copy( blas::adjoint( T ) );

    blas::scale( value_t(1) / norm_USV, S1 );
    blas::scale( value_t(1) / norm_WTV, T1 );
    
    //
    // new row basis is computed as the left singular vectors of
    //
    //   (U W) ⎛S⎞ V' = (U W) (V·(S' T'))'
    //         ⎝T⎠
    //
    // of which V is orthogonal and can be omitted. 
    //
    // With QR decomposition Q·R = (S' T') one ends up with
    // the left singular vectors of (U W) R'.
    //

    auto  Q = blas::join_row< value_t >( { S1, T1 } );
    auto  R = blas::matrix< value_t >();
    
    blas::qr( Q, R, false );
                
    // extended bases and coupling
    auto  Ue = blas::join_row< value_t >( { U, W } );
    auto  Us = blas::prod( Ue, blas::adjoint( R ) );
    auto  Ss = blas::vector< real_t >();

    blas::svd( Us, Ss );
                    
    auto  rank_U = acc.trunc_rank( Ss );
    auto  U_rank = blas::matrix( Us, blas::range::all, blas::range( 0, rank_U-1 ) );
    auto  Un     = std::move( blas::copy( U_rank ) );
    
    //
    // new coupling matrix is Un'·Ue·⎛S⎞
    //                               ⎝T⎠
    //
                
    auto  Se = blas::join_col< value_t >( { S, T } );
    auto  TU = blas::prod( blas::adjoint( Un ), Ue );
    auto  Sn = blas::prod( TU, Se );

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

    return { std::move( Un ), std::move( Sn ) };
}
      
//
// compute M=U·S·V' + U·T·X'
// - ASSUMPTION: X is orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,
            blas::matrix< value_t > >
add_col ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      T,
          const blas::matrix< value_t > &      X,
          const hpro::TTruncAcc &              acc )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
    //
    // scale coupling matrices w.r.t. norm
    //
    
    const auto  norm_USV = blas::norm2( S );
    const auto  norm_UTX = blas::norm2( T );
    auto        S1       = blas::copy( S );
    auto        T1       = blas::copy( T );

    blas::scale( value_t(1) / norm_USV, S1 );
    blas::scale( value_t(1) / norm_UTX, T1 );
    
    //
    // new column basis is computed as the left singular vectors of
    //
    //   (V X) ⎛S'⎞ U' = (V X) (U (S T))'
    //         ⎝T'⎠
    //
    // of which U is orthogonal and can be omitted. 
    //
    // With QR decomposition Q·R = (S T) one ends up with the left
    // singular vectors of (V X) R'.
    //

    auto  Q = blas::join_row< value_t >( { S1, T1 } );
    auto  R = blas::matrix< value_t >();
                
    blas::qr( Q, R, false );
                    
    auto  Ve = blas::join_row< value_t >( { V, X } );
    auto  Vs = blas::prod( Ve, blas::adjoint( R ) );
    auto  Ss = blas::vector< real_t >();
    
    blas::svd( Vs, Ss );
    
    auto  rank_V = acc.trunc_rank( Ss );
    auto  V_rank = blas::matrix( Vs, blas::range::all, blas::range( 0, rank_V-1 ) );
    auto  Vn     = std::move( blas::copy( V_rank ) );
    
    //
    // new coupling matrix is ⎛S⎞·(V X)'·Vn
    //                        ⎝T⎠
                
    auto  Se = blas::join_row< value_t >( { S, T } );
    auto  TV = blas::prod( blas::adjoint( Ve ), Vn );
    auto  Sn = blas::prod( Se, TV );

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

    return { std::move( Sn ), std::move( Vn ) };
}

//
// compute new row basis for block row of M with M being replaced by W·T·X'
// assuming all involved bases are orthogonal (X is not needed for computation)
//
template < typename value_t >
blas::matrix< value_t >
compute_updated_row_basis ( const uniform_lrmatrix< value_t > &  M,
                            const blas::matrix< value_t > &      W,
                            const blas::matrix< value_t > &      T,
                            const hpro::TTruncAcc &              acc,
                            const uniform_map_t &                rowmap )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

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

    // determine number of rows of matrix S below (sum of column ranks)
    size_t  nrows_S = T.ncols();  // known apriori
    
    for ( auto  M_ik : rowmap.at( M.row_is() ) )
    {
        if ( matrix::is_uniform_lowrank( M_ik ) && ( M_ik->col_is() != M.col_is() ))
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
        auto  U  = M.row_cb().basis();
        auto  Ue = blas::join_row< value_t >( { U, W } );

        // compute QR of column basis for each block in row and assemble
        // all results into common matrix Q
        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
        size_t  pos = 0;

        for ( auto  M_ik : rowmap.at( M.row_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_ik ) )
                continue;
        
            if ( M_ik->col_is() == M.col_is() )
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

        return std::move( blas::copy( U_rank ) );
    }// else
}

//
// compute new column basis for block column of M with M being replaced by W·T·X'
// assuming all involved bases are orthogonal (W is not needed for computation)
//
template < typename value_t >
blas::matrix< value_t >
compute_updated_col_basis ( const uniform_lrmatrix< value_t > &  M,
                            const blas::matrix< value_t > &      T,
                            const blas::matrix< value_t > &      X,
                            const hpro::TTruncAcc &              acc,
                            const uniform_map_t &                colmap )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

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
                                  
    // determine number of rows of matrix S below (sum of row ranks)
    size_t  nrows_S = T.nrows(); // known apriori
    
    for ( auto  M_kj : colmap.at( M.col_is() ) )
    {
        if ( matrix::is_uniform_lowrank( M_kj ) && ( M_kj != &M ))
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
            
        auto  V  = M.col_cb().basis();
        auto  Ve = blas::join_row< value_t >( { V, X } );
    
        // assemble normalized coefficient matrices into common matrix S
        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
        size_t  pos = 0;

        for ( auto  M_kj : colmap.at( M.col_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;

            if ( M_kj == &M )
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

        return std::move( blas::copy( V_rank ) );
    }// else
}

//
// replace U·S·V' of M by W·T·X' and update row/column bases
// - ASSUMPTION: W and X are orthogonal
//
template < typename value_t >
void
update_row_col_basis ( uniform_lrmatrix< value_t > &    M,
                       const blas::matrix< value_t > &  W,
                       const blas::matrix< value_t > &  T,
                       const blas::matrix< value_t > &  X,
                       const hpro::TTruncAcc &          acc,
                       const uniform_map_t &            rowmap,
                       const uniform_map_t &            colmap )
{
    // io::matlab::write( W, "W" );
    // io::matlab::write( T, "T" );
    // io::matlab::write( X, "X" );

    auto  Vn = compute_updated_col_basis( M, T, X, acc, colmap );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        auto  V  = M.col_cb().basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( auto  M_kj : colmap.at( M.col_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;
                    
            if ( M_kj != &M )
            {
                auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
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
    }

    auto  Un = compute_updated_row_basis( M, W, T, acc, rowmap );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   Un'·U·S_i = TU·S_i  with TU = Un'·U
        //

        auto  U  = M.row_cb().basis();
        auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( auto  M_ik : rowmap.at( M.row_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_ik ) )
                continue;
                    
            if ( M_ik != &M )
            {
                auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
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
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M.col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::cluster_basis< value_t > * >( & M.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace M=U·S·V' by W·T·V' and update row bases of
// all other blocks in block row
// - ASSUMPTION: W is orthogonal
//
template < typename value_t >
void
update_row_basis ( uniform_lrmatrix< value_t > &    M,
                   const blas::matrix< value_t > &  W,
                   const blas::matrix< value_t > &  T,
                   const hpro::TTruncAcc &          acc,
                   const uniform_map_t &            rowmap )
{
    auto  Un = compute_updated_row_basis( M, W, T, acc, rowmap );

    // io::matlab::write( Un, "Un" );
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   TU ⎛S_kj⎞  or  TU ⎛  0 ⎞
    //      ⎝ 0  ⎠         ⎝S_kj⎠
    //

    auto  U  = M.row_cb().basis();
    auto  TU = blas::prod( blas::adjoint( Un ), U );

    for ( auto  M_ik : rowmap.at( M.row_is() ) )
    {
        if ( ! matrix::is_uniform_lowrank( M_ik ) )
            continue;
                    
        if ( M_ik != &M )
        {
            auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
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

    //
    // compute coupling of M_ij as Un' W T
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  Sn = blas::prod( TW, T );

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
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace M=U·S·V' by U·T·X' and update row bases of
// all other blocks in block column
// - ASSUMPTION: X is orthogonal
//
template < typename value_t >
void
update_col_basis ( uniform_lrmatrix< value_t > &    M,
                   const blas::matrix< value_t > &  T,
                   const blas::matrix< value_t > &  X,
                   const hpro::TTruncAcc &          acc,
                   const uniform_map_t &            colmap )
{
    auto  Vn = compute_updated_col_basis( M, T, X, acc, colmap );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        auto  V  = M.col_cb().basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( auto  M_kj : colmap.at( M.col_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;
                    
            if ( M_kj != &M )
            {
                auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
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
    }

    //
    // compute coupling of M as T X' Vn
    //

    auto  TX = blas::prod( T,  blas::adjoint( X ) );
    auto  Sn = blas::prod( TX, Vn );

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
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster bases
    //

    const_cast< matrix::cluster_basis< value_t > * >( & M.col_cb() )->set_basis( std::move( Vn ) );
}

//
// add U·S·V' to M
// - ASSUMPTION: U and V are orthogonal
//
template < typename value_t >
void
add ( hpro::TMatrix &                  M,
      const blas::matrix< value_t > &  U,
      const blas::matrix< value_t > &  S,
      const blas::matrix< value_t > &  V,
      const hpro::TTruncAcc &          acc,
      const uniform_map_t &            rowmap,
      const uniform_map_t &            colmap )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  U_i = blas::matrix( U, B_ij->row_is() - B->row_ofs(), blas::range::all );
                    auto  V_j = blas::matrix( V, B_ij->col_is() - B->col_ofs(), blas::range::all );
                    
                    add( *B_ij, U_i, S, V_j, acc, rowmap, colmap );
                }// if
            }// for
        }// for
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R = ptrcast( &M, uniform_lrmatrix< value_t > );

        auto [ Un, Sn, Vn ] = add( *R, U, S, V, acc );
        
        update_row_col_basis( *R, Un, Sn, Vn, acc, rowmap, colmap );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D  = ptrcast( &M, hpro::TDenseMatrix );
        auto  US = blas::prod( U, S );

        blas::prod( value_t(1), US, blas::adjoint( V ), value_t(1), blas::mat< value_t >( D ) );
    }// if
}

//
// forward decl. of general version
//
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
           const uniform_map_t &    colmap );

//
// matrix multiplication C := α·A·B + C
//
template < typename value_t >
void
multiply ( const value_t               alpha,
           const hpro::matop_t         op_A,
           const hpro::TBlockMatrix &  A,
           const hpro::matop_t         op_B,
           const hpro::TBlockMatrix &  B,
           hpro::TBlockMatrix &        C,
           const hpro::TTruncAcc &     acc,
           const uniform_map_t &       rowmap,
           const uniform_map_t &       colmap )
{
    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                    multiply< value_t >( alpha,
                                         op_A, *A.block( i, l, op_A ),
                                         op_B, *B.block( l, j, op_B ),
                                         *C.block( i, j ), acc, rowmap, colmap );
            }// if       
        }// for
    }// for
}

template < typename value_t >
void
multiply ( const value_t                  alpha,
           const hpro::matop_t            op_A,
           const hpro::TBlockMatrix &     A,
           const hpro::matop_t            op_B,
           const hpro::TBlockMatrix &     B,
           uniform_lrmatrix< value_t > &  C,
           const hpro::TTruncAcc &        acc,
           const uniform_map_t &          rowmap,
           const uniform_map_t &          colmap )
{
    //
    // compute temporary standard low-rank block matrix BC
    // and sub blocks BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< hpro::TBlockMatrix >( C.row_is(), C.col_is() );

    BC->set_block_struct( A.nblock_rows( op_A ), B.nblock_cols( op_B ) );
    
    for ( uint  i = 0; i < A.nblock_rows( op_A ); ++i )
    {
        for ( uint  j = 0; j < B.nblock_cols( op_B ); ++j )
        {
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                {
                    auto  A_il = A.block( i, l, op_A );
                    auto  B_lj = B.block( l, j, op_B );

                    if ( is_null( BC->block( i, j ) ) )
                        BC->set_block( i, j, new hpro::TRkMatrix( A_il->row_is( op_A ), B_lj->col_is( op_B ),
                                                                  hpro::value_type< value_t >::value ) );
                    
                    hlr::multiply< value_t >( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx::SVD< value_t >() );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    //
    // convert to lowrank format and construct U·S·V' via QR
    //
    
    auto  R = hlr::matrix::convert_to_lowrank( *BC, acc, approx::SVD< value_t >() );
    auto  U  = blas::mat_U< value_t >( R );
    auto  V  = blas::mat_V< value_t >( R );
    auto  RU = blas::matrix< value_t >();
    auto  RV = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    auto [ Un, Sn, Vn ] = add( C, U, S, V, acc );
    
    update_row_col_basis( C, Un, Sn, Vn, acc, rowmap, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const hpro::TBlockMatrix &           A,
           const hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           hpro::TBlockMatrix &                 C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                rowmap,
           const uniform_map_t &                colmap )
{
    // (A·U)·S·V' + C
    auto  U  = B.row_cb( op_B ).basis();
    auto  AU = blas::matrix< value_t >( C.nrows(), U.ncols() );

    hlr::multiply< value_t >( alpha, op_A, A, U, AU );

    add( C, AU, B.coeff(), B.col_cb().basis(), acc, rowmap, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const hpro::TBlockMatrix &           A,
           const hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           uniform_lrmatrix< value_t > &        C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                rowmap,
           const uniform_map_t &                /* colmap */ )
{
    // A·B + C = (A·U)·S·V' + W·T·V'
    auto  U  = B.row_cb( op_B ).basis();
    auto  AU = blas::matrix< value_t >( C.nrows(), U.ncols() );

    hlr::multiply< value_t >( alpha, op_A, A, U, AU );

    auto [ Un, Sn ] = add_row( C, AU, B.coeff(), acc );

    detail::update_row_basis( C, Un, Sn, acc, rowmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                  op_B,
           const hpro::TBlockMatrix &           B,
           hpro::TBlockMatrix &                 C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                rowmap,
           const uniform_map_t &                colmap )
{
    // U·S·(V'·B) + C with V'·B computed as B'·V
    auto  V  = A.col_cb( op_A ).basis();
    auto  BV = blas::matrix< value_t >( C.ncols(), V.ncols() );

    hlr::multiply< value_t >( alpha, blas::adjoint( op_B ), B, V, BV );

    add( C, A.row_cb().basis(), A.coeff(), BV, acc, rowmap, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                  op_B,
           const hpro::TBlockMatrix &           B,
           uniform_lrmatrix< value_t > &        C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                /* rowmap */,
           const uniform_map_t &                colmap )
{
    // U·S·(V'·B) + U·T·X' with V'·B computed as B'·V
    auto  V  = A.col_cb( op_A ).basis();
    auto  BV = blas::matrix< value_t >( C.ncols(), V.ncols() );

    hlr::multiply< value_t >( alpha, blas::adjoint( op_B ), B, V, BV );

    auto [ Sn, Vn ] = add_col( C, A.coeff(), BV, acc );
    
    detail::update_col_basis( C, Sn, Vn, acc, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           hpro::TBlockMatrix &                 C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                rowmap,
           const uniform_map_t &                colmap )
{
    // U·(S·V' × W·T)·X' + C = U·R·X' + C
    auto  R = blas::matrix< value_t >();

    {
        auto  VW   = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
        auto  SVW  = blas::prod( alpha, blas::mat_view( op_A, A.coeff() ), VW );

        R = std::move( blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) ) );
    }

    add( C, A.row_cb().basis(), R, B.col_cb().basis(), acc, rowmap, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           uniform_lrmatrix< value_t > &        C,
           const hpro::TTruncAcc &              /* acc */,
           const uniform_map_t &                /* rowmap */,
           const uniform_map_t &                /* colmap */ )
{
    // A·B + C = U·(S·V' × W·T)·X' + U·R·X'
    auto  VW   = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
    auto  SVW  = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );

    blas::prod( alpha, SVW, blas::mat_view( op_B, B.coeff() ), value_t(1), C.coeff() );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const hpro::matop_t                  op_B,
           const hpro::TDenseMatrix &           B,
           uniform_lrmatrix< value_t > &        C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                /* rowmap */,
           const uniform_map_t &                colmap )
{
    // A×B + C = U·S·(V' × B) + U·T·X' = U·S·(B' × V)' + U·T·X'
    auto  BV = blas::prod( alpha,
                           blas::mat_view( blas::adjoint( op_B ), blas::mat< value_t >( B) ),
                           A.col_cb( op_A ).basis() );

    auto [ Sn, Vn ] = add_col( C, A.coeff(), BV, acc );
    
    detail::update_col_basis( C, Sn, Vn, acc, colmap );
}

template < typename value_t >
void
multiply ( const value_t                        alpha,
           const hpro::matop_t                  op_A,
           const hpro::TDenseMatrix &           A,
           const hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           uniform_lrmatrix< value_t > &        C,
           const hpro::TTruncAcc &              acc,
           const uniform_map_t &                rowmap,
           const uniform_map_t &                /* colmap */ )
{
    // A×B + C = (A × U)·S·V' + W·T·V'
    auto  AU = blas::prod( alpha,
                           blas::mat_view( op_A, blas::mat< value_t >( A ) ),
                           B.row_cb( op_B ).basis() );

    auto [ Un, Sn ] = add_row( C, AU, B.coeff(), acc );
    
    detail::update_row_basis( C, Un, Sn, acc, rowmap );
}

template < typename value_t >
void
multiply ( const value_t                  alpha,
           const hpro::matop_t            op_A,
           const hpro::TDenseMatrix &     A,
           const hpro::matop_t            op_B,
           const hpro::TDenseMatrix &     B,
           uniform_lrmatrix< value_t > &  C,
           const hpro::TTruncAcc &        acc,
           const uniform_map_t &          rowmap,
           const uniform_map_t &          colmap )
{
    HLR_MULT_PRINT;
    
    auto  AB       = blas::prod( alpha,
                                 blas::mat_view( op_A, hpro::blas_mat< value_t >( A ) ),
                                 blas::mat_view( op_B, hpro::blas_mat< value_t >( B ) ) );
    auto  [ U, V ] = approx::svd( AB, acc );
    auto  RU       = blas::matrix< value_t >();
    auto  RV       = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    auto [ Un, Sn, Vn ] = add( C, U, S, V, acc );
    
    detail::update_row_col_basis( C, Un, Sn, Vn, acc, rowmap, colmap );
}



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
    auto  DA = matrix::convert_to_dense< value_t >( A );
    auto  DB = matrix::convert_to_dense< value_t >( B );
    auto  DC = matrix::convert_to_dense< value_t >( C );

    hlr::multiply< value_t >( alpha, op_A, *DA, op_B, *DB, *DC );
    
    if ( is_blocked( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha, 
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TBlockMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                          op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, uniform_lrmatrix< value_t > ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_dense( A ) )
    {
        if ( is_blocked( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TBlockMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, uniform_lrmatrix< value_t > ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_dense(   B ) )
        {
            if ( is_blocked( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, hpro::TBlockMatrix ),
                                     acc, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) )
                multiply< value_t >( alpha,
                                     op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                     op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                     * ptrcast( &C, uniform_lrmatrix< value_t > ),
                                     acc, rowmap, colmap );
            else if ( is_dense(   C ) )
                hlr::multiply< value_t >( alpha,
                                          op_A, * cptrcast( &A, hpro::TDenseMatrix ),
                                          op_B, * cptrcast( &B, hpro::TDenseMatrix ),
                                          * ptrcast( &C, hpro::TDenseMatrix ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    auto  DD = matrix::convert_to_dense< value_t >( C );

    hlr::add( value_t(-1), *DC, *DD );

    std::cout << "multiply : " << A.id() << " × " << B.id() << " = " << C.id() << " : " << norm::frobenius( *DD ) << std::endl;
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
solve_lower_tri ( const eval_side_t              side,
                  const diag_type_t              diag,
                  const hpro::TMatrix &          L,
                  uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &        acc,
                  const uniform_map_t &          rowmap,
                  const uniform_map_t &          /* colmap */ )
{
    if ( is_dense( L ) && ( diag == unit_diag ))
        return;
    
    if ( side == from_left )
    {
        //
        // solve L×M = L×W·T·X' = U·S·V' as L×W = U
        //

        auto  W = blas::copy( M.row_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.row_is(), hpro::is( 0, W.ncols()-1 ), W );

        hlr::solve_lower_tri< value_t >( side, diag, L, D );

        // orthogonalise W, compute T and update row basis
        auto  R = blas::matrix< value_t >();

        blas::qr( W, R );

        auto  T = blas::prod( R, M.coeff() );

        update_row_basis( M, W, T, acc, rowmap );
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
    auto  DL = hlr::seq::matrix::copy_nonuniform< value_t >( L );
    auto  DM = matrix::convert_to_dense< value_t >( M );

    hlr::solve_lower_tri< value_t >( side, diag, *DL, *DM );
    
    if ( is_blocked( L ) )
    {
        if ( is_blocked( M ) )
            solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( is_dense( L ) )
    {
        if ( is_uniform_lowrank( M ) )
            solve_lower_tri< value_t >( side, diag, L, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_lower_tri< value_t >( side, diag, * cptrcast( & L, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    auto  DX = matrix::convert_to_dense< value_t >( M );

    hlr::add( value_t(-1), *DX, *DM );
    std::cout << "solve_lower_tri: " << M.id() << " : " << hlr::norm::frobenius( *DM ) << std::endl;
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
solve_upper_tri ( const eval_side_t              side,
                  const diag_type_t              diag,
                  const hpro::TMatrix &          U,
                  uniform_lrmatrix< value_t > &  M,
                  const hpro::TTruncAcc &        acc,
                  const uniform_map_t &          /* rowmap */,
                  const uniform_map_t &          colmap )
{
    if ( is_dense( U ) && ( diag == unit_diag ))
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve W·T·X'×R = U·S·V', e.g., X'×R = V', as R'×X = V
        //

        auto  X = blas::copy( M.col_cb().basis() );
        auto  D = hpro::TDenseMatrix( M.col_is(), hpro::is( 0, X.ncols()-1 ), X );

        hlr::solve_upper_tri< value_t >( from_left, diag, U, D );

        // orthogonalise X, compute T and update column basis
        auto  R = blas::matrix< value_t >();
        
        blas::qr( X, R );

        auto  T = blas::prod( M.coeff(), blas::adjoint( R ) );
        
        update_col_basis( M, T, X, acc, colmap );
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
    auto  DU = hlr::seq::matrix::copy_nonuniform< value_t >( U );
    auto  DM = matrix::convert_to_dense< value_t >( M );

    hlr::solve_upper_tri< value_t >( side, diag, *DU, *DM );
    
    if ( is_blocked( U ) )
    {
        if ( is_blocked( M ) )
            solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TBlockMatrix ), acc, rowmap, colmap );
        else if ( is_uniform_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TBlockMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }//if
    else if ( is_dense( U ) )
    {
        if ( is_uniform_lowrank( M ) )
            solve_upper_tri< value_t >( side, diag, U, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, rowmap, colmap );
        else if ( is_dense( M ) )
            hlr::solve_upper_tri< value_t >( side, diag, * cptrcast( & U, hpro::TDenseMatrix ), * ptrcast( & M, hpro::TDenseMatrix ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );

    auto  DX = matrix::convert_to_dense< value_t >( M );

    hlr::add( value_t(-1), *DX, *DM );
    std::cout << "solve_upper_tri: " << M.id() << " : " << hlr::norm::frobenius( *DM ) << std::endl;
}

//
// recursive LU factorization
//
template < typename value_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const uniform_map_t &    rowmap,
     const uniform_map_t &    colmap,
     hpro::TMatrix &          REF )
{
    if ( is_blocked( A ) )
    {
        auto  BA   = ptrcast( &A,   hpro::TBlockMatrix );
        auto  BREF = ptrcast( &REF, hpro::TBlockMatrix );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), acc, rowmap, colmap, *BREF->block( i, i ) );

            {
                auto  D1 = matrix::convert_to_dense< value_t >( *BA->block(i,i) );
                auto  D2 = matrix::convert_to_dense< value_t >( *BREF->block(i,i) );

                hlr::add( value_t(-1), *D1, *D2 );
                std::cout << BA->block(i,i)->id() << " : " << norm::frobenius( *D2 ) << std::endl;
            }

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag,
                                                *BA->block( i, i ), *BA->block( j, i ),
                                                acc, rowmap, colmap );

                {
                    auto  D1 = matrix::convert_to_dense< value_t >( *BA->block(j,i) );
                    auto  D2 = matrix::convert_to_dense< value_t >( *BREF->block(j,i) );
                    
                    hlr::add( value_t(-1), *D1, *D2 );
                    std::cout << BA->block(j,i)->id() << " : " << norm::frobenius( *D2 ) << std::endl;
                }
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag,
                                                *BA->block( i, i ), *BA->block( i, j ),
                                                acc, rowmap, colmap );

                {
                    auto  D1 = matrix::convert_to_dense< value_t >( *BA->block(i,j) );
                    auto  D2 = matrix::convert_to_dense< value_t >( *BREF->block(i,j) );
                    
                    hlr::add( value_t(-1), *D1, *D2 );
                    std::cout << BA->block(i,j)->id() << " : " << norm::frobenius( *D2 ) << std::endl;
                }
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

}}}// namespace hlr::uniform::detail

#endif // __HLR_ARITH_DETAIL_UNIFORM_HH
