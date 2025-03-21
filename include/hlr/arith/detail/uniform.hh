#ifndef __HLR_ARITH_DETAIL_UNIFORM_HH
#define __HLR_ARITH_DETAIL_UNIFORM_HH
//
// Project     : HLR
// Module      : arith/uniform
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/norm.hh>
#include <hlr/arith/multiply.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/invert.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/matrix/convert.hh>
#include <hlr/matrix/level_hierarchy.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>

#include <hlr/arith/detail/uniform_basis.hh>

namespace hlr { namespace uniform { namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// mat-vec : y = y + α op( M ) x
//
////////////////////////////////////////////////////////////////////////////////

using matrix::shared_cluster_basis;
using matrix::shared_cluster_basis_hierarchy;
using matrix::uniform_lrmatrix;
using matrix::uniform_lr2matrix;
using matrix::is_uniform_lowrank;
using matrix::is_uniform_lowrank2;
using matrix::level_hierarchy;
using vector::scalar_vector;
using vector::uniform_vector;
using vector::uniform_vector_hierarchy;

using indexset = Hpro::TIndexSet;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec ( const value_t                                              alpha,
          const Hpro::matop_t                                        op_M,
          const Hpro::TMatrix< value_t > &                           M,
          const uniform_vector< shared_cluster_basis< value_t > > &  x,
          uniform_vector< shared_cluster_basis< value_t > > &        y,
          const scalar_vector< value_t > &                           sx,
          scalar_vector< value_t > &                                 sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

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
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        //
        // y = U S V^H x
        //

        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        //
        // y = U S_r S_c^H V^H x
        //

        auto  R = cptrcast( &M, uniform_lr2matrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< shared_cluster_basis< value_t > > >
scalar_to_uniform ( const shared_cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &         v )
{
    auto  u = std::make_unique< uniform_vector< shared_cluster_basis< value_t > > >( cb );

    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
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

template < typename value_t >
using  is_veccoeff_map_t  = std::unordered_map< indexset, blas::vector< value_t >, indexset_hash >;

template < typename value_t >
void
scalar_to_uniform ( const shared_cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &         v,
                    is_veccoeff_map_t< value_t > &           coeffmap )
{
    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
        auto  s    = cb.transform_forward( v_cb );

        coeffmap[ cb.is() ] = std::move( s );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            scalar_to_uniform( *cb.son(i), v, coeffmap );
    }// if
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t >
void
add_uniform_to_scalar ( const uniform_vector< shared_cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                                 v )
{
    if ( u.basis().rank() > 0 )
    {
        auto  v_u = blas::vector< value_t >( blas::vec( v ), u.is() - v.ofs() );

        u.basis().transform_backward( u.coeffs(), v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        for ( uint  i = 0; i < u.nblocks(); ++i )
            add_uniform_to_scalar( *u.block(i), v );
    }// if
}

template < typename value_t >
void
mul_vec2 ( const value_t                         alpha,
           const Hpro::matop_t                   op_M,
           const Hpro::TMatrix< value_t > &      M,
           const is_veccoeff_map_t< value_t > &  x_cmap,
           const scalar_vector< value_t > &      x,
           scalar_vector< value_t > &            y,
           const is_matrix_cmap_t< value_t > &   matmap )
{
    //
    // go over all clusters in <matmap> and compute local update to y
    //

    for ( auto  entry : matmap )
    {
        const auto                               is    = entry.first;
        auto                                     y_j   = blas::vector< value_t >();
        auto                                     ly    = blas::vector< value_t >();
        auto                                     sy    = blas::vector< value_t >();
        const shared_cluster_basis< value_t > *  rowcb = nullptr;
        
        for ( auto  M : matmap.at( is ) )
        {
            if ( is_uniform_lowrank( M ) )
            {
                auto  R   = cptrcast( M, uniform_lrmatrix< value_t > );
                auto  x_i = x_cmap.at( R->col_is( op_M ) );

                if ( sy.length() == 0 )
                    sy = std::move( blas::vector< value_t >( R->row_cb( op_M ).rank() ) );

                if ( is_null( rowcb ) )
                    rowcb = & R->row_cb( op_M );

                if ( y_j.length() == 0 )
                    y_j = std::move( blas::vector< value_t >( blas::vec( y ), M->row_is( op_M ) - y.ofs() ) );
                
                switch ( op_M )
                {
                    case Hpro::apply_normal :
                        blas::mulvec( alpha, R->coeff(), x_i, value_t(1), sy );
                        break;

                    case Hpro::apply_conjugate :
                        HLR_ASSERT( false );

                    case Hpro::apply_transposed :
                        HLR_ASSERT( false );

                    case Hpro::apply_adjoint :
                        blas::mulvec( alpha, blas::adjoint(R->coeff()), x_i, value_t(1), sy );
                        break;

                    default:
                        HLR_ERROR( "unsupported matrix operator" );
                }// switch
            }// if
            else if ( matrix::is_dense( M ) )
            {
                auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );

                if ( ly.length() == 0 )
                    ly = std::move( blas::vector< value_t >( M->row_is( op_M ).size() ) );
                
                if ( y_j.length() == 0 )
                    y_j = std::move( blas::vector< value_t >( blas::vec( y ), M->row_is( op_M ) - y.ofs() ) );
                
                M->apply_add( alpha, x_i, ly, op_M );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// for

        //
        // apply updates to y
        //

        if ( ly.length() > 0 )
            blas::add( value_t(1), ly, y_j );
            
        if ( ! is_null( rowcb ) )
        {
            HLR_ASSERT( sy.length() > 0 );
            
            auto  t = rowcb->transform_backward( sy );

            blas::add( value_t(1), t, y_j );
        }// if
    }// for
}

//
// uniform mat-vec with level-wise approach
//
template < typename value_t >
std::unique_ptr< uniform_vector_hierarchy< shared_cluster_basis< value_t > > >
scalar_to_uniform ( const shared_cluster_basis_hierarchy< value_t > &  cb,
                    const scalar_vector< value_t > &                   v )
{
    auto        hier = std::make_unique< uniform_vector_hierarchy< shared_cluster_basis< value_t > > >();
    const auto  nlvl = cb.hierarchy().size();
    
    hier->set_nlevel( nlvl );

    for ( uint  lvl = 0; lvl < nlvl; ++lvl )
    {
        const auto  ncb = cb.hierarchy()[lvl].size();
        
        hier->hierarchy()[lvl].resize( ncb );

        for ( uint  i = 0; i < ncb; ++i )
        {
            auto  cb_i = cb.hierarchy()[lvl][i];

            if ( ! is_null( cb_i ) && ( cb_i->rank() > 0 ))
            {
                auto  u_i  = std::make_unique< uniform_vector< shared_cluster_basis< value_t > > >( *cb_i );
                auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb_i->is() - v.ofs() );
                auto  s    = cb_i->transform_forward( v_cb );

                u_i->set_coeffs( std::move( s ) );
                hier->hierarchy()[lvl][i] = u_i.release();
            }// if
        }// for
    }// for
    
    return hier;
}

template < typename value_t >
void
mul_vec_hier ( const value_t                                                        alpha,
               const Hpro::matop_t                                                  op_M,
               const level_hierarchy< value_t > &                                   M,
               const uniform_vector_hierarchy< shared_cluster_basis< value_t > > &  x,
               const scalar_vector< value_t > &                                     sx,
               scalar_vector< value_t > &                                           sy,
               const shared_cluster_basis_hierarchy< value_t > &                    rowcb )
{
    using  cb_t = shared_cluster_basis< value_t >;
    
    HLR_ASSERT( op_M == apply_normal );
    
    const auto  nlvl = M.nlevel();

    for ( uint  lvl = 0; lvl < nlvl; ++lvl )
    {
        for ( uint  row = 0; row < M.row_ptr[lvl].size()-1; ++row )
        {
            const auto  lb = M.row_ptr[lvl][row];
            const auto  ub = M.row_ptr[lvl][row+1];

            if ( lb == ub )
                continue;
            
            cb_t *  ycb = nullptr;
            auto    s   = blas::vector< value_t >();
            auto    y_j = blas::vector< value_t >( blas::vec( sy ), M.row_mat[lvl][lb]->row_is( op_M ) - sy.ofs() );

            for ( uint  j = lb; j < ub; ++j )
            {
                auto  col_idx = M.col_idx[lvl][j];
                auto  mat     = M.row_mat[lvl][j];

                if ( matrix::is_uniform_lowrank( mat ) )
                {
                    auto  R  = cptrcast( mat, uniform_lrmatrix< value_t > );
                    auto  ux = x.hierarchy()[lvl][col_idx];
                    
                    if ( is_null( ycb ) )
                    {
                        ycb = rowcb.hierarchy()[lvl][row];
                        s   = blas::vector< value_t >( ycb->rank() );
                    }// if

                    R->uni_apply_add( alpha, ux->coeffs(), s, op_M );
                }// if
                else if ( matrix::is_uniform_lowrank2( mat ) )
                {
                    auto        R  = cptrcast( mat, uniform_lr2matrix< value_t > );
                    auto        ux = x.hierarchy()[lvl][col_idx];
                    // const auto  k  = R->rank();
                    // auto        t  = blas::vector< value_t >( k );
                    
                    if ( is_null( ycb ) )
                    {
                        ycb = rowcb.hierarchy()[lvl][row];
                        s   = blas::vector< value_t >( ycb->rank() );
                    }// if

                    R->uni_apply_add( alpha, ux->coeffs(), s, op_M );
                }// if
                else if ( matrix::is_dense( mat ) )
                {
                    auto  x_i = blas::vector< value_t >( blas::vec( sx ), mat->col_is( op_M ) - sx.ofs() );
        
                    mat->apply_add( alpha, x_i, y_j, op_M );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + mat->typestr() );
            }// for

            //
            // add uniform part to y
            //

            if ( ! is_null( ycb ) )
            {
                ycb->transform_backward( s, y_j );
            }// if
        }// for
    }// for
}

//
// construct mapping id -> clusterbasis
//
template < typename value_t >
void
build_id2cb ( shared_cluster_basis< value_t > &                   cb,
              std::vector< shared_cluster_basis< value_t > * > &  idmap )
{
    HLR_ASSERT( cb.id() != -1 );
    HLR_ASSERT( cb.id() < idmap.size() );

    idmap[ cb.id() ] = & cb;

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
                build_id2cb( * cb.son( i ), idmap );
        }// for
    }// if
}

//
// construct mapping cluster basis id (cluster) -> list of matrix blocks in M
//
template < typename value_t >
void
build_id2blocks ( const shared_cluster_basis< value_t > &                         cb,
                  const Hpro::TMatrix< value_t > &                                M,
                  std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap,
                  const bool                                                      transposed )
{
    HLR_ASSERT( cb.id() != -1 );
    HLR_ASSERT( cb.id() < blockmap.size() );

    if ( ! is_blocked( M ) )
    {
        blockmap[ cb.id() ].push_back( & M );
    }// else
    else
    {
        auto  op = ( transposed ? apply_transposed : apply_normal );
        auto  B  = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( B->nblock_rows( op ) == cb.nsons() );

        for ( uint  i = 0; i < B->nblock_rows( op ); ++i )
        {
            auto  cb_i = cb.son( i );

            HLR_ASSERT( ! is_null( cb_i ) );
            
            for ( uint  j = 0; j < B->nblock_cols( op ); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    build_id2blocks( *cb_i, * B->block( i, j ), blockmap, transposed );
            }// for
        }// if
    }// else
}

template < typename value_t >
void
scalar_to_uniform ( const shared_cluster_basis< value_t > &   cb,
                    const scalar_vector< value_t > &          v,
                    std::vector< blas::vector< value_t > > &  vcoeff )
{
    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
        
        HLR_ASSERT( cb.id() < vcoeff.size() );
        
        vcoeff[ cb.id() ] = std::move( cb.transform_forward( v_cb ) );
    }// if

    for ( uint  i = 0; i < cb.nsons(); ++i )
    {
        if ( ! is_null( cb.son(i) ) )
            scalar_to_uniform( *cb.son(i), v, vcoeff );
    }// for
}

template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const Hpro::matop_t                                                   op_M,
              const shared_cluster_basis< value_t > &                               rowcb,
              const std::vector< shared_cluster_basis< value_t > * > &              colcb,
              const std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap,
              const std::vector< blas::vector< value_t > > &                        xcoeff,
              const scalar_vector< value_t > &                                      sx,
              scalar_vector< value_t > &                                            sy )
{
    //
    // perform matvec with all blocks in block row
    //

    auto &  blockrow = blockmap[ rowcb.id() ];

    if ( blockrow.size() > 0 )
    {
        auto  rowis = rowcb.is();
        auto  y_j   = blas::vector< value_t >( blas::vec( sy ), rowis - sy.ofs() );
        auto  t_j   = blas::vector< value_t >( y_j.length() );
        auto  s     = blas::vector< value_t >( rowcb.rank() );

        for ( auto  M : blockrow )
        {
            if ( matrix::is_uniform_lowrank( M ) )
            {
                auto    R       = cptrcast( M, matrix::uniform_lrmatrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, s, op_M );
            }// if
            else if ( matrix::is_uniform_lowrank2( M ) )
            {
                auto    R       = cptrcast( M, matrix::uniform_lr2matrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, s, op_M );
            }// if
            else if ( matrix::is_dense( M ) )
            {
                auto  x_i = blas::vector< value_t >( blas::vec( sx ), M->col_is( op_M ) - sx.ofs() );
                        
                M->apply_add( alpha, x_i, t_j, op_M );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// for

        //
        // add uniform part to y
        //
        
        if ( s.length() > 0 )
            rowcb.transform_backward( s, t_j );
        
        blas::add( value_t(1), t_j, y_j );
    }// if

    //
    // recurse
    //

    if ( rowcb.nsons() > 0 )
    {
        for ( uint  i = 0; i < rowcb.nsons(); ++i )
        {
            if ( ! is_null( rowcb.son( i ) ) )
                mul_vec_row( alpha, op_M, * rowcb.son( i ), colcb, blockmap, xcoeff, sx, sy );
        }// for
    }// if
}

//
// return FLOPs needed to convert vector into uniform basis
//
template < typename cluster_basis_t >
flops_t
scalar_to_uniform_flops ( const cluster_basis_t &  cb )
{
    flops_t  flops = 0;
    
    if ( cb.rank() > 0 )
    {
        if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
            flops += FLOPS_ZGEMV( cb.rank(), cb.is().size() );
        else
            flops += FLOPS_DGEMV( cb.rank(), cb.is().size() );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            flops += scalar_to_uniform_flops( *cb.son(i) );
    }// if

    return flops;
}

//
// return FLOPs needed for computing y = y + α op( M ) x
// with M in uniform-H format
//
template < typename value_t >
flops_t
mul_vec_flops ( const Hpro::matop_t               op_M,
                const Hpro::TMatrix< value_t > &  M )
{
    using namespace hlr::matrix;
    
    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        const auto  row_ofs = M.row_is( op_M ).first();
        const auto  col_ofs = M.col_is( op_M ).first();
        flops_t     flops   = 0;
    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
            
                if ( ! is_null( B_ij ) )
                    flops += mul_vec_flops( op_M, *B_ij );
            }// for
        }// for

        return flops;
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        const auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( R->row_rank( op_M ), R->col_rank( op_M ) );
        else
            return FLOPS_DGEMV( R->row_rank( op_M ), R->col_rank( op_M ) );
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        const auto  R = cptrcast( &M, matrix::uniform_lr2matrix< value_t > );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( R->row_rank( op_M ), R->rank() ) + FLOPS_ZGEMV( R->col_rank( op_M ), R->rank() );
        else
            return FLOPS_DGEMV( R->row_rank( op_M ), R->rank() ) + FLOPS_DGEMV( R->col_rank( op_M ), R->rank() );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( nrows, ncols );
        else
            return FLOPS_DGEMV( nrows, ncols );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + M.typestr() );

    return 0;
}

//
// return FLOPs for adding vector in uniform cluster basis to scalar vector
//
template < typename cluster_basis_t >
flops_t
add_uniform_to_scalar_flops ( const cluster_basis_t &  cb )
{
    flops_t  flops = 0;
    
    if ( cb.rank() > 0 )
    {
        if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
            flops += FLOPS_ZGEMV( cb.is().size(), cb.rank() );
        else
            flops += FLOPS_DGEMV( cb.is().size(), cb.rank() );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            flops += add_uniform_to_scalar_flops( *cb.son(i) );
    }// if

    return flops;
}

}// namespace detail


////////////////////////////////////////////////////////////
//
// LU factorization
//
////////////////////////////////////////////////////////////

namespace detail
{

//
// compute M=U·S·V' + W·T·X' = (U W)⎛S  ⎞(V X)'
//                                  ⎝  T⎠
//
// - ASSUMPTION: W and X are orthogonal
//
template < typename value_t,
           typename approx_t >
std::tuple< blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
add ( const uniform_lrmatrix< value_t > &  M,
      const blas::matrix< value_t > &      W,
      const blas::matrix< value_t > &      T,
      const blas::matrix< value_t > &      X,
      const accuracy &                     acc,
      const approx_t &                     approx )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();

    //
    // extended bases and coupling
    //
    // Remark: no scaling of S/T since need to scale both by norm of M
    //
    
    const auto  Ue = blas::join_row< value_t >( { U, W } );
    auto        Se = blas::diag< value_t >( { S, T } );
    const auto  Ve = blas::join_row< value_t >( { V, X } );

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
        auto  Q  = blas::copy( blas::adjoint( Se ) );
                
        blas::qr( Q, R, false );
                
        auto  Us = blas::prod( Ue, blas::adjoint( R ) );

        Un = std::move( approx.column_basis( Us, acc ) );
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
        auto  Q  = blas::copy( Se ); // need to copy since modified during QR
                
        blas::qr( Q, R, false );
                    
        auto  Vs = blas::prod( Ve, blas::adjoint( R ) );

        Vn = std::move( approx.column_basis( Vs, acc ) );
    }
    
    //
    // new coupling matrix is
    //
    //   Un' Ue Se Ve' Vn = TU Se TVj'
    //
    // with TU = Un' Ue and TV = Vn' Ve
    //
                
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
    //         std::cout << "          add Un   : " << boost::format( "%.4e" ) % ( blas::norm_F( UUM ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  MV  = blas::prod( M1, Vn );
    //         auto  MVV = blas::prod( MV, blas::adjoint( Vn ) );
            
    //         blas::add( value_t(-1), M1, MVV );
    //         std::cout << "          add Vn   : " << boost::format( "%.4e" ) % ( blas::norm_F( MVV ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  US2 = blas::prod( Un, Sn );
    //         auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

    //         blas::add( value_t(-1), M1, M2 );
    //         std::cout << "          add     : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    //     }
    // }
    // // DEBUG }

    return { std::move( Un ), std::move( Sn ), std::move( Vn ) };
}
      
//
// compute M = U·S·V' + W·T·V' = (U W)·⎛S⎞·V'
//                                     ⎝T⎠
//
// - return orthogonal row basis and new coefficients
// - no approximation
// - W does not need to be orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,  // new row basis
            blas::matrix< value_t > > // new coupling
add_row ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      W,
          const blas::matrix< value_t > &      T )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
    //
    // new row basis is orthogonalized (U W) with updated coupling
    //

    auto  Ue = blas::join_row< value_t >( { U, W } );
    auto  R  = blas::matrix< value_t >();
    
    blas::qr( Ue, R ); // Ue is orthogonal afterwards
                
    auto  Se = blas::join_col< value_t >( { S, T } );
    auto  Sn = blas::prod( R, Se );

    return { std::move( Ue ), std::move( Sn ) };
}

//
// compute M=U·S·V' + W·T·V' = (U W)·⎛S⎞·V'
//                                   ⎝T⎠
// - ASSUMPTION: W is orthogonal
//
template < typename value_t,
           typename approx_t >
std::tuple< blas::matrix< value_t >,   // new row basis
            blas::matrix< value_t > >  // new coupling
add_row ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      W,
          const blas::matrix< value_t > &      T,
          const accuracy &                     acc,
          const approx_t &                     approx )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
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
    // Remark: no scaling of S/T since need to scale both by norm of M.
    //

    auto  Se = blas::join_col< value_t >( { S, T } );
    auto  Q  = blas::copy( blas::adjoint( Se ) );
    auto  R  = blas::matrix< value_t >();
    
    blas::qr( Q, R, false );
                
    // extended bases and coupling
    auto  Ue = blas::join_row< value_t >( { U, W } );
    auto  Us = blas::prod( Ue, blas::adjoint( R ) );
    auto  Un = approx.column_basis( Us, acc );
    
    //
    // new coupling matrix is Un'·Ue·⎛S⎞
    //                               ⎝T⎠
    //
                
    auto  TU = blas::prod( blas::adjoint( Un ), Ue );
    auto  Sn = blas::prod( TU, Se );

    // // DEBUG {
    // {
    //     auto  US1 = blas::prod( Ue, Se );
    //     auto  M1  = blas::prod( US1, blas::adjoint( V ) );

    //     {
    //         auto  UM  = blas::prod( blas::adjoint( Un ), M1 );
    //         auto  UUM = blas::prod( Un, UM );
            
    //         blas::add( value_t(-1), M1, UUM );
    //         std::cout << "          add_row Un   : " << boost::format( "%.4e" ) % ( blas::norm_F( UUM ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  US2 = blas::prod( Un, Sn );
    //         auto  M2  = blas::prod( US2, blas::adjoint( V ) );

    //         blas::add( value_t(-1), M1, M2 );
    //         std::cout << "          add_row     : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    //     }
    // }
    // // DEBUG }

    return { std::move( Un ), std::move( Sn ) };
}
      
//
// compute M = U·S·V' + U·T·X', M = U·(S T)·(V X)'
//
// - return new coefficients and orthogonal column basis
// - no approximation
// - X does not need to be orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,   // new coupling
            blas::matrix< value_t > >  // new column basis
add_col ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      T,
          const blas::matrix< value_t > &      X )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();

    //
    // new column basis is orthogonalized (V X) with updated coupling
    //

    auto  Ve = blas::join_row< value_t >( { V, X } );
    auto  R  = blas::matrix< value_t >();

    blas::qr( Ve, R ); // Ve is orthogonal afterwards
    
    auto  Se = blas::join_row< value_t >( { S, T } );
    auto  Sn = blas::prod( Se, blas::adjoint( R ) );

    return { std::move( Sn ), std::move( Ve ) };
}

//
// compute M=U·S·V' + U·T·X', M=U·(S T)·(V X)'
// - ASSUMPTION: X is orthogonal
//
template < typename value_t,
           typename approx_t >
std::tuple< blas::matrix< value_t >,   // new coupling
            blas::matrix< value_t > >  // new column basis
add_col ( const uniform_lrmatrix< value_t > &  M,
          const blas::matrix< value_t > &      T,
          const blas::matrix< value_t > &      X,
          const accuracy &                     acc,
          const approx_t &                     approx )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();

    // io::matlab::write( U, "U" );
    // io::matlab::write( S, "S" );
    // io::matlab::write( V, "V" );
    // io::matlab::write( T, "T" );
    // io::matlab::write( X, "X" );
    
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
    // Remark: no scaling of S/T since need to scale both by norm of M.
    //

    auto  Se = blas::join_row< value_t >( { S, T } );
    auto  Q  = blas::copy( Se );
    auto  R  = blas::matrix< value_t >();
    
    blas::qr( Q, R, false );

    // io::matlab::write( Q, "Q" );
    // io::matlab::write( R, "R" );
    
    auto  Ve = blas::join_row< value_t >( { V, X } );
    auto  Vs = blas::prod( Ve, blas::adjoint( R ) );
    auto  Vn = approx.column_basis( Vs, acc );
    
    //
    // new coupling matrix is ⎛S⎞·(V X)'·Vn
    //                        ⎝T⎠
                
    auto  TV = blas::prod( blas::adjoint( Ve ), Vn );
    auto  Sn = blas::prod( Se, TV );
    
    // { // DEBUG {
    //     auto  US1 = blas::prod( U, Se );
    //     auto  M1  = blas::prod( US1, blas::adjoint( Ve ) );

    //     {
    //         auto  MV  = blas::prod( M1, Vn );
    //         auto  MVV = blas::prod( MV, blas::adjoint( Vn ) );
            
    //         blas::add( value_t(-1), M1, MVV );
    //         std::cout << "          add_col Vn   : " << boost::format( "%.4e" ) % ( blas::norm_F( MVV ) / blas::norm_F( M1 ) ) << std::endl;
    //     }

    //     {
    //         auto  US2 = blas::prod( U, Sn );
    //         auto  M2  = blas::prod( US2, blas::adjoint( Vn ) );

    //         blas::add( value_t(-1), M1, M2 );
    //         std::cout << "          add_col     : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    //     }
    // } // DEBUG }

    return { std::move( Sn ), std::move( Vn ) };
}

//
// compute M = U·S·V' + W·T·X' = (U W)·⎛S  ⎞·(V X)'
//                                     ⎝  T⎠
//
// - return orthogonal row/column basis and new coefficients
// - no approximation
// - W does not need to be orthogonal
//
template < typename value_t >
std::tuple< blas::matrix< value_t >,   // new row basis
            blas::matrix< value_t >,   // new coupling
            blas::matrix< value_t > >  // new column basis
add_row_col ( const uniform_lrmatrix< value_t > &  M,
              const blas::matrix< value_t > &      W,
              const blas::matrix< value_t > &      T,
              const blas::matrix< value_t > &      X )
{
    auto  U = M.row_cb().basis();
    auto  S = M.coeff();
    auto  V = M.col_cb().basis();
    
    auto  Ue = blas::join_row< value_t >( { U, W } );
    auto  RU = blas::matrix< value_t >();

    blas::qr( Ue, RU ); // Ue is orthogonal afterwards
                
    auto  Ve = blas::join_row< value_t >( { V, X } );
    auto  RV = blas::matrix< value_t >();

    blas::qr( Ve, RV ); // Ve is orthogonal afterwards
    
    auto  Se = blas::diag< value_t >( { S, T } );
    auto  S1 = blas::prod( RU, Se );
    auto  Sn = blas::prod( S1, blas::adjoint( RV ) );

    return { std::move( Ue ), std::move( Sn ), std::move( Ve ) };
}
    
//
// add U·S·V' to M
// - ASSUMPTION: U and V are orthogonal
//
template < typename value_t,
           typename approx_t >
void
add ( Hpro::TMatrix< value_t > &          M,
      const blas::matrix< value_t > &     U,
      const blas::matrix< value_t > &     S,
      const blas::matrix< value_t > &     V,
      const accuracy &                    acc,
      const approx_t &                    approx,
      const is_matrix_map_t< value_t > &  rowmap,
      const is_matrix_map_t< value_t > &  colmap )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  U_i = blas::matrix< value_t >( U, B_ij->row_is() - B->row_ofs(), blas::range::all );
                    auto  V_j = blas::matrix< value_t >( V, B_ij->col_is() - B->col_ofs(), blas::range::all );
                    
                    add( *B_ij, U_i, S, V_j, acc, approx, rowmap, colmap );
                }// if
            }// for
        }// for
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R  = ptrcast( &M, uniform_lrmatrix< value_t > );
        auto  Uc = blas::copy( U );
        auto  Vc = blas::copy( V );
        auto  RU = blas::matrix< value_t >();
        auto  RV = blas::matrix< value_t >();

        blas::qr( Uc, RU );
        blas::qr( Vc, RV );

        auto  RUS            = blas::prod( RU, S );
        auto  T              = blas::prod( RUS, blas::adjoint(RV) );
        auto  [ Un, Sn, Vn ] = add( *R, Uc, T, Vc, acc, approx );
        
        update_row_col_basis( *R, Un, Sn, Vn, acc, approx, rowmap, colmap );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D  = ptrcast( &M, matrix::dense_matrix< value_t > );
        auto  Dm = D->mat();
        auto  US = blas::prod( U, S );

        blas::prod( value_t(1), US, blas::adjoint( V ), value_t(1), Dm );

        if ( D->is_compressed() )
            D->compress( acc );
    }// if
}

//
// add D to M
//
template < typename value_t,
           typename approx_t >
void
add ( Hpro::TMatrix< value_t > &          M,
      const blas::matrix< value_t > &     D,
      const accuracy &                    acc,
      const approx_t &                    approx,
      const is_matrix_map_t< value_t > &  rowmap,
      const is_matrix_map_t< value_t > &  colmap )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  D_ij = blas::matrix< value_t >( D,
                                                          B_ij->row_is() - B->row_ofs(),
                                                          B_ij->col_is() - B->col_ofs() );
                    
                    add( *B_ij, D_ij, acc, approx, rowmap, colmap );
                }// if
            }// for
        }// for
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R        = ptrcast( &M, uniform_lrmatrix< value_t > );
        auto  C        = blas::copy( D ); // need to copy because modified during SVD
        auto  [ U, V ] = approx( C, acc );
        auto  RU       = blas::matrix< value_t >();
        auto  RV       = blas::matrix< value_t >();

        blas::qr( U, RU );
        blas::qr( V, RV );

        auto  S = blas::prod( RU, blas::adjoint( RV ) );

        auto [ Un, Sn, Vn ] = add( *R, U, S, V, acc, approx );
        
        update_row_col_basis( *R, Un, Sn, Vn, acc, approx, rowmap, colmap );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  DM = ptrcast( &M, matrix::dense_matrix< value_t > );
        auto  Dm = DM->mat();

        blas::add( value_t(1), D, Dm );

        if ( DM->is_compressed() )
            DM->compress( acc );
    }// if
}

//
// matrix multiplication C := α·A·B + C
// (forward decl.)
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const Hpro::matop_t                 op_A,
           const Hpro::TMatrix< value_t > &    A,
           const Hpro::matop_t                 op_B,
           const Hpro::TMatrix< value_t > &    B,
           Hpro::TMatrix< value_t > &          C,
           const accuracy &                    acc,
           const approx_t &                    approx,
           const is_matrix_map_t< value_t > &  rowmap,
           const is_matrix_map_t< value_t > &  colmap );

//
// blocked x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const Hpro::TBlockMatrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const Hpro::TBlockMatrix< value_t > & B,
           Hpro::TBlockMatrix< value_t > &       C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    for ( uint  i = 0; i < C.nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C.nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C.block( i, j ) ) );
                
            for ( uint  l = 0; l < A.nblock_cols( op_A ); ++l )
            {
                if ( ! is_null_any( A.block( i, l, op_A ), B.block( l, j, op_B ) ) )
                    multiply( alpha,
                                         op_A, *A.block( i, l, op_A ),
                                         op_B, *B.block( l, j, op_B ),
                                         *C.block( i, j ), acc, approx,
                                         rowmap, colmap );
            }// if       
        }// for
    }// for
}

//
// blocked x blocked = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const Hpro::TBlockMatrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const Hpro::TBlockMatrix< value_t > & B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    //
    // compute temporary standard low-rank block matrix BC
    // and sub blocks BC_ij for each  i,j ∈ nblocks(A) × ncols(B)
    // and combine all for update of C
    //

    auto  BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( C.row_is(), C.col_is() );

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
                        BC->set_block( i, j, new Hpro::TRkMatrix< value_t >( A_il->row_is( op_A ), B_lj->col_is( op_B ) ) );
                    
                    hlr::multiply( alpha, op_A, *A_il, op_B, *B_lj, *BC->block( i, j ), acc, approx );
                }// if
            }// if       
        }// for
    }// for

    // ensure correct value type of BC
    BC->adjust_value_type();

    //
    // convert to lowrank format and construct U·S·V' via QR
    //
    
    auto  R  = hlr::matrix::convert_to_lowrank( *BC, acc, approx );
    auto  U  = blas::mat_U( R );
    auto  V  = blas::mat_V( R );
    auto  RU = blas::matrix< value_t >();
    auto  RV = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    auto [ Un, Sn, Vn ] = add( C, U, S, V, acc, approx );
    
    update_row_col_basis( C, Un, Sn, Vn, acc, approx, rowmap, colmap );
}

//
// blocked x dense = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const Hpro::TBlockMatrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const matrix::dense_matrix< value_t > & B,
           matrix::uniform_lrmatrix< value_t > & C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    auto  D = matrix::convert_to_dense( C );

    hlr::multiply( alpha, op_A, A, op_B, B, *D );
    
    auto  [ U, V ] = approx( blas::mat( *D ), acc );
    auto  RU       = blas::matrix< value_t >();
    auto  RV       = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    detail::update_row_col_basis( C, U, S, V, acc, approx, rowmap, colmap );
}

//
// blocked x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const Hpro::TBlockMatrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const uniform_lrmatrix< value_t > &   B,
           Hpro::TBlockMatrix< value_t > &       C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    // (A·U)·S·V' + C
    auto  U  = B.row_cb( op_B ).basis();
    auto  AU = blas::matrix< value_t >( C.nrows(), U.ncols() );

    hlr::multiply( alpha, op_A, A, U, AU );

    add( C, AU, B.coeff(), B.col_cb().basis(), acc, approx, rowmap, colmap );
}

//
// blocked x uniform = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const Hpro::TBlockMatrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const uniform_lrmatrix< value_t > &   B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap )
{
    // A·B + C = (A·U)·S·V' + W·T·V'
    auto  U  = B.row_cb( op_B ).basis();
    auto  AU = blas::matrix< value_t >( C.nrows(), U.ncols() );

    hlr::multiply( alpha, op_A, A, U, AU );

    auto  R = blas::matrix< value_t >();

    blas::qr( AU, R );

    auto  T          = blas::prod( R, B.coeff() );
    auto  [ Un, Sn ] = add_row( C, AU, T, acc, approx );

    detail::update_row_basis( C, Un, Sn, acc, approx, rowmap );
}

//
// dense x blocked = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const matrix::dense_matrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const Hpro::TBlockMatrix< value_t > & B,
           matrix::uniform_lrmatrix< value_t > & C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    auto  D = matrix::convert_to_dense( C );

    hlr::multiply( alpha, op_A, A, op_B, B, *D );
    
    auto  [ U, V ] = approx( blas::mat( *D ), acc );
    auto  RU       = blas::matrix< value_t >();
    auto  RV       = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    detail::update_row_col_basis( C, U, S, V, acc, approx, rowmap, colmap );
}

//
// uniform x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const uniform_lrmatrix< value_t > &   A,
           const Hpro::matop_t                   op_B,
           const Hpro::TBlockMatrix< value_t > & B,
           Hpro::TBlockMatrix< value_t > &       C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    // U·S·(V'·B) + C with V'·B computed as B'·V
    auto  V  = A.col_cb( op_A ).basis();
    auto  BV = blas::matrix< value_t >( C.ncols(), V.ncols() );

    hlr::multiply( alpha, blas::adjoint( op_B ), B, V, BV );

    add( C, A.row_cb().basis(), A.coeff(), BV, acc, approx, rowmap, colmap );
}

//
// uniform x blocked = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const uniform_lrmatrix< value_t > &   A,
           const Hpro::matop_t                   op_B,
           const Hpro::TBlockMatrix< value_t > & B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    colmap )
{
    // U·S·(V'·B) + U·T·X' with V'·B computed as B'·V
    auto  V  = A.col_cb( op_A ).basis();
    auto  BV = blas::matrix< value_t >( C.ncols(), V.ncols() );

    hlr::multiply( alpha, blas::adjoint( op_B ), B, V, BV );

    auto  R = blas::matrix< value_t >();

    blas::qr( BV, R );

    auto  T          = blas::prod( A.coeff(), blas::adjoint( R ) );
    auto  [ Sn, Vn ] = add_col( C, T, BV, acc, approx );
    
    detail::update_col_basis( C, Sn, Vn, acc, approx, colmap );
}

//
// dense x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::uniform_lrmatrix< value_t > &     B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const accuracy &                                acc,
           const approx_t &                                approx,
           const is_matrix_map_t< value_t > &              rowmap,
           const is_matrix_map_t< value_t > &              colmap )
{
    auto  U = B.row_basis( op_B );
    auto  S = B.coeff();
    auto  V = B.col_basis( op_B );

    auto  AU = blas::prod( alpha, blas::mat_view( op_A, A->mat() ), U );
    auto  R  = blas::matrix< value_t >();

    blas::qr( AU, R );

    auto  SR = blas::prod( R, blas::mat_view( op_B, S ) );
    
    add( C, AU, SR, V, acc, approx, rowmap, colmap );
}

//
// uniform x uniform = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           Hpro::TBlockMatrix< value_t > &      C,
           const accuracy &                     acc,
           const approx_t &                     approx,
           const is_matrix_map_t< value_t > &   rowmap,
           const is_matrix_map_t< value_t > &   colmap )
{
    // U·(S·V' × W·T)·X' + C = U·R·X' + C
    auto  R = blas::matrix< value_t >();

    {
        auto  VW   = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
        auto  SVW  = blas::prod( alpha, blas::mat_view( op_A, A.coeff() ), VW );

        R = std::move( blas::prod( SVW, blas::mat_view( op_B, B.coeff() ) ) );
    }

    add( C, A.row_cb().basis(), R, B.col_cb().basis(), acc, approx, rowmap, colmap );
}

//
// uniform x uniform = uniform
//
template < typename value_t >
void
multiply ( const value_t                        alpha,
           const Hpro::matop_t                  op_A,
           const uniform_lrmatrix< value_t > &  A,
           const Hpro::matop_t                  op_B,
           const uniform_lrmatrix< value_t > &  B,
           uniform_lrmatrix< value_t > &        C )
{
    // A·B + C = U·(S·V' × W·T)·X' + U·R·X'
    auto  VW   = blas::prod( blas::adjoint( A.col_cb( op_A ).basis() ), B.row_cb( op_B ).basis() );
    auto  SVW  = blas::prod( blas::mat_view( op_A, A.coeff() ), VW );

    blas::prod( alpha, SVW, blas::mat_view( op_B, B.coeff() ), value_t(1), C.coeff() );
}

//
// uniform x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                               alpha,
           const Hpro::matop_t                         op_A,
           const matrix::uniform_lrmatrix< value_t > & A,
           const Hpro::matop_t                         op_B,
           const matrix::dense_matrix< value_t > &       B,
           Hpro::TBlockMatrix< value_t > &             C,
           const accuracy &                            acc,
           const approx_t &                            approx,
           const is_matrix_map_t< value_t > &          rowmap,
           const is_matrix_map_t< value_t > &          colmap )
{
    auto  U = A.row_basis( op_A );
    auto  S = A.coeff();
    auto  V = A.col_basis( op_A );

    auto  VB = blas::prod( alpha, blas::mat_view( blas::adjoint( op_B ), B->mat() ), V );
    auto  R  = blas::matrix< value_t >();

    blas::qr( VB, R );

    auto  SR = blas::prod( blas::mat_view( op_A, S ), blas::adjoint( R ) );
    
    add( C, U, SR, VB, acc, approx, rowmap, colmap );
}

//
// uniform x dense = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const uniform_lrmatrix< value_t > &   A,
           const Hpro::matop_t                   op_B,
           const matrix::dense_matrix< value_t > & B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    colmap )
{
    // A×B + C = U·S·(V' × B) + U·T·X' = U·S·(B' × V)' + U·T·X'
    auto  BV = blas::prod( alpha,
                           blas::mat_view( blas::adjoint( op_B ), B->mat() ),
                           A.col_cb( op_A ).basis() );

    auto  R = blas::matrix< value_t >();

    blas::qr( BV, R );

    auto  T          = blas::prod( A.coeff(), blas::adjoint( R ) );
    auto  [ Sn, Vn ] = add_col( C, T, BV, acc, approx );
    
    detail::update_col_basis( C, Sn, Vn, acc, approx, colmap );
}

//
// dense x uniform = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const matrix::dense_matrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const uniform_lrmatrix< value_t > &   B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap )
{
    // A×B + C = (A × U)·T·V' + W·S·V'
    auto  AU = blas::prod( alpha,
                           blas::mat_view( op_A, A->mat() ),
                           B.row_cb( op_B ).basis() );

    auto  R = blas::matrix< value_t >();

    blas::qr( AU, R );

    auto  T          = blas::prod( R, blas::mat_view( op_B, B.coeff() ) );
    auto  [ Un, Sn ] = add_row( C, AU, T, acc, approx );
    
    detail::update_row_basis( C, Un, Sn, acc, approx, rowmap );
}

//
// dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const matrix::dense_matrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const matrix::dense_matrix< value_t > & B,
           Hpro::TBlockMatrix< value_t > &       C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    auto  AB = blas::prod( alpha,
                           blas::mat_view( op_A, A->mat() ),
                           blas::mat_view( op_B, B->mat() ) );

    add( C, AB, acc, approx, rowmap, colmap );
}

//
// dense x dense = uniform
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                         alpha,
           const Hpro::matop_t                   op_A,
           const matrix::dense_matrix< value_t > & A,
           const Hpro::matop_t                   op_B,
           const matrix::dense_matrix< value_t > & B,
           uniform_lrmatrix< value_t > &         C,
           const accuracy &                      acc,
           const approx_t &                      approx,
           const is_matrix_map_t< value_t > &    rowmap,
           const is_matrix_map_t< value_t > &    colmap )
{
    auto  AB       = blas::prod( alpha,
                                 blas::mat_view( op_A, A->mat() ),
                                 blas::mat_view( op_B, B->mat() ) );
    auto  [ U, V ] = approx( AB, acc );
    auto  RU       = blas::matrix< value_t >();
    auto  RV       = blas::matrix< value_t >();

    blas::qr( U, RU );
    blas::qr( V, RV );

    auto  S = blas::prod( RU, blas::adjoint( RV ) );

    auto [ Un, Sn, Vn ] = add( C, U, S, V, acc, approx );
    
    detail::update_row_col_basis( C, Un, Sn, Vn, acc, approx, rowmap, colmap );
}

//
// general function
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const Hpro::matop_t                 op_A,
           const Hpro::TMatrix< value_t > &    A,
           const Hpro::matop_t                 op_B,
           const Hpro::TMatrix< value_t > &    B,
           Hpro::TMatrix< value_t > &          C,
           const accuracy &                    acc,
           const approx_t &                    approx,
           const is_matrix_map_t< value_t > &  rowmap,
           const is_matrix_map_t< value_t > &  colmap )
{
    // // DEBUG {
    // auto  DA = hlr::seq::matrix::copy_nonuniform( A );
    // auto  DB = hlr::seq::matrix::copy_nonuniform( B );
    // auto  DC = hlr::seq::matrix::copy_nonuniform( C );

    // hlr::multiply( alpha, op_A, *DA, op_B, *DB, *DC, acc, approx::SVD< value_t >() );
    // // DEBUG }
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * BA, op_B, * BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * BA, op_B, * BB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( matrix::is_dense( C ) )   hlr::multiply( alpha, op_A, * BA, op_B, * BB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            auto  UB = cptrcast( &B, uniform_lrmatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * BA, op_B, * UB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * BA, op_B, * UB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap );
            else if ( matrix::is_dense( C ) )   hlr::multiply( alpha, op_A, * BA, op_B, * UB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, matrix::dense_matrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * BA, op_B, * DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * BA, op_B, * DB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( matrix::is_dense( C ) )   hlr::multiply( alpha, op_A, * BA, op_B, * DB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( is_uniform_lowrank( A ) )
    {
        auto  UA = cptrcast( &A, uniform_lrmatrix< value_t > );
        
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * UA, op_B, * BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * UA, op_B, * BB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, colmap );
            else if ( matrix::is_dense(   C ) ) hlr::multiply( alpha, op_A, * UA, op_B, * BB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            auto  UB = cptrcast( &B, uniform_lrmatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * UA, op_B, * UB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * UA, op_B, * UB, * ptrcast( &C, uniform_lrmatrix< value_t > ) ); 
            else if ( matrix::is_dense( C ) )   hlr::multiply( alpha, op_A, * UA, op_B, * UB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, matrix::dense_matrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * UA, op_B, * DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * UA, op_B, * DB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, colmap );
            else if ( matrix::is_dense( C ) )   hlr::multiply( alpha, op_A, * UA, op_B, * DB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  DA = cptrcast( &A, matrix::dense_matrix< value_t > );
        
        if ( is_blocked( B ) )
        {
            auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * DA, op_B, * BB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * DA, op_B, * BB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( matrix::is_dense(   C ) ) hlr::multiply( alpha, op_A, * DA, op_B, * BB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_uniform_lowrank( B ) )
        {
            auto  UB = cptrcast( &B, uniform_lrmatrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * DA, op_B, * UB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * DA, op_B, * UB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap );
            else if ( matrix::is_dense(   C ) ) hlr::multiply( alpha, op_A, * DA, op_B, * UB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( matrix::is_dense( B ) )
        {
            auto  DB = cptrcast( &B, matrix::dense_matrix< value_t > );
            
            if      ( is_blocked( C ) )         multiply( alpha, op_A, * DA, op_B, * DB, * ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( is_uniform_lowrank( C ) ) multiply( alpha, op_A, * DA, op_B, * DB, * ptrcast( &C, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
            else if ( matrix::is_dense(   C ) ) hlr::multiply( alpha, op_A, * DA, op_B, * DB, * ptrcast( &C, matrix::dense_matrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + B.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    // // DEBUG {
    // auto  DD = hlr::seq::matrix::copy_nonuniform( C );
    // auto  TC = matrix::convert_to_dense( *DC );
    // auto  TD = matrix::convert_to_dense( *DD );

    // hlr::add( value_t(-1), *TC, *TD );

    // std::cout << "multiply : " << A.id() << " × " << B.id() << " = " << C.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *TD ) / norm::frobenius( *TC ) ) << std::endl;
    // // DEBUG }
}

//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    L,
                  Hpro::TMatrix< value_t > &          M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  rowmap,
                  const is_matrix_map_t< value_t > &  colmap );

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                     side,
                  const diag_type_t                     diag,
                  const Hpro::TBlockMatrix< value_t > & L,
                  Hpro::TBlockMatrix< value_t > &       M,
                  const accuracy &                      acc,
                  const approx_t &                      approx,
                  const is_matrix_map_t< value_t > &    rowmap,
                  const is_matrix_map_t< value_t > &    colmap )
{
    HLR_LOG( 4, Hpro::to_string( "svltr( B %d, B %d )", L.id(), M.id() ) );

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
                    solve_lower_tri( side, diag, *L_ii, *M_ij, acc, approx, rowmap, colmap );
            }// for

            for ( uint  k = i+1; k < M.nblock_rows(); ++k )
            {
                for ( uint  j = 0; j < M.nblock_cols(); ++j )
                {
                    if ( ! is_null_any( L.block(k,i), M.block(i,j) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(k,j) ) );
                        
                        multiply( value_t(-1),
                                             apply_normal, *L.block(k,i),
                                             apply_normal, *M.block(i,j),
                                             *M.block(k,j),
                                             acc, approx, rowmap, colmap );
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
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    L,
                  uniform_lrmatrix< value_t > &       M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  rowmap,
                  const is_matrix_map_t< value_t > &  /* colmap */ )
{
    if ( matrix::is_dense( L ) && ( diag == unit_diag ))
        return;
    
    if ( side == from_left )
    {
        //
        // solve L×M = L×W·T·X' = U·S·V' as L×W = U
        //

        auto  W = blas::copy( M.row_cb().basis() );
        auto  D = matrix::dense_matrix< value_t >( M.row_is(), Hpro::is( 0, W.ncols()-1 ), W );

        hlr::solve_lower_tri( side, diag, L, D );

        // orthogonalise W, compute T and update row basis
        auto  R = blas::matrix< value_t >();

        blas::qr( W, R );

        auto  T = blas::prod( R, M.coeff() );

        update_row_basis( M, W, T, acc, approx, rowmap );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    L,
                  Hpro::TMatrix< value_t > &          M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  rowmap,
                  const is_matrix_map_t< value_t > &  colmap )
{
    // // DEBUG {
    // auto  DL = hlr::seq::matrix::copy_nonuniform( L );
    // auto  DM = hlr::seq::matrix::copy_nonuniform( M );

    // hlr::solve_lower_tri( side, diag, *DL, *DM, acc, approx::SVD< value_t >() );
    // // DEBUG }
    
    if ( is_blocked( L ) )
    {
        auto  BL = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( M ) )         solve_lower_tri( side, diag, * BL, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( is_uniform_lowrank( M ) ) solve_lower_tri( side, diag, L, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( matrix::is_dense( M ) )   hlr::solve_lower_tri( side, diag, * BL, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else if ( matrix::is_dense( L ) )
    {
        auto  DL = cptrcast( & L, matrix::dense_matrix< value_t > );
        
        if      ( is_uniform_lowrank( M ) ) solve_lower_tri( side, diag, L, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( matrix::is_dense( M ) )   hlr::solve_lower_tri( side, diag, * DL, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + L.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );

    // // DEBUG {
    // auto  DX = hlr::seq::matrix::copy_nonuniform( M );
    // auto  TM = matrix::convert_to_dense( *DM );
    // auto  TX = matrix::convert_to_dense( *DX );
    
    // hlr::add( value_t(-1), *TM, *TX );
    // std::cout << "solve_lower: " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *TX ) / norm::frobenius( *TM ) ) << std::endl;
    // // DEBUG }
}

//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  Hpro::TMatrix< value_t > &          M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  rowmap,
                  const is_matrix_map_t< value_t > &  colmap );

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                     side,
                  const diag_type_t                     diag,
                  const Hpro::TBlockMatrix< value_t > & U,
                  Hpro::TBlockMatrix< value_t > &       M,
                  const accuracy &                      acc,
                  const approx_t &                      approx,
                  const is_matrix_map_t< value_t > &    rowmap,
                  const is_matrix_map_t< value_t > &    colmap )
{
    HLR_LOG( 4, Hpro::to_string( "svutr( B %d, B %d )", U.id(), M.id() ) );
    
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
                    solve_upper_tri( side, diag, *U_jj, *M_ij, acc, approx, rowmap, colmap );
            }// for
            
            for ( uint  k = j+1; k < M.nblock_cols(); ++k )
            {
                for ( uint  i = 0; i < M.nblock_rows(); ++i )
                {
                    if ( ! is_null_any( M.block(i,j), U.block(j,k) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(i,k) ) );
                        
                        multiply( value_t(-1),
                                             apply_normal, *M.block(i,j),
                                             apply_normal, *U.block(j,k),
                                             *M.block(i,k),
                                             acc, approx, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  uniform_lrmatrix< value_t > &       M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  /* rowmap */,
                  const is_matrix_map_t< value_t > &  colmap )
{
    if ( matrix::is_dense( U ) && ( diag == unit_diag ))
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
        auto  D = matrix::dense_matrix< value_t >( M.col_is(), Hpro::is( 0, X.ncols()-1 ), X );

        hlr::solve_upper_tri( from_left, diag, U, D );

        // orthogonalise X, compute T and update column basis
        auto  R = blas::matrix< value_t >();
        
        blas::qr( X, R );

        auto  T = blas::prod( M.coeff(), blas::adjoint( R ) );
        
        update_col_basis( M, T, X, acc, approx, colmap );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  Hpro::TMatrix< value_t > &          M,
                  const accuracy &                    acc,
                  const approx_t &                    approx,
                  const is_matrix_map_t< value_t > &  rowmap,
                  const is_matrix_map_t< value_t > &  colmap )
{
    // // DEBUG {
    // auto  DU = hlr::seq::matrix::copy_nonuniform( U );
    // auto  DM = hlr::seq::matrix::copy_nonuniform( M );

    // hlr::solve_upper_tri( side, diag, *DU, *DM, acc, approx::SVD< value_t >() );
    // // DEBUG }
    
    if ( is_blocked( U ) )
    {
        auto  BU = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( M ) )         solve_upper_tri( side, diag, * BU, * ptrcast( & M, Hpro::TBlockMatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( is_uniform_lowrank( M ) ) solve_upper_tri( side, diag, U, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( matrix::is_dense( M ) )   hlr::solve_upper_tri( side, diag, * BU, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }//if
    else if ( matrix::is_dense( U ) )
    {
        auto  DU = cptrcast( & U, matrix::dense_matrix< value_t > );
        
        if      ( is_uniform_lowrank( M ) ) solve_upper_tri( side, diag, U, * ptrcast( & M, uniform_lrmatrix< value_t > ), acc, approx, rowmap, colmap );
        else if ( matrix::is_dense( M ) )   hlr::solve_upper_tri( side, diag, * DU, * ptrcast( & M, matrix::dense_matrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type for M : " + M.typestr() );
    }//if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );

    // // DEBUG {
    // auto  DX = hlr::seq::matrix::copy_nonuniform( M );
    // auto  TM = matrix::convert_to_dense( *DM );
    // auto  TX = matrix::convert_to_dense( *DX );
    
    // hlr::add( value_t(-1), *TM, *TX );
    // std::cout << "solve_upper: " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *TX ) / norm::frobenius( *TM ) ) << std::endl;
    // // DEBUG }
}

//
// recursive LU factorization
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &          A,
     const accuracy &                    acc,
     const approx_t &                    approx,
     const is_matrix_map_t< value_t > &  rowmap,
     const is_matrix_map_t< value_t > &  colmap )
// Hpro::TMatrix< value_t > &              REF )
{
    if ( is_blocked( A ) )
    {
        auto  BA   = ptrcast( &A,   Hpro::TBlockMatrix< value_t > );
        // auto  BREF = ptrcast( &REF, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            // lu( * BA->block( i, i ), acc, rowmap, colmap, *BREF->block( i, i ) );
            lu( * BA->block( i, i ), acc, approx, rowmap, colmap );

            // // DEBUG {
            // {
            //     auto  D1 = matrix::convert_to_dense( *BA->block(i,i) );
            //     auto  D2 = matrix::convert_to_dense( *BREF->block(i,i) );

            //     hlr::add( value_t(-1), *D2, *D1 );
            //     std::cout << "ref error " << BA->block(i,i)->id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
            // }
            // // DEBUG }

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri( from_right, general_diag,
                                                *BA->block( i, i ), *BA->block( j, i ),
                                                acc, approx, rowmap, colmap );

                // // DEBUG {
                // {
                //     auto  D1 = matrix::convert_to_dense( *BA->block(j,i) );
                //     auto  D2 = matrix::convert_to_dense( *BREF->block(j,i) );
                    
                //     hlr::add( value_t(-1), *D2, *D1 );
                //     std::cout << "ref error " << BA->block(j,i)->id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
                // }
                // // DEBUG }
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri( from_left, unit_diag,
                                                *BA->block( i, i ), *BA->block( i, j ),
                                                acc, approx, rowmap, colmap );

                // DEBUG {
                // {
                //     auto  D1 = matrix::convert_to_dense( *BA->block(i,j) );
                //     auto  D2 = matrix::convert_to_dense( *BREF->block(i,j) );
                    
                //     hlr::add( value_t(-1), *D2, *D1 );
                //     std::cout << "ref error " << BA->block(i,j)->id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
                // }
                // DEBUG }
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
                                  acc, approx, rowmap, colmap );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  D  = ptrcast( &A, matrix::dense_matrix< value_t > );
        auto  Dm = D->mat();

        blas::invert( Dm );

        if ( D->is_compressed() )
            D->set_matrix( Dm, acc );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}}}// namespace hlr::uniform::detail

#endif // __HLR_ARITH_DETAIL_UNIFORM_HH
