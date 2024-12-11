#ifndef __HLR_TBB_DETAIL_ARITH_H2_HH
#define __HLR_TBB_DETAIL_ARITH_H2_HH
//
// Project     : HLR
// Module      : tbb/detail/h2_mvm.hh
// Description : matrix-vector multiplication for H² matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_for.h>

#include <hlr/arith/h2.hh>

namespace hlr { namespace tbb { namespace h2 { namespace detail {

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

using  mutex_map_t = std::vector< std::mutex >;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_mtx ( const value_t                              alpha,
              const Hpro::matop_t                        op_M,
              const Hpro::TMatrix< value_t > &           M,
              const uniform_vector< cluster_basis_t > &  x,
              uniform_vector< cluster_basis_t > &        y,
              const scalar_vector< value_t > &           sx,
              scalar_vector< value_t > &                 sy,
              mutex_map_t &                              mtx_map )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, B->nblock_rows( op_M ),
                                              0, B->nblock_cols( op_M ) ),
            [&,alpha,op_M,B] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j, op_M );
            
                        if ( ! is_null( B_ij ) )
                        {
                            auto  x_j = x.block( j );
                            auto  y_i = y.block( i );
                            
                            mul_vec_mtx( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy, mtx_map );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i  = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  lock = std::scoped_lock( mtx_map[ y.basis().id() ] );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R    = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        auto  lock = std::scoped_lock( y.mutex() );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else if ( matrix::is_h2_lowrank2( M ) )
    {
        auto  R    = cptrcast( &M, matrix::h2_lr2matrix< value_t > );
        auto  lock = std::scoped_lock( y.mutex() );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t,
           typename cluster_basis_t >
void
mul_vec_row ( const value_t                              alpha,
              const Hpro::matop_t                        op_M,
              const Hpro::TMatrix< value_t > &           M,
              const uniform_vector< cluster_basis_t > &  x,
              uniform_vector< cluster_basis_t > &        y,
              const scalar_vector< value_t > &           sx,
              scalar_vector< value_t > &                 sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT(( B->nblock_rows( op_M ) == y.nblocks() ) &&
                   ( B->nblock_cols( op_M ) == x.nblocks() ));

        //
        // parallelise only block rows, then only one task will access y
        //
        
        ::tbb::parallel_for< uint >(
            0, B->nblock_rows( op_M ),
            [&,alpha,op_M,B] ( const auto  i )
            {
                auto  y_i = y.block( i );
                
                for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
                {
                    auto  B_ij = B->block( i, j, op_M );
                    
                    if ( ! is_null( B_ij ) )
                    {
                        auto  x_j = x.block( j );
                        
                        mul_vec_row( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                    }// if
                }// for
            } );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else if ( hlr::matrix::is_h2_lowrank2( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lr2matrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t,
           typename cluster_basis_t >
std::unique_ptr< uniform_vector< cluster_basis_t > >
scalar_to_uniform ( const cluster_basis_t &           cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis_t > >( cb );

    if ( cb.nsons() == 0 )
    {
        auto  V = cb.basis();
        
        if ( V.nrows() == 0 )
            return u;
        
        //
        // s ≔ V'·v
        //
        
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
        auto  s    = blas::mulvec( blas::adjoint( V ), v_cb );

        u->set_coeffs( std::move( s ) );
    }// if
    else
    {
        //
        // s ≔ V'·v = (∑_i V_i E_i)' v = ∑_i E_i' V_i' v
        // - first recurse, then sum up
        // - lock free with sequential summation of coefficients
        //

        auto  Si = std::vector< blas::vector< value_t > >( cb.nsons() );
        
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                auto  u_i = scalar_to_uniform( *cb.son(i), v );
                
                if ( cb.rank() > 0 )
                    Si[i] = std::move( blas::mulvec( blas::adjoint( cb.transfer_mat(i) ), u_i->coeffs() ) );
                
                u->set_block( i, u_i.release() );
            } );

        if ( cb.rank() > 0 )
        {
            auto  s = blas::vector< value_t >( cb.rank() );
        
            for ( auto  si : Si )
                blas::add( 1, si, s );
        
            u->set_coeffs( std::move( s ) );
        }// if
    }// else

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t,
           typename cluster_basis_t >
std::unique_ptr< uniform_vector< cluster_basis_t > >
make_uniform ( const cluster_basis_t &  cb )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis_t > >( cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for< uint >(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                u->set_block( i, make_uniform< value_t, cluster_basis_t >( *cb.son(i) ).release() );
            } );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t,
           typename cluster_basis_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis_t > &  u,
                        scalar_vector< value_t > &                 v,
                        blas::vector< value_t > &                  s )
{
    if ( s.length() > 0 )
        blas::add( value_t(1), u.coeffs(), s );
    else
        s = blas::copy( u.coeffs() );

    auto &  cb = u.basis();

    if ( cb.nsons() == 0 )
    {
        auto  v_loc = blas::vector< value_t >( blas::vec< value_t >( v ), cb.is() - v.ofs() );

        blas::mulvec( value_t(1), cb.basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        ::tbb::parallel_for< uint >(
            uint(0), u.nblocks(),
            [&] ( const uint  i )
            {
                auto  u_i = u.block( i );
                auto  s_i = cb.transfer_to_son( i, s );
                
                add_uniform_to_scalar( *u_i, v, s_i );
            } );
    }// else
}

//
// construct mapping id -> clusterbasis
//
template < typename value_t >
void
build_id2cb ( matrix::nested_cluster_basis< value_t > &                   cb,
              std::vector< matrix::nested_cluster_basis< value_t > * > &  idmap )
{
    HLR_ASSERT( cb.id() != -1 );
    HLR_ASSERT( cb.id() < idmap.size() );

    idmap[ cb.id() ] = & cb;

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for< uint >(
            0, cb.nsons(),
            [&] ( const auto  i )
            {
                if ( ! is_null( cb.son( i ) ) )
                    build_id2cb( * cb.son( i ), idmap );
            } );
    }// if
}

//
// construct mapping cluster basis id (cluster) -> list of matrix blocks in M
//
template < typename value_t >
void
build_id2blocks ( const matrix::nested_cluster_basis< value_t > &                 cb,
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

//
// no construction of explicit, hierarchical representation of uniform vector
// but just store coefficients in std::vector indexed by cluster basis id
//
template < typename value_t >
void
scalar_to_uniform ( const matrix::nested_cluster_basis< value_t > &   cb,
                    const vector::scalar_vector< value_t > &          v,
                    std::vector< blas::vector< value_t > > &          vcoeff )
{
    if ( cb.nsons() == 0 )
    {
        if ( cb.rank() > 0 )
        {
            auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );

            HLR_ASSERT( cb.id() < vcoeff.size() );
            
            vcoeff[ cb.id() ] = std::move( cb.transform_forward( v_cb ) );
        }// if
    }// if
    else
    {
        if ( cb.rank() > 0 )
        {
            //
            // recurse and transfer coefficients from sons
            //
            
            auto  Si = std::vector< blas::vector< value_t > >( cb.nsons() );
        
            ::tbb::parallel_for< uint >(
                0, cb.nsons(),
                [&] ( const uint  i )
                {
                    auto  cb_i = cb.son(i);
                    
                    scalar_to_uniform( *cb.son(i), v, vcoeff );

                    Si[i] = std::move( cb.transfer_from_son( i, vcoeff[ cb_i->id() ] ) );
                } );

            auto  s = blas::vector< value_t >( cb.rank() );
        
            for ( auto  s_i : Si )
                blas::add( value_t(1), s_i, s );
        
            vcoeff[ cb.id() ] = std::move( s );
        }// if
        else
        {
            //
            // just recurse (no transfer matrices)
            //
            
            ::tbb::parallel_for< uint >(
                0, cb.nsons(),
                [&] ( const uint  i )
                {
                    scalar_to_uniform( *cb.son(i), v, vcoeff );
                } );
        }// else
    }// else
}

template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const Hpro::matop_t                                                   op_M,
              const matrix::nested_cluster_basis< value_t > &                       rowcb,
              const std::vector< matrix::nested_cluster_basis< value_t > * > &      colcb,
              const std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap,
              const std::vector< blas::vector< value_t > > &                        xcoeff,
              const blas::vector< value_t > &                                       y_parent,
              const vector::scalar_vector< value_t > &                              sx,
              vector::scalar_vector< value_t > &                                    sy )
{
    //
    // start with coefficients from parent
    //

    auto  y_loc = blas::copy( y_parent );
    auto  y_son = std::vector< blas::vector< value_t > >( rowcb.nsons() );

    if (( y_loc.length() == 0 ) && ( rowcb.rank() != 0 ))
        y_loc = std::move( blas::vector< value_t >( rowcb.rank() ) );

    HLR_ASSERT( y_loc.length() == rowcb.rank() );
    
    //
    // perform matvec with all blocks in block row
    //

    auto &  blockrow = blockmap[ rowcb.id() ];

    if ( blockrow.size() > 0 )
    {
        auto  rowis  = rowcb.is();
        auto  y_j    = blas::vector< value_t >( blas::vec( sy ), rowis - sy.ofs() );
        auto  t_j    = blas::vector< value_t >( y_j.length() );
        bool  update = false;
            
        for ( auto  M : blockrow )
        {
            if ( matrix::is_h2_lowrank( M ) )
            {
                auto    R       = cptrcast( M, matrix::h2_lrmatrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, y_loc, op_M );
            }// if
            else if ( matrix::is_h2_lowrank2( M ) )
            {
                auto    R       = cptrcast( M, matrix::h2_lr2matrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, y_loc, op_M );
            }// if
            else if ( matrix::is_dense( M ) )
            {
                auto  x_i = blas::vector< value_t >( blas::vec( sx ), M->col_is( op_M ) - sx.ofs() );
                        
                M->apply_add( alpha, x_i, t_j, op_M );
                update = true;
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// for

        //
        // handle local uniform part
        //
        
        if ( y_loc.length() > 0 )
        {
            if ( rowcb.nsons() == 0 )
            {
                // apply to local update
                rowcb.transform_backward( y_loc, t_j );
                update = true;
            }// if
            else
            {
                // transfer local part to sons
                for ( uint  i = 0; i < rowcb.nsons(); ++i )
                    y_son[i] = std::move( rowcb.transfer_to_son( i, y_loc ) );
            }// else
        }// if

        //
        // finally add local update to destination
        //

        if ( update )
            blas::add( value_t(1), t_j, y_j );
    }// if

    //
    // recurse
    //

    if ( rowcb.nsons() > 0 )
    {
        ::tbb::parallel_for< uint >(
            0, rowcb.nsons(),
            [&] ( const auto  i )
            {
                if ( ! is_null( rowcb.son( i ) ) )
                    mul_vec_row( alpha, op_M, * rowcb.son( i ), colcb, blockmap, xcoeff, y_son[i], sx, sy );
            } );
    }// if
}

}}}} // namespace hlr::tbb::h2::detail

#endif // __HLR_ARITH_H2_HH
