#ifndef __HLR_TBB_ARITH_UNIFORM_MVM_HH
#define __HLR_TBB_ARITH_UNIFORM_MVM_HH
//
// Project     : HLR
// Module      : tbb/detail/uniform_mvm.hh
// Description : matrix-vector product for uniform matrices with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/uniform.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/utils/hash.hh>

namespace hlr { namespace tbb { namespace uniform { namespace detail {

using hlr::matrix::shared_cluster_basis;
using hlr::matrix::shared_cluster_basis_hierarchy;
using hlr::matrix::uniform_lrmatrix;
using hlr::matrix::uniform_lr2matrix;
using hlr::matrix::level_hierarchy;
using hlr::vector::scalar_vector;
using hlr::vector::uniform_vector;
using hlr::vector::uniform_vector_hierarchy;

using  mutex_map_t = std::unordered_map< indexset, std::unique_ptr< std::mutex >, indexset_hash >;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec_mtx ( const value_t                                              alpha,
              const Hpro::matop_t                                        op_M,
              const Hpro::TMatrix< value_t > &                           M,
              const uniform_vector< shared_cluster_basis< value_t > > &  x,
              uniform_vector< shared_cluster_basis< value_t > > &        y,
              const scalar_vector< value_t > &                           sx,
              scalar_vector< value_t > &                                 sy,
              mutex_map_t &                                              mtx_map )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT(( B->nblock_rows( op_M ) == y.nblocks() ) &&
                   ( B->nblock_cols( op_M ) == x.nblocks() ));
            
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
    else if ( hlr::matrix::is_dense( M ) )
    {
        auto  x_i  = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j  = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  mtx  = mtx_map.at( M.row_is( op_M ) ).get();
        auto  lock = std::scoped_lock( *mtx );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R    = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  lock = std::scoped_lock( y.mutex() );
        
        #if defined(HLR_HAS_ZBLAS_DIRECT)
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                default               : HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     : blas::mulvec( alpha, R->coeff(), x.coeffs(), value_t(1), y.coeffs() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : blas::mulvec( alpha, blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() ); break;
                default               :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
    }// if
    else if ( hlr::matrix::is_uniform_lowrank2( M ) )
    {
        auto  R    = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  k    = R->rank();
        auto  t    = blas::vector< value_t >( k );
        auto  lock = std::scoped_lock( y.mutex() );

        #if defined(HLR_HAS_ZBLAS_DIRECT)
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     :
                    compress::zblas::mulvec( R->col_rank(), k, apply_adjoint, value_t(1), R->zcol_coupling(), x.coeffs().data(), t.data() );
                    compress::zblas::mulvec( R->row_rank(), k, apply_normal,       alpha, R->zrow_coupling(), t.data(), y.coeffs().data() );
                    break;
                    
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                    
                case apply_adjoint    :
                    compress::zblas::mulvec( R->row_rank(), k, apply_adjoint, value_t(1), R->zrow_coupling(), x.coeffs().data(), t.data() );
                    compress::zblas::mulvec( R->col_rank(), k, apply_normal,       alpha, R->zcol_coupling(), t.data(), y.coeffs().data() );
                    break;
                    
                default :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     :
                    blas::mulvec( value_t(1), blas::adjoint( R->col_coupling() ), x.coeffs(), value_t(1), t );
                    blas::mulvec( alpha, R->row_coupling(), t, value_t(1), y.coeffs() );
                    break;
                    
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                    
                case apply_adjoint    :
                    blas::mulvec( value_t(1), blas::adjoint( R->row_coupling() ), x.coeffs(), value_t(1), t );
                    blas::mulvec( alpha, R->col_coupling(), t, value_t(1), y.coeffs() );
                    break;
                    
                default               :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t >
void
mul_vec_row ( const value_t                                              alpha,
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
    else if ( hlr::matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );

        #if defined(HLR_HAS_ZBLAS_DIRECT)
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), x.coeffs().data(), y.coeffs().data() ); break;
                default               : HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     : blas::mulvec( alpha, R->coeff(), x.coeffs(), value_t(1), y.coeffs() ); break;
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                case apply_adjoint    : blas::mulvec( alpha, blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() ); break;
                default               :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
    }// if
    else if ( hlr::matrix::is_uniform_lowrank2( M ) )
    {
        auto  R = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  k = R->rank();
        auto  t = blas::vector< value_t >( k );

        #if defined(HLR_HAS_ZBLAS_DIRECT)
        if ( R->is_compressed() )
        {
            switch ( op_M )
            {
                case apply_normal     :
                    compress::zblas::mulvec( R->col_rank(), k, apply_adjoint, value_t(1), R->zcol_coupling(), x.coeffs().data(), t.data() );
                    compress::zblas::mulvec( R->row_rank(), k, apply_normal,       alpha, R->zrow_coupling(), t.data(), y.coeffs().data() );
                    break;
                    
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                    
                case apply_adjoint    :
                    compress::zblas::mulvec( R->row_rank(), k, apply_adjoint, value_t(1), R->zrow_coupling(), x.coeffs().data(), t.data() );
                    compress::zblas::mulvec( R->col_rank(), k, apply_normal,       alpha, R->zcol_coupling(), t.data(), y.coeffs().data() );
                    break;
                    
                default :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// if
        else
        #endif
        {
            switch ( op_M )
            {
                case apply_normal     :
                    blas::mulvec( value_t(1), blas::adjoint( R->col_coupling() ), x.coeffs(), value_t(1), t );
                    blas::mulvec( alpha, R->row_coupling(), t, value_t(1), y.coeffs() );
                    break;
                    
                case apply_conjugate  : { HLR_ASSERT( false ); }
                case apply_transposed : { HLR_ASSERT( false ); }
                    
                case apply_adjoint    :
                    blas::mulvec( value_t(1), blas::adjoint( R->row_coupling() ), x.coeffs(), value_t(1), t );
                    blas::mulvec( alpha, R->col_coupling(), t, value_t(1), y.coeffs() );
                    break;
                    
                default               :
                    HLR_ERROR( "unsupported matrix operator" );
            }// switch
        }// else
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

    ::tbb::parallel_invoke(
        [&] ()
        {                    
            if ( cb.rank() > 0 )
            {
                auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
                auto  s    = cb.transform_forward( v_cb );
                
                u->set_coeffs( std::move( s ) );
            }// if
        },

        [&] ()
        {
            if ( cb.nsons() > 0 )
            {
                ::tbb::parallel_for(
                    uint(0), cb.nsons(),
                    [&] ( const uint  i )
                    {
                        u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
                    }
                );
            }// if
        } );

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< shared_cluster_basis< value_t > > >
make_uniform ( const shared_cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< shared_cluster_basis< value_t > > >( cb );

    if ( cb.nsons() > 0 )
    {
        ::tbb::parallel_for(
            uint(0), cb.nsons(),
            [&] ( const uint  i )
            {
                u->set_block( i, make_uniform( *cb.son(i) ).release() );
            }
        );
    }// if

    return u;
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
        // auto  v_u = blas::vector< value_t >( blas::vec( v ), u.is() - v.ofs() );

        // u.basis().transform_backward( u.coeffs(), v_u );

        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( blas::vec( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        ::tbb::parallel_for(
            uint(0), u.nblocks(),
            [&] ( const uint  i )
            {
                add_uniform_to_scalar( *u.block(i), v );
            }
        );
    }// if
}

//
// generate mapping of index set to mutices for leaf clusters
//
template < typename value_t >
void
build_mutex_map ( const shared_cluster_basis< value_t > &  cb,
                  mutex_map_t &                            mtx_map )
{
    mtx_map[ cb.is() ] = std::make_unique< std::mutex >();

    for ( uint  i = 0; i < cb.nsons(); ++i )
        build_mutex_map( *cb.son(i), mtx_map );
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

    ::tbb::parallel_for< uint >(
        0, nlvl,
        [&] ( const uint  lvl )
        // for ( uint  lvl = 0; lvl < nlvl; ++lvl )
        {
            const auto  ncb = cb.hierarchy()[lvl].size();
            
            hier->hierarchy()[lvl].resize( ncb );
            
            ::tbb::parallel_for< uint >(
                0, ncb,
                [&,lvl] ( const uint  i )
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
                } );
        } ); // }// for
    
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
    
    for ( uint  lvl = 0; lvl < M.nlevel(); ++lvl )
    {
        ::tbb::parallel_for< uint >(
            0, M.row_ptr[lvl].size()-1,
            [&,alpha,op_M,lvl] ( const uint  row )
            {
                const auto  lb = M.row_ptr[lvl][row];
                const auto  ub = M.row_ptr[lvl][row+1];

                if ( lb == ub )
                    return;
            
                cb_t *      ycb    = nullptr;
                auto        s      = blas::vector< value_t >();
                const auto  row_is = M.row_mat[lvl][lb]->row_is( op_M );
                auto        y_j    = blas::vector< value_t >( blas::vec( sy ), row_is - sy.ofs() );
                auto        t_j    = blas::vector< value_t >( y_j.length() );
                
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
                        
                        #if defined(HLR_HAS_ZBLAS_DIRECT)
                        if ( R->is_compressed() )
                        {
                            switch ( op_M )
                            {
                                case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), ux->coeffs().data(), s.data() ); break;
                                case apply_conjugate  : { HLR_ASSERT( false ); }
                                case apply_transposed : { HLR_ASSERT( false ); }
                                case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), ux->coeffs().data(), s.data() ); break;
                                default               : HLR_ERROR( "unsupported matrix operator" );
                            }// switch
                        }// if
                        else
                        #endif
                        {
                            switch ( op_M )
                            {
                                case apply_normal     : blas::mulvec( alpha, R->coupling(), ux->coeffs(), value_t(1), s ); break;
                                case apply_conjugate  : HLR_ASSERT( false );
                                case apply_transposed : HLR_ASSERT( false );
                                case apply_adjoint    : blas::mulvec( alpha, blas::adjoint( R->coupling() ), ux->coeffs(), value_t(1), s ); break;
                                default               : HLR_ERROR( "unsupported matrix operator" );
                            }// switch
                        }// else
                    }// if
                    else if ( matrix::is_uniform_lowrank2( mat ) )
                    {
                        auto  R  = cptrcast( mat, uniform_lr2matrix< value_t > );
                        auto  k  = R->rank();
                        auto  t  = blas::vector< value_t >( k );
                        auto  ux = x.hierarchy()[lvl][col_idx];
                        
                        if ( is_null( ycb ) )
                        {
                            ycb = rowcb.hierarchy()[lvl][row];
                            s   = blas::vector< value_t >( ycb->rank() );
                        }// if
                        
                        #if defined(HLR_HAS_ZBLAS_DIRECT)
                        if ( R->is_compressed() )
                        {
                            switch ( op_M )
                            {
                                case apply_normal     :
                                    compress::zblas::mulvec( R->col_rank(), k, apply_adjoint, value_t(1), R->zcol_coupling(), ux->coeffs().data(), t.data() );
                                    compress::zblas::mulvec( R->row_rank(), k, apply_normal,       alpha, R->zrow_coupling(), t.data(), s.data() );
                                    break;
                    
                                case apply_conjugate  : { HLR_ASSERT( false ); }
                                case apply_transposed : { HLR_ASSERT( false ); }
                    
                                case apply_adjoint    :
                                    compress::zblas::mulvec( R->row_rank(), k, apply_adjoint, value_t(1), R->zrow_coupling(), ux->coeffs().data(), t.data() );
                                    compress::zblas::mulvec( R->col_rank(), k, apply_normal,       alpha, R->zcol_coupling(), t.data(), s.data() );
                                    break;
                    
                                default :
                                    HLR_ERROR( "unsupported matrix operator" );
                            }// switch
                        }// if
                        else
                        #endif
                        {
                            switch ( op_M )
                            {
                                case apply_normal     :
                                    blas::mulvec( value_t(1), blas::adjoint( R->col_coupling() ), ux->coeffs(), value_t(1), t );
                                    blas::mulvec( alpha, R->row_coupling(), t, value_t(1), s );
                                    break;
                    
                                case apply_conjugate  : { HLR_ASSERT( false ); }
                                case apply_transposed : { HLR_ASSERT( false ); }
                    
                                case apply_adjoint    :
                                    blas::mulvec( value_t(1), blas::adjoint( R->row_coupling() ), ux->coeffs(), value_t(1), t );
                                    blas::mulvec( alpha, R->col_coupling(), t, value_t(1), s );
                                    break;
                    
                                default               :
                                    HLR_ERROR( "unsupported matrix operator" );
                            }// switch
                        }// else
                    }// if
                    else if ( matrix::is_dense( mat ) )
                    {
                        auto  x_i = blas::vector< value_t >( blas::vec( sx ), mat->col_is( op_M ) - sx.ofs() );
                        
                        mat->apply_add( alpha, x_i, t_j, op_M );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + mat->typestr() );
                }// for

                //
                // add uniform part to y
                //

                if ( ! is_null( ycb ) )
                    ycb->transform_backward( s, t_j );

                blas::add( 1, t_j, y_j );
            } );
    }// for
}

}}}}// namespace hlr::tbb::uniform::detail

namespace hlr { namespace tbb { namespace uniform { namespace tlr { namespace detail {

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                                    alpha,
          const Hpro::matop_t                              op_M,
          const Hpro::TMatrix< value_t > &                 M,
          const vector::scalar_vector< value_t > &         x,
          vector::scalar_vector< value_t > &               y,
          const matrix::shared_cluster_basis< value_t > &  rowcb,
          const matrix::shared_cluster_basis< value_t > &  colcb )
{
    HLR_ASSERT( is_blocked( M ) );

    auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
    
    //
    // construct uniform vector for x
    //

    const auto  nrowblocks = B->nblock_rows();
    const auto  ncolblocks = B->nblock_cols();
    auto        ux         = std::vector< blas::vector< value_t > >( ncolblocks );

    ::tbb::parallel_for< size_t >(
        0, ncolblocks,
        [&,alpha,op_M] ( const auto  j )
        {
            auto  colcb_j = colcb.son( j );
            
            if ( colcb_j->rank() > 0 )
            {
                const auto  x_j = blas::vector< value_t >( blas::vec( x ), colcb_j->is() - x.ofs() );
                
                ux[j] = std::move( colcb_j->transform_forward( x_j ) );
            }// if
        } );
        

    //
    // multiply while going over block rows
    //   - collect updates in row (separate for uniform and dense)
    //

    ::tbb::parallel_for< size_t >(
        0, nrowblocks,
        [&,alpha,op_M] ( const auto  i )
        {
            auto  rowcb_i = rowcb.son( i );
            auto  y_i     = blas::vector< value_t >( blas::vec( y ), rowcb_i->is() - y.ofs() );
            auto  t_i     = blas::vector< value_t >( y_i.length() );
            auto  s_i     = blas::vector< value_t >( rowcb_i->rank() );

            for ( size_t  j = 0; j < ncolblocks; ++j )
            {
                auto  B_ij = B->block( i, j );

                if ( hlr::matrix::is_uniform_lowrank( B_ij ) )
                {
                    auto  R = cptrcast( B_ij, hlr::matrix::uniform_lrmatrix< value_t > );

                    #if defined(HLR_HAS_ZBLAS_DIRECT)
                    if ( R->is_compressed() )
                    {
                        switch ( op_M )
                        {
                            case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), ux[j].data(), s_i.data() ); break;
                            case apply_conjugate  : { HLR_ASSERT( false ); }
                            case apply_transposed : { HLR_ASSERT( false ); }
                            case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoeff(), ux[j].data(), s_i.data() ); break;
                            default               : HLR_ERROR( "unsupported matrix operator" );
                        }// switch
                    }// if
                    else
                        #endif
                    {
                        switch ( op_M )
                        {
                            case apply_normal     : blas::mulvec( alpha, R->coupling(), ux[j], value_t(1), s_i ); break;
                            case apply_conjugate  : HLR_ASSERT( false );
                            case apply_transposed : HLR_ASSERT( false );
                            case apply_adjoint    : blas::mulvec( alpha, blas::adjoint( R->coupling() ), ux[j], value_t(1), s_i ); break;
                            default               : HLR_ERROR( "unsupported matrix operator" );
                        }// switch
                    }// else
                }// if
                else if ( hlr::matrix::is_dense( B_ij ) )
                {
                    auto  x_j = blas::vector< value_t >( blas::vec( x ), B_ij->col_is( op_M ) - x.ofs() );
        
                    B_ij->apply_add( alpha, x_j, t_i, op_M );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
            }// for

            //
            // add uniform contribution to local result
            //

            if ( s_i.length() > 0 )
            {
                rowcb_i->transform_backward( s_i, t_i );
            }// if

            //
            // add update to y
            //

            blas::add( value_t(1), t_i, y_i );
        } );
        
}

}}}}}// hlr::tbb::uniform::tlr::detail

#endif // __HLR_TBB_ARITH_UNIFORM_MVM_HH
