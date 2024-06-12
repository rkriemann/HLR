#ifndef __HLR_TBB_DETAIL_UNIFORM_MATRIX_HH
#define __HLR_TBB_DETAIL_UNIFORM_MATRIX_HH
//
// Project     : HLR
// Module      : uniform/matrix.hh
// Description : uniform matrix construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

#include <hlr/tbb/detail/uniform_basis.hh>

namespace hlr { namespace tbb { namespace matrix { namespace detail {

using namespace hlr::matrix;

using hlr::uniform::is_matrix_map_t;
using hlr::tbb::uniform::detail::compute_extended_basis;
using hlr::tbb::uniform::detail::update_coupling;

template < typename value_t >
void
init_cluster_bases ( const Hpro::TBlockCluster *        bct,
                     shared_cluster_basis< value_t > &  rowcb,
                     shared_cluster_basis< value_t > &  colcb )
{
    //
    // decide upon cluster type, how to construct matrix
    //

    if ( ! bct->is_leaf() )
    {
        //
        // build cluster bases for next level
        //
        
        {
            for ( uint  i = 0; i < bct->nrows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
            
                for ( uint  j = 0; j < bct->ncols(); ++j )
                {
                    auto  bc_ij = bct->son( i, j );
                    
                    if ( ! is_null( bc_ij ) )
                    {
                        if ( is_null( rowcb_i ) )
                        {
                            rowcb_i = new shared_cluster_basis< value_t >( bc_ij->is().row_is() );
                            rowcb.set_son( i, rowcb_i );

                            rowcb_i->set_nsons( bc_ij->rowcl()->nsons() );
                        }// if
                    }// if
                }// for
            }// for
        }

        {
            for ( uint  j = 0; j < bct->ncols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                for ( uint  i = 0; i < bct->nrows(); ++i )
                {
                    auto  bc_ij = bct->son( i, j );
                    
                    if ( ! is_null( bc_ij ) )
                    {
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new shared_cluster_basis< value_t >( bc_ij->is().col_is() );
                            colcb.set_son( j, colcb_j );

                            colcb_j->set_nsons( bc_ij->colcl()->nsons() );
                        }// if
                    }// if
                }// for
            }// for
        }

        //
        // recurse
        //
        
        for ( uint  i = 0; i < bct->nrows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < bct->ncols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( bct->son( i, j ) ) )
                    init_cluster_bases( bct->son( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
}

template < typename value_t >
void
init_cluster_bases ( const Hpro::TMatrix< value_t > &   M,
                     shared_cluster_basis< value_t > &  rowcb,
                     shared_cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        {
            // auto  lock = std::scoped_lock( rowcb.mutex() );
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
            
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  M_ij    = B->block( i, j );
                
                    if ( ! is_null( M_ij ) )
                    {
                        if ( is_null( rowcb_i ) )
                        {
                            rowcb_i = new shared_cluster_basis< value_t >( M_ij->row_is() );
                            rowcb.set_son( i, rowcb_i );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
                            rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );

                        continue;
                    }// if
                }// for
            }// for
        }

        {
            // auto  lock = std::scoped_lock( colcb.mutex() );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
            
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    auto  M_ij = B->block( i, j );
                
                    if ( ! is_null( M_ij ) )
                    {
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new shared_cluster_basis< value_t >( M_ij->col_is() );
                            colcb.set_son( j, colcb_j );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                            colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );

                        continue;
                    }// if
                }// for
            }// for
        }

        //
        // recurse
        //
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( B->block( i, j ) ) )
                    init_cluster_bases( *B->block( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< hlr::matrix::lrmatrix< value_t > * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< Hpro::idx_t, Hpro::TBlockMatrix< value_t > * >;

    //
    // go BFS-style through block cluster tree and construct leaves per level
    // then convert lowrank to uniform lowrank while constructing bases
    //

    // TODO: handle case of global lowrank matrix
    HLR_ASSERT( ! bct->is_adm() );
    
    auto  rowcb_root = std::unique_ptr< cluster_basis >();
    auto  colcb_root = std::unique_ptr< cluster_basis >();

    auto  rowcb_map  = basis_map_t();
    auto  colcb_map  = basis_map_t();

    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >();

    auto  nodes      = std::deque< const Hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    auto  bmtx       = std::mutex(); // for bmat_map
    auto  cmtx       = std::mutex(); // for children list
    auto  lmtx       = std::mutex(); // for row/col map lists
    auto  cbmtx      = std::mutex(); // for rowcb/colcb map lists

    //
    // local function to set up hierarchy (parent <-> M)
    //
    auto  insert_hier = [&] ( const Hpro::TBlockCluster *                    node,
                              std::unique_ptr< Hpro::TMatrix< value_t > > &  M )
    {
        if ( is_null( node->parent() ) )
        {
            M_root = std::move( M );
        }// if
        else
        {
            auto  parent   = node->parent();
            auto  M_parent = typename bmat_map_t::mapped_type( nullptr );

            {
                auto  lock = std::scoped_lock( bmtx );
                        
                M_parent = bmat_map.at( parent->id() );
            }

            for ( uint  i = 0; i < parent->nrows(); ++i ) 
            {
                for ( uint  j = 0; j < parent->ncols(); ++j )
                {
                    if ( parent->son( i, j ) == node )
                    {
                        M_parent->set_block( i, j, M.release() );
                        return;
                    }// if
                }// for
            }// for
        }// if
    };

    //
    // local function to create cluster basis objects (with hierarchy)
    //
    auto  create_cb = [&] ( const Hpro::TBlockCluster *  node )
    {
        //
        // build row/column cluster basis objects and set up
        // cluster bases hierarchy
        //

        auto             rowcl = node->rowcl();
        auto             colcl = node->colcl();
        cluster_basis *  rowcb = nullptr;
        cluster_basis *  colcb = nullptr;
        auto             lock  = std::scoped_lock( cbmtx );
                    
        if ( rowcb_map.find( *rowcl ) == rowcb_map.end() )
        {
            rowcb = new cluster_basis( *rowcl );
            rowcb->set_nsons( rowcl->nsons() );

            rowcb_map.emplace( *rowcl, rowcb );
        }// if
        else
            rowcb = rowcb_map.at( *rowcl );
                    
        if ( colcb_map.find( *colcl ) == colcb_map.end() )
        {
            colcb = new cluster_basis( *colcl );
            colcb->set_nsons( colcl->nsons() );
            colcb_map.emplace( *colcl, colcb );
        }// if
        else
            colcb = colcb_map.at( *colcl );

        if ( is_null( node->parent() ) )
        {
            rowcb_root.reset( rowcb_map[ *rowcl ] );
            colcb_root.reset( colcb_map[ *colcl ] );
        }// if
        else
        {
            auto  parent     = node->parent();
            auto  row_parent = parent->rowcl();
            auto  col_parent = parent->colcl();

            for ( uint  i = 0; i < row_parent->nsons(); ++i )
            {
                if ( row_parent->son( i ) == rowcl )
                {
                    rowcb_map.at( *row_parent )->set_son( i, rowcb );
                    break;
                }// if
            }// for

            for ( uint  i = 0; i < col_parent->nsons(); ++i )
            {
                if ( col_parent->son( i ) == colcl )
                {
                    colcb_map.at( *col_parent )->set_son( i, colcb );
                    break;
                }// if
            }// for
        }// else
    };

    //
    // level-wise iteration for matrix construction
    //
    
    while ( ! nodes.empty() )
    {
        auto  children = decltype( nodes )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::deque< Hpro::TMatrix< value_t > * >();
        
        ::tbb::parallel_for_each(
            nodes,
            [&] ( auto  node )
            {
                auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

                if ( node->is_leaf() )
                {
                    // handled above
                    if ( node->is_adm() )
                    {
                        M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( node, acc ) );

                        {
                            auto  lock = std::scoped_lock( lmtx );

                            if ( hlr::matrix::is_lowrank( *M ) )
                            {
                                auto  R = ptrcast( M.get(), hlr::matrix::lrmatrix< value_t > );
                                    
                                rowmap[ M->row_is() ].push_back( R );
                                colmap[ M->col_is() ].push_back( R );
                            }// if
                            else
                                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
                                
                            // store always to maintain affinity
                            lrmat.push_back( M.get() );
                        }
                            
                        M->set_id( node->id() );
                        M->set_procs( node->procs() );

                        // insert_hier( node, M );
                        // create_cb( node );
                    }// if
                    else
                    {
                        M = coeff.build( node->is().row_is(), node->is().col_is() );
                        
                        if ( hlr::matrix::is_dense( *M ) )
                        {
                            // all is good
                        }// if
                        else if ( Hpro::is_dense( *M ) )
                        {
                            auto  D = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );

                            M = std::move( std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + M->typestr() );
                    }// else
                }// if
                else
                {
                    // collect children
                    {
                        auto  lock = std::scoped_lock( cmtx );
                            
                        for ( uint  i = 0; i < node->nrows(); ++i )
                            for ( uint  j = 0; j < node->ncols(); ++j )
                                if ( node->son( i, j ) != nullptr )
                                    children.push_back( node->son( i, j ) );
                    }

                    M = std::make_unique< Hpro::TBlockMatrix< value_t > >( node );
        
                    auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

                    // make sure, block structure is correct
                    if (( B->nblock_rows() != node->nrows() ) ||
                        ( B->nblock_cols() != node->ncols() ))
                        B->set_block_struct( node->nrows(), node->ncols() );

                    // remember all block matrices for setting up hierarchy
                    {
                        auto  lock = std::scoped_lock( bmtx );
                        
                        bmat_map[ node->id() ] = B;
                    }
                }// else

                M->set_id( node->id() );
                M->set_procs( node->procs() );

                insert_hier( node, M );
                create_cb( node );
            } );

        nodes = std::move( children );
        
        ::tbb::parallel_invoke(

            [&] ()
            {
                //
                // construct row bases for all block rows constructed on this level
                //

                auto  rowiss = std::deque< indexset >();

                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                ::tbb::parallel_for_each(
                    rowiss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = rowmap.at( is );
                    
                        if ( matrices.size() == 0 )
                            return;

                        //
                        // compute column basis for
                        //
                        //   ( U₀·V₀'  U₁·V₁'  U₂·V₂'  … ) =
                        //
                        //                  ⎛ V₀'        ⎞
                        //   ( U₀ U₁ U₂ … ) ⎜    V₁'     ⎟ =
                        //                  ⎜       V₂'  ⎟
                        //                  ⎝          … ⎠
                        //
                        //                  ⎛ Q₀·R₀             ⎞'
                        //   ( U₀ U₁ U₂ … ) ⎜      Q₁·R₁        ⎟ =
                        //                  ⎜           Q₂·R₂   ⎟
                        //                  ⎝                 … ⎠
                        //
                        //                  ⎛⎛Q₀     ⎞ ⎛R₀     ⎞⎞'
                        //   ( U₀ U₁ U₂ … ) ⎜⎜  Q₁   ⎟·⎜  R₁   ⎟⎟ =
                        //                  ⎜⎜    Q₂ ⎟ ⎜    R₂ ⎟⎟
                        //                  ⎝⎝      …⎠ ⎝      …⎠⎠
                        //
                        // Since diag(Q_i) is orthogonal, it can be omitted for row bases
                        // computation, leaving
                        //
                        //                  ⎛R₀     ⎞'                 
                        //   ( U₀ U₁ U₂ … ) ⎜  R₁   ⎟ = ( U₀·R₀' U₁·R₁' U₂·R₂' … )
                        //                  ⎜    R₂ ⎟                  
                        //                  ⎝      …⎠                  
                        //
                        // of which a column basis is computed.
                        //

                        //
                        // form U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
                        //
            
                        size_t  nrows_U = is.size();
                        size_t  ncols_U = 0;

                        for ( auto &  R : matrices )
                            ncols_U += R->rank();

                        auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
                        size_t  pos = 0;

                        for ( auto &  R : matrices )
                        {
                            // R = U·V' = W·T·X'
                            auto  U_i = R->U_direct();
                            auto  V_i = blas::copy( R->V_direct() );
                            auto  R_i = blas::matrix< value_t >();
                            auto  k   = R->rank();
                
                            blas::qr( V_i, R_i );

                            auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                            auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                            blas::copy( UR_i, U_sub );
                
                            pos += k;
                        }// for

                        //
                        // QR of S and computation of row basis
                        //

                        auto  Un = basisapx.column_basis( U, acc );
            
                        // finally assign to cluster basis object
                        // (no change to "rowcb_map", therefore no lock)
                        rowcb_map.at( is )->set_basis( std::move( Un ) );
                    } );
            },

            [&] ()
            {
                //
                // construct column bases for all block columns constructed on this level
                //

                auto  coliss = std::deque< indexset >();
            
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                ::tbb::parallel_for_each(
                    coliss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = colmap.at( is );

                        if ( matrices.size() == 0 )
                            return;

                        //
                        // compute column basis for
                        //
                        //   ⎛U₀·V₀'⎞ 
                        //   ⎜U₁·V₁'⎟
                        //   ⎜U₂·V₂'⎟
                        //   ⎝  …   ⎠
                        //
                        // or row basis of
                        //
                        //   ⎛U₀·V₀'⎞' 
                        //   ⎜U₁·V₁'⎟ = ( V₀·U₀'  V₁·U₁'  V₂·U₂'  … ) =
                        //   ⎜U₂·V₂'⎟
                        //   ⎝  …   ⎠
                        //
                        //                  ⎛ U₀      ⎞'
                        //   ( V₀ V₁ V₂ … ) ⎜   U₁    ⎟ =
                        //                  ⎜     U₂  ⎟
                        //                  ⎝       … ⎠
                        //
                        //                  ⎛ Q₀·R₀               ⎞'
                        //   ( V₀ V₁ V₂ … ) ⎜       Q₁·R₁         ⎟ =
                        //                  ⎜             Q₂·R₂   ⎟
                        //                  ⎝                   … ⎠
                        //
                        //                  ⎛⎛Q₀     ⎞ ⎛R₀     ⎞⎞'
                        //   ( V₀ V₁ V₂ … ) ⎜⎜  Q₁   ⎟·⎜  R₁   ⎟⎟ =
                        //                  ⎜⎜    Q₂ ⎟ ⎜    R₂ ⎟⎟
                        //                  ⎝⎝      …⎠ ⎝      …⎠⎠
                        //
                        // Since diag(Q_i) is orthogonal, it can be omitted for column bases
                        // computation, leaving
                        //
                        //                  ⎛R₀     ⎞'                
                        //   ( V₀ V₁ V₂ … ) ⎜  R₁   ⎟ = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
                        //                  ⎜    R₂ ⎟                
                        //                  ⎝      …⎠
                        //
                        // of which a column basis is computed.
                        //

                        //
                        // form matrix V = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
                        //

                        size_t  nrows_V = is.size();
                        size_t  ncols_V = 0;

                        for ( auto &  R : matrices )
                            ncols_V += R->rank();

                        auto    V   = blas::matrix< value_t >( nrows_V, ncols_V );
                        size_t  pos = 0;

                        for ( auto &  R : matrices )
                        {
                            // R' = (U·V')' = V·U' = X·T'·W'
                            auto  V_i = blas::copy( R->V_direct() );
                            auto  U_i = blas::copy( R->U_direct() );
                            auto  R_i = blas::matrix< value_t >();
                            auto  k   = R->rank();
                
                            blas::qr( U_i, R_i );

                            auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                            auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                            blas::copy( VR_i, V_sub );
                
                            pos += k;
                        }// for

                        auto  Vn = basisapx.column_basis( V, acc );

                        // finally assign to cluster basis object
                        // (no change to "colcb_map", therefore no lock)
                        colcb_map.at( is )->set_basis( std::move( Vn ) );
                    } );
            }
        );

        //
        // now convert all blocks on this level
        //

        ::tbb::parallel_for_each(
            lrmat,
            [&] ( auto  M )                           
            {
                auto  R     = ptrcast( M, hlr::matrix::lrmatrix< value_t > );
                auto  rowcb = rowcb_map.at( R->row_is() );
                auto  colcb = colcb_map.at( R->col_is() );
                auto  Un    = rowcb->basis();
                auto  Vn    = colcb->basis();

                //
                // R = U·V' ≈ Un (Un' U V' Vn) Vn'
                //          = Un S Vn'  with  S = Un' U V' Vn
                //

                auto  UnU = blas::prod( blas::adjoint( Un ), R->U_direct() );
                auto  VnV = blas::prod( blas::adjoint( Vn ), R->V_direct() );
                auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

                auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                          R->col_is(),
                                                                                          *rowcb,
                                                                                          *colcb,
                                                                                          std::move( S ) );
            
                // replace standard lowrank block by uniform lowrank block
                R->parent()->replace_block( R, RU.release() );
                delete R;
            } );
    }// while
    
    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// level-wise construction of uniform-H matrix from given H-matrix
//
template < typename basisapx_t >
std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > >
build_uniform_lvl ( const Hpro::TMatrix< typename basisapx_t::value_t > &   A,
                    const basisapx_t &                                      basisapx,
                    const accuracy &                                        acc,
                    shared_cluster_basis< typename basisapx_t::value_t > &  rowcb_root,
                    shared_cluster_basis< typename basisapx_t::value_t > &  colcb_root )
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< const hlr::matrix::lrmatrix< value_t > * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< Hpro::idx_t, Hpro::TBlockMatrix< value_t > * >;

    //
    // go BFS-style through matrix and construct leaves per level
    // then convert lowrank to uniform lowrank while constructing bases
    //

    // TODO: handle case of global lowrank matrix
    HLR_ASSERT( ! hlr::matrix::is_lowrank( A ) );
    
    auto  rowcb_map = basis_map_t();
    auto  colcb_map = basis_map_t();

    auto  M_root    = std::unique_ptr< Hpro::TMatrix< value_t > >();

    auto  matrices  = std::list< const Hpro::TMatrix< value_t > * >{ &A };
    auto  bmat_map  = bmat_map_t();

    auto  bmtx      = std::mutex(); // for bmat_map
    auto  cmtx      = std::mutex(); // for children list
    auto  lmtx      = std::mutex(); // for row/col map lists
    auto  cbmtx     = std::mutex(); // for rowcb/colcb map lists

    //
    // level-wise iteration for matrix construction
    //
    
    rowcb_map[ A.row_is() ] = & rowcb_root;
    colcb_map[ A.col_is() ] = & colcb_root;
    
    while ( ! matrices.empty() )
    {
        auto  children = decltype( matrices )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::deque< const hlr::matrix::lrmatrix< value_t > * >();
        
        ::tbb::parallel_for_each(
            matrices,
            [&] ( auto  mat )
            {
                auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

                if ( hlr::matrix::is_lowrank( mat ) )
                {
                    auto  R    = cptrcast( mat, hlr::matrix::lrmatrix< value_t > );
                    auto  lock = std::scoped_lock( lmtx );
                        
                    rowmap[ R->row_is() ].push_back( R );
                    colmap[ R->col_is() ].push_back( R );
                    lrmat.push_back( R );
                }// if
                else if ( hlr::matrix::is_dense( mat ) )
                {
                    M = mat->copy();
                }// if
                else if ( is_blocked( mat ) )
                {
                    auto  B = cptrcast( mat, Hpro::TBlockMatrix< value_t > );
                
                    // collect sub-blocks
                    {
                        auto  lock = std::scoped_lock( cmtx );

                        for ( uint  i = 0; i < B->nblock_rows(); ++i )
                            for ( uint  j = 0; j < B->nblock_cols(); ++j )
                                if ( ! is_null( B->block( i, j ) ) )
                                    children.push_back( B->block( i, j ) );
                    }

                    M = B->copy_struct();

                    // remember all block matrices for setting up hierarchy
                    {
                        auto  lock = std::scoped_lock( bmtx );
                        
                        bmat_map[ mat->id() ] = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );
                    }
                }// else

                //
                // set up hierarchy (parent <-> M)
                //

                if ( ! is_null( M ) )
                {
                    if ( mat == &A )
                    {
                        M_root = std::move( M );
                    }// if
                    else
                    {
                        auto                  mat_parent = mat->parent();
                        Hpro::TBlockMatrix< value_t > *  M_parent   = nullptr;

                        {
                            auto  lock_bmap = std::scoped_lock( bmtx );
                            
                            M_parent = bmat_map.at( mat_parent->id() );
                        }
                        
                        for ( uint  i = 0; i < mat_parent->nblock_rows(); ++i )
                        {
                            for ( uint  j = 0; j < mat_parent->nblock_cols(); ++j )
                            {
                                if ( mat_parent->block( i, j ) == mat )
                                {
                                    M_parent->set_block( i, j, M.release() );
                                    break;
                                }// if
                            }// for
                        }// for
                    }// if
                }// if

                //
                // fill mapping for row/column cluster basis
                //

                {
                    auto  lock_cb = std::scoped_lock( cbmtx );
                    
                    HLR_ASSERT( rowcb_map.find( mat->row_is() ) != rowcb_map.end() );
            
                    auto  rowcb = rowcb_map[ mat->row_is() ];

                    for ( uint  i = 0; i < rowcb->nsons(); ++i )
                    {
                        auto  son_i = rowcb->son(i);
                
                        if ( ! is_null( son_i ) )
                            rowcb_map[ son_i->is() ] = son_i;
                    }// for
            
                    HLR_ASSERT( colcb_map.find( mat->col_is() ) != colcb_map.end() );
            
                    auto  colcb = colcb_map[ mat->col_is() ];

                    for ( uint  i = 0; i < colcb->nsons(); ++i )
                    {
                        auto  son_i = colcb->son(i);
                
                        if ( ! is_null( son_i ) )
                            colcb_map[ son_i->is() ] = son_i;
                    }// for
                }
            } );

        matrices = std::move( children );
        
        ::tbb::parallel_invoke(

            [&] ()
            {
                //
                // construct row bases for all block rows constructed on this level
                //

                auto  rowiss = std::deque< indexset >();

                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                ::tbb::parallel_for_each(
                    rowiss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = rowmap.at( is );
                    
                        if ( matrices.size() == 0 )
                            return;

                        //
                        // form U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
                        //
            
                        size_t  nrows_U = is.size();
                        size_t  ncols_U = 0;

                        for ( auto &  R : matrices )
                            ncols_U += R->rank();

                        auto    U       = blas::matrix< value_t >( nrows_U, ncols_U );
                        size_t  pos     = 0;
                        auto    pos_mtx = std::mutex();
                        
                        ::tbb::parallel_for_each(
                            matrices,
                            [&] ( auto &  R )
                            {
                                // R = U·V' = W·T·X'
                                auto  U_i = R->U_direct();
                                auto  V_i = blas::copy( R->V_direct() );
                                auto  R_i = blas::matrix< value_t >();
                                auto  k   = R->rank();
                
                                blas::qr( V_i, R_i );

                                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );

                                {
                                    auto  lock  = std::scoped_lock( pos_mtx );
                                    auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                                    blas::copy( UR_i, U_sub );
                
                                    pos += k;
                                }
                            } );

                        //
                        // QR of S and computation of row basis
                        //

                        auto  Un = basisapx.column_basis( U, acc );
            
                        // finally assign to cluster basis object
                        // (no change to "rowcb_map", therefore no lock)
                        rowcb_map.at( is )->set_basis( std::move( Un ) );
                    } );
            },

            [&] ()
            {
                //
                // construct column bases for all block columns constructed on this level
                //

                auto  coliss = std::deque< indexset >();
            
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                ::tbb::parallel_for_each(
                    coliss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = colmap.at( is );

                        if ( matrices.size() == 0 )
                            return;

                        //
                        // form matrix V = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
                        //

                        size_t  nrows_V = is.size();
                        size_t  ncols_V = 0;

                        for ( auto &  R : matrices )
                            ncols_V += R->rank();

                        auto    V       = blas::matrix< value_t >( nrows_V, ncols_V );
                        size_t  pos     = 0;
                        auto    pos_mtx = std::mutex();
                        
                        ::tbb::parallel_for_each (
                            matrices,
                            [&] ( auto &  R )
                            {
                                // R' = (U·V')' = V·U' = X·T'·W'
                                auto  V_i = blas::copy( R->V_direct() );
                                auto  U_i = blas::copy( R->U_direct() );
                                auto  R_i = blas::matrix< value_t >();
                                auto  k   = R->rank();
                
                                blas::qr( U_i, R_i );

                                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );

                                {
                                    auto  lock = std::scoped_lock( pos_mtx );
                                    auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                                    blas::copy( VR_i, V_sub );
                
                                    pos += k;
                                }
                            } );

                        auto  Vn = basisapx.column_basis( V, acc );

                        // finally assign to cluster basis object
                        // (no change to "colcb_map", therefore no lock)
                        colcb_map.at( is )->set_basis( std::move( Vn ) );
                    } );
            }
        );

        //
        // now convert all blocks on this level
        //

        ::tbb::parallel_for_each(
            lrmat,
            [&] ( auto  R )
            {
                auto  rowcb = rowcb_map.at( R->row_is() );
                auto  colcb = colcb_map.at( R->col_is() );
                auto  Un    = rowcb->basis();
                auto  Vn    = colcb->basis();

                //
                // R = U·V' ≈ Un (Un' U V' Vn) Vn'
                //          = Un S Vn'  with  S = Un' U V' Vn
                //

                auto  UnU = blas::prod( blas::adjoint( Un ), R->U_direct() );
                auto  VnV = blas::prod( blas::adjoint( Vn ), R->V_direct() );
                auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

                auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                          R->col_is(),
                                                                                          *rowcb,
                                                                                          *colcb,
                                                                                          std::move( S ) );
            
                // replace standard lowrank block by uniform lowrank block
                auto  R_parent = R->parent();
                auto  U_parent = bmat_map.at( R_parent->id() );

                for ( uint  i = 0; i < R_parent->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < R_parent->nblock_cols(); ++j )
                    {
                        if ( R_parent->block( i, j ) == R )
                        {
                            U_parent->set_block( i, j, RU.release() );
                            break;
                        }// if
                    }// for
                }// for
            } );
    }// while
    
    return M_root;
}

//
// Build mapping from index set to set of lowrank matrices in block row/column
// together with computing QR factorization of each.
// Also set up structure of cluster bases.
// 
// template < typename value_t >
// using  lr_coupling_map_t  = std::unordered_map< indexset, std::list< std::pair< const hlr::matrix::lrmatrix< value_t > *, blas::matrix< value_t > > >, indexset_hash >;

template < typename value_t >
using cond_mat_t = std::pair< const hlr::matrix::lrmatrix< value_t > *, blas::matrix< value_t > >;

template < typename value_t >
using cond_mat_list_t = std::list< cond_mat_t< value_t > >;

template < typename value_t >
using  lr_coupling_map_t  = ::tbb::concurrent_hash_map< indexset, cond_mat_list_t< value_t >, indexset_hash >;

template < typename value_t >
using  accessor_t = typename lr_coupling_map_t< value_t >::accessor;

template < typename value_t >
void
build_mat_map ( const Hpro::TMatrix< value_t > &   A,
                shared_cluster_basis< value_t > &  rowcb,
                shared_cluster_basis< value_t > &  colcb,
                lr_coupling_map_t< value_t > &     row_map,
                std::mutex &                       row_mtx,
                lr_coupling_map_t< value_t > &     col_map,
                std::mutex &                       col_mtx )
{
    using namespace hlr::matrix;
    
    //
    // decide upon cluster type, how to construct matrix
    //
    
    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // for  M = U·V' compute [ Q, Cu ] = U  and  [ Q, Cv ] = V
        // and remember Cu and Cv
        //
        // (see "build_cluster_basis" below for explanation)
        //
        
        auto  R  = cptrcast( &A, hlr::matrix::lrmatrix< value_t > );
        auto  U  = blas::copy( R->U_direct() );
        auto  V  = blas::copy( R->V_direct() );
        auto  Cu = blas::matrix< value_t >();
        auto  Cv = blas::matrix< value_t >();
        
        ::tbb::parallel_invoke( [&] () { blas::qr( U, Cu, false ); },  // only need R, not Q
                                [&] () { blas::qr( V, Cv, false ); } );
        
        HLR_ASSERT( Cu.ncols() != 0 );
        HLR_ASSERT( Cv.ncols() != 0 );

        //
        // add matrix to block row/column together with respectively other(!) semi-coupling
        //
        
        {
            auto  accessor = accessor_t< value_t >();
            auto  entry    = cond_mat_t< value_t >{ R, std::move( Cv ) };

            row_map.insert( accessor, A.row_is() );
            accessor->second.push_back( std::move( entry ) );
        }

        {
            auto  accessor = accessor_t< value_t >();
            auto  entry    = cond_mat_t< value_t >{ R, std::move( Cu ) };

            col_map.insert( accessor, A.col_is() );
            accessor->second.push_back( std::move( entry ) );
        }
    }// if
    else if ( is_blocked( A ) )
    {
        auto  B = cptrcast( &A, Hpro::TBlockMatrix< value_t > );

        //
        // set up (empty) cluster bases if not yet done
        //

        // ::tbb::parallel_invoke(
        //     [&] ()
        //     {
        //         auto  lock = std::scoped_lock( rowcb.mutex() );
            
        //         for ( uint  i = 0; i < B->nblock_rows(); ++i )
        //         {
        //             auto  rowcb_i = rowcb.son( i );
            
        //             for ( uint  j = 0; j < B->nblock_cols(); ++j )
        //             {
        //                 auto  M_ij = B->block( i, j );
                
        //                 if ( ! is_null( M_ij ) )
        //                 {
        //                     if ( is_null( rowcb_i ) )
        //                     {
        //                         rowcb_i = new shared_cluster_basis< value_t >( M_ij->row_is() );
        //                         rowcb.set_son( i, rowcb_i );
        //                     }// if
            
        //                     if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
        //                         rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );

        //                     continue;
        //                 }// if
        //             }// for
        //         }// for
        //     },

        //     [&] ()
        //     {
        //         auto  lock = std::scoped_lock( colcb.mutex() );
            
        //         for ( uint  j = 0; j < B->nblock_cols(); ++j )
        //         {
        //             auto  colcb_j = colcb.son( j );
            
        //             for ( uint  i = 0; i < B->nblock_rows(); ++i )
        //             {
        //                 auto  M_ij = B->block( i, j );
                
        //                 if ( ! is_null( M_ij ) )
        //                 {
        //                     if ( is_null( colcb_j ) )
        //                     {
        //                         colcb_j = new shared_cluster_basis< value_t >( M_ij->col_is() );
        //                         colcb.set_son( j, colcb_j );
        //                     }// if
            
        //                     if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
        //                         colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );

        //                     continue;
        //                 }// if
        //             }// for
        //         }// for
        //     }
        // );

        //
        // recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowcb_i = rowcb.son( i );
                    
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  colcb_j = colcb.son( j );
                
                        if ( ! is_null( B->block( i, j ) ) )
                            build_mat_map( *B->block( i, j ),
                                           *rowcb_i, *colcb_j,
                                           row_map, row_mtx,
                                           col_map, col_mtx );
                    }// for
                }// for
            } );
    }// if
}

//
// build cluster basis using precomputed QR decomposition of lowrank matrices
// in block row/columns
//
template < typename value_t,
           typename basisapx_t >
void
build_cluster_basis ( shared_cluster_basis< value_t > &     cb,
                      const basisapx_t &                    basisapx,
                      const accuracy &                      acc,
                      const lr_coupling_map_t< value_t > &  mat_map,
                      const bool                            transposed )
{
    using  real_t  = Hpro::real_type_t< value_t >;

    const matop_t  op = ( transposed ? apply_transposed : apply_normal );

    //
    // construct cluster basis for all precollected blocks
    //

    auto  accessor = accessor_t< value_t >();
    
    if ( mat_map.find( accessor, cb.is() ) )
    {
        //
        // compute column basis for block row
        //
        //  ( U₀·V₀'  U₁·V₁'  U₂·V₂'  … )
        //
        // as 
        //
        //   ( U₀·C₀'·Q₀'  U₁·C₁'·Q₁'  U₂'·C₂'·Q₂' … )
        //
        // with QR decomposition V_i = Q_i C_i
        // (precomputed in "build_mat_map" above)
        //
        // As Q_i is orthogonal, it can be neglected in column basis computation!
        //

        const uint  nrows = cb.is().size();
        uint        ncols = 0;

        // determine total number of columns
        for ( const auto  [ M_i, C_i ] : accessor->second )
            ncols += C_i.nrows();

        // build ( U_0·C_0'  U_1·C_1'  U_2'·C_2' … )
        auto  X   = blas::matrix< value_t >( nrows, ncols );
        uint  pos = 0;

        for ( const auto  [ R_i, C_i ] : accessor->second )
        {
            auto  U_i = blas::prod( R_i->U( op ), blas::adjoint( C_i ) );
            auto  X_i = blas::matrix< value_t >( X, blas::range::all, blas::range( pos, pos + C_i.nrows() - 1 ) );

            blas::copy( U_i, X_i );
            pos += C_i.nrows();
        }// for

        // actually build cluster basis
        auto  Ws = blas::vector< real_t >(); // singular values corresponding to basis vectors
        auto  W  = basisapx.column_basis( X, acc, & Ws );

        cb.set_basis( std::move( W ), std::move( Ws ) );
    }// if

    //
    // recurse
    //
    
    ::tbb::parallel_for< uint >(
        0, cb.nsons(),
        [&,transposed] ( const uint  i )
        {
            if ( ! is_null( cb.son( i ) ) )
                build_cluster_basis( *cb.son( i ), basisapx, acc, mat_map, transposed );
        } );
}

//
// build cluster basis using precomputed QR decomposition of 
// all lowrank matrices in M
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_uniform ( const Hpro::TMatrix< value_t > &   A,
                shared_cluster_basis< value_t > &  rowcb,
                shared_cluster_basis< value_t > &  colcb )
{
    using namespace hlr::matrix;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // compute coupling matrix as W'·U·(X'·V)'
        // with cluster basis W and X
        //
        
        auto  R  = cptrcast( &A, hlr::matrix::lrmatrix< value_t > );
        auto  SU = blas::prod( blas::adjoint( rowcb.basis() ), R->U_direct() );
        auto  SV = blas::prod( blas::adjoint( colcb.basis() ), R->V_direct() );
        auto  S  = blas::prod( SU, blas::adjoint( SV ) );

        M = std::make_unique< uniform_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BA );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowcb_i = rowcb.son( i );
                    
                    HLR_ASSERT( ! is_null( rowcb_i ) );

                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  colcb_j = colcb.son( j );
                        auto  A_ij    = BA->block( i, j );

                        HLR_ASSERT( ! is_null( colcb_j ) );
                        
                        if ( ! is_null( A_ij ) )
                        {
                            auto  B_ij = build_uniform( *A_ij, *rowcb_i, *colcb_j );

                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );

        return  std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
    {
        M = A.copy();
    }// else

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

}}}}// namespace hlr::tbb::matrix::detail

#endif // __HLR_TBB_DETAIL_UNIFORM_MATRIX_HH
