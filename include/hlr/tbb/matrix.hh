#ifndef __HLR_TBB_MATRIX_HH
#define __HLR_TBB_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>
#include <deque>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"
#include "hlr/matrix/restrict.hh"
#include "hlr/matrix/convert.hh"

namespace hlr
{

namespace hpro = HLIB;
    
namespace tbb
{

namespace matrix
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const hpro::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    assert( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< hpro::TMatrix >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M.reset( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc, nseq );
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // copy properties from the cluster
    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// build representation of sparse matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <M>. If low rank blocks are not zero, they are
// truncated to accuracy <acc> using approximation method <apx>
//
template < typename approx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const hpro::TSparseMatrix &  S,
        const hpro::TTruncAcc &      acc,
        const approx_t &             apx,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using  value_t = typename approx_t::value_t;
    
    // static_assert( std::is_same< typename coeff_t::value_t,
    //                              typename lrapx_t::value_t >::value,
    //                "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        //
        // restrict to local cluster and convert to desired format
        //

        auto  S_bct = hlr::matrix::restrict( S, bct->is() );
        
        if ( bct->is_adm() )
        {
            M = hlr::matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else
        {
            M = hlr::matrix::convert_to_dense< value_t >( *S_bct );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build( bct->son( i, j ), S, acc, apx, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}
    
//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform ( const hpro::TBlockCluster *  bct,
                const coeff_t &              coeff,
                const lrapx_t &              lrapx,
                const hpro::TTruncAcc &      acc,
                const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< hpro::TRkMatrix * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< hpro::idx_t, hpro::TBlockMatrix * >;

    //
    // TODO: argument for approximation object
    //

    const auto  approx = approx::SVD< value_t >();
    
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

    auto  M_root     = std::unique_ptr< hpro::TMatrix >();

    auto  nodes      = std::deque< const hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    auto  bmtx       = std::mutex(); // for bmat_map
    auto  cmtx       = std::mutex(); // for children list
    auto  lmtx       = std::mutex(); // for row/col map lists
    auto  cbmtx      = std::mutex(); // for rowcb/colcb map lists

    //
    // local function to set up hierarchy (parent <-> M)
    //
    auto  insert_hier = [&] ( const hpro::TBlockCluster *         node,
                              std::unique_ptr< hpro::TMatrix > &  M )
    {
        if ( is_null( node->parent() ) )
        {
            M_root = std::move( M );
        }// if
        else
        {
            auto  parent   = node->parent();
            auto  M_parent = bmat_map_t::mapped_type( nullptr );

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
    auto  create_cb = [&] ( const hpro::TBlockCluster *  node )
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
        auto  lrnodes  = std::deque< const hpro::TBlockCluster * >();
        auto  lrmat    = std::deque< hpro::TMatrix * >();
        auto  aff_part = ::tbb::affinity_partitioner();
        
        // filter out admissible nodes to initialize affinity_partitinioner
        for ( auto  node : nodes )
            if ( node->is_leaf() && node->is_adm() )
                lrnodes.push_back( node );

        ::tbb::parallel_invoke(
            [&] ()
            {
                ::tbb::parallel_for(
                    ::tbb::blocked_range< size_t >( 0, lrnodes.size() ),
                    [&] ( const auto &  r )
                    {
                        for ( auto  idx = r.begin(); idx != r.end(); ++idx )
                        {
                            auto  node = lrnodes[ idx ];
                            auto  M    = std::unique_ptr< hpro::TMatrix >( lrapx.build( node, acc ) );

                            {
                                auto  lock = std::scoped_lock( lmtx );

                                if ( is_lowrank( *M ) )
                                {
                                    auto  R = ptrcast( M.get(), hpro::TRkMatrix );
                                    
                                    rowmap[ M->row_is() ].push_back( R );
                                    colmap[ M->col_is() ].push_back( R );
                                }// if
                                
                                // store always to maintain affinity
                                lrmat.push_back( M.get() );
                            }
                            
                            M->set_id( node->id() );
                            M->set_procs( node->procs() );

                            insert_hier( node, M );
                            create_cb( node );
                        }// for
                    },
                    aff_part
                );
            },

            [&] ()
            {
                ::tbb::parallel_for< size_t >(
                    0, nodes.size(),
                    [&] ( const auto  idx )
                    {
                        auto  node = nodes[idx];
                        auto  M    = std::unique_ptr< hpro::TMatrix >();

                        if ( node->is_leaf() )
                        {
                            // handled above
                            if ( node->is_adm() )
                                return;
                    
                            M = coeff.build( node->is().row_is(), node->is().col_is() );
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

                            M = std::make_unique< hpro::TBlockMatrix >( node );
        
                            auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

                            // make sure, block structure is correct
                            if (( B->nblock_rows() != node->nrows() ) ||
                                ( B->nblock_cols() != node->ncols() ))
                                B->set_block_struct( node->nrows(), node->ncols() );

                            // make value type consistent in block matrix and sub blocks
                            B->adjust_value_type();

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
                    }
                );
            }
        );

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

                ::tbb::parallel_for< size_t >(
                    0, rowiss.size(),
                    [&] ( const auto  idx )                           
                    {
                        auto  is       = rowiss[ idx ];
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
                            auto  U_i = blas::mat_U< value_t >( R );
                            auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
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

                        auto  Un = approx.column_basis( U, acc );
            
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

                ::tbb::parallel_for< size_t >(
                    0, coliss.size(),
                    [&] ( const auto  idx )                           
                    {
                        auto  is       = coliss[ idx ];
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
                            auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
                            auto  U_i = blas::copy( blas::mat_U< value_t >( R ) );
                            auto  R_i = blas::matrix< value_t >();
                            auto  k   = R->rank();
                
                            blas::qr( U_i, R_i );

                            auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                            auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                            blas::copy( VR_i, V_sub );
                
                            pos += k;
                        }// for

                        auto  Vn = approx.column_basis( V, acc );

                        // finally assign to cluster basis object
                        // (no change to "colcb_map", therefore no lock)
                        colcb_map.at( is )->set_basis( std::move( Vn ) );
                    } );
            }
        );

        //
        // now convert all blocks on this level
        //

        ::tbb::parallel_for(
            ::tbb::blocked_range< size_t >( 0, lrmat.size() ),
            [&] ( const auto &  r )                           
            {
                for ( auto  idx = r.begin(); idx != r.end(); ++idx )
                {
                    auto  M = lrmat[ idx ];

                    if ( ! is_lowrank( M ) )
                        return;

                    auto  R     = ptrcast( M, hpro::TRkMatrix );
                    auto  rowcb = rowcb_map.at( R->row_is() );
                    auto  colcb = colcb_map.at( R->col_is() );
                    auto  Un    = rowcb->basis();
                    auto  Vn    = colcb->basis();

                    //
                    // R = U·V' ≈ Un (Un' U V' Vn) Vn'
                    //          = Un S Vn'  with  S = Un' U V' Vn
                    //

                    auto  UnU = blas::prod( blas::adjoint( Un ), blas::mat_U< value_t >( R ) );
                    auto  VnV = blas::prod( blas::adjoint( Vn ), blas::mat_V< value_t >( R ) );
                    auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

                    auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                              R->col_is(),
                                                                                              *rowcb,
                                                                                              *colcb,
                                                                                              std::move( S ) );
            
                    // replace standard lowrank block by uniform lowrank block
                    R->parent()->replace_block( R, RU.release() );
                    delete R;
                }// for
            },
            aff_part
        );
    }// while
    
    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// assign block cluster to matrix
//
inline
void
assign_cluster ( hpro::TMatrix &              M,
                 const hpro::TBlockCluster &  bc )
{
    hlr::seq::matrix::assign_cluster( M, bc );
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
inline
std::unique_ptr< hpro::TMatrix >
copy ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = copy( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

//
// return copy of matrix with TRkMatrix replaced by tiled_lrmatrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_tiled ( const hpro::TMatrix &  M,
             const size_t           ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM,ntile] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = copy_tiled< value_t >( * BM->block( i, j ), ntile );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hpro::TRkMatrix );
        auto  R  = std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                               RM->col_is(),
                                                                               ntile,
                                                                               hpro::blas_mat_A< value_t >( RM ),
                                                                               hpro::blas_mat_B< value_t >( RM ) );

        R->set_id( RM->id() );

        return R;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of matrix with tiled_lrmatrix replaced by TRkMatrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_nontiled ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = copy_nontiled< value_t >( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else if ( IS_TYPE( & M, tiled_lrmatrix ) )
    {
        //
        // copy low-rank data into tiled form
        //

        assert( M.is_real() );
        
        auto  RM = cptrcast( & M, hlr::matrix::tiled_lrmatrix< real > );
        auto  R  = std::make_unique< hpro::TRkMatrix >( RM->row_is(), RM->col_is() );
        auto  U  = hlr::matrix::to_dense( RM->U() );
        auto  V  = hlr::matrix::to_dense( RM->V() );

        R->set_lrmat( U, V );
        R->set_id( RM->id() );

        return R;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of (block-wise) lower-left part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_ll ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM,diag] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  T = M.copy();

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// return copy of (block-wise) upper-right part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_ur ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM,diag] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  T = M.copy();

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
inline
void
copy_to ( const hpro::TMatrix &  A,
          hpro::TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BA->nblock_rows(),
                                            0, BA->nblock_cols() ),
            [BA,BB] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BA->block( i, j ) != nullptr )
                        {
                            assert( ! is_null( BB->block( i, j ) ) );
                            
                            copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
inline
std::unique_ptr< hpro::TMatrix >
realloc ( hpro::TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, hpro::TBlockMatrix );
        auto  C  = std::make_unique< hpro::TBlockMatrix >();
        auto  BC = ptrcast( C.get(), hpro::TBlockMatrix );

        C->copy_struct_from( B );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BC] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  C_ij = realloc( B->block( i, j ) );

                        BC->set_block( i, j, C_ij.release() );
                        B->set_block( i, j, nullptr );
                    }// for
                }// for
            } );

        delete B;

        return C;
    }// if
    else
    {
        auto  C = A->copy();

        delete A;

        return C;
    }// else
}

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_uniform ( const hpro::TMatrix &                    M,
               hlr::matrix::cluster_basis< value_t > &  rowcb,
               hlr::matrix::cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = copy_uniform( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (V·V^H·A) (U·U^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H

        auto  R  = cptrcast( &M, hpro::TRkMatrix );

        auto  UA = rowcb.transform_forward( hpro::blas_mat_A< value_t >( R ) );
        auto  VB = colcb.transform_forward( hpro::blas_mat_B< value_t >( R ) );
        auto  S  = blas::prod( value_t(1), UA, blas::adjoint( VB ) );

        // auto  M1 = blas::prod( value_t(1), hpro::blas_mat_A< value_t >( R ), blas::adjoint( hpro::blas_mat_B< value_t >( R ) ) );
        // auto  T  = blas::prod( value_t(1), rowcb.basis(), S );
        // auto  M2 = blas::prod( value_t(1), T, blas::adjoint( colcb.basis() ) );

        // blas::add( value_t(-1), M2, M1 );
        
        // std::cout << blas::norm_F( M1 ) << std::endl;
        
        return std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                             rowcb, colcb,
                                                                             std::move( S ) );
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// convert given matrix into lowrank format
//
template < typename approx_t >
std::unique_ptr< hpro::TRkMatrix >
convert_to_lowrank ( const hpro::TMatrix &    M,
                     const hpro::TTruncAcc &  acc,
                     const approx_t &         approx )
{
    using  value_t = typename approx_t::value_t;
    
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto        B  = cptrcast( &M, hpro::TBlockMatrix );
        auto        Us = std::list< blas::matrix< value_t > >();
        auto        Vs = std::list< blas::matrix< value_t > >();
        std::mutex  mtx;

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j );
                
                        if ( is_null( B_ij ) )
                            continue;

                        auto  R_ij = convert_to_lowrank( *B_ij, acc, approx );
                        auto  U    = blas::matrix< value_t >( M.nrows(), R_ij->rank() );
                        auto  V    = blas::matrix< value_t >( M.ncols(), R_ij->rank() );
                        auto  U_i  = blas::matrix< value_t >( U, R_ij->row_is() - M.row_ofs(), blas::range::all );
                        auto  V_j  = blas::matrix< value_t >( V, R_ij->col_is() - M.col_ofs(), blas::range::all );

                        blas::copy( hpro::blas_mat_A< value_t >( R_ij ), U_i );
                        blas::copy( hpro::blas_mat_B< value_t >( R_ij ), V_j );

                        std::scoped_lock  lock( mtx );
                            
                        Us.push_back( std::move( U ) );
                        Vs.push_back( std::move( V ) );
                    }// for
                }// for
            } );

        auto  [ U, V ] = approx( Us, Vs, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D        = cptrcast( &M, hpro::TDenseMatrix );
        auto  T        = std::move( blas::copy( hpro::blas_mat< value_t >( D ) ) );
        auto  [ U, V ] = approx( T, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, hpro::TRkMatrix );
        auto  [ U, V ] = approx( hpro::blas_mat_A< value_t >( R ),
                                 hpro::blas_mat_B< value_t >( R ),
                                 acc );
        
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}// namespace matrix

}// namespace tbb

}// namespace hlr

#endif // __HLR_TBB_MATRIX_HH
