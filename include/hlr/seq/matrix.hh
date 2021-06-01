#ifndef __HLR_SEQ_MATRIX_HH
#define __HLR_SEQ_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/utils/checks.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/norm.hh"
#include "hlr/approx/svd.hh" // DEBUG
#include "hlr/matrix/cluster_basis.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/matrix/convert.hh"
#include "hlr/matrix/restrict.hh"

namespace hlr { namespace seq { namespace matrix {

namespace hpro = HLIB;

using namespace hlr::matrix;

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
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    assert( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
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
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

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
            M = matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else
        {
            M = matrix::convert_to_dense< value_t >( *S_bct );
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
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    auto  B_ij = build( bct->son( i, j ), S, acc, apx, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

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

    size_t  base_mem = hpro::Mem::usage();
    size_t  max_mem  = hpro::Mem::usage();

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

    auto  nodes      = std::list< const hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    while ( ! nodes.empty() )
    {
        auto  children = decltype( nodes )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::list< hpro::TRkMatrix * >();
        
        for ( auto  node : nodes )
        {
            auto  M = std::unique_ptr< hpro::TMatrix >();

            // std::cout << node->id() << std::endl;
            
            if ( node->is_leaf() )
            {
                if ( node->is_adm() )
                {
                    M = std::unique_ptr< hpro::TMatrix >( lrapx.build( node, acc ) );

                    if ( is_lowrank( *M ) )
                    {
                        auto  R = ptrcast( M.get(), hpro::TRkMatrix );
                        
                        rowmap[ M->row_is() ].push_back( R );
                        colmap[ M->col_is() ].push_back( R );
                        lrmat.push_back( R );
                    }// if
                }// if
                else
                {
                    M = coeff.build( node->is().row_is(), node->is().col_is() );
                }// else
            }// if
            else
            {
                // collect children
                for ( uint  i = 0; i < node->nrows(); ++i )
                    for ( uint  j = 0; j < node->ncols(); ++j )
                        if ( node->son( i, j ) != nullptr )
                            children.push_back( node->son( i, j ) );

                M = std::make_unique< hpro::TBlockMatrix >( node );
        
                auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

                // make sure, block structure is correct
                if (( B->nblock_rows() != node->nrows() ) ||
                    ( B->nblock_cols() != node->ncols() ))
                    B->set_block_struct( node->nrows(), node->ncols() );

                // make value type consistent in block matrix and sub blocks
                B->adjust_value_type();

                // remember all block matrices for setting up hierarchy
                bmat_map[ node->id() ] = B;
            }// else

            M->set_id( node->id() );
            M->set_procs( node->procs() );

            //
            // set up hierarchy (parent <-> M)
            //

            if ( is_null( node->parent() ) )
            {
                M_root = std::move( M );
            }// if
            else
            {
                auto  parent   = node->parent();
                auto  M_parent = bmat_map.at( parent->id() );

                for ( uint  i = 0; i < parent->nrows(); ++i ) 
                {
                    for ( uint  j = 0; j < parent->ncols(); ++j )
                    {
                        if ( parent->son( i, j ) == node )
                        {
                            M_parent->set_block( i, j, M.release() );
                            break;
                        }// if
                    }// for
                }// for
            }// if

            //
            // build row/column cluster basis objects and set up
            // cluster bases hierarchy
            //

            auto             rowcl = node->rowcl();
            auto             colcl = node->colcl();
            cluster_basis *  rowcb = nullptr;
            cluster_basis *  colcb = nullptr;

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
        }// for

        nodes = std::move( children );
        
        //
        // construct row bases for all block rows constructed on this level
        //

        for ( auto  [ is, matrices ] : rowmap )
        {
            if ( matrices.size() == 0 )
                continue;

            #if 1

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

            #else
            
            //
            // compute column basis for
            //
            //   ( U₀·S₀·V₀'  U₁·S₁·V₁'  U₂·S₂·V₂'  … ) =
            //
            //                  ⎛ S₀·V₀'              ⎞
            //   ( U₀ U₁ U₂ … ) ⎜       S₁·V₁'        ⎟ =
            //                  ⎜             S₂·V₂'  ⎟
            //                  ⎝                   … ⎠
            //
            //                  ⎛ V₀·S₀'              ⎞'
            //   ( U₀ U₁ U₂ … ) ⎜       V₁·S₁'        ⎟ =
            //                  ⎜             V₂·S₂'  ⎟
            //                  ⎝                   … ⎠
            //
            //                  ⎛⎛V₀     ⎞ ⎛S₀'        ⎞⎞'
            //   ( U₀ U₁ U₂ … ) ⎜⎜  V₁   ⎟·⎜   S₁'     ⎟⎟ =
            //                  ⎜⎜    V₂ ⎟ ⎜      S₂'  ⎟⎟
            //                  ⎝⎝      …⎠ ⎝         … ⎠⎠
            //
            // Since diag(V_i) is orthogonal, it can be omitted for row bases
            // computation, leaving
            //
            //                  ⎛S₀'       ⎞'                 ⎛  ⎛S₀'        ⎞⎞'
            //   ( U₀ U₁ U₂ … ) ⎜   S₁'    ⎟ = ( U₀ U₁ U₂ … ) ⎜qr⎜   S₁'     ⎟⎟ =
            //                  ⎜     S₂'  ⎟                  ⎜  ⎜      S₂'  ⎟⎟
            //                  ⎝        … ⎠                  ⎝  ⎝         … ⎠⎠
            //
            //   ( U₀ U₁ U₂ … ) ( Q·R )'
            //
            // of which again Q can be omitted due to orthogonality.
            // Finally one needs to compute the row basis of
            //
            //   ( U₀ U₁ U₂ … ) R'
            //
            // Also, the local coupling matrices S_i are scaled w.r.t. spectral norm
            // to achieve even approximation for all blocks.
            //

            //                                          ⎛S₀'        ⎞
            // form matrices U = ( U₀ U₁ U₂ … ) and S = ⎜   S₁'     ⎟
            //                                          ⎜      S₂'  ⎟
            //                                          ⎝         … ⎠
            //
            
            size_t  nrows_U = is.size();
            size_t  ncols_U = 0;
            size_t  ncols_S = 0;

            for ( auto &  R : matrices )
            {
                ncols_U += R->rank();
                ncols_S  = std::max( ncols_S, R->rank() );
            }// for

            auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
            auto    S   = blas::matrix< value_t >( ncols_U, ncols_S );
            size_t  pos = 0;

            for ( auto &  R : matrices )
            {
                // R = U·V' = W·T·X'
                auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
                auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
                auto  RW = blas::matrix< value_t >();
                auto  RX = blas::matrix< value_t >();
                auto  k  = R->rank();
                
                blas::qr( W, RW );
                blas::qr( X, RX );

                auto  T = blas::prod( RW, blas::adjoint( RX ) );

                blas::scale( value_t(1) / norm::spectral( T ), T );
                
                auto  U_i = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );
                auto  S_i = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + k - 1 ),
                                                     blas::range( 0, k - 1 ) );

                blas::copy( W, U_i );
                blas::copy( blas::adjoint( T ), S_i );
                
                pos += k;
            }// for

            //
            // QR of S and computation of row basis
            //

            auto  R = blas::matrix< value_t >();
        
            blas::qr( S, R, false );

            auto  UR = blas::prod( U, blas::adjoint( R ) );
            auto  Un = approx.column_basis( UR, acc );

            #endif
            
            // finally assign to cluster basis object
            rowcb_map.at( is )->set_basis( std::move( Un ) );
        }// for

        //
        // construct column bases for all block columns constructed on this level
        //

        for ( auto  [ is, matrices ] : colmap )
        {
            if ( matrices.size() == 0 )
                continue;

            #if 1

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

            #else
            
            //
            // compute column basis for
            //
            //   ⎛U₀·S₀·V₀'⎞ 
            //   ⎜U₁·S₁·V₁'⎟
            //   ⎜U₂·S₂·V₂'⎟
            //   ⎝    …    ⎠
            //
            // or row basis of
            //
            //   ⎛U₀·S₀·V₀'⎞ 
            //   ⎜U₁·S₁·V₁'⎟ = ( V₀·S₀'·U₀'  V₁·S₁'·U₁'  V₂·S₂'·U₂'  … ) =
            //   ⎜U₂·S₂·V₂'⎟
            //   ⎝    …    ⎠
            //
            //                  ⎛ S₀'·U₀'                ⎞
            //   ( V₀ V₁ V₂ … ) ⎜        S₁'·U₁'         ⎟ =
            //                  ⎜               S₂'·U₂'  ⎟
            //                  ⎝                      … ⎠
            //
            //                  ⎛ U₀·S₀               ⎞'
            //   ( V₀ V₁ V₂ … ) ⎜       U₁·S₁         ⎟ =
            //                  ⎜             U₂·S₂   ⎟
            //                  ⎝                   … ⎠
            //
            //                  ⎛⎛U₀     ⎞ ⎛S₀⎞⎞'
            //   ( V₀ V₁ V₂ … ) ⎜⎜  U₁   ⎟·⎜S₁⎟⎟ =
            //                  ⎜⎜    U₂ ⎟ ⎜S₂⎟⎟
            //                  ⎝⎝      …⎠ ⎝… ⎠⎠
            //
            // Since diag(U_i) is orthogonal, it can be omitted for column bases
            // computation, leaving
            //
            //                  ⎛S₀⎞'                 ⎛  ⎛V₀⎞⎞'
            //   ( V₀ V₁ V₂ … ) ⎜S₁⎟ = ( V₀ V₁ V₂ … ) ⎜qr⎜V₁⎟⎟ =
            //                  ⎜S₂⎟                  ⎜  ⎜V₂⎟⎟
            //                  ⎝… ⎠                  ⎝  ⎝… ⎠⎠
            //
            //   ( V₀ V₁ V₂ … ) ( Q·R )'
            //
            // of which again Q can be omitted due to orthogonality.
            // Finally one needs to compute the row basis of
            //
            //   ( V₀ V₁ V₂ … ) R'
            //
            // Also, the local coupling matrices S_i are scaled w.r.t. spectral norm
            // to achieve even approximation for all blocks.
            //

            //                                          ⎛S₀⎞
            // form matrices V = ( V₀ V₁ V₂ … ) and S = ⎜S₁⎟
            //                                          ⎜S₂⎟
            //                                          ⎝… ⎠
            //

            size_t  nrows_V = is.size();
            size_t  ncols_V = 0;
            size_t  ncols_S = 0;

            for ( auto &  R : matrices )
            {
                ncols_V += R->rank();
                ncols_S  = std::max( ncols_S, R->rank() );
            }// for

            auto    V   = blas::matrix< value_t >( nrows_V, ncols_V );
            auto    S   = blas::matrix< value_t >( ncols_V, ncols_S );
            size_t  pos = 0;

            for ( auto &  R : matrices )
            {
                // R' = (U·V')' = V·U' = X·T'·W'
                auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
                auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
                auto  RX = blas::matrix< value_t >();
                auto  RW = blas::matrix< value_t >();
                auto  k  = R->rank();
                
                blas::qr( X, RX );
                blas::qr( W, RW );

                auto  T = blas::prod( RW, blas::adjoint( RX ) );

                blas::scale( value_t(1) / norm::spectral( T ), T );
                
                auto  V_i = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );
                auto  S_i = blas::matrix< value_t >( S,
                                                     blas::range( pos, pos + k - 1 ),
                                                     blas::range( 0, k - 1 ) );

                blas::copy( X, V_i );
                blas::copy( T, S_i );
                
                pos += k;
            }// for

            //
            // QR of S and computation of row basis (column basis of initial problem)
            //

            auto  R = blas::matrix< value_t >();
        
            blas::qr( S, R, false );

            auto  VR = blas::prod( V, blas::adjoint( R ) );
            auto  Vn = approx.column_basis( VR, acc );

            #endif
            
            // finally assign to cluster basis object
            colcb_map.at( is )->set_basis( std::move( Vn ) );
        }// for

        //
        // now convert all blocks on this level
        //

        max_mem = std::max( max_mem, hpro::Mem::usage() );
        
        for ( auto  R : lrmat )
        {
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

            // {
            //     auto  M1 = blas::prod( blas::mat_U< value_t >( R ), blas::adjoint( blas::mat_V< value_t >( R ) ) );
            //     auto  T  = blas::prod( RU->row_basis(), RU->coeff() );
            //     auto  M2 = blas::prod( T, blas::adjoint( RU->col_basis() ) );

            //     blas::add( value_t(-1), M1, M2 );

            //     std::cout << R->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            // }
            
            // replace standard lowrank block by uniform lowrank block
            R->parent()->replace_block( R, RU.release() );
            delete R;
        }// for
    }// while

    std::cout << max_mem - base_mem << std::endl;
    
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
    M.set_cluster_force( & bc );
    
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );

        HLR_ASSERT( ( B->nblock_rows() == bc.nrows() ) &&
                    ( B->nblock_cols() == bc.ncols() ) );
                    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( B->block( i, j ) == nullptr )
                    continue;

                if ( bc.son( i, j ) == nullptr )
                    HLR_ERROR( "null cluster for non-null sub-block" );
                
                assign_cluster( * B->block( i, j ), * bc.son( i, j ) );
            }// for
        }// for
    }// if
}

//
// return copy of matrix
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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of matrix with TRkMatrix replaced by tiled_lrmatrix
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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_tiled< value_t >( * BM->block( i, j ), ntile );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hpro::TRkMatrix );
        auto  R  = std::make_unique< tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                  RM->col_is(),
                                                                  ntile,
                                                                  blas::mat_U< value_t >( RM ),
                                                                  blas::mat_V< value_t >( RM ) );

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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_nontiled< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( IS_TYPE( & M, tiled_lrmatrix ) )
    {
        //
        // copy low-rank data into tiled form
        //

        assert( M.is_real() );
        
        auto  RM = cptrcast( & M, tiled_lrmatrix< real > );
        auto  R  = std::make_unique< hpro::TRkMatrix >( RM->row_is(), RM->col_is() );
        auto  U  = to_dense( RM->U() );
        auto  V  = to_dense( RM->V() );

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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
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
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy lower-left data of A to matrix B
// - ASSUMPTION: identical matrix structure in lower-left part
//
inline
void
copy_to_ll ( const hpro::TMatrix &  A,
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
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ll( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy upper-right data of A to matrix B
// - ASSUMPTION: identical matrix structure in upper-right part
//
inline
void
copy_to_ur ( const hpro::TMatrix &  A,
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
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ur( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
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

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  C_ij = realloc( B->block( i, j ) );

                BC->set_block( i, j, C_ij.release() );
                B->set_block( i, j, nullptr );
            }// for
        }// for

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
// nullify data in matrix, e.g., M := 0
//
inline
void
clear ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( & M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    clear( * BM->block( i, j ) );
                }// if
            }// for
        }// for
    }// if
    else if ( is_lowrank( & M ) )
    {
        auto  R = ptrcast( & M, hpro::TRkMatrix );

        R->set_rank( 0 );
    }// if
    else if ( is_dense( & M ) )
    {
        auto  D = ptrcast( & M, hpro::TDenseMatrix );

        if ( D->is_complex() )
            blas::fill( hpro::blas_mat< hpro::complex >( D ), hpro::complex(0) );
        else
            blas::fill( hpro::blas_mat< hpro::real >( D ), hpro::real(0) );
    }// if
    else
        assert( false );
}

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_uniform ( const hpro::TMatrix &       M,
               cluster_basis< value_t > &  rowcb,
               cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_uniform( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (U·U^H·A) (V·V^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H
        //
        
        auto  R  = cptrcast( &M, hpro::TRkMatrix );

        auto  UA = rowcb.transform_forward( blas::mat_U< value_t >( R ) );
        auto  VB = colcb.transform_forward( blas::mat_V< value_t >( R ) );
        auto  S  = blas::prod( UA, blas::adjoint( VB ) );
        auto  UR = std::make_unique< uniform_lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                    rowcb, colcb,
                                                                    std::move( S ) );

        UR->set_id( R->id() );

        return UR;
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// return copy of matrix with uniform low-rank matrices converted
// to standard lowrank matrices
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_nonuniform ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_nonuniform< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R  = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  U  = blas::prod( R->row_cb().basis(), R->coeff() );
        auto  V  = blas::copy( R->col_cb().basis() );
        auto  SR = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

        SR->set_id( R->id() );

        return SR;
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// import functions from matrix module
//
using hlr::matrix::convert_to_lowrank;
using hlr::matrix::convert_to_dense;

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH
