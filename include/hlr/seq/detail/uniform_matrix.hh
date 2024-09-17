#ifndef __HLR_SEQ_DETAIL_UNIFORM_MATRIX_HH
#define __HLR_SEQ_DETAIL_UNIFORM_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <unordered_map>

#include <hlr/approx/accuracy.hh>
#include <hlr/approx/traits.hh>
#include <hlr/bem/traits.hh>
#include <hlr/arith/detail/uniform_basis.hh>
#include <hlr/matrix/dense_matrix.hh>

namespace hlr { namespace seq { namespace matrix { namespace detail {

using namespace hlr::matrix;

using hlr::uniform::is_matrix_map_t;

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
                    const accuracy &             acc,
                    const bool                   compress )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using real_t        = Hpro::real_type_t< value_t >;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< lrmatrix< value_t > * >, indexset_hash >;
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

    auto  nodes      = std::list< const Hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    while ( ! nodes.empty() )
    {
        auto  children = decltype( nodes )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::list< lrmatrix< value_t > * >();
        
        for ( auto  node : nodes )
        {
            auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

            // std::cout << node->id() << std::endl;
            
            if ( node->is_leaf() )
            {
                if ( node->is_adm() )
                {
                    M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( node, acc ) );

                    if ( hlr::matrix::is_lowrank( *M ) )
                    {
                        auto  R = ptrcast( M.get(), lrmatrix< value_t > );
                        
                        rowmap[ M->row_is() ].push_back( R );
                        colmap[ M->col_is() ].push_back( R );
                        lrmat.push_back( R );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + M->typestr() );
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
                        auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                        auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) );

                        if ( compress )
                            DD->compress( acc );

                        M = std::move( DD );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + M->typestr() );
                }// else
            }// if
            else
            {
                // collect children
                for ( uint  i = 0; i < node->nrows(); ++i )
                    for ( uint  j = 0; j < node->ncols(); ++j )
                        if ( ! is_null( node->son( i, j ) ) )
                            children.push_back( node->son( i, j ) );

                M = std::make_unique< Hpro::TBlockMatrix< value_t > >( node );
        
                auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

                // make sure, block structure is correct
                if (( B->nblock_rows() != node->nrows() ) ||
                    ( B->nblock_cols() != node->ncols() ))
                    B->set_block_struct( node->nrows(), node->ncols() );

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
                auto  U_i = R->U();
                auto  V_i = blas::copy( R->V() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// for

            //
            // QR of S and computation of row basis
            //

            auto  Us = blas::vector< real_t >();
            auto  Un = basisapx.column_basis( U, acc, & Us );

            #else
            
            //
            // compute scaled column basis for
            //
            //   ( U₀×V₀'               U₁×V₁'                U₂×V₂'               … ) =
            //
            //   ( Qu₀·Ru₀×(Qv₀·Rv₀)'   Qu₁·Ru₁×·(Qv₁·Rv₁)'   Qu₂·Ru₂×(Qv₂·Rv₂)'   … ) =
            //
            //   ( Qu₀·(Ru₀×Rv₀')·Qv₀'  Qu₁·(Ru₁×·Rv₁')·Qv₁'  Qu₂·(Ru₂×Rv₂')·Qv₂'  … ) =
            //
            //   ( Qu₀·S₀·Qv₀'          Qu₁·S₁·Qv₁'           Qu₂·S₂·Qv₂'          … ) =   with S_i = Ru_i × Rv_i'
            //
            //                     ⎛ Qv₀·S₀             ⎞'
            //   ( Qu₀ Qu₁ Qu₂ … ) ⎜      Qv₁·S₁        ⎟ =
            //                     ⎜           Qv₂·S₂   ⎟
            //                     ⎝                  … ⎠
            //
            //                     ⎛⎛Qv₀       ⎞ ⎛S₀     ⎞⎞'
            //   ( Qu₀ Qu₁ Qu₂ … ) ⎜⎜   Qv₁    ⎟·⎜  S₁   ⎟⎟ =
            //                     ⎜⎜      Qv₂ ⎟ ⎜    S₂ ⎟⎟
            //                     ⎝⎝         …⎠ ⎝      …⎠⎠
            //
            // Since diag(Qv_i) is orthogonal, it can be omitted for row bases
            // computation, leaving
            //
            //                     ⎛S₀     ⎞'                 
            //   ( Qu₀ Qu₁ Qu₂ … ) ⎜  S₁   ⎟ = ( Qu₀·S₀' Qu₁·S₁' Qu₂·S₂' … )
            //                     ⎜    S₂ ⎟                  
            //                     ⎝      …⎠                  
            //
            // of which a column basis is computed.
            //
            // Also: S_i is scaled with respect to spectral norm.
            //

            //
            // form U = ( Qu₀·S₀' Qu₁·S₁' Qu₂·S₁' … )
            //
            
            size_t  nrows_U = is.size();
            size_t  ncols_U = 0;

            for ( auto &  R : matrices )
                ncols_U += R->rank();

            auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
            size_t  pos = 0;

            for ( auto &  R : matrices )
            {
                // R = U·V' = Qu·Ru×Rv'·Qv'
                auto  U_i  = blas::copy( R->U() );
                auto  V_i  = blas::copy( R->V() );
                auto  Ru_i = blas::matrix< value_t >();
                auto  Rv_i = blas::matrix< value_t >();
                auto  k    = R->rank();
                
                blas::qr( U_i, Ru_i, false );
                blas::qr( V_i, Rv_i, false );

                auto  S_i    = blas::prod( Ru_i, blas::adjoint( Rv_i ) );
                auto  norm_i = blas::norm_2( S_i );

                blas::scale( value_t(1) / norm_i, S_i );

                auto  US_i   = blas::prod( U_i, blas::adjoint( S_i ) );
                auto  U_sub  = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( US_i, U_sub );
                
                pos += k;
            }// for

            auto  Us = blas::vector< real_t >();
            auto  Un = basisapx.column_basis( U, acc, & Us );

            #endif
            
            // finally assign to cluster basis object
            rowcb_map.at( is )->set_basis( std::move( Un ), std::move( Us ) );

            if ( compress )
                rowcb_map.at( is )->compress( acc );
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
                auto  V_i = blas::copy( R->V() );
                auto  U_i = blas::copy( R->U() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( U_i, R_i, false );

                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VR_i, V_sub );
                
                pos += k;
            }// for

            auto  Vs = blas::vector< real_t >();
            auto  Vn = basisapx.column_basis( V, acc, & Vs );

            #else
            
            //
            // compute scaled column basis for
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
            //   ( Qv₀·Rv₀×(Qu₀·Ru₀)'   Qv₁·Rv₁×(Qu₁·Ru₁)'   Qv₂·Rv₂×(Qu₂·Ru₂)'  … ) =
            //
            //   ( Qv₀·(Rv₀×Ru₀')·Qu₀'  Qv₁·(Rv₁×Ru₁')·Qu₁'  Qv₂·(Rv₂×Ru₂')·Qu₂' … ) =
            //
            //   ( Qv₀·S₀·Qu₀'          Qv₁·S₁·Qu₁'          Qv₂·S₂·Qu₂'         … ) =   with  S_i = Rv_i × Ru_i'
            //
            //                     ⎛ Qu₀·S₀'                ⎞'
            //   ( Qv₀ Qv₁ Qv₂ … ) ⎜        Qu₁·S₁'         ⎟ =
            //                     ⎜               Qu₂·S₂'  ⎟
            //                     ⎝                      … ⎠
            //
            //                     ⎛⎛Qu₀       ⎞ ⎛S₀'       ⎞⎞'
            //   ( Qv₀ Qv₁ Qv₂ … ) ⎜⎜   Qu₁    ⎟·⎜   S₁'    ⎟⎟ =
            //                     ⎜⎜      Qu₂ ⎟ ⎜      S₂' ⎟⎟
            //                     ⎝⎝         …⎠ ⎝         …⎠⎠
            //
            // Since diag(Qu_i) is orthogonal, it can be omitted for column bases
            // computation, leaving
            //
            //                     ⎛S₀'       ⎞'                
            //   ( Qv₀ Qv₁ Qv₂ … ) ⎜   S₁'    ⎟ = ( Qv₀·S₀ Qv₁·S₁ Qv₂·S₂ … )
            //                     ⎜      S₂' ⎟                
            //                     ⎝         …⎠
            //
            // of which a column basis is computed.
            //
            // Also: the matrices S_i are scaled with respect to their spectral norm.
            //

            //
            // form matrix V = ( Qv₀·S₀ Qv₁·S₁ Qv₂·S₂ … )
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
                auto  V_i  = blas::copy( R->V() );
                auto  U_i  = blas::copy( R->U() );
                auto  Rv_i = blas::matrix< value_t >();
                auto  Ru_i = blas::matrix< value_t >();
                auto  k    = R->rank();
                
                blas::qr( V_i, Rv_i, false );
                blas::qr( U_i, Ru_i, false );

                auto  S_i    = blas::prod( Rv_i, blas::adjoint( Ru_i ) );
                auto  norm_i = blas::norm_2( S_i );

                blas::scale( value_t(1) / norm_i, S_i );
                
                auto  VS_i   = blas::prod( V_i, S_i );
                auto  V_sub  = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VS_i, V_sub );
                
                pos += k;
            }// for

            auto  Vs = blas::vector< real_t >();
            auto  Vn = basisapx.column_basis( V, acc, & Vs );

            #endif
            
            // finally assign to cluster basis object
            colcb_map.at( is )->set_basis( std::move( Vn ), std::move( Vs ) );

            if ( compress )
                colcb_map.at( is )->compress( acc );
        }// for

        //
        // now convert all blocks on this level
        //

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

            auto  UnU = blas::prod( blas::adjoint( Un ), R->U() );
            auto  VnV = blas::prod( blas::adjoint( Vn ), R->V() );
            auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

            auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                      R->col_is(),
                                                                                      *rowcb,
                                                                                      *colcb,
                                                                                      std::move( S ) );

            RU->set_id( R->id() );
            
            if ( compress )
                RU->compress( acc );
            
            // {
            //     auto  M1 = blas::prod( R->U(), blas::adjoint( R->V() ) );
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

    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
build_uniform_lvl_sep ( const Hpro::TBlockCluster *  bct,
                        const coeff_t &              coeff,
                        const lrapx_t &              lrapx,
                        const basisapx_t &           basisapx,
                        const accuracy &             acc,
                        const bool                   compress )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using real_t        = Hpro::real_type_t< value_t >;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< lrmatrix< value_t > * >, indexset_hash >;
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

    auto  nodes      = std::list< const Hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    while ( ! nodes.empty() )
    {
        auto  children = decltype( nodes )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::list< lrmatrix< value_t > * >();
        
        for ( auto  node : nodes )
        {
            auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

            // std::cout << node->id() << std::endl;
            
            if ( node->is_leaf() )
            {
                if ( node->is_adm() )
                {
                    M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( node, acc ) );

                    if ( hlr::matrix::is_lowrank( *M ) )
                    {
                        auto  R = ptrcast( M.get(), lrmatrix< value_t > );
                        
                        rowmap[ M->row_is() ].push_back( R );
                        colmap[ M->col_is() ].push_back( R );
                        lrmat.push_back( R );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + M->typestr() );
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
                        auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                        auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) );
                        
                        if ( compress )
                            DD->compress( acc );

                        M = std::move( DD );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + M->typestr() );
                }// else
            }// if
            else
            {
                // collect children
                for ( uint  i = 0; i < node->nrows(); ++i )
                    for ( uint  j = 0; j < node->ncols(); ++j )
                        if ( ! is_null( node->son( i, j ) ) )
                            children.push_back( node->son( i, j ) );

                M = std::make_unique< Hpro::TBlockMatrix< value_t > >( node );
        
                auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

                // make sure, block structure is correct
                if (( B->nblock_rows() != node->nrows() ) ||
                    ( B->nblock_cols() != node->ncols() ))
                    B->set_block_struct( node->nrows(), node->ncols() );

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

            //
            // compute column basis for
            //
            //   ( U₀·V₀'  U₁·V₁'  U₂·V₂'  … ) =
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
                auto  U_i = R->U();
                auto  V_i = blas::copy( R->V() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// for

            //
            // QR of S and computation of row basis
            //

            auto  Us = blas::vector< real_t >();
            auto  Un = basisapx.column_basis( U, acc, & Us );

            // finally assign to cluster basis object
            rowcb_map.at( is )->set_basis( std::move( Un ), std::move( Us ) );

            if ( compress )
                rowcb_map.at( is )->compress( acc );
        }// for

        //
        // construct column bases for all block columns constructed on this level
        //

        for ( auto  [ is, matrices ] : colmap )
        {
            if ( matrices.size() == 0 )
                continue;

            //
            // compute column basis for
            //
            //   ⎛U₀·V₀'⎞ 
            //   ⎜U₁·V₁'⎟
            //   ⎜U₂·V₂'⎟
            //   ⎝  …   ⎠
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
                auto  V_i = blas::copy( R->V() );
                auto  U_i = blas::copy( R->U() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( U_i, R_i, false );

                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VR_i, V_sub );
                
                pos += k;
            }// for

            auto  Vs = blas::vector< real_t >();
            auto  Vn = basisapx.column_basis( V, acc, & Vs );
            
            // finally assign to cluster basis object
            colcb_map.at( is )->set_basis( std::move( Vn ), std::move( Vs ) );

            if ( compress )
                colcb_map.at( is )->compress( acc );
        }// for

        //
        // now convert all blocks on this level
        //

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

            auto  Sr = blas::prod( blas::adjoint( Un ), R->U() );
            auto  Sc = blas::prod( blas::adjoint( Vn ), R->V() );
            auto  RU = std::make_unique< hlr::matrix::uniform_lr2matrix< value_t > >( R->row_is(), R->col_is(),
                                                                                      *rowcb, *colcb,
                                                                                      std::move( Sr ), std::move( Sc ) );

            RU->set_id( R->id() );
            
            if ( compress )
                RU->compress( acc );
            
            // replace standard lowrank block by uniform lowrank block
            R->parent()->replace_block( R, RU.release() );
            delete R;
        }// for
    }// while

    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

template < typename cluster_basis_t >
void
set_ids ( cluster_basis_t &  cb,
          int &              id )
{
    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
                set_ids( * cb.son( i ), id );
        }// for
    }// if

    // id(parent) > id(sons)
    cb.set_id( id++ );
}

//
// build lists of blocks in block row per cluster
//
inline
void
build_block_lists ( const Hpro::TBlockCluster *                                bt,
                    std::vector< std::list< const Hpro::TBlockCluster * > > &  blocks,
                    const bool                                                 adjoint,
                    const bool                                                 also_inner = false )
{
    HLR_ASSERT( ! is_null( bt ) );
    
    if ( bt->is_leaf() || also_inner )
    {
        auto  cl = ( adjoint ? bt->colcl() : bt->rowcl() );

        HLR_ASSERT( cl->id() >= 0 );
        
        blocks[ cl->id() ].push_back( bt );
    }// if

    for ( uint  i = 0; i < bt->nsons(); ++i )
    {
        if ( ! is_null( bt->son(i) ) )
            build_block_lists( bt->son(i), blocks, adjoint, also_inner );
    }// for
}

// template < typename coeff_t,
//            typename lrapx_t,
//            typename basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
// build_uniform_cl ( const Hpro::TBlockCluster *                                bt,
//                    const coeff_t &                                            coeff,
//                    const lrapx_t &                                            lrapx,
//                    const basisapx_t &                                         basisapx,
//                    const accuracy &                                           acc,
//                    const bool                                                 compress,
//                    std::vector< std::list< const Hpro::TBlockCluster * > > &  row_blocks,
//                    std::vector< std::list< const Hpro::TBlockCluster * > > &  col_blocks )
// {
//     using value_t       = typename coeff_t::value_t;
//     using real_t        = Hpro::real_type_t< value_t >;
//     using cluster       = Hpro::TCluster;
//     using block_cluster = Hpro::TBlockCluster;
//     using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
//     using matrix        = Hpro::TMatrix< value_t >;

//     //
//     // go level-wise through cluster trees, construct matrices for all row/column blocks
//     // and build corresponding cluster basis
//     //

//     auto        root_row     = bt->rowcl();
//     auto        root_col     = bt->colcl();
//     const auto  nrowcl       = root_row->id() + 1;
//     const auto  ncolcl       = root_col->id() + 1;
//     const auto  nblocks      = bt->id() + 1;
//     auto        lrmatrices   = std::vector< lrmatrix< value_t > * >( nblocks );            // allows access to lrmatrices via id
//     auto        parent_mat   = std::vector< Hpro::TBlockMatrix< value_t > * >( nblocks );  // stores pointer to parent matrix
//     auto        parent_rowcl = std::vector< cluster * >( nrowcl );                         // to access cluster parents
//     auto        parent_rowcb = std::vector< cluster_basis * >( nrowcl );                   // store pointer to parent of cluster basis
//     auto        parent_colcl = std::vector< cluster * >( ncolcl );
//     auto        parent_colcb = std::vector< cluster_basis * >( ncolcl );
//     auto        root_M       = std::unique_ptr< matrix * >();

//     auto        rowcls       = std::list< cluster * >{ root_row };
//     auto        colcls       = std::list< cluster * >{ root_col };

//     while ( ! ( rowcls.empty() && colcls.empty() ) )
//     {
//         for ( auto  rowcl : rowcls )
//         {
//             //
//             // build cluster basis object
//             //

//             auto  rowcb = std::make_unique< cluster_basis >( *rowcl );

//             rowcb->set_nsons( rowcl->nsons() );

//             // remember for sons
//             for ( uint  i = 0; i < rowcl->nsons(); ++i )
//             {
//                 if ( ! is_null( rowcl->son(i) ) )
//                 {
//                     parent_rowcl[ rowcl->son(i) ] = rowcl;
//                     parent_rowcb[ rowcl->son(i) ] = rowcb.get();
//                 }// if
//             }// for
            
//             // put into parent
//             auto  parent_cb = parent_rowcb[ rowcl->id() ];
//             auto  parent_cl = parent_rowcl[ rowcl->id() ];

//             HLR_ASSERT( is_null( parent_cb ) == is_null( parent_cl ) );

//             if ( ! is_null( parent_cl ) )
//             {
//                 for ( uint  i = 0; i < parent_cl->nsons(); ++i )
//                 {
//                     if ( parent_cl->son(i) == rowcl )
//                     {
//                         parent_cb->set_son( i, rowcb.get() );
//                         break;
//                     }// if
//                 }// for
//             }// if
                
//             //
//             // construct all matrix blocks in local list
//             //

//             {
//                 auto  blocks   = row_blocks[ rowcl->id() ];
//                 auto  lrblocks = std::list< lrmatrix< value_t > >();

//                 for ( auto  block : blocks )
//                 {
//                     auto  M = std::unique_ptr< matrix >();

//                     if ( block->is_leaf() )
//                     {
//                         if ( block->is_adm() )
//                         {
//                             M = lrapx.build( block, acc );

//                             if ( hlr::matrix::is_lowrank( *M ) )
//                             {
//                                 auto  R = ptrcast( M.get(), lrmatrix< value_t > );

//                                 // remember for cluster basis
//                                 lrblocks.push_back( R );
//                             }// if
//                             else
//                                 HLR_ERROR( "unsupported matrix type : " + M->typestr() );
//                         }// if
//                         else
//                         {
//                             M = coeff.build( block->is().row_is(), block->is().col_is() );
                        
//                             if ( hlr::matrix::is_dense( *M ) )
//                             {
//                                 // all is good
//                             }// if
//                             else if ( Hpro::is_dense( *M ) )
//                             {
//                                 auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
//                                 auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) );

//                                 if ( compress )
//                                     DD->compress( acc );

//                                 M = std::move( DD );
//                             }// if
//                             else
//                                 HLR_ERROR( "unsupported matrix type : " + M->typestr() );
//                         }// else
//                     }// if
//                     else
//                     {
//                         auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( block );

//                         // make sure, block structure is correct
//                         if (( B->nblock_rows() != block->nrows() ) || ( B->nblock_cols() != block->ncols() ))
//                             B->set_block_struct( block->nrows(), block->ncols() );

//                         // set matrix as parent for all sub blocks
//                         for ( uint  i = 0; i < block->nsons(); ++i )
//                         {
//                             if ( ! is_null( block->son(i) ) )
//                                 parent_mat[ block->son(i)->id() ] = B.get();
//                         }// for
//                     }// else

//                     M->set_id( block->id() );
//                     M->set_procs( block->procs() );

//                     //
//                     // insert into parent
//                     //

//                     auto  parent_M     = parent_mat[ M->id() ];
//                     auto  parent_block = block->parent();

//                     HLR_ASSERT( is_null( parent_block ) == is_null( parent_M ) );
                    
//                     if ( ! is_null( parent_block ) )
//                     {
//                         for ( uint  i = 0; i < parent_block->nrows(); ++i )
//                         {
//                             for ( uint  j = 0; j < parent_block->ncols(); ++j )
//                             {
//                                 if ( parent_block->son(i,j) == block )
//                                 {
//                                     parent_M->set_block( i, j, M.release() );
//                                     break;
//                                 }// if
//                             }// for
//                         }// for
//                     }// if
//                     else
//                     {
//                         //
//                         // without parent => root
//                         //

//                         M_root = std::move( M );
//                     }// else
//                 }// for

//                 //
//                 // now construct cluster basis
//                 //

//                 if ( ! lrblocks.empty() )
//                 {
//                     //
//                     // form U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
//                     //
            
//                     size_t  nrows_U = rowcl->size();
//                     size_t  ncols_U = 0;

//                     for ( auto  R : lrblocks )
//                         ncols_U += R->rank();

//                     auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
//                     size_t  pos = 0;

//                     for ( auto  R : lrblocks )
//                     {
//                         // R = U·V' = W·T·X'
//                         auto  U_i = R->U();
//                         auto  V_i = blas::copy( R->V() );
//                         auto  R_i = blas::matrix< value_t >();
//                         auto  k   = R->rank();
                
//                         blas::qr( V_i, R_i, false );

//                         auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
//                         auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

//                         blas::copy( UR_i, U_sub );
                
//                         pos += k;
//                     }// for

//                     //
//                     // QR of S and computation of row basis
//                     //

//                     auto  Us = blas::vector< real_t >();
//                     auto  Un = basisapx.column_basis( U, acc, & Us );

//                     rowcb->set_basis( Un, Us );
//                 }// if
//             }
//         }// for
//     }// while
// }

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
    using lrmat_map_t   = std::unordered_map< indexset, std::list< const lrmatrix< value_t > * >, indexset_hash >;
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

    rowcb_map[ A.row_is() ] = & rowcb_root;
    colcb_map[ A.col_is() ] = & colcb_root;

    while ( ! matrices.empty() )
    {
        auto  children = decltype( matrices )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::list< const lrmatrix< value_t > * >();
        
        for ( auto  mat : matrices )
        {
            auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
            
            if ( hlr::matrix::is_lowrank( mat ) )
            {
                auto  R = cptrcast( mat, lrmatrix< value_t > );
                        
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
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            children.push_back( B->block( i, j ) );

                M = B->copy_struct();

                // remember all block matrices for setting up hierarchy
                bmat_map[ mat->id() ] = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + mat->typestr() );

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
                    auto  mat_parent = mat->parent();
                    auto  M_parent   = bmat_map.at( mat_parent->id() );

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
        }// for

        matrices = std::move( children );
        
        //
        // construct row bases for all block rows constructed on this level
        //

        for ( auto  [ is, matrices ] : rowmap )
        {
            if ( matrices.size() == 0 )
                continue;

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
                auto  U_i = R->U();
                auto  V_i = blas::copy( R->V() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// for

            //
            // QR of S and computation of row basis
            //

            // std::cout << is.to_string() << " : " << U.nrows() << " x " << U.ncols() << std::endl;
            
            auto  Un = basisapx.column_basis( U, acc );

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
                auto  V_i = blas::copy( R->V() );
                auto  U_i = blas::copy( R->U() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( U_i, R_i, false );

                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VR_i, V_sub );
                
                pos += k;
            }// for

            // std::cout << is.to_string() << " : " << V.nrows() << " x " << V.ncols() << std::endl;
            
            auto  Vn = basisapx.column_basis( V, acc );
            
            // finally assign to cluster basis object
            colcb_map.at( is )->set_basis( std::move( Vn ) );
        }// for

        //
        // now convert all blocks on this level
        //

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

            auto  UnU = blas::prod( blas::adjoint( Un ), R->U() );
            auto  VnV = blas::prod( blas::adjoint( Vn ), R->V() );
            auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

            auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                      R->col_is(),
                                                                                      *rowcb,
                                                                                      *colcb,
                                                                                      std::move( S ) );

            RU->set_id( R->id() );
            
            // put uniform matrix in parent matrix
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
        }// for
    }// while

    return M_root;
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
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_uniform_rec ( const Hpro::TBlockCluster *                          bct,
                    const coeff_t &                                      coeff,
                    const lrapx_t &                                      lrapx,
                    const basisapx_t &                                   basisapx,
                    const accuracy &                                     acc,
                    shared_cluster_basis< typename coeff_t::value_t > &  rowcb,
                    shared_cluster_basis< typename coeff_t::value_t > &  colcb,
                    is_matrix_map_t< typename coeff_t::value_t > &       rowmap,
                    is_matrix_map_t< typename coeff_t::value_t > &       colmap )
{
    using value_t = typename coeff_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc ) );

            if ( hlr::matrix::is_lowrank( *M ) )
            {
                //
                // form U·V' = W·T·X' with orthogonal W/X
                //

                auto  R  = ptrcast( M.get(), hlr::matrix::lrmatrix< value_t > );
                auto  W  = R->U();
                auto  X  = R->V();
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                blas::qr( W, Rw );
                blas::qr( X, Rx );

                auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
                //
                // update cluster bases
                //

                auto  Us = blas::vector< real_t >(); // singular values corresponding to basis vectors
                auto  Vs = blas::vector< real_t >();
                
                auto  Un = hlr::uniform::detail::compute_extended_row_basis< value_t, basisapx_t >( rowcb, W, T, acc, basisapx, rowmap, nullptr, & Us );
                auto  Vn = hlr::uniform::detail::compute_extended_col_basis< value_t, basisapx_t >( colcb, T, X, acc, basisapx, colmap, nullptr, & Vs );

                hlr::uniform::detail::update_row_coupling( rowcb, Un, rowmap );
                hlr::uniform::detail::update_col_coupling( colcb, Vn, colmap );

                //
                // compute coupling matrix with new row/col bases Un/Vn
                //

                auto  UW = blas::prod( blas::adjoint( Un ), W );
                auto  VX = blas::prod( blas::adjoint( Vn ), X );
                auto  T1 = blas::prod( UW, T );
                auto  S  = blas::prod( T1, blas::adjoint( VX ) );

                // update bases in cluster bases objects (only now since Un/Vn are used before)
                rowcb.set_basis( std::move( Un ), std::move( Us ) );
                colcb.set_basis( std::move( Vn ), std::move( Vs ) );
                
                auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), rowcb, colcb, std::move( S ) );

                rowmap[ RU->row_is() ].push_back( RU.get() );
                colmap[ RU->col_is() ].push_back( RU.get() );

                // {// DEBUG {
                //     auto  M1 = blas::prod( U, blas::adjoint( V ) );
                //     auto  T2 = blas::prod( W, T );
                //     auto  M2 = blas::prod( T2, blas::adjoint( X ) );
                //     auto  T3 = blas::prod( rowcb.basis(), RU->coeff() );
                //     auto  M3 = blas::prod( T3, blas::adjoint( colcb.basis() ) );

                //     blas::add( value_t(-1), M1, M2 );
                //     blas::add( value_t(-1), M1, M3 );

                //     std::cout << blas::norm_F( M2 ) / blas::norm_F( M1 ) << "    "
                //               << blas::norm_F( M3 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }
                
                M = std::move( RU );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );

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
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( bct->son( i, j ) ) )
                {
                    if ( is_null( rowcb_i ) )
                    {
                        rowcb_i = new shared_cluster_basis< value_t >( bct->son( i, j )->is().row_is() );
                        rowcb_i->set_nsons( bct->son( i, j )->rowcl()->nsons() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new shared_cluster_basis< value_t >( bct->son( i, j )->is().col_is() );
                        colcb_j->set_nsons( bct->son( i, j )->colcl()->nsons() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    auto  B_ij = build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, *rowcb_i, *colcb_j, rowmap, colmap );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
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

template < typename basisapx_t >
std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > >
build_uniform_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &   A,
                    const basisapx_t &                                      basisapx,
                    const accuracy &                                        acc,
                    shared_cluster_basis< typename basisapx_t::value_t > &  rowcb,
                    shared_cluster_basis< typename basisapx_t::value_t > &  colcb,
                    is_matrix_map_t< typename basisapx_t::value_t > &       rowmap,
                    is_matrix_map_t< typename basisapx_t::value_t > &       colmap )
{
    using value_t = typename basisapx_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // form U·V' = W·T·X' with orthogonal W/X
        //
        
        auto  R  = cptrcast( &A, lrmatrix< value_t > );
        auto  W  = blas::copy( R->U() );
        auto  X  = blas::copy( R->V() );
        auto  Rw = blas::matrix< value_t >();
        auto  Rx = blas::matrix< value_t >();
            
        blas::qr( W, Rw );
        blas::qr( X, Rx );

        auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
        //
        // update cluster bases
        //

        auto  Us = blas::vector< real_t >(); // singular values corresponding to basis vectors
        auto  Vs = blas::vector< real_t >();
            
        auto  Un = hlr::uniform::detail::compute_extended_row_basis< value_t, basisapx_t >( rowcb, W, T, acc, basisapx, rowmap, nullptr, & Us );
        auto  Vn = hlr::uniform::detail::compute_extended_col_basis< value_t, basisapx_t >( colcb, T, X, acc, basisapx, colmap, nullptr, & Vs );
            
        hlr::uniform::detail::update_row_coupling( rowcb, Un, rowmap );
        hlr::uniform::detail::update_col_coupling( colcb, Vn, colmap );

        //
        // compute coupling matrix with new row/col bases Un/Vn
        //

        auto  UW = blas::prod( blas::adjoint( Un ), W );
        auto  VX = blas::prod( blas::adjoint( Vn ), X );
        auto  T1 = blas::prod( UW, T );
        auto  S  = blas::prod( T1, blas::adjoint( VX ) );

        // update bases in cluster bases objects (only now since Un/Vn are used before)
        rowcb.set_basis( std::move( Un ), std::move( Us ) );
        colcb.set_basis( std::move( Vn ), std::move( Vs ) );
                
        auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), rowcb, colcb, std::move( S ) );

        // std::cout << R->id() << " " << R->rank() << " " << RU->row_rank() << " " << RU->col_rank() << std::endl;

        rowmap[ rowcb.is() ].push_back( RU.get() );
        colmap[ colcb.is() ].push_back( RU.get() );

        M = std::move( RU );
    }// if
    else if ( hlr::matrix::is_lowrank_sv( A ) )
    {
        //
        // matrix already is W·S·X' with orthogonal W/X
        //
        
        auto  R  = cptrcast( &A, lrsvmatrix< value_t > );
        auto  k  = R->rank();
        auto  W  = R->U();
        auto  X  = R->V();
        auto  T  = blas::diag< value_t >( R->S() );

        //
        // update cluster bases
        //

        auto  Us = blas::vector< real_t >(); // singular values corresponding to basis vectors
        auto  Vs = blas::vector< real_t >();
            
        auto  Un = hlr::uniform::detail::compute_extended_row_basis< value_t, basisapx_t >( rowcb, W, T, acc, basisapx, rowmap, nullptr, & Us );
        auto  Vn = hlr::uniform::detail::compute_extended_col_basis< value_t, basisapx_t >( colcb, T, X, acc, basisapx, colmap, nullptr, & Vs );
            
        hlr::uniform::detail::update_row_coupling( rowcb, Un, rowmap );
        hlr::uniform::detail::update_col_coupling( colcb, Vn, colmap );

        //
        // compute coupling matrix with new row/col bases Un/Vn
        //

        auto  UW = blas::prod( blas::adjoint( Un ), W );
        auto  VX = blas::prod( blas::adjoint( Vn ), X );

        blas::prod_diag_ip( UW, R->S() );
        
        auto  S  = blas::prod( UW, blas::adjoint( VX ) );

        // update bases in cluster bases objects (only now since Un/Vn are used before)
        rowcb.set_basis( std::move( Un ), std::move( Us ) );
        colcb.set_basis( std::move( Vn ), std::move( Vs ) );
                
        auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), rowcb, colcb, std::move( S ) );

        // std::cout << R->id() << " " << R->rank() << " " << RU->row_rank() << " " << RU->col_rank() << std::endl;

        rowmap[ rowcb.is() ].push_back( RU.get() );
        colmap[ colcb.is() ].push_back( RU.get() );

        M = std::move( RU );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BA );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                auto  A_ij    = BA->block( i, j );
                
                if ( ! is_null( A_ij ) )
                {
                    if ( is_null( rowcb_i ) )
                    {
                        rowcb_i = new shared_cluster_basis< value_t >( A_ij->row_is() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_blocked( A_ij ) && ( rowcb_i->nsons() == 0 ))
                        rowcb_i->set_nsons( cptrcast( A_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
                        
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new shared_cluster_basis< value_t >( A_ij->col_is() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    if ( is_blocked( A_ij ) && ( colcb_j->nsons() == 0 ))
                        colcb_j->set_nsons( cptrcast( A_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );

                    auto  B_ij = build_uniform_rec( *A_ij, basisapx, acc, *rowcb_i, *colcb_j, rowmap, colmap );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );

        M = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

//
// special construction in BLR2 format (blr clustering)
//
template < coefficient_function_type coeff_t,
           lowrank_approx_type       lrapx_t,
           approx::approximation_type        basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_blr2 ( const Hpro::TBlockCluster *  bc,
             const coeff_t &              coeff,
             const lrapx_t &              lrapx,
             const basisapx_t &           basisapx,
             const accuracy &             acc )
{
    using value_t = typename basisapx_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;

    HLR_ASSERT( ! is_null( bc ) );
    
    //
    // initialize empty cluster bases
    //

    auto  rowcb = std::make_unique< shared_cluster_basis< value_t > >( bc->rowis() );
    auto  colcb = std::make_unique< shared_cluster_basis< value_t > >( bc->colis() );

    rowcb->set_nsons( bc->nrows() );
    colcb->set_nsons( bc->ncols() );
    
    for ( size_t  i = 0; i < bc->nrows(); ++i )
    {
        auto  rowis_i = indexset();
        
        for ( size_t  j = 0; j < bc->ncols(); ++j )
        {
            if ( ! is_null( bc->son( i, j ) ) )
            {
                rowis_i = bc->son( i, j )->rowis();
                break;
            }// if
        }// for

        HLR_ASSERT( rowis_i.size() > 0 );
        
        auto  rowcb_i = std::make_unique< shared_cluster_basis< value_t > >( rowis_i );

        rowcb->set_son( i, rowcb_i.release() );
    }// for

    for ( size_t  j = 0; j < bc->ncols(); ++j )
    {
        auto  colis_j = indexset();
        
        for ( size_t  i = 0; i < bc->nrows(); ++i )
        {
            if ( ! is_null( bc->son( i, j ) ) )
            {
                colis_j = bc->son( i, j )->colis();
                break;
            }// if
        }// for
                
        HLR_ASSERT( colis_j.size() > 0 );
        
        auto  colcb_j = std::make_unique< shared_cluster_basis< value_t > >( colis_j );

        colcb->set_son( j, colcb_j.release() );
    }// for

    //
    // construct blocks and update bases
    //

    auto  B       = std::make_unique< Hpro::TBlockMatrix< value_t > >( bc->rowis(), bc->colis() );
    auto  weights = tensor2< real_t >( bc->nrows(), bc->ncols() );

    B->set_block_struct( bc->nrows(), bc->ncols() );
    
    for ( size_t  i = 0; i < bc->nrows(); ++i )
    {
        for ( size_t  j = 0; j < bc->ncols(); ++j )
        {
            auto  bc_ij = bc->son( i, j );

            if ( is_null( bc_ij ) )
                continue;

            auto  B_ij    = std::unique_ptr< Hpro::TMatrix< value_t > >();
            auto  rowcb_i = rowcb->son( i );
            auto  colcb_j = colcb->son( j );
            
            if ( bc_ij->is_adm() )
            {
                B_ij = lrapx.build( bc_ij, acc );
                
                if ( ! hlr::matrix::is_lowrank( *B_ij ) )
                    HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
                    
                //
                // form U·V' = W·T·X' with orthogonal W/X
                //

                auto  R  = ptrcast( B_ij.get(), hlr::matrix::lrmatrix< value_t > );
                auto  W  = R->U();
                auto  X  = R->V();
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                blas::qr( W, Rw );
                blas::qr( X, Rx );

                auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );

                // remember norm of block as weight for bases updates
                weights(i,j) = norm::spectral( T );
                
                //
                // compute extended row cluster basis
                // - for details see "compute_extended_row_basis"
                //

                auto  Un = blas::matrix< value_t >();
                
                {
                    size_t  nrows_S = T.ncols();

                    for ( size_t  jj = 0; jj < j; ++jj )
                    {
                        auto  B_ij = B->block( i, jj );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                            nrows_S += cptrcast( B_ij, uniform_lrmatrix< value_t > )->col_rank();
                    }// for

                    if ( nrows_S == T.ncols() )
                        Un = std::move( blas::copy( W ) );
                    else
                    {
                        auto    U   = rowcb_i->basis();
                        auto    Ue  = blas::join_row< value_t >( { U, W } );
                        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
                        size_t  pos = 0;

                        for ( size_t  jj = 0; jj < j; ++jj )
                        {
                            auto  B_ij = B->block( i, jj );

                            if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                            {
                                const auto  R_ij  = cptrcast( B_ij, uniform_lrmatrix< value_t > );
                                const auto  rank  = R_ij->col_rank();
                                auto        S_ij  = blas::copy( R_ij->coeff() );
                                auto        w_ij  = weights(i,jj);
                                auto        S_sub = blas::matrix< value_t >( S,
                                                                             blas::range( pos, pos + rank-1 ),
                                                                             blas::range( 0, U.ncols() - 1 ) );

                                if ( w_ij != real_t(0) )
                                    blas::scale( value_t(1) / w_ij, S_ij );
            
                                blas::copy( blas::adjoint( S_ij ), S_sub );
                                pos += rank;
                            }// else
                        }// for

                        {
                            const auto  rank  = T.ncols();
                            auto        S_ij  = blas::copy( T );
                            auto        w_ij  = weights(i,j);
                            auto        S_sub = blas::matrix< value_t >( S,
                                                                         blas::range( pos, pos + rank-1 ),
                                                                         blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
                            if ( w_ij != real_t(0) )
                                blas::scale( value_t(1) / w_ij, S_ij );
            
                            blas::copy( blas::adjoint( S_ij ), S_sub );
                        }
        
                        // apply QR to extended coupling and compute column basis approximation
                        auto  R = blas::matrix< value_t >();
        
                        blas::qr( S, R, false );

                        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );

                        Un = basisapx.column_basis( UeR, acc );
                    }// else
                }

                //
                // compute extended column cluster basis
                //

                auto  Vn = blas::matrix< value_t >();

                {
                    size_t  nrows_S = T.nrows();
    
                    for ( size_t  ii = 0; ii < i; ++ii )
                    {
                        auto  B_ij = B->block( ii, j );
                    
                        if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                            nrows_S += cptrcast( B_ij, uniform_lrmatrix< value_t > )->row_rank();
                    }// for

                    if ( nrows_S == T.nrows() )
                    {
                        Vn = std::move( blas::copy( X ) );
                    }// if
                    else
                    {
                        auto    V   = colcb_j->basis();
                        auto    Ve  = blas::join_row< value_t >( { V, X } );
                        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
                        size_t  pos = 0;

                        for ( size_t  ii = 0; ii < i; ++ii )
                        {
                            auto  B_ij = B->block( ii, j );

                            if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                            {
                                const auto  R_ij  = cptrcast( B_ij, uniform_lrmatrix< value_t > );
                                const auto  rank  = R_ij->row_rank();
                                auto        S_ij  = blas::copy( R_ij->coeff() );
                                auto        w_ij  = weights(ii,j);
                                auto        S_sub = blas::matrix< value_t >( S,
                                                                             blas::range( pos, pos + rank-1 ),
                                                                             blas::range( 0, V.ncols() - 1 ) );

                                if ( w_ij != real_t(0) )
                                    blas::scale( value_t(1) / w_ij, S_ij );

                                blas::copy( S_ij, S_sub );
                                pos += rank;
                            }// else
                        }// for

                        {
                            const auto  rank  = T.nrows();
                            auto        S_ij  = blas::copy( T );
                            auto        w_ij  = weights(i,j);
                            auto        S_sub = blas::matrix< value_t >( S,
                                                                         blas::range( pos, pos + rank-1 ),
                                                                         blas::range( V.ncols(), Ve.ncols() - 1 ) );

                            if ( w_ij != real_t(0) )
                                blas::scale( value_t(1) / w_ij, S_ij );
                
                            blas::copy( S_ij, S_sub );
                            pos += rank;
                        }

                        // apply QR to extended coupling and compute column basis approximation
                        auto  R = blas::matrix< value_t >();

                        blas::qr( S, R, false );

                        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );

                        Vn = basisapx.column_basis( VeR, acc );
                    }// else
                }// for
                
                //
                // update couplings of previous blocks
                //

                if ( rowcb_i->rank() > 0 )
                {
                    auto  U  = rowcb_i->basis();
                    auto  TU = blas::prod( blas::adjoint( Un ), U );
                
                    for ( size_t  jj = 0; jj < j; ++jj )
                    {
                        auto  B_ij = B->block( i, jj );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                        {
                            auto  R_ij  = ptrcast( B_ij, uniform_lrmatrix< value_t > );
                            auto  Sn_ij = blas::prod( TU, R_ij->coupling() );

                            R_ij->set_coupling_unsafe( std::move( Sn_ij ) );
                        }// if
                    }// for
                }// if

                if ( colcb_j->rank() > 0 )
                {
                    auto  V  = colcb_j->basis();
                    auto  TV = blas::prod( blas::adjoint( Vn ), V );

                    for ( size_t  ii = 0; ii < i; ++ii )
                    {
                        auto  B_ij = B->block( ii, j );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank( B_ij ) )
                        {
                            auto  R_ij  = ptrcast( B_ij, uniform_lrmatrix< value_t > );
                            auto  Sn_ij = blas::prod( R_ij->coupling(), blas::adjoint( TV ) );

                            R_ij->set_coupling_unsafe( std::move( Sn_ij ) );
                        }// if
                    }// for
                }// if

                //
                // compute coupling matrix with new row/col bases Un/Vn
                //

                auto  UW = blas::prod( blas::adjoint( Un ), W );
                auto  VX = blas::prod( blas::adjoint( Vn ), X );
                auto  T1 = blas::prod( UW, T );
                auto  S  = blas::prod( T1, blas::adjoint( VX ) );

                // update bases in cluster bases objects (only now since Un/Vn are used before)
                rowcb_i->set_basis( std::move( Un ) );
                colcb_j->set_basis( std::move( Vn ) );
                
                auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), *rowcb_i, *colcb_j, std::move( S ) );

                // {// DEBUG {
                //     auto  M1 = blas::prod( U, blas::adjoint( V ) );
                //     auto  T2 = blas::prod( W, T );
                //     auto  M2 = blas::prod( T2, blas::adjoint( X ) );
                //     auto  T3 = blas::prod( rowcb.basis(), RU->coeff() );
                //     auto  M3 = blas::prod( T3, blas::adjoint( colcb.basis() ) );

                //     blas::add( value_t(-1), M1, M2 );
                //     blas::add( value_t(-1), M1, M3 );

                //     std::cout << blas::norm_F( M2 ) / blas::norm_F( M1 ) << "    "
                //               << blas::norm_F( M3 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }
                
                B_ij = std::move( RU );
            }// if
            else
            {
                B_ij = coeff.build( bc_ij->rowis(), bc_ij->colis() );
                
                if ( hlr::matrix::is_dense( *B_ij ) )
                {
                    // all is good
                }// if
                else if ( Hpro::is_dense( *B_ij ) )
                {
                    auto  D = ptrcast( B_ij.get(), Hpro::TDenseMatrix< value_t > );

                    B_ij = std::move( std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
            }// else

            B->set_block( i, j, B_ij.release() );
        }// for
    }// for

    B->set_id( bc->id() );
    B->set_procs( bc->procs() );
    
    return { std::move( rowcb ), std::move( colcb ), std::move( B ) };
}

template < coefficient_function_type coeff_t,
           lowrank_approx_type       lrapx_t,
           approx::approximation_type        basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_blr2_sep ( const Hpro::TBlockCluster *  bc,
                 const coeff_t &              coeff,
                 const lrapx_t &              lrapx,
                 const basisapx_t &           basisapx,
                 const accuracy &             acc,
                 const bool                   compress )
{
    using value_t = typename basisapx_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;

    HLR_ASSERT( ! is_null( bc ) );
    
    //
    // initialize empty cluster bases
    //

    auto  rowcb = std::make_unique< shared_cluster_basis< value_t > >( bc->rowis() );
    auto  colcb = std::make_unique< shared_cluster_basis< value_t > >( bc->colis() );

    rowcb->set_nsons( bc->nrows() );
    colcb->set_nsons( bc->ncols() );
    
    for ( size_t  i = 0; i < bc->nrows(); ++i )
    {
        auto  rowis_i = indexset();
        
        for ( size_t  j = 0; j < bc->ncols(); ++j )
        {
            if ( ! is_null( bc->son( i, j ) ) )
            {
                rowis_i = bc->son( i, j )->rowis();
                break;
            }// if
        }// for

        HLR_ASSERT( rowis_i.size() > 0 );
        
        auto  rowcb_i = std::make_unique< shared_cluster_basis< value_t > >( rowis_i );

        rowcb->set_son( i, rowcb_i.release() );
    }// for

    for ( size_t  j = 0; j < bc->ncols(); ++j )
    {
        auto  colis_j = indexset();
        
        for ( size_t  i = 0; i < bc->nrows(); ++i )
        {
            if ( ! is_null( bc->son( i, j ) ) )
            {
                colis_j = bc->son( i, j )->colis();
                break;
            }// if
        }// for
                
        HLR_ASSERT( colis_j.size() > 0 );
        
        auto  colcb_j = std::make_unique< shared_cluster_basis< value_t > >( colis_j );

        colcb->set_son( j, colcb_j.release() );
    }// for

    //
    // construct blocks and update bases
    //

    auto  B       = std::make_unique< Hpro::TBlockMatrix< value_t > >( bc->rowis(), bc->colis() );
    auto  weights = tensor2< real_t >( bc->nrows(), bc->ncols() );

    B->set_block_struct( bc->nrows(), bc->ncols() );
    
    for ( size_t  i = 0; i < bc->nrows(); ++i )
    {
        for ( size_t  j = 0; j < bc->ncols(); ++j )
        {
            auto  bc_ij = bc->son( i, j );

            if ( is_null( bc_ij ) )
                continue;

            auto  B_ij    = std::unique_ptr< Hpro::TMatrix< value_t > >();
            auto  rowcb_i = rowcb->son( i );
            auto  colcb_j = colcb->son( j );
            
            if ( bc_ij->is_adm() )
            {
                B_ij = lrapx.build( bc_ij, acc );
                
                if ( ! hlr::matrix::is_lowrank( *B_ij ) )
                    HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
                    
                //
                // form U·V' = W·T·X' with orthogonal W/X
                //

                auto  R  = ptrcast( B_ij.get(), hlr::matrix::lrmatrix< value_t > );
                auto  W  = R->U();
                auto  X  = R->V();
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                blas::qr( W, Rw );
                blas::qr( X, Rx );

                auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
                // remember norm of block as weight for bases updates
                weights(i,j) = norm::spectral( T );
                
                //
                // compute extended row cluster basis
                // - for details see "compute_extended_row_basis"
                //

                auto  Us = blas::vector< real_t >();
                auto  Un = blas::matrix< value_t >();
                
                {
                    size_t  nrows_S = T.ncols();

                    for ( size_t  jj = 0; jj < j; ++jj )
                    {
                        auto  B_ij = B->block( i, jj );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                            nrows_S += cptrcast( B_ij, uniform_lr2matrix< value_t > )->col_rank();
                    }// for

                    if ( nrows_S == T.ncols() )
                    {
                        Un = std::move( blas::copy( W ) );
                        Us = blas::sv( T );
                    }// if
                    else
                    {
                        auto    U   = rowcb_i->basis();
                        auto    Ue  = blas::join_row< value_t >( { U, W } );
                        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
                        size_t  pos = 0;

                        for ( size_t  jj = 0; jj < j; ++jj )
                        {
                            auto  B_ij = B->block( i, jj );

                            if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                            {
                                const auto  R_ij  = cptrcast( B_ij, uniform_lr2matrix< value_t > );
                                const auto  rank  = R_ij->col_rank();
                                auto        S_ij  = blas::prod( R_ij->col_coupling(), blas::adjoint( R_ij->row_coupling() ) );
                                auto        w_ij  = weights(i,jj);
                                auto        S_sub = blas::matrix< value_t >( S,
                                                                             blas::range( pos, pos + rank-1 ),
                                                                             blas::range( 0, U.ncols() - 1 ) );

                                if ( w_ij != real_t(0) )
                                    blas::scale( value_t(1) / w_ij, S_ij );
            
                                blas::copy( S_ij, S_sub );
                                pos += rank;
                            }// else
                        }// for

                        {
                            const auto  rank  = T.ncols();
                            auto        S_ij  = blas::copy( T );
                            auto        w_ij  = weights(i,j);
                            auto        S_sub = blas::matrix< value_t >( S,
                                                                         blas::range( pos, pos + rank-1 ),
                                                                         blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
                            if ( w_ij != real_t(0) )
                                blas::scale( value_t(1) / w_ij, S_ij );
            
                            blas::copy( blas::adjoint( S_ij ), S_sub );
                        }
        
                        // apply QR to extended coupling and compute column basis approximation
                        auto  R = blas::matrix< value_t >();
        
                        blas::qr( S, R, false );

                        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );

                        Un = basisapx.column_basis( UeR, acc );
                    }// else
                }

                //
                // compute extended column cluster basis
                //

                auto  Vs = blas::vector< real_t >();
                auto  Vn = blas::matrix< value_t >();

                {
                    size_t  nrows_S = T.nrows();
    
                    for ( size_t  ii = 0; ii < i; ++ii )
                    {
                        auto  B_ij = B->block( ii, j );
                    
                        if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                            nrows_S += cptrcast( B_ij, uniform_lr2matrix< value_t > )->row_rank();
                    }// for

                    if ( nrows_S == T.nrows() )
                    {
                        Vn = std::move( blas::copy( X ) );
                        Vs = blas::sv( T );
                    }// if
                    else
                    {
                        auto    V   = colcb_j->basis();
                        auto    Ve  = blas::join_row< value_t >( { V, X } );
                        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
                        size_t  pos = 0;

                        for ( size_t  ii = 0; ii < i; ++ii )
                        {
                            auto  B_ij = B->block( ii, j );

                            if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                            {
                                const auto  R_ij  = cptrcast( B_ij, uniform_lr2matrix< value_t > );
                                const auto  rank  = R_ij->row_rank();
                                auto        S_ij  = blas::prod( R_ij->row_coupling(), blas::adjoint( R_ij->col_coupling() ) );
                                auto        w_ij  = weights(ii,j);
                                auto        S_sub = blas::matrix< value_t >( S,
                                                                             blas::range( pos, pos + rank-1 ),
                                                                             blas::range( 0, V.ncols() - 1 ) );

                                if ( w_ij != real_t(0) )
                                    blas::scale( value_t(1) / w_ij, S_ij );

                                blas::copy( S_ij, S_sub );
                                pos += rank;
                            }// else
                        }// for

                        {
                            const auto  rank  = T.nrows();
                            auto        S_ij  = blas::copy( T );
                            auto        w_ij  = weights(i,j);
                            auto        S_sub = blas::matrix< value_t >( S,
                                                                         blas::range( pos, pos + rank-1 ),
                                                                         blas::range( V.ncols(), Ve.ncols() - 1 ) );

                            if ( w_ij != real_t(0) )
                                blas::scale( value_t(1) / w_ij, S_ij );
                
                            blas::copy( S_ij, S_sub );
                            pos += rank;
                        }

                        // apply QR to extended coupling and compute column basis approximation
                        auto  R = blas::matrix< value_t >();

                        blas::qr( S, R, false );

                        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );

                        Vn = basisapx.column_basis( VeR, acc );
                    }// else
                }// for
                
                //
                // update couplings of previous blocks
                //

                if ( rowcb_i->rank() > 0 )
                {
                    auto  U  = rowcb_i->basis();
                    auto  TU = blas::prod( blas::adjoint( Un ), U );
                
                    for ( size_t  jj = 0; jj < j; ++jj )
                    {
                        auto  B_ij = B->block( i, jj );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                        {
                            auto  R_ij  = ptrcast( B_ij, uniform_lr2matrix< value_t > );
                            auto  Sn_ij = blas::prod( TU, R_ij->row_coupling() );

                            R_ij->set_row_coupling_unsafe( std::move( Sn_ij ) );
                        }// if
                    }// for
                }// if

                if ( colcb_j->rank() > 0 )
                {
                    auto  V  = colcb_j->basis();
                    auto  TV = blas::prod( blas::adjoint( Vn ), V );

                    for ( size_t  ii = 0; ii < i; ++ii )
                    {
                        auto  B_ij = B->block( ii, j );
                        
                        if ( ! is_null( B_ij ) && is_uniform_lowrank2( B_ij ) )
                        {
                            auto  R_ij  = ptrcast( B_ij, uniform_lr2matrix< value_t > );
                            auto  Sn_ij = blas::prod( TV, R_ij->col_coupling() );

                            R_ij->set_col_coupling_unsafe( std::move( Sn_ij ) );
                        }// if
                    }// for
                }// if

                //
                // compute coupling matrix with new row/col bases Un/Vn
                //

                auto  UW    = blas::prod( blas::adjoint( Un ), W );
                auto  VX    = blas::prod( blas::adjoint( Vn ), X );
                auto  S_row = blas::prod( UW, Rw );
                auto  S_col = blas::prod( VX, Rx );

                // update bases in cluster bases objects (only now since Un/Vn are used before)
                rowcb_i->set_basis( std::move( Un ) );
                colcb_j->set_basis( std::move( Vn ) );
                
                auto  RU = std::make_unique< uniform_lr2matrix< value_t > >( R->row_is(), R->col_is(), *rowcb_i, *colcb_j, std::move( S_row ), std::move( S_col ) );

                // {// DEBUG {
                //     auto  M1 = blas::prod( U, blas::adjoint( V ) );
                //     auto  T2 = blas::prod( W, T );
                //     auto  M2 = blas::prod( T2, blas::adjoint( X ) );
                //     auto  T3 = blas::prod( rowcb.basis(), RU->coeff() );
                //     auto  M3 = blas::prod( T3, blas::adjoint( colcb.basis() ) );

                //     blas::add( value_t(-1), M1, M2 );
                //     blas::add( value_t(-1), M1, M3 );

                //     std::cout << blas::norm_F( M2 ) / blas::norm_F( M1 ) << "    "
                //               << blas::norm_F( M3 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }
                
                B_ij = std::move( RU );
            }// if
            else
            {
                B_ij = coeff.build( bc_ij->rowis(), bc_ij->colis() );
                
                if ( hlr::matrix::is_dense( *B_ij ) )
                {
                    // all is good
                }// if
                else if ( Hpro::is_dense( *B_ij ) )
                {
                    auto  D = ptrcast( B_ij.get(), Hpro::TDenseMatrix< value_t > );

                    B_ij = std::move( std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) ) );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );

                if ( compress )
                    ptrcast( B_ij.get(), dense_matrix< value_t > )->compress( acc );
            }// else

            B->set_block( i, j, B_ij.release() );
        }// for
    }// for

    B->set_id( bc->id() );
    B->set_procs( bc->procs() );
    
    return { std::move( rowcb ), std::move( colcb ), std::move( B ) };
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_blr2 ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
             const basisapx_t &                                     basisapx,
             const accuracy &                                       acc )
{
    using value_t = typename basisapx_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;

    if ( ! is_blocked( A ) )
        HLR_ERROR( "TODO" );
    
    auto  B = cptrcast( &A, Hpro::TBlockMatrix< value_t > );

    //
    // construct row cluster bases for each block row
    //

    auto  rowcb = std::make_unique< shared_cluster_basis< value_t > >( A.row_is() );

    rowcb->set_nsons( B->nblock_rows() );
    
    for ( size_t  i = 0; i < B->nblock_rows(); ++i )
    {
        //
        // determine rank of extended cluster basis
        //

        auto  rowis = indexset();
        bool  first = true;
        uint  k     = 0;
        
        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( first )
            {
                rowis = B_ij->row_is();
                first = false;
            }// if
            
            if ( hlr::matrix::is_lowrank( B_ij ) )
                k += cptrcast( B_ij, lrmatrix< value_t > )->rank();
        }// for

        //
        // build extended cluster basis
        //
        //   U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
        //
        // with R_i from Q_V R_i = V_i
        // (Q_V can be omitted since orthogonal)
        //
        
        auto  U   = blas::matrix< value_t >( rowis.size(), k );
        uint  pos = 0;

        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  U_i = R->U();
                auto  V_i = blas::copy( R->V() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// if
        }// for

        //
        // truncate extended basis to form cluster basis
        //

        auto  sv      = blas::vector< real_t >();
        auto  Un      = basisapx.column_basis( U, acc, & sv );
        auto  rowcb_i = std::make_unique< shared_cluster_basis< value_t > >( rowis );

        rowcb_i->set_basis( std::move( Un ), std::move( sv ) );
        rowcb->set_son( i, rowcb_i.release() );
    }// for

    //
    // construct column cluster bases for each block column
    //

    auto  colcb = std::make_unique< shared_cluster_basis< value_t > >( A.col_is() );

    colcb->set_nsons( B->nblock_cols() );
    
    for ( size_t  j = 0; j < B->nblock_cols(); ++j )
    {
        //
        // determine rank of extended cluster basis
        //

        auto  colis = indexset();
        bool  first = true;
        uint  k     = 0;
        
        for ( size_t  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( first )
            {
                colis = B_ij->col_is();
                first = false;
            }// if
            
            if ( hlr::matrix::is_lowrank( B_ij ) )
                k += cptrcast( B_ij, lrmatrix< value_t > )->rank();
        }// for

        //
        // build extended cluster basis
        //
        //   V = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
        //
        // with R_i from Q_U R_i = U_i
        // (Q_U can be omitted since orthogonal)
        //
        
        auto  V   = blas::matrix< value_t >( colis.size(), k );
        uint  pos = 0;

        for ( size_t  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  V_i = blas::copy( R->V() );
                auto  U_i = blas::copy( R->U() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( U_i, R_i, false );

                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VR_i, V_sub );
                
                pos += k;
            }// if
        }// for

        //
        // truncate extended basis to form cluster basis
        //

        auto  sv      = blas::vector< real_t >();
        auto  Vn      = basisapx.column_basis( V, acc, & sv );
        auto  colcb_j = std::make_unique< shared_cluster_basis< value_t > >( colis );

        colcb_j->set_basis( std::move( Vn ), std::move( sv ) );
        colcb->set_son( j, colcb_j.release() );
    }// for

    //
    // build uniform H-matrix by converting all lowrank blocks to uniform blocks
    //
    
    auto  M = std::make_unique< Hpro::TBlockMatrix< value_t > >( A.row_is(), A.col_is() );

    M->copy_struct_from( B );

    for ( size_t  i = 0; i < B->nblock_rows(); ++i )
    {
        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                //
                // R = U·V' ≈ Un (Un' U V' Vn) Vn'
                //          = Un S Vn'  with  S = Un' U V' Vn
                //

                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  Un  = rowcb->son( i )->basis();
                auto  Vn  = colcb->son( j )->basis();
                auto  UnU = blas::prod( blas::adjoint( Un ), R->U() );
                auto  VnV = blas::prod( blas::adjoint( Vn ), R->V() );
                auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

                auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                          R->col_is(),
                                                                                          * ( rowcb->son( i ) ),
                                                                                          * ( colcb->son( j ) ),
                                                                                          std::move( S ) );

                M->set_block( i, j, RU.release() );
            }// if
            else if ( hlr::matrix::is_dense( B_ij ) )
            {
                auto  D  = cptrcast( B_ij, dense_matrix< value_t > );
                auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( blas::copy( D->mat() ) ) );

                M->set_block( i, j, DD.release() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
        }// for
    }// for

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

template < typename basisapx_t >
blas::matrix< typename basisapx_t::value_t >
row_basis_reduction ( const uint  lb,
                      const uint  ub,
                      std::list< const lrmatrix< typename basisapx_t::value_t > * > &  matrices,
                      const basisapx_t &  basisapx,
                      const accuracy &    acc )
{
    using value_t = typename basisapx_t::value_t;
    
    if ( ub == lb )
    {
        auto  R   = matrices[ lb ];
        auto  U_i = R->U();
        auto  V_i = blas::copy( R->V() );
        auto  R_i = blas::matrix< value_t >();
        auto  k   = R->rank();
                
        blas::qr( V_i, R_i, false );

        // auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
        // auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

        // blas::copy( UR_i, U_sub );
    }// if
    else
    {
        const auto  mid = ( ub + lb ) / 2;
        auto        B0  = row_basis_reduction( lb, mid,   matrices, basisapx, acc );
        auto        B1  = row_basis_reduction( mid+1, ub, matrices, basisapx, acc );

        
    }// else
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_blr2_red ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                 const basisapx_t &                                     basisapx,
                 const accuracy &                                       acc )
{
    using value_t = typename basisapx_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    using namespace hlr::matrix;

    if ( ! is_blocked( A ) )
        HLR_ERROR( "TODO" );
    
    auto  B = cptrcast( &A, Hpro::TBlockMatrix< value_t > );

    //
    // construct row cluster bases for each block row
    //

    auto  rowcb = std::make_unique< shared_cluster_basis< value_t > >( A.row_is() );

    rowcb->set_nsons( B->nblock_rows() );
    
    for ( size_t  i = 0; i < B->nblock_rows(); ++i )
    {
        //
        // determine rank of extended cluster basis
        //

        auto  rowis = indexset();
        bool  first = true;
        uint  k     = 0;
        
        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( first )
            {
                rowis = B_ij->row_is();
                first = false;
            }// if
            
            if ( hlr::matrix::is_lowrank( B_ij ) )
                k += cptrcast( B_ij, lrmatrix< value_t > )->rank();
        }// for

        //
        // build extended cluster basis
        //
        //   U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
        //
        // with R_i from Q_V R_i = V_i
        // (Q_V can be omitted since orthogonal)
        //
        
        auto  U   = blas::matrix< value_t >( rowis.size(), k );
        uint  pos = 0;

        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  U_i = R->U();
                auto  V_i = blas::copy( R->V() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// if
        }// for

        //
        // truncate extended basis to form cluster basis
        //

        auto  sv      = blas::vector< real_t >();
        auto  Un      = basisapx.column_basis( U, acc, & sv );
        auto  rowcb_i = std::make_unique< shared_cluster_basis< value_t > >( rowis );

        rowcb_i->set_basis( std::move( Un ), std::move( sv ) );
        rowcb->set_son( i, rowcb_i.release() );
    }// for

    //
    // construct column cluster bases for each block column
    //

    auto  colcb = std::make_unique< shared_cluster_basis< value_t > >( A.col_is() );

    colcb->set_nsons( B->nblock_cols() );
    
    for ( size_t  j = 0; j < B->nblock_cols(); ++j )
    {
        //
        // determine rank of extended cluster basis
        //

        auto  colis = indexset();
        bool  first = true;
        uint  k     = 0;
        
        for ( size_t  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( first )
            {
                colis = B_ij->col_is();
                first = false;
            }// if
            
            if ( hlr::matrix::is_lowrank( B_ij ) )
                k += cptrcast( B_ij, lrmatrix< value_t > )->rank();
        }// for

        //
        // build extended cluster basis
        //
        //   V = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
        //
        // with R_i from Q_U R_i = U_i
        // (Q_U can be omitted since orthogonal)
        //
        
        auto  V   = blas::matrix< value_t >( colis.size(), k );
        uint  pos = 0;

        for ( size_t  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  V_i = blas::copy( R->V() );
                auto  U_i = blas::copy( R->U() );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( U_i, R_i, false );

                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VR_i, V_sub );
                
                pos += k;
            }// if
        }// for

        //
        // truncate extended basis to form cluster basis
        //

        auto  sv      = blas::vector< real_t >();
        auto  Vn      = basisapx.column_basis( V, acc, & sv );
        auto  colcb_j = std::make_unique< shared_cluster_basis< value_t > >( colis );

        colcb_j->set_basis( std::move( Vn ), std::move( sv ) );
        colcb->set_son( j, colcb_j.release() );
    }// for

    //
    // build uniform H-matrix by converting all lowrank blocks to uniform blocks
    //
    
    auto  M = std::make_unique< Hpro::TBlockMatrix< value_t > >( A.row_is(), A.col_is() );

    M->copy_struct_from( B );

    for ( size_t  i = 0; i < B->nblock_rows(); ++i )
    {
        for ( size_t  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );

            if ( is_null( B_ij ) )
                continue;

            if ( hlr::matrix::is_lowrank( B_ij ) )
            {
                //
                // R = U·V' ≈ Un (Un' U V' Vn) Vn'
                //          = Un S Vn'  with  S = Un' U V' Vn
                //

                auto  R   = cptrcast( B_ij, lrmatrix< value_t > );
                auto  Un  = rowcb->son( i )->basis();
                auto  Vn  = colcb->son( j )->basis();
                auto  UnU = blas::prod( blas::adjoint( Un ), R->U() );
                auto  VnV = blas::prod( blas::adjoint( Vn ), R->V() );
                auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

                auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                          R->col_is(),
                                                                                          * ( rowcb->son( i ) ),
                                                                                          * ( colcb->son( j ) ),
                                                                                          std::move( S ) );

                M->set_block( i, j, RU.release() );
            }// if
            else if ( hlr::matrix::is_dense( B_ij ) )
            {
                auto  D  = cptrcast( B_ij, dense_matrix< value_t > );
                auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( blas::copy( D->mat() ) ) );

                M->set_block( i, j, DD.release() );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + B_ij->typestr() );
        }// for
    }// for

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// set up initial recursive structure of cluster bases
//
template < typename value_t >
void
init_cluster_bases ( const Hpro::TMatrix< value_t > &   M,
                     shared_cluster_basis< value_t > &  rowcb,
                     shared_cluster_basis< value_t > &  colcb,
                     int &                              row_id,
                     int &                              col_id )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
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
                        
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new shared_cluster_basis< value_t >( M_ij->col_is() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                        colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
                }// if
            }// for
        }// for

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
                    init_cluster_bases( *B->block( i, j ), *rowcb_i, *colcb_j, row_id, col_id );
            }// for
        }// for
    }// if

    // set IDs last to ensure id(parent) > id(sons)
    if ( rowcb.id() == -1 ) rowcb->set_id( row_id++ );
    if ( colcb.id() == -1 ) colcb->set_id( col_id++ );
}

//
// Build mapping from index set to set of lowrank matrices in block row/column
// together with computing QR factorization of each.
// Also set up structure of cluster bases.
// 
template < typename value_t >
using  lr_coupling_map_t  = std::unordered_map< indexset, std::list< std::pair< const lrmatrix< value_t > *, blas::matrix< value_t > > >, indexset_hash >;

template < typename value_t >
void
build_mat_map ( const Hpro::TMatrix< value_t > &   A,
                shared_cluster_basis< value_t > &  rowcb,
                shared_cluster_basis< value_t > &  colcb,
                lr_coupling_map_t< value_t > &     row_map,
                lr_coupling_map_t< value_t > &     col_map )
{
    using namespace hlr::matrix;
    
    //
    // decide upon cluster type, how to construct matrix
    //
    
    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // compute semi-coupling, e.g. QR factorization of U/V
        // (see "build_cluster_basis" for more details)
        //
        
        auto  R  = cptrcast( &A, lrmatrix< value_t > );
        auto  W  = blas::copy( R->U() );
        auto  X  = blas::copy( R->V() );
        auto  Cw = blas::matrix< value_t >();
        auto  Cx = blas::matrix< value_t >();
        
        blas::qr( W, Cw, false ); // only need R, not Q
        blas::qr( X, Cx, false );
        
        HLR_ASSERT( Cw.ncols() != 0 );
        HLR_ASSERT( Cx.ncols() != 0 );

        // add matrix to block row/column together with other(!) semi-coupling
        row_map[ A.row_is() ].push_back( { R, std::move( Cx ) } );
        col_map[ A.col_is() ].push_back( { R, std::move( Cw ) } );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  B = cptrcast( &A, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
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
                        
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new shared_cluster_basis< value_t >( M_ij->col_is() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                        colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
                }// if
            }// for
        }// for

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
                    build_mat_map( *B->block( i, j ), *rowcb_i, *colcb_j, row_map, col_map );
            }// for
        }// for
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

    if ( mat_map.find( cb.is() ) != mat_map.end() )
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
        for ( const auto  [ M_i, C_i ] : mat_map.at( cb.is() ) )
            ncols += C_i.nrows();

        // build ( U_0·C_0'  U_1·C_1'  U_2'·C_2' … )
        auto  X   = blas::matrix< value_t >( nrows, ncols );
        uint  pos = 0;

        for ( const auto  [ R_i, C_i ] : mat_map.at( cb.is() ) )
        {
            auto  X_i   = blas::prod( R_i->U( op ), blas::adjoint( C_i ) );
            auto  X_sub = blas::matrix< value_t >( X, blas::range::all, blas::range( pos, pos + C_i.nrows() - 1 ) );

            blas::copy( X_i, X_sub );
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
    
    for ( uint  i = 0; i < cb.nsons(); ++i )
    {
        if ( ! is_null( cb.son( i ) ) )
            build_cluster_basis( *cb.son( i ), basisapx, acc, mat_map, transposed );
    }// for
}

//
// build uniform-H representation of A by converting all lowrank matrices
// into uniform low-rank matrices using given cluster bases rowcb/colcb.
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
        
        auto  R  = cptrcast( &A, lrmatrix< value_t > );
        auto  SU = rowcb.transform_forward( R->U() );
        auto  SV = colcb.transform_forward( R->V() );
        auto  S  = blas::prod( SU, blas::adjoint( SV ) );
        auto  UR = std::make_unique< uniform_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );

        HLR_ASSERT( UR->row_rank() == rowcb.rank() );
        HLR_ASSERT( UR->col_rank() == colcb.rank() );

        M = std::move( UR );
    }// if
    else if ( hlr::matrix::is_lowrank_sv( A ) )
    {
        //
        // compute coupling matrix as W'·U·(X'·V)'
        // with cluster basis W and X
        //
        
        auto  R  = cptrcast( &A, lrsvmatrix< value_t > );
        auto  SU = rowcb.transform_forward( R->U() );
        auto  SV = colcb.transform_forward( R->V() );
        auto  S  = blas::prod( SU, blas::adjoint( SV ) );
        auto  UR = std::make_unique< uniform_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );

        HLR_ASSERT( UR->row_rank() == rowcb.rank() );
        HLR_ASSERT( UR->col_rank() == colcb.rank() );

        M = std::move( UR );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BA );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );

            HLR_ASSERT( ! is_null( rowcb_i ) );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
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
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );

        M = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_uniform_sep ( const Hpro::TMatrix< value_t > &   A,
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
        
        auto  R  = cptrcast( &A, lrmatrix< value_t > );
        auto  SU = rowcb.transform_forward( R->U() );
        auto  SV = colcb.transform_forward( R->V() );
        auto  UR = std::make_unique< uniform_lr2matrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( SU ), std::move( SV ) );

        M = std::move( UR );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BA );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );

            HLR_ASSERT( ! is_null( rowcb_i ) );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                auto  A_ij    = BA->block( i, j );
                
                HLR_ASSERT( ! is_null( colcb_j ) );

                if ( ! is_null( A_ij ) )
                {
                    auto  B_ij = build_uniform_sep( *A_ij, *rowcb_i, *colcb_j );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );

        M = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

////////////////////////////////////////////////////////////////////////////////
//
//
//
////////////////////////////////////////////////////////////////////////////////

//
// collect all clusters within cluster tree
// - largest clusters first (BFS style walkthrough)
//
void
collect_clusters ( const Hpro::TCluster *                 cl,
                   std::list< const Hpro::TCluster * > &  cllist )
{
    auto  nodes = std::list< const Hpro::TCluster * >{ cl };
    
    while ( ! nodes.empty() )
    {
        auto  sons = std::list< const Hpro::TCluster * >();
        
        for ( auto  node : nodes )
        {
            cllist.push_back( node );

            for ( uint  i = 0; i < node->nsons(); ++i )
                sons.push_back( node->son(i) );
        }// for

        nodes = std::move( sons );
    }// while
}

//
// set up mapping from cluster to blocks in block row/column
//
template < typename value_t >
void
build_block_map ( const Hpro::TBlockCluster *                                bc,
                  std::vector< std::list< const Hpro::TBlockCluster * > > &  row_map,
                  std::vector< std::list< const Hpro::TBlockCluster * > > &  col_map )
{
    HLR_ASSERT( ! is_null( bc ) );
    
    row_map[ bc->rowcl()->id() ].push_back( bc );
    col_map[ bc->colcl()->id() ].push_back( bc );

    if ( ! bc->is_leaf() )
    {
        for ( uint  i = 0; i < bc->nsons(); ++i )
            if ( ! is_null( bc->son(i) ) )
                build_block_map< value_t >( bc->son(i), row_map, col_map );
    }// if
}

//
// build single matrix block for given block cluster
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_matrix ( const Hpro::TBlockCluster *  bc,
               const coeff_t &              coeff,
               const lrapx_t &              lrapx,
               const accuracy &             acc,
               const bool                   compress )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( ! is_null( bc ) );
    
    using value_t = typename coeff_t::value_t;
    
    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
        
    if ( bc->is_leaf() )
    {
        if ( bc->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bc, acc ) );
            
            if ( hlr::matrix::is_lowrank( *M ) ) { /* all is good */ }// if
            else if ( Hpro::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                auto  RR = std::make_unique< hlr::matrix::lrmatrix< value_t > >( R->row_is(), R->col_is(),
                                                                                 std::move( R->blas_mat_A() ),
                                                                                 std::move( R->blas_mat_B() ) );
                
                M = std::move( RR );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// if
        else
        {
            M = coeff.build( bc->is().row_is(), bc->is().col_is() );
                        
            if      ( hlr::matrix::is_dense( *M ) ) { /* all is good */ }// if
            else if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( D->blas_mat() ) );
                
                if ( compress )
                    DD->compress( acc );
                
                M = std::move( DD );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// else
    }// if
    else
    {
        auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( bc );
        
        if (( B->nblock_rows() != bc->nrows() ) ||
            ( B->nblock_cols() != bc->ncols() ))
            B->set_block_struct( bc->nrows(), bc->ncols() );
        
        M = std::move( B );
    }// else
    
    M->set_id( bc->id() );
    M->set_procs( bc->procs() );
    
    return  M;
}

//
// correctly insert M into hierarchy (parent matrix)
//
template < typename value_t >
void
add_to_parent ( const Hpro::TBlockCluster *                  bc,
                Hpro::TMatrix< value_t > *                   M,
                std::vector< Hpro::TMatrix< value_t > * > &  mat_map )
{
    HLR_ASSERT( ! is_null( bc ) );
    
    if ( ! is_null( bc->parent() ) )
    {
        auto  bc_parent = bc->parent();
        auto  M_parent  = mat_map[ bc_parent->id() ];
        auto  B         = ptrcast( M_parent, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < bc_parent->nrows(); ++i )
        {
            for ( uint  j = 0; j < bc_parent->ncols(); ++j )
            {
                if ( bc_parent->son(i,j) == bc )
                {
                    B->set_block( i, j, M );
                    i = bc_parent->nrows();
                    j = bc_parent->ncols();
                }// if
            }// for
        }// for
    }// if
}

//
// replace matrix R by matrix U in parent of R
//
template < typename value_t >
void
replace_in_parent ( hlr::matrix::lrmatrix< value_t > *          R,
                    hlr::matrix::uniform_lrmatrix< value_t > *  U )
{
    auto  B = R->parent();

    if ( ! is_null( B ) )
        B->replace_block( R, U );
}

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
void
build_uniform ( const Hpro::TCluster *                                            cl,
                hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > *  cb,
                const coeff_t &                                                   coeff,
                const lrapx_t &                                                   lrapx,
                const basisapx_t &                                                basisapx,
                const accuracy &                                                  acc,
                const bool                                                        compress,
                std::vector< std::list< const Hpro::TBlockCluster * > > &         block_map,
                std::vector< Hpro::TMatrix< typename coeff_t::value_t > * > &     mat_map_H,
                std::vector< Hpro::TMatrix< typename coeff_t::value_t > * > &     mat_map_U,
                std::vector< blas::matrix< typename coeff_t::value_t > > &        row_coup,
                std::vector< blas::matrix< typename coeff_t::value_t > > &        col_coup,
                const matop_t                                                     op )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );

    using value_t       = typename coeff_t::value_t;
    using real_t        = Hpro::real_type_t< value_t >;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    HLR_ASSERT( cl->id() < block_map.size() );
    
    //
    // construct all blocks in current block row/column
    //

    for ( auto  bc : block_map[ cl->id() ] )
    {
        // only compute, if not already done
        if ( is_null( mat_map_H[ bc->id() ] ) )
        {
            auto  M = build_matrix( bc, coeff, lrapx, acc, compress );
        
            // std::cout << "building " << bc->id() << std::endl;
            
            if ( is_blocked( *M ) )
                mat_map_U[ bc->id() ] = M.get();

            mat_map_H[ bc->id() ] = M.get();
            add_to_parent( bc, M.release(), mat_map_H );
        }// if
        // else
        //     std::cout << "exists " << bc->id() << std::endl;
    }// for

    //
    // build row cluster basis
    //

    HLR_ASSERT( ! is_null( cb ) );

    {
        auto    lrmat   = std::list< hlr::matrix::lrmatrix< value_t > * >();
        size_t  totrank = 0;

        for ( auto  bc : block_map[ cl->id() ] )
        {
            auto  M = mat_map_H[ bc->id() ];
        
            if ( hlr::matrix::is_lowrank( M ) )
            {
                auto  R = ptrcast( M, hlr::matrix::lrmatrix< value_t > );
            
                totrank += R->rank();
                lrmat.push_back( R );
            }// if
        }// for

        if ( ! lrmat.empty() )
        {
            // form full cluster basis
            //
            //     U = ( M₀ M₁ M₂ … ) = ( U₀·V₀' U₁·V₁' U₂·V₂' … )
            //
            // in condensed form
            //
            //    ( U₀·C₀' U₁·C₁' U₂·C₁' … )
            //
            // with
            //
            //    C_i = R_i  and  qr(V_i) = Q_i R_i
            //

            size_t  nrows_U = cl->size();
            auto    U       = blas::matrix< value_t >( cl->size(), totrank );
            size_t  pos     = 0;
        
            for ( auto  R : lrmat )
            {
                auto  U_i = R->U( op );
                auto  V_i = blas::copy( R->V( op ) );
                auto  R_i = blas::matrix< value_t >();
                auto  k   = R->rank();
                
                blas::qr( V_i, R_i, false );

                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( UR_i, U_sub );
                
                pos += k;
            }// for

            //
            // approximate basis
            //
        
            auto  Us = blas::vector< real_t >();
            auto  Un = basisapx.column_basis( U, acc, & Us );

            // finally assign to cluster basis object
            cb->set_basis( std::move( Un ), std::move( Us ) );

            //
            // compute row coupling for all lowrank matrices in block row
            //

            for ( auto  R : lrmat )
            {
                //
                // create uniform lr matrix, if not yet present
                //
                     
                auto  U = ptrcast( mat_map_U[ R->id() ], hlr::matrix::uniform_lrmatrix< value_t > );
                
                if ( is_null( U ) )
                {
                    auto  Up = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(), R->col_is() );

                    U = Up.get();
                    U->set_id( R->id() );
                    U->set_procs( R->procs() );
                    
                    mat_map_U[ R->id() ] = U;
                    replace_in_parent( R, Up.release() );
                }// if

                // already assign row basis ("unsafe" due to missing couplings)
                if ( op == apply_normal ) U->set_row_basis_unsafe( *cb );
                else                      U->set_col_basis_unsafe( *cb );

                // compute row coupling
                auto  U_i = R->U( op );
                auto  S_r = blas::prod( blas::adjoint( cb->basis() ), U_i );

                // finalize U if both couplings are present or otherwise just remember row coupling 
                if ( col_coup[ R->id() ].nrows() != 0 )
                {
                    if ( op == apply_normal )
                        U->set_coupling( std::move( blas::prod( S_r, blas::adjoint( col_coup[ R->id() ] ) ) ) );
                    else
                        U->set_coupling( std::move( blas::prod( col_coup[ R->id() ], blas::adjoint( S_r ) ) ) );

                    // no longer needed
                    col_coup[ R->id() ] = std::move( blas::matrix< value_t >() );

                    // std::cout << "deleting " << R->id() << std::endl;
                    mat_map_H[ R->id() ] = nullptr;
                    delete R;
                }// if
                else
                {
                    row_coup[ R->id() ] = std::move( S_r );

                    // std::cout << "coupling " << R->id() << std::endl;
                    // // reset matrix U in R
                    // R->clear_U();
                }// else
            }// for
        }// if
    }
}

}}}}// namespace hlr::seq::detail::matrix

#endif // __HLR_SEQ_DETAIL_UNIFORM_MATRIX_HH
