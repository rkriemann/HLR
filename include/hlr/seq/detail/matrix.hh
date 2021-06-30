#ifndef __HLR_SEQ_DETAIL_MATRIX_HH
#define __HLR_SEQ_DETAIL_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <unordered_map>

#include <hlr/arith/detail/uniform.hh>

namespace hlr { namespace seq { namespace matrix { namespace detail {

namespace hpro = HLIB;

using namespace hlr::matrix;

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_lvl ( const hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< hpro::TRkMatrix * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< hpro::idx_t, hpro::TBlockMatrix * >;

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

            auto  Un = basisapx.column_basis( U, acc );

            #else
            
            //
            // compute column basis for
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
                auto  U_i  = blas::copy( blas::mat_U< value_t >( R ) );
                auto  V_i  = blas::copy( blas::mat_V< value_t >( R ) );
                auto  Ru_i = blas::matrix< value_t >();
                auto  Rv_i = blas::matrix< value_t >();
                auto  k    = R->rank();
                
                blas::qr( U_i, Ru_i );
                blas::qr( V_i, Rv_i );

                auto  S_i    = blas::prod( Ru_i, blas::adjoint( Rv_i ) );
                auto  norm_i = blas::norm_2( S_i );

                blas::scale( value_t(1) / norm_i, S_i );

                auto  US_i   = blas::prod( U_i, blas::adjoint( S_i ) );
                auto  U_sub  = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( US_i, U_sub );
                
                pos += k;
            }// for

            auto  Un = basisapx.column_basis( U, acc );

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

            auto  Vn = basisapx.column_basis( V, acc );

            #else
            
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
                auto  V_i  = blas::copy( blas::mat_V< value_t >( R ) );
                auto  U_i  = blas::copy( blas::mat_U< value_t >( R ) );
                auto  Rv_i = blas::matrix< value_t >();
                auto  Ru_i = blas::matrix< value_t >();
                auto  k    = R->rank();
                
                blas::qr( V_i, Rv_i );
                blas::qr( U_i, Ru_i );

                auto  S_i    = blas::prod( Rv_i, blas::adjoint( Ru_i ) );
                auto  norm_i = blas::norm_2( S_i );

                blas::scale( value_t(1) / norm_i, S_i );
                
                auto  VS_i   = blas::prod( V_i, S_i );
                auto  V_sub  = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                blas::copy( VS_i, V_sub );
                
                pos += k;
            }// for

            auto  Vn = basisapx.column_basis( V, acc );

            #endif
            
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

    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::unique_ptr< hpro::TMatrix >
build_uniform_rec ( const hpro::TBlockCluster *                   bct,
                    const coeff_t &                               coeff,
                    const lrapx_t &                               lrapx,
                    const basisapx_t &                            basisapx,
                    const hpro::TTruncAcc &                       acc,
                    cluster_basis< typename coeff_t::value_t > &  rowcb,
                    cluster_basis< typename coeff_t::value_t > &  colcb,
                    uniform_map_t &                               rowmap,
                    uniform_map_t &                               colmap )
{
    using value_t = typename coeff_t::value_t;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );

            if ( is_lowrank( *M ) )
            {
                //
                // form U·V' = W·T·X' with orthogonal W/X
                //

                auto  R  = ptrcast( M.get(), hpro::TRkMatrix );
                auto  W  = blas::mat_U< value_t >( R );
                auto  X  = blas::mat_V< value_t >( R );
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                blas::qr( W, Rw );
                blas::qr( X, Rx );

                auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
                //
                // update cluster bases
                //

                auto  Un = hlr::uniform::detail::compute_extended_row_basis( rowcb, W, T, acc, basisapx, rowmap );
                auto  Vn = hlr::uniform::detail::compute_extended_col_basis( colcb, T, X, acc, basisapx, colmap );

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
                rowcb.set_basis( std::move( Un ) );
                colcb.set_basis( std::move( Vn ) );
                
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
            auto  rowcb_i = rowcb.son( i );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( bct->son( i, j ) ) )
                {
                    if ( is_null( rowcb_i ) )
                    {
                        rowcb_i = new cluster_basis< value_t >( bct->son( i, j )->is().row_is() );
                        rowcb_i->set_nsons( bct->son( i, j )->rowcl()->nsons() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new cluster_basis< value_t >( bct->son( i, j )->is().col_is() );
                        colcb_j->set_nsons( bct->son( i, j )->colcl()->nsons() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    auto  B_ij = build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, *rowcb_i, *colcb_j, rowmap, colmap );

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

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

template < typename basisapx_t >
std::unique_ptr< hpro::TMatrix >
build_uniform_rec ( const hpro::TMatrix &                            A,
                    const basisapx_t &                               basisapx,
                    const hpro::TTruncAcc &                          acc,
                    cluster_basis< typename basisapx_t::value_t > &  rowcb,
                    cluster_basis< typename basisapx_t::value_t > &  colcb,
                    uniform_map_t &                                  rowmap,
                    uniform_map_t &                                  colmap )
{
    using value_t = typename basisapx_t::value_t;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( is_lowrank( A ) )
    {
        //
        // form U·V' = W·T·X' with orthogonal W/X
        //
        
        auto  R  = cptrcast( &A, hpro::TRkMatrix );
        auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
        auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
        auto  Rw = blas::matrix< value_t >();
        auto  Rx = blas::matrix< value_t >();
            
        blas::qr( W, Rw );
        blas::qr( X, Rx );

        auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
        //
        // update cluster bases
        //
            
        auto  Un = hlr::uniform::detail::compute_extended_row_basis( rowcb, W, T, acc, basisapx, rowmap );
        auto  Vn = hlr::uniform::detail::compute_extended_col_basis( colcb, T, X, acc, basisapx, colmap );
            
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
        rowcb.set_basis( std::move( Un ) );
        colcb.set_basis( std::move( Vn ) );
                
        M = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), rowcb, colcb, std::move( S ) );

        rowmap[ rowcb.is() ].push_back( M.get() );
        colmap[ colcb.is() ].push_back( M.get() );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        
        M = std::make_unique< hpro::TBlockMatrix >();

        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                        rowcb_i = new cluster_basis< value_t >( A_ij->row_is() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_blocked( A_ij ) && ( rowcb_i->nsons() == 0 ))
                        rowcb_i->set_nsons( cptrcast( A_ij, hpro::TBlockMatrix )->nblock_rows() );
                        
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new cluster_basis< value_t >( A_ij->col_is() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    if ( is_blocked( A_ij ) && ( colcb_j->nsons() == 0 ))
                        colcb_j->set_nsons( cptrcast( A_ij, hpro::TBlockMatrix )->nblock_cols() );

                    auto  B_ij = build_uniform_rec( *A_ij, basisapx, acc, *rowcb_i, *colcb_j, rowmap, colmap );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        // TODO: really needed when constructing from existing matrix?
        B->adjust_value_type();
    }// if
    else
    {
        M = A.copy();
    }// else

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

}}}}// namespace hlr::seq::detail::matrix

#endif // __HLR_SEQ_DETAIL_MATRIX_HH
