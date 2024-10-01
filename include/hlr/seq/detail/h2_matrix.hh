#ifndef __HLR_SEQ_DETAIL_H2_MATRIX_HH
#define __HLR_SEQ_DETAIL_H2_MATRIX_HH
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
#include <hlr/matrix/h2_lr2matrix.hh>

#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace seq { namespace matrix { namespace detail {

using namespace hlr::matrix;

//
// collect lowrank matrices per block row/column and corresponding semi-coupling matrices.
// Also set up structure of cluster bases.
// 
template < typename value_t >
using  lr_mat_map_t   = std::unordered_map< indexset, std::list< const lrmatrix< value_t > * >, indexset_hash >;

template < typename value_t >
using  coupling_map_t = std::unordered_map< const lrmatrix< value_t > *, blas::matrix< value_t > >;

template < typename value_t >
void
build_mat_map ( const Hpro::TMatrix< value_t > &   A,
                nested_cluster_basis< value_t > &  rowcb,
                nested_cluster_basis< value_t > &  colcb,
                lr_mat_map_t< value_t > &          row_map,
                coupling_map_t< value_t > &        row_coupling,
                lr_mat_map_t< value_t > &          col_map,
                coupling_map_t< value_t > &        col_coupling )
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

        // add matrix to block row/column
        row_map[ A.row_is() ].push_back( R );
        col_map[ A.col_is() ].push_back( R );

        // also remember the semi-coupling (see "build_cluster_basis" why the "other")
        row_coupling[ R ] = std::move( Cx );
        col_coupling[ R ] = std::move( Cw );
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
                        rowcb_i = new nested_cluster_basis< value_t >( M_ij->row_is() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
                        rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
                        
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new nested_cluster_basis< value_t >( M_ij->col_is() );
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
                    build_mat_map( *B->block( i, j ), *rowcb_i, *colcb_j, row_map, row_coupling, col_map, col_coupling );
            }// for
        }// for
    }// if
}

//
// build nested cluster basis using precomputed semi-coupling of lowrank matrices
// in block row/columns
//
template < typename value_t >
using  lr_mat_list_t  = std::list< const lrmatrix< value_t > * >;

template < typename value_t,
           typename basisapx_t >
blas::matrix< value_t >
build_nested_cluster_basis ( nested_cluster_basis< value_t > &  cb,
                             const basisapx_t &                 basisapx,
                             const accuracy &                   acc,
                             const lr_mat_map_t< value_t > &    lrmat_map,
                             const coupling_map_t< value_t > &  coupling_map,
                             const lr_mat_list_t< value_t > &   parent_matrices,
                             const bool                         transposed,
                             const bool                         compress )
{
    using  real_t  = Hpro::real_type_t< value_t >;

    const matop_t  op = ( transposed ? apply_transposed : apply_normal );

    //
    // set up list of lowrank matrices contributing to local basis
    //
    
    auto  mat_list = lr_mat_list_t< value_t >( parent_matrices );
    auto  is_sort  = [op] ( auto  M1, auto  M2 ) { return M1->row_is( op ).is_strictly_left_of( M2->row_is( op ) ); };

    if ( lrmat_map.find( cb.is() ) != lrmat_map.end() )
    {
        const auto  local_mats = lrmat_map.at( cb.is() );
        
        mat_list.insert( mat_list.end(), local_mats.begin(), local_mats.end() );
    }// if

    mat_list.sort( is_sort );
    
    // std::cout << cb.is() << " : " << mat_list.size() << std::endl;
    
    //
    // determine local "rank"
    //
    
    uint  ncols = 0;

    // determine total number of columns
    for ( const auto  R_i : mat_list )
    {
        const auto  C_i = coupling_map.at( R_i );
                
        ncols += C_i.nrows();
    }// for
    
    //
    //
    // construct cluster basis
    //

    if ( cb.nsons() == 0 )
    {
        // check for empty basis
        if ( mat_list.empty() )
            return blas::matrix< value_t >();
        
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

        // build ( U_0·C_0'  U_1·C_1'  U_2'·C_2' … )
        const uint  nrows = cb.is().size();
        auto        X     = blas::matrix< value_t >( nrows, ncols );
        uint        pos   = 0;

        for ( const auto  M_i : mat_list )
        {
            const auto  C_i   = coupling_map.at( M_i );
            auto        U_i   = M_i->U( op );
            auto        U_sub = blas::matrix< value_t >( U_i, cb.is() - M_i->row_ofs( op ), blas::range::all );
            auto        X_i   = blas::prod( U_sub, blas::adjoint( C_i ) );
            auto        X_sub = blas::matrix< value_t >( X, blas::range::all, blas::range( pos, pos + C_i.nrows() - 1 ) );

            blas::copy( X_i, X_sub );
            pos += C_i.nrows();
        }// for

        // actually build cluster basis
        auto  Ws = blas::vector< real_t >(); // singular values corresponding to basis vectors
        auto  W  = basisapx.column_basis( X, acc, & Ws );
        auto  R  = blas::prod( blas::adjoint( W ), X );

        cb.set_basis( std::move( W ), std::move( Ws ) );

        if ( compress )
            cb.compress( acc );

        return std::move( R );
    }// if
    else
    {
        //
        // recurse
        //

        uint  nrows    = 0;
        auto  son_data = std::vector< std::pair< nested_cluster_basis< value_t > *, blas::matrix< value_t > > >( cb.nsons() );
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
            {
                auto  R_i = build_nested_cluster_basis( *cb.son( i ), basisapx, acc, lrmat_map, coupling_map, mat_list, transposed, compress );

                nrows += R_i.nrows();
                son_data[i]  = { cb.son(i), std::move( R_i ) };
            }// if
        }// for

        // check for empty basis
        if ( mat_list.empty() )
            return blas::matrix< value_t >();
        
        //
        // compute transfer matrices by joining the R_i
        //

        auto  V   = blas::matrix< value_t >( nrows, ncols );
        uint  pos = 0;

        for ( const auto  [ cb_i, R_i ] : son_data )
        {
            //
            // build mapping from lowrank matrices of son cluster to R_i
            //

            auto  list_i  = lr_mat_list_t< value_t >( mat_list.begin(), mat_list.end() );
            auto  mat_idx = std::unordered_map< const lrmatrix< value_t > *, idx_t >();

            if ( lrmat_map.find( cb_i->is() ) != lrmat_map.end() )
            {
                auto  map_i = lrmat_map.at( cb_i->is() );
                
                list_i.insert( list_i.end(), map_i.begin(), map_i.end() );
            }// if

            list_i.sort( is_sort );

            // offset is defined by position in son list ...
            uint  idx = 0;
            
            for ( auto  M_j : list_i )
            {
                const auto  C_j = coupling_map.at( M_j );

                mat_idx[ M_j ] = idx;
                idx           += C_j.nrows();
            }// for

            const auto  rrange_R = blas::range( 0, idx_t( R_i.nrows() ) - 1 );
            const auto  rrange_V = rrange_R + pos;
            uint        cpos     = 0;

            // ... but we use only those also within local list
            for ( auto  M_j : mat_list )
            {
                const auto  C_j      = coupling_map.at( M_j );
                const auto  ncols_j  = C_j.nrows(); // always used as C'
                const auto  ofs_j    = mat_idx[ M_j ];
                const auto  crange_R = blas::range( ofs_j, ofs_j + ncols_j - 1 );
                const auto  crange_V = blas::range( cpos, cpos + ncols_j - 1 );
                const auto  R_sub    = blas::matrix< value_t >( R_i, rrange_R, crange_R );
                auto        V_sub    = blas::matrix< value_t >( V,   rrange_V, crange_V );
                
                blas::copy( R_sub, V_sub );
                cpos += ncols_j;
            }// for

            pos += R_i.nrows();
        }// for

        // std::cout << cb.is() << " : |V| = " << blas::norm_F( V ) << std::endl;
        // io::matlab::write( V, "V2" );
        
        auto  [ Q, R ] = blas::factorise_ortho( V, acc );
        auto  E        = std::vector< blas::matrix< value_t > >( cb.nsons() );

        // std::cout << cb.is() << " : |V| = " << blas::norm_F( Q ) << std::endl;
        // std::cout << cb.is() << " : |R| = " << blas::norm_F( R ) << std::endl;
        // io::matlab::write( V, "V2" );
        
        pos = 0;
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            const auto  R_i = son_data[i].second;
            const auto  Q_i = blas::matrix< value_t >( Q,
                                                       blas::range( pos, pos + R_i.nrows() - 1 ),
                                                       blas::range::all );

            // std::cout << cb.is() << " : |E| = " << blas::norm_F( Q_i ) << std::endl;
            E[i] = std::move( blas::copy( Q_i ) );
            pos += R_i.nrows();
        }// for

        cb.set_transfer( std::move( E ) );

        if ( compress )
            cb.compress( acc );
        
        return std::move( R );
    }// else

    return blas::matrix< value_t >();
}

template < typename value_t >
void
build_uniform_map ( const Hpro::TMatrix< value_t > &                                   M,
                    const shared_cluster_basis< value_t > &                            rowcb,
                    const shared_cluster_basis< value_t > &                            colcb,
                    std::vector< std::list< const uniform_lrmatrix< value_t > * > > &  row_map,
                    std::vector< std::list< const uniform_lrmatrix< value_t > * > > &  col_map )
{
    if ( is_uniform_lowrank( M ) )
    {
        row_map[ rowcb.id() ].push_back( cptrcast( &M, uniform_lrmatrix< value_t > ) );
        col_map[ colcb.id() ].push_back( cptrcast( &M, uniform_lrmatrix< value_t > ) );
    }// if
    else if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block(i,j) ) )
                    build_uniform_map( * B->block(i,j), * rowcb.son(i), * colcb.son(j), row_map, col_map );
            }// for
        }// for
    }// if
}

template < typename value_t >
void
build_uniform_sep_map ( const Hpro::TMatrix< value_t > &                                    M,
                        const shared_cluster_basis< value_t > &                             rowcb,
                        const shared_cluster_basis< value_t > &                             colcb,
                        std::vector< std::list< const uniform_lr2matrix< value_t > * > > &  row_map,
                        std::vector< std::list< const uniform_lr2matrix< value_t > * > > &  col_map )
{
    if ( is_uniform_lowrank2( M ) )
    {
        row_map[ rowcb.id() ].push_back( cptrcast( &M, uniform_lr2matrix< value_t > ) );
        col_map[ colcb.id() ].push_back( cptrcast( &M, uniform_lr2matrix< value_t > ) );
    }// if
    else if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block(i,j) ) )
                    build_uniform_sep_map( * B->block(i,j), * rowcb.son(i), * colcb.son(j), row_map, col_map );
            }// for
        }// for
    }// if
}

//
// build nested cluster basis out of shared cluster basis
//
template < typename value_t,
           typename basisapx_t >
std::pair< std::unique_ptr< nested_cluster_basis< value_t > >,
           blas::matrix< value_t > >
build_nested_cluster_basis ( const shared_cluster_basis< value_t > &                                  scb,
                             const std::vector< std::list< const uniform_lrmatrix< value_t > * > > &  lrblocks,
                             const std::list< const uniform_lrmatrix< value_t > * > &                 pblocks,
                             const basisapx_t &                                                       basisapx,
                             const accuracy &                                                         acc,
                             const bool                                                               compress,
                             const bool                                                               transposed )
{
    using  real_t     = Hpro::real_type_t< value_t >;
    using  mat_list_t = std::remove_reference_t< decltype( pblocks ) >;

    HLR_ASSERT( scb.id() != -1 );
    
    // std::cout << scb.id() << " " << scb.is().to_string() << std::endl;
    
    //
    // set up empty cluster basis
    //
    
    auto  ncb = std::make_unique< nested_cluster_basis< value_t > >( scb.is() );

    ncb->set_id( scb.id() );
    ncb->set_nsons( scb.nsons() );
    
    //
    // set up list of lowrank matrices contributing to local basis
    //

    auto  op       = ( transposed ? apply_transposed : apply_normal );
    auto  mat_list = mat_list_t( pblocks );
    auto  lblocks  = lrblocks[ scb.id() ];
        
    mat_list.insert( mat_list.end(), lblocks.begin(), lblocks.end() );
    
    //
    // construct cluster basis
    //

    // local rank is sum of row rank of lowrank blocks
    auto  ncols = std::accumulate( mat_list.begin(), mat_list.end(), 0, [op] ( int v, auto  M ) { return v + M->col_rank( op ); } );
    
    if ( scb.nsons() == 0 )
    {
        // check for empty basis
        if ( mat_list.empty() )
            return { std::move( ncb ), blas::matrix< value_t >() };
        
        //
        // compute column basis for block row
        //
        //  ( U₀·S₀'  U₁·S₁'  U₂·S₂'  … )
        //
        // where U_i is restricted to local row index set (e.g., of larger blocks)
        //
        // TODO: collect couplings per shared bases??? same complexity but maybe faster
        //

        const uint  nrows = scb.is().size();
        auto        X     = blas::matrix< value_t >( nrows, ncols );
        uint        pos   = 0;

        for ( const auto  M_i : mat_list )
        {
            auto  S_i   = M_i->coupling();
            auto  k_i   = M_i->col_rank( op );
            auto  U_i   = M_i->row_basis( op );
            auto  U_loc = blas::matrix< value_t >( U_i, scb.is() - M_i->row_ofs( op ), blas::range::all );
            auto  X_i   = blas::prod( U_loc, blas::mat_view( op, S_i ) );
            auto  X_sub = blas::matrix< value_t >( X, blas::range::all, blas::range( pos, pos + k_i - 1 ) );

            blas::copy( X_i, X_sub );
            pos += k_i;
        }// for

        //
        // compress total basis
        //
        
        auto  Ws = blas::vector< real_t >(); // sing. val. for each basis vector
        auto  W  = basisapx.column_basis( X, acc, & Ws );
        auto  R  = blas::prod( blas::adjoint( W ), X );

        ncb->set_basis( std::move( W ), std::move( Ws ) );

        if ( compress )
            ncb->compress( acc );

        return { std::move( ncb ), std::move( R ) };
    }// if
    else
    {
        //
        // recurse
        //

        size_t  nrows  = 0;
        auto    R_sons = std::vector< blas::matrix< value_t > >( scb.nsons() );
        
        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            auto  scb_i = scb.son( i );
            
            HLR_ASSERT( ! is_null( scb.son(i) ) );
            
            // construct son basis
            auto  [ ncb_i, R_i ] = build_nested_cluster_basis( *scb_i, lrblocks, mat_list, basisapx, acc, compress, transposed );

            ncb->set_son( i, ncb_i.release() );

            // io::matlab::write( R_i, Hpro::to_string( "R%d", i ) );
            
            nrows    += R_i.nrows();
            R_sons[i] = std::move( R_i );
        }// for

        // check for empty basis
        if ( mat_list.empty() )
            return { std::move( ncb ), std::move( blas::matrix< value_t >() ) };
        
        //
        // extract components of R_i which also apply to blocks in local list
        // (remove parts only in sub basis)
        //

        auto  V    = blas::matrix< value_t >( nrows, ncols );
        uint  rofs = 0;

        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            const auto  son_i = scb.son(i);
            const auto  R_i   = R_sons[i];
            
            //
            // now go through list and extract local block parts of R
            //

            const auto  rows_R = blas::range( 0, idx_t( R_sons[i].nrows() ) - 1 );
            const auto  rows_V = rows_R + rofs;
            uint        cofs   = 0;
            
            for ( auto  M_j : mat_list )
            {
                const auto  rowis_j = M_j->row_is( op );
                const auto  ncols_j = M_j->col_rank( op );
                const auto  cols_R  = blas::range( cofs, cofs + ncols_j - 1 );
                const auto  cols_V  = blas::range( cofs, cofs + ncols_j - 1 );
                const auto  R_sub   = blas::matrix< value_t >( R_i, rows_R, cols_R );
                auto        V_sub   = blas::matrix< value_t >( V,   rows_V, cols_V );
                
                blas::copy( R_sub, V_sub );
                cofs += ncols_j;
            }// for
            
            rofs += R_i.nrows();
        }// for

        // io::matlab::write( V, "V" );
        
        //
        // compute transfer matrices by joining the R_i
        //

        auto  [ Q, R ] = blas::factorise_ortho( V, acc );
        auto  E        = std::vector< blas::matrix< value_t > >( scb.nsons() );

        rofs = 0;

        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            const auto  k_i = R_sons[i].nrows();
            const auto  Q_i = blas::matrix< value_t >( Q, blas::range( rofs, rofs + k_i - 1 ), blas::range::all );

            E[i]  = std::move( blas::copy( Q_i ) );
            rofs += k_i;
        }// for

        ncb->set_transfer( std::move( E ) );

        if ( compress )
            ncb->compress( acc );
        
        return { std::move( ncb ), std::move( R ) };
    }// else

    return { std::unique_ptr< nested_cluster_basis< value_t > >(), blas::matrix< value_t >() };
}

template < typename value_t,
           typename basisapx_t >
std::pair< std::unique_ptr< nested_cluster_basis< value_t > >,
           blas::matrix< value_t > >
build_nested_cluster_basis_sep ( const shared_cluster_basis< value_t > &                                   scb,
                                 const std::vector< std::list< const uniform_lr2matrix< value_t > * > > &  lrblocks,
                                 const std::list< const uniform_lr2matrix< value_t > * > &                 pblocks,
                                 const basisapx_t &                                                        basisapx,
                                 const accuracy &                                                          acc,
                                 const bool                                                                compress,
                                 const bool                                                                transposed )
{
    using  real_t     = Hpro::real_type_t< value_t >;
    using  mat_list_t = std::remove_reference_t< decltype( pblocks ) >;

    HLR_ASSERT( scb.id() != -1 );
    
    // std::cout << scb.id() << " " << scb.is().to_string() << std::endl;
    
    //
    // set up empty cluster basis
    //
    
    auto  ncb = std::make_unique< nested_cluster_basis< value_t > >( scb.is() );

    ncb->set_id( scb.id() );
    ncb->set_nsons( scb.nsons() );
    
    //
    // set up list of lowrank matrices contributing to local basis
    //

    auto  op       = ( transposed ? apply_transposed : apply_normal );
    auto  mat_list = mat_list_t( pblocks );
    auto  lblocks  = lrblocks[ scb.id() ];
        
    mat_list.insert( mat_list.end(), lblocks.begin(), lblocks.end() );
    
    //
    // construct cluster basis
    //

    // local rank is sum of row rank of lowrank blocks
    auto  ncols = std::accumulate( mat_list.begin(), mat_list.end(), 0, [op] ( int v, auto  M ) { return v + M->col_rank( op ); } );
    
    if ( scb.nsons() == 0 )
    {
        // check for empty basis
        if ( mat_list.empty() )
            return { std::move( ncb ), blas::matrix< value_t >() };
        
        //
        // compute column basis for block row
        //
        //  ( U₀·S₀'  U₁·S₁'  U₂·S₂'  … )
        //
        // where U_i is restricted to local row index set (e.g., of larger blocks)
        //
        // TODO: collect couplings per shared bases??? same complexity but maybe faster
        //

        const uint  nrows = scb.is().size();
        auto        X     = blas::matrix< value_t >( nrows, ncols );
        uint        pos   = 0;

        for ( const auto  M_i : mat_list )
        {
            auto  S_i   = blas::prod( M_i->row_coupling( op ), blas::adjoint( M_i->col_coupling( op ) ) );
            auto  k_i   = M_i->col_rank( op );
            auto  U_i   = M_i->row_basis( op );
            auto  U_loc = blas::matrix< value_t >( U_i, scb.is() - M_i->row_ofs( op ), blas::range::all );
            auto  X_i   = blas::prod( U_loc, blas::mat_view( op, S_i ) );
            auto  X_sub = blas::matrix< value_t >( X, blas::range::all, blas::range( pos, pos + k_i - 1 ) );

            blas::copy( X_i, X_sub );
            pos += k_i;
        }// for

        //
        // compress total basis
        //
        
        auto  Ws = blas::vector< real_t >(); // sing. val. for each basis vector
        auto  W  = basisapx.column_basis( X, acc, & Ws );
        auto  R  = blas::prod( blas::adjoint( W ), X );

        ncb->set_basis( std::move( W ), std::move( Ws ) );

        if ( compress )
            ncb->compress( acc );

        return { std::move( ncb ), std::move( R ) };
    }// if
    else
    {
        //
        // recurse
        //

        size_t  nrows  = 0;
        auto    R_sons = std::vector< blas::matrix< value_t > >( scb.nsons() );
        
        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            auto  scb_i = scb.son( i );
            
            HLR_ASSERT( ! is_null( scb.son(i) ) );
            
            // construct son basis
            auto  [ ncb_i, R_i ] = build_nested_cluster_basis_sep( *scb_i, lrblocks, mat_list, basisapx, acc, compress, transposed );

            ncb->set_son( i, ncb_i.release() );

            // io::matlab::write( R_i, Hpro::to_string( "R%d", i ) );
            
            nrows    += R_i.nrows();
            R_sons[i] = std::move( R_i );
        }// for

        // check for empty basis
        if ( mat_list.empty() )
            return { std::move( ncb ), std::move( blas::matrix< value_t >() ) };
        
        //
        // extract components of R_i which also apply to blocks in local list
        // (remove parts only in sub basis)
        //

        auto  V    = blas::matrix< value_t >( nrows, ncols );
        uint  rofs = 0;

        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            const auto  son_i = scb.son(i);
            const auto  R_i   = R_sons[i];
            
            //
            // now go through list and extract local block parts of R
            //

            const auto  rows_R = blas::range( 0, idx_t( R_sons[i].nrows() ) - 1 );
            const auto  rows_V = rows_R + rofs;
            uint        cofs   = 0;
            
            for ( auto  M_j : mat_list )
            {
                const auto  rowis_j = M_j->row_is( op );
                const auto  ncols_j = M_j->col_rank( op );
                const auto  cols_R  = blas::range( cofs, cofs + ncols_j - 1 );
                const auto  cols_V  = blas::range( cofs, cofs + ncols_j - 1 );
                const auto  R_sub   = blas::matrix< value_t >( R_i, rows_R, cols_R );
                auto        V_sub   = blas::matrix< value_t >( V,   rows_V, cols_V );
                
                blas::copy( R_sub, V_sub );
                cofs += ncols_j;
            }// for
            
            rofs += R_i.nrows();
        }// for

        // io::matlab::write( V, "V" );
        
        //
        // compute transfer matrices by joining the R_i
        //

        auto  [ Q, R ] = blas::factorise_ortho( V, acc );
        auto  E        = std::vector< blas::matrix< value_t > >( scb.nsons() );

        rofs = 0;

        for ( uint  i = 0; i < scb.nsons(); ++i )
        {
            const auto  k_i = R_sons[i].nrows();
            const auto  Q_i = blas::matrix< value_t >( Q, blas::range( rofs, rofs + k_i - 1 ), blas::range::all );

            E[i]  = std::move( blas::copy( Q_i ) );
            rofs += k_i;
        }// for

        ncb->set_transfer( std::move( E ) );

        if ( compress )
            ncb->compress( acc );
        
        return { std::move( ncb ), std::move( R ) };
    }// else

    return { std::unique_ptr< nested_cluster_basis< value_t > >(), blas::matrix< value_t >() };
}

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
build_h2 ( const Hpro::TBlockCluster *  bct,
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
    HLR_ERROR( "todo" );
}

//
// build H² representation of A by converting all lowrank matrices
// into uniform low-rank matrices using given cluster bases rowcb/colcb.
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_h2 ( const Hpro::TMatrix< value_t > &   A,
           nested_cluster_basis< value_t > &  rowcb,
           nested_cluster_basis< value_t > &  colcb,
           const accuracy &                   acc,
           const bool                         compress )
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
        
        auto  R = cptrcast( &A, lrmatrix< value_t > );
        auto  W = rowcb.transform_forward( R->U() );
        auto  X = colcb.transform_forward( R->V() );
        auto  S = blas::prod( W, blas::adjoint( X ) );
        auto  H = std::make_unique< h2_lrmatrix< value_t > >( rowcb, colcb, std::move( S ) );

        if ( compress )
            H->compress( acc );

        M = std::move( H );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( A ) )
    {
        //
        // compute coupling matrix as W'·U·S·(X'·V)'
        // with cluster basis W and X
        //
        
        auto  R   = cptrcast( &A, uniform_lrmatrix< value_t > );
        auto  U   = R->row_basis();
        auto  V   = R->col_basis();
        auto  S   = R->coupling();
        auto  W   = rowcb.transform_forward( U ); // todo: cache/precompute transformed basis
        auto  X   = colcb.transform_forward( V );
        auto  WS  = blas::prod( W, S );
        auto  WSX = blas::prod( WS, blas::adjoint( X ) );
        auto  H   = std::make_unique< h2_lrmatrix< value_t > >( rowcb, colcb, std::move( WSX ) );

        if ( compress )
            H->compress( acc );

        M = std::move( H );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank2( A ) )
    {
        //
        // compute coupling matrix as (W'·U·Sr)·(Sc'·V'·X')
        // with cluster basis W and X
        //
        
        auto  R   = cptrcast( &A, uniform_lr2matrix< value_t > );
        auto  U   = R->row_basis();
        auto  V   = R->col_basis();
        auto  Sr  = R->row_coupling();
        auto  Sc  = R->col_coupling();
        auto  W   = rowcb.transform_forward( U ); // todo: cache/precompute transformed basis
        auto  X   = colcb.transform_forward( V );
        auto  WSr = blas::prod( W, Sr );
        auto  XSc = blas::prod( X, Sc );
        auto  H   = std::make_unique< h2_lr2matrix< value_t > >( rowcb, colcb, std::move( WSr ), std::move( XSc ) );

        if ( compress )
            H->compress( acc );

        M = std::move( H );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B  = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

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
                    auto  B_ij = build_h2( *A_ij, *rowcb_i, *colcb_j, acc, compress );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  BD = blas::copy( D->mat() );
        auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( BD ) );

        if ( compress )
            DD->compress( acc );

        M = std::move( DD );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_h2_sep ( const Hpro::TMatrix< value_t > &   A,
               nested_cluster_basis< value_t > &  rowcb,
               nested_cluster_basis< value_t > &  colcb,
               const accuracy &                   acc,
               const bool                         compress )
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
        auto  Sr = rowcb.transform_forward( R->U() );
        auto  Sc = colcb.transform_forward( R->V() );
        auto  H  = std::make_unique< h2_lr2matrix< value_t > >( rowcb, colcb, std::move( Sr ), std::move( Sc ) );
        
        if ( compress )
            H->compress( acc );

        M = std::move( H );
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
                    auto  B_ij = build_h2_sep( *A_ij, *rowcb_i, *colcb_j, acc, compress );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, dense_matrix< value_t > );
        auto  BD = blas::copy( D->mat() );
        auto  DD = std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( BD ) );

        if ( compress )
            DD->compress( acc );

        M = std::move( DD );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + A.typestr() );

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

}}}}// namespace hlr::seq::detail::matrix

#endif // __HLR_SEQ_DETAIL_H2_MATRIX_HH
