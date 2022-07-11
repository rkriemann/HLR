#ifndef __HLR_BEM_TILED_HCA_HH
#define __HLR_BEM_TILED_HCA_HH
//
// Project     : HLR
// File        : tiled_hca.hh
// Description : HCA implementation using tiled low-rank matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>

#include <hlr/matrix/tiling.hh>
#include <hlr/matrix/tiled_lrmatrix.hh>

#include <hlr/bem/base_hca.hh>

#include <hlr/seq/arith_tiled_v2.hh>

namespace hlr { namespace bem {

//////////////////////////////////////////////////////////////////////
//
// implement hybrid cross approximation but construct
// low-rank matrices in tiled format
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
struct tiled_hca : public base_hca< T_coeff, T_generator_fn >
{
    //
    // template arguments as internal types
    //

    using  base_class         = base_hca< T_coeff, T_generator_fn >;
    using  coeff_fn_t         = T_coeff;
    using  value_t            = typename coeff_fn_t::value_t;
    using  real_t             = typename real_type< value_t >::type_t;
    using  generator_fn_t     = T_generator_fn;
    using  interpolation_fn_t = std::function< std::vector< double > ( const size_t ) >;

    // row/column tile indexsets
    matrix::tile_is_map_t &  _row_tile_map;
    matrix::tile_is_map_t &  _col_tile_map;
    
    //////////////////////////////////////
    //
    // constructor and destructor
    //

    tiled_hca ( const coeff_fn_t &       acoeff,
                const generator_fn_t &   agenerator,
                const real_t             aaca_eps,
                const uint               aipol_order,
                matrix::tile_is_map_t &  arow_tile_map,
                matrix::tile_is_map_t &  acol_tile_map,
                interpolation_fn_t       aipol_fn = chebyshev_points )
            : base_class( acoeff, agenerator, aaca_eps, aipol_order, aipol_fn )
            , _row_tile_map( arow_tile_map )
            , _col_tile_map( acol_tile_map )
    {}

    //
    // actual HCA algorithm
    // - <acc> only used for recompression
    //
    TMatrix *
    approx ( const TGeomCluster &  rowcl,
             const TGeomCluster &  colcl,
             const TTruncAcc &     acc ) const
    {
        if (( rowcl.bbox().max().dim() != 3 ) ||
            ( colcl.bbox().max().dim() != 3 ))
            HLR_ERROR( "tiled_hca::approx : unsupported dimension of cluster" );

        // generate grid for local clusters for evaluation of kernel generator function
        const auto  row_grid = tensor_grid< real_t >( rowcl.bbox(), base_class::ipol_points() );
        const auto  col_grid = tensor_grid< real_t >( colcl.bbox(), base_class::ipol_points() );

        // determine ACA pivot elements for the kernel generator matrix
        const auto  pivots   = base_class::comp_aca_pivots( row_grid, col_grid, base_class::aca_eps() );
        const auto  k        = pivots.size();

        // immediately return empty matrix
        if ( k == 0 )
            return new TRkMatrix( rowcl, colcl );
    
        //
        // compute G = (S|_pivot_row,pivot_col)^-1
        //
    
        const auto  G = base_class::compute_G( pivots, row_grid, col_grid );

        //
        // compute low-rank matrix as (U·G) × V^H
        //

        auto  U = compute_U( rowcl, k, pivots, col_grid, G );
        auto  V = compute_V( colcl, k, pivots, row_grid    );

        // recompression
        auto [ U_tr, V_tr ] = seq::tiled2::truncate( rowcl, colcl, U, V, acc );
        
        // auto  U = compute_U( rowcl, k, pivots, col_grid, order, G );
        // auto  V = compute_V( colcl, k, pivots, row_grid, order );
        auto  R = std::make_unique< matrix::tiled_lrmatrix< value_t > >( rowcl, 
                                                                         colcl,
                                                                         std::move( U_tr ),
                                                                         std::move( V_tr ) );

        return R.release();
    }
    
    //
    // compute collocation matrices
    //

    matrix::tile_storage< value_t >
    compute_U  ( const TIndexSet &                rowis,
                 const size_t                     rank,
                 const pivot_arr_t &              pivots,
                 const tensor_grid< real_t > &    col_grid,
                 const blas::matrix< value_t > &  G ) const
    {
        //
        // set up collocation points and evaluate
        // - also multiply with G
        //

        matrix::tile_storage< value_t >  U;
        std::vector< T3Point >           y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( col_grid.fold( pivots[j].second ) );

        HLR_ASSERT( _row_tile_map.find( rowis ) != _row_tile_map.end() );

        const auto &  tiles = _row_tile_map.at( rowis );

        for ( auto  is : tiles )
        {
            blas::matrix< value_t >  U_is( is.size(), rank );
            
            base_class::generator_fn().integrate_dx( is, y_pts, U_is );
            
            U[ is ] = std::move( blas::prod( value_t(1), U_is, G ) );
        }// for

        return U;
    }
    
    matrix::tile_storage< value_t >
    compute_V  ( const TIndexSet &              colis,
                 const size_t                   rank,
                 const pivot_arr_t &            pivots,
                 const tensor_grid< real_t > &  row_grid ) const
    {
        //
        // set up collocation points and evaluate
        //
        
        matrix::tile_storage< value_t >  V;
        std::vector< T3Point >           x_pts( rank );
    
        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( row_grid.fold( pivots[j].first ) );

        HLR_ASSERT( _col_tile_map.find( colis ) != _col_tile_map.end() );
        
        const auto &  tiles = _col_tile_map.at( colis );

        for ( auto  is : tiles )
        {
            blas::matrix< value_t >  V_is( is.size(), rank );
            
            base_class::generator_fn().integrate_dy( is, x_pts, V_is );
            blas::conj( V_is );
            
            V[ is ] = std::move( V_is );
        }// for
        
        return V;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_TILED_HCA_HH
