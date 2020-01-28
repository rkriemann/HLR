#ifndef __HLR_TBB_HCA_HH
#define __HLR_TBB_HCA_HH
//
// Project     : HLR
// File        : hca.hh
// Description : various HCA algorithms using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/bem/hca.hh>
#include <hlr/bem/tiled_hca.hh>

namespace hlr { namespace tbb { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

using hlr::bem::pivot_arr_t;
using hlr::bem::tensor_grid;
using hlr::matrix::tile_storage;

//////////////////////////////////////////////////////////////////////
//
// standard HCA
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
struct hca : public hlr::bem::hca< T_coeff, T_generator_fn >
{
    //
    // template arguments as internal types
    //

    using  base_class         = hlr::bem::hca< T_coeff, T_generator_fn >;
    using  coeff_fn_t         = T_coeff;
    using  value_t            = typename coeff_fn_t::value_t;
    using  real_t             = typename real_type< value_t >::type_t;
    using  generator_fn_t     = T_generator_fn;
    using  interpolation_fn_t = std::function< std::vector< double > ( const size_t ) >;

    //////////////////////////////////////
    //
    // constructor
    //

    hca ( const coeff_fn_t &       acoeff,
          const generator_fn_t &   agenerator,
          const real_t             aaca_eps,
          const uint               aipol_order,
          interpolation_fn_t       aipol_fn = hlr::bem::chebyshev_points )
            : hlr::bem::hca< T_coeff, T_generator_fn >( acoeff, agenerator, aaca_eps, aipol_order, aipol_fn )
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
            HLR_ERROR( "hca::approx : unsupported dimension of cluster" );

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

        blas::Matrix< value_t >  U, V;

        ::tbb::parallel_invoke( [&,k] { U = std::move( base_class::compute_U( rowcl, k, pivots, col_grid, G ) ); },
                                [&,k] { V = std::move( base_class::compute_V( colcl, k, pivots, row_grid    ) ); } );

        // auto  U = compute_U( rowcl, k, pivots, col_grid, G );
        // auto  V = compute_V( colcl, k, pivots, row_grid );
        auto  R = std::make_unique< TRkMatrix >( rowcl, colcl, hpro::value_type< value_t >::value );

        // std::move not working above for TRkMatrix ???
        R->set_lrmat( U, V );
        R->truncate( acc );

        return R.release();
    }
};

//////////////////////////////////////////////////////////////////////
//
// tiled HCA
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
struct tiled_hca : public hlr::bem::tiled_hca< T_coeff, T_generator_fn >
{
    //
    // template arguments as internal types
    //

    using  base_class         = hlr::bem::tiled_hca< T_coeff, T_generator_fn >;
    using  coeff_fn_t         = T_coeff;
    using  value_t            = typename coeff_fn_t::value_t;
    using  real_t             = typename real_type< value_t >::type_t;
    using  generator_fn_t     = T_generator_fn;
    using  interpolation_fn_t = std::function< std::vector< double > ( const size_t ) >;

    //////////////////////////////////////
    //
    // constructor and destructor
    //

    tiled_hca ( const coeff_fn_t &            acoeff,
                const generator_fn_t &        agenerator,
                const real_t                  aaca_eps,
                const uint                    aipol_order,
                hlr::matrix::tile_is_map_t &  arow_tile_map,
                hlr::matrix::tile_is_map_t &  acol_tile_map,
                interpolation_fn_t            aipol_fn = hlr::bem::chebyshev_points )
            : base_class( acoeff, agenerator, aaca_eps, aipol_order, arow_tile_map, acol_tile_map, aipol_fn )
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

        tile_storage< value_t >  U, V;

        ::tbb::parallel_invoke( [&,k] { U = std::move( compute_U( rowcl, k, pivots, col_grid, G ) ); },
                                [&,k] { V = std::move( compute_V( colcl, k, pivots, row_grid    ) ); } );

        // recompression
        auto [ U_tr, V_tr ] = tbb::tiled2::truncate( rowcl, colcl, U, V, acc );
        
        auto  R = std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( rowcl, 
                                                                              colcl,
                                                                              std::move( U_tr ),
                                                                              std::move( V_tr ) );

        return R.release();
    }
    
    //
    // compute collocation matrices
    //

    tile_storage< value_t >
    compute_U ( const TIndexSet &                rowis,
                const size_t                     rank,
                const pivot_arr_t &              pivots,
                const tensor_grid< real_t > &    col_grid,
                const blas::Matrix< value_t > &  G ) const
    {
        //
        // set up collocation points and evaluate
        // - also multiply with G
        //

        tile_storage< value_t >  U;
        std::vector< T3Point >   y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( col_grid.fold( pivots[j].second ) );

        HLR_ASSERT( base_class::_row_tile_map.contains( rowis ) );

        const auto &  tiles = base_class::_row_tile_map.at( rowis );

        // for ( auto  is : tiles )
        ::tbb::parallel_for( size_t(0), tiles.size(),
            [&,rank] ( const auto  i )
            {
                const auto               is = tiles[ i ];
                blas::Matrix< value_t >  U_is( is.size(), rank );
            
                base_class::generator_fn().integrate_dx( is, y_pts, U_is );

                U[ is ] = std::move( blas::prod( value_t(1), U_is, G ) );
            } );

        return U;
    }
    
    tile_storage< value_t >
    compute_V ( const TIndexSet &              colis,
                const size_t                   rank,
                const pivot_arr_t &            pivots,
                const tensor_grid< real_t > &  row_grid ) const
    {
        //
        // set up collocation points and evaluate
        //
        
        tile_storage< value_t >  V;
        std::vector< T3Point >   x_pts( rank );
    
        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( row_grid.fold( pivots[j].first ) );

        HLR_ASSERT( base_class::_col_tile_map.contains( colis ) );
        
        const auto &  tiles = base_class::_col_tile_map.at( colis );

        // for ( auto  is : _col_tile_map.at( colis ) )
        ::tbb::parallel_for( size_t(0), tiles.size(),
            [&,rank] ( const auto  i )
            {
                const auto               is = tiles[ i ];
                blas::Matrix< value_t >  V_is( is.size(), rank );
            
                base_class::generator_fn().integrate_dy( is, x_pts, V_is );
                blas::conj( V_is );

               V[ is ] = std::move( V_is );
            } );
        
        return V;
    }
};

}}}// namespace hlr::tbb::bem

#endif // __HLR_TBB_HCA_HH
