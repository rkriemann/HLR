#ifndef __HLR_BEM_HCA_HH
#define __HLR_BEM_HCA_HH
//
// Project     : HLR
// File        : hca.hh
// Description : HCA implementation using tiled low-rank matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>

#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>

#include <hlr/bem/base_hca.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

//////////////////////////////////////////////////////////////////////
//
// implement hybrid cross approximation
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
struct hca : public base_hca< T_coeff, T_generator_fn >
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

    //////////////////////////////////////
    //
    // constructor
    //

    hca ( const coeff_fn_t &       acoeff,
          const generator_fn_t &   agenerator,
          const real_t             aaca_eps,
          const uint               aipol_order,
          interpolation_fn_t       aipol_fn = chebyshev_points )
            : base_hca< T_coeff, T_generator_fn >( acoeff, agenerator, aaca_eps, aipol_order, aipol_fn )
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

        ::tbb::parallel_invoke( [&,k] { U = std::move( compute_U( rowcl, k, pivots, col_grid, G ) ); },
                                [&,k] { V = std::move( compute_V( colcl, k, pivots, row_grid    ) ); } );

        // auto  U = compute_U( rowcl, k, pivots, col_grid, G );
        // auto  V = compute_V( colcl, k, pivots, row_grid );
        auto  R = std::make_unique< TRkMatrix >( rowcl, colcl, hpro::value_type< value_t >::value );

        // std::move not working above for TRkMatrix ???
        R->set_lrmat( U, V );
        R->truncate( acc );

        return R.release();
    }

    //
    // compute collocation matrices, e.g. evaluate collocation
    // integrals at index positions/pivot elements
    //

    blas::Matrix< value_t >
    compute_U  ( const TIndexSet &                rowis,
                 const size_t                     rank,
                 const pivot_arr_t &              pivots,
                 const tensor_grid< real_t > &    col_grid,
                 const blas::Matrix< value_t > &  G ) const
    {
        std::vector< T3Point >  y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( col_grid.fold( pivots[j].second ) );

        blas::Matrix< value_t >  U( rowis.size(), rank );

        base_class::generator_fn().integrate_dx( rowis, y_pts, U );

        // multiply with G
        return blas::prod( value_t(1), U, G );
    }

    blas::Matrix< value_t >
    compute_V  ( const TIndexSet &              colis,
                 const size_t                   rank,
                 const pivot_arr_t &            pivots,
                 const tensor_grid< real_t > &  row_grid ) const
    {
        std::vector< T3Point >  x_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( row_grid.fold( pivots[j].first ) );

        blas::Matrix< value_t >  V( colis.size(), rank );

        base_class::generator_fn().integrate_dy( colis, x_pts, V );

        blas::conj( V );

        return V;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_HCA_HH
