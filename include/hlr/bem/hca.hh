#ifndef __HLR_BEM_HCA_HH
#define __HLR_BEM_HCA_HH
//
// Project     : HLR
// Module      : hca.hh
// Description : HCA implementation using tiled low-rank matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>

#include <hlr/bem/base_hca.hh>

#include <hlr/utils/io.hh> // DEBUG
#include <hlr/utils/timer.hh> // DEBUG

namespace hlr { namespace bem {

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
    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    approx ( const Hpro::TGeomCluster &  rowcl,
             const Hpro::TGeomCluster &  colcl,
             const Hpro::TTruncAcc &     acc ) const 
    {
        if (( rowcl.bbox().dim() != 3 ) || ( colcl.bbox().dim() != 3 ))
            HLR_ERROR( "unsupported dimension of cluster" );

        // generate grid for local clusters for evaluation of kernel generator function
        const auto  row_grid = tensor_grid< real_t >( rowcl.bbox(), base_class::ipol_order(), base_class::ipol_func() );
        const auto  col_grid = tensor_grid< real_t >( colcl.bbox(), base_class::ipol_order(), base_class::ipol_func() );
        
        // determine ACA pivot elements for the kernel generator matrix
        const auto  [ row_pivots, col_pivots, W, X ] = base_class::comp_aca_pivots( row_grid, col_grid, base_class::aca_eps() );
        const auto  k                                = row_pivots.size();

        // immediately return empty matrix
        if ( k == 0 )
            return std::make_unique< matrix::lrmatrix< value_t > >( rowcl, colcl );
        
        #if 1

        //
        // compute G = (S|_pivot_row,pivot_col)^-1
        //
    
        const auto  G = base_class::compute_G( row_pivots, col_pivots, row_grid, col_grid );

        //
        // compute low-rank matrix as U×G×V^H
        //

        auto  U = compute_U( rowcl, k, col_pivots, col_grid );
        auto  V = compute_V( colcl, k, row_pivots, row_grid );

        auto  UG = blas::prod( U, G );
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( rowcl, colcl, std::move( UG ), std::move( V ) );

        R->truncate( acc );
        
        return R;

        #else

        //
        // R = U · (W_k X_k')^-1 · V'
        //   = (U.(X_k')^-1 (V·(W_k^-1)')'
        //
        
        auto  Wk = copy_lower( W, row_pivots ); // ACA leads to triangular shape if ordered w.r.t. pivots
        auto  Xk = copy_upper( X, col_pivots ); // (W is also unit diagonal)
        auto  U  = compute_U( rowcl, k, col_pivots, col_grid );
        auto  V  = compute_V( colcl, k, row_pivots, row_grid );
        
        blas::solve_tri( blas::from_right, blas::upper_triangular, blas::general_diag, value_t(1), blas::adjoint( Xk ), U );
        blas::solve_tri( blas::from_right, blas::lower_triangular, blas::general_diag, value_t(1), blas::adjoint( Wk ), V );
        
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( rowcl, colcl, std::move( U ), std::move( V ) );

        R->truncate( acc );

        return R;

        #endif
    }

    //
    // compute collocation matrices, e.g. evaluate collocation
    // integrals at index positions/pivot elements
    //

    blas::matrix< value_t >
    compute_U  ( const Hpro::TIndexSet &             rowis,
                 const size_t                        rank,
                 const std::vector< Hpro::idx_t > &  pivots,
                 const tensor_grid< real_t > &       col_grid ) const
    {
        auto  y_pts = std::vector< Hpro::T3Point >( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( col_grid.fold( pivots[j] ) );

        auto  U = blas::matrix< value_t >( rowis.size(), rank );

        base_class::generator_fn().integrate_dx( rowis, y_pts, U );

        return U;
    }

    blas::matrix< value_t >
    compute_V  ( const Hpro::TIndexSet &             colis,
                 const size_t                        rank,
                 const std::vector< Hpro::idx_t > &  pivots,
                 const tensor_grid< real_t > &       row_grid ) const
    {
        auto  x_pts = std::vector< Hpro::T3Point >( rank );

        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( row_grid.fold( pivots[j] ) );

        auto  V = blas::matrix< value_t >( colis.size(), rank );

        base_class::generator_fn().integrate_dy( colis, x_pts, V );
        blas::conj( V );

        return V;
    }

    blas::matrix< value_t >
    copy_lower ( const blas::matrix< value_t > &     M,
                 const std::vector< Hpro::idx_t > &  pivots ) const
    {
        auto  k  = pivots.size();
        auto  Mk = blas::matrix< value_t >( k, M.ncols() );

        for ( uint  i = 0; i < k; ++i )
        {
            auto  pi = pivots[i];
            
            for ( uint  j = 0; j < i+1; ++j )
                // for ( uint  j = 0; j < M.ncols(); ++j )
                Mk(i,j) = M(pi,j );
        }// for

        return  Mk;
    }

    blas::matrix< value_t >
    copy_upper ( const blas::matrix< value_t > &     M,
                 const std::vector< Hpro::idx_t > &  pivots ) const
    {
        auto  k  = pivots.size();
        auto  Mk = blas::matrix< value_t >( k, M.ncols() );

        for ( uint  i = 0; i < k; ++i )
        {
            auto  pi = pivots[i];
            
            for ( uint  j = i; j < M.ncols(); ++j )
                // for ( uint  j = 0; j < M.ncols(); ++j )
                Mk(i,j) = M(pi,j );
        }// for

        return  Mk;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_HCA_HH
