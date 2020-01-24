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
#include <array>
#include <cmath>

#include <boost/math/constants/constants.hpp>

#include <hpro/cluster/TGeomCluster.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/blas/Algebra.hh>

#include <hlr/matrix/tiling.hh>
#include <hlr/matrix/tiled_lrmatrix.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

//////////////////////////////////////////////////////////////////////
//
// different interpolation points
//
//////////////////////////////////////////////////////////////////////

inline
std::vector< double >
chebyshev_points ( const size_t  k )
{
    //
    //     ⎛ π · ( 2·i + 1)⎞
    // cos ⎜───────────────⎟ , ∀ i ∈ {0,…,k-1}
    //     ⎝      2·k      ⎠
    //

    std::vector< double >  pts( k );
    constexpr auto         pi = boost::math::constants::pi< double >();
    
    for ( uint i = 0; i < k; i++ )
        pts[i] = std::cos( pi * double( 2 * i + 1 ) / double( 2 * k ) );

    return pts;
}

inline
std::vector< double >
chebplusb_points ( const size_t  k )
{
    //
    //     ⎛π · ( 2·(i-1) + 1)⎞
    // cos ⎜──────────────────⎟ , ∀ i ∈ {0,…,k-2}
    //     ⎝      2·(k-2)     ⎠
    //

    std::vector< double >  pts( k );
    constexpr auto         pi = boost::math::constants::pi< double >();

    if ( k > 1 )
    {
        pts[0]   = double(-1);
        pts[k-1] = double(1);
        
        for( uint i = 0; i < k-2; i++ )
            pts[i+1] = std::cos( pi * double( 2 * i + 1 ) / double( 2 * ( k - 2 ) ) );
    }// if

    return pts;
}

inline
std::vector< double >
cheblobat_points ( const size_t  k )
{
    //
    //     ⎛ 2·π·i ⎞
    // cos ⎜───────⎟ , ∀ i ∈ {0,…,k-2}
    //     ⎝2·(k-1)⎠
    //

    std::vector< double >  pts( k );
    constexpr auto         pi = boost::math::constants::pi< double >();

    if ( k > 1 )
    {
        for( uint i = 0; i < k; i++ )
            pts[i] = std::cos( pi * double( 2 * i ) / double( 2 * ( k - 1 ) ) );
    }// if

    return pts;
}

//////////////////////////////////////////////////////////////////////
//
// implement hybrid cross approximation but construct
// low-rank matrices in tiled format
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn,
           typename T_interpolation_fn >
class tiled_hca : public hpro::TLowRankApx
{
    static_assert( std::is_same< typename T_coeff::value_t, typename T_generator_fn::value_t >::value,
                   "value types of coefficient and generator must be equal" );
    
public:
    //
    // template arguments as internal types
    //

    using  coeff_fn_t         = T_coeff;
    using  value_t            = typename coeff_fn_t::value_t;
    using  real_t             = typename real_type< value_t >::type_t;
    using  generator_fn_t     = T_generator_fn;
    using  interpolation_fn_t = T_interpolation_fn;
    using  pivot_arr_t        = std::vector< std::pair< idx_t, idx_t > >;

protected:
    //
    // multi-index of dimension three
    //

    using  idx3_t = std::array< idx_t, 3 >;
    
    // unfold given index i into (i₀,i₁,i₂) with
    // constant dimension <dim> per index dimension
    idx3_t
    unfold ( const idx_t   i,
             const size_t  dim ) const
    {
        idx_t   idx = i;
        idx3_t  midx;
        
        midx[0] = idx % dim; idx /= dim;
        midx[1] = idx % dim; idx /= dim;
        midx[2] = idx % dim;

        return  midx;
    };
    
    //
    // store tensor grid, e.g. grid defined by (x_i,y_j,z_k)
    //
    struct  tensor_grid
    {
        std::vector< real_t >  x, y, z;

        // setup grid for given bbox and interpolation points
        tensor_grid ( const TBBox &                  bbox,
                      const std::vector< real_t > &  ipol_points )
        {
            const auto  middle = 0.5 * ( T3Point( bbox.max() ) + T3Point( bbox.min() ) );
            const auto  diam   = 0.5 * ( T3Point( bbox.max() ) - T3Point( bbox.min() ) );
            const auto  n      = ipol_points.size();

            x.reserve( n );
            y.reserve( n );
            z.reserve( n );
            
            for( auto  p : ipol_points )
            {
                x.push_back( middle.x() + p * diam.x() );
                y.push_back( middle.y() + p * diam.y() );
                z.push_back( middle.z() + p * diam.z() );
            }// for
        }
        
        // return point at index (i,j,k)
        T3Point
        operator () ( const idx_t  i,
                      const idx_t  j,
                      const idx_t  k ) const
        {
            return T3Point( x[i], y[j], z[k] );
        }

        // return point at (multi-) index midx = (i,j,k)
        T3Point
        operator () ( const idx3_t  midx ) const
        {
            return T3Point( x[ midx[0] ], y[ midx[1] ], z[ midx[2] ] );
        }
    };
    
protected:
    // function for matrix evaluation
    const coeff_fn_t &       _coeff;
    
    // function for kernel evaluation
    const generator_fn_t &   _generator_fn;

    // accuracy of ACA inside HCA
    const real_t             _aca_eps;

    // interpolation order to use
    const uint               _ipol_order;

    // function generating 1D interpolation points
    interpolation_fn_t &     _ipol_fn;
    
    // row/column tile indexsets
    matrix::tile_is_map_t &  _row_tile_map;
    matrix::tile_is_map_t &  _col_tile_map;
    
public:
    //////////////////////////////////////
    //
    // constructor and destructor
    //

    tiled_hca ( const coeff_fn_t &       acoeff,
                const generator_fn_t &   agenerator,
                const real_t             aaca_eps,
                const uint               aipol_order,
                interpolation_fn_t &&    aipol_fn,
                matrix::tile_is_map_t &  arow_tile_map,
                matrix::tile_is_map_t &  acol_tile_map )
            : _coeff( acoeff )
            , _generator_fn( agenerator )
            , _aca_eps( aaca_eps )
            , _ipol_order( aipol_order )
            , _ipol_fn( aipol_fn )
            , _row_tile_map( arow_tile_map )
            , _col_tile_map( acol_tile_map )
    {}

    virtual ~tiled_hca () {}

    //////////////////////////////////////
    //
    // build low-rank matrix
    //

    // build low rank matrix for block cluster bct with
    // rank defined by accuracy acc
    virtual TMatrix * build ( const TBlockCluster *   bc,
                              const TTruncAcc &       acc ) const
    {
        HLR_ASSERT( IS_TYPE( bc->rowcl(), TGeomCluster ) && IS_TYPE( bc->colcl(), TGeomCluster ) );

        return hca( * cptrcast( bc->rowcl(), TGeomCluster ),
                    * cptrcast( bc->colcl(), TGeomCluster ),
                    acc );
    }

    virtual TMatrix * build ( const TBlockIndexSet & ,
                              const TTruncAcc & ) const
    {
        HLR_ERROR( "tiled_hca::build : block index set not supported" );
    }
    
protected:
    //
    // actual HCA algorithm
    //
    TMatrix *
    hca ( const TGeomCluster &  rowcl,
          const TGeomCluster &  colcl,
          const TTruncAcc &     acc ) const
    {
        if (( rowcl.bbox().max().dim() != 3 ) ||
            ( colcl.bbox().max().dim() != 3 ))
            HLR_ERROR( "tiled_hca::hca : unsupported dimension of cluster" );

        const uint  order      = _ipol_order;

        // generate grid for local clusters for evaluation of kernel generator function
        const auto  rowcl_grid = tensor_grid( rowcl.bbox(), _ipol_fn( order ) );
        const auto  colcl_grid = tensor_grid( colcl.bbox(), _ipol_fn( order ) );

        // determine ACA pivot elements for the kernel generator matrix
        const auto   pivots    = comp_aca_pivots( rowcl_grid, colcl_grid, _aca_eps, order );
        const auto   k         = pivots.size();

        // immediately return empty matrix
        if ( k == 0 )
            return new TRkMatrix( rowcl, colcl );
    
        //
        // if rank is too high compute matrix by other means
        //

        // if (( k >= std::min( n, m ) ) || ( order4 >= n * m ))
        // {
        //     TSVDLRApx< value_t >   svd( _coeff );
        //     unique_ptr< TMatrix >  M( svd.build( bcl, acc ) );
        
        //     return M.release();
        // }// if

        //
        // otherwise continue with HCA by computing S and G:
        //   G = (S|_pivot_row,pivot_col)^-1
        //
    
        blas::Matrix< value_t >  G( k, k );
    
        for ( idx_t  j = 0; j < idx_t(k); j++ )
        {
            const idx3_t   mj = unfold( pivots[j].second, order );
            const T3Point  y  = colcl_grid( mj );
        
            for ( idx_t  i = 0; i < idx_t(k); i++ )
            {
                const idx3_t   mi = unfold( pivots[i].first, order );
                const T3Point  x  = rowcl_grid( mi );
            
                G(i,j) = _generator_fn.eval( x, y );
            }// for
        }// for

        blas::invert( G );

        //
        // compute low-rank matrix as (U·G) × V^H
        //
    
        auto  U = compute_U( rowcl, k, pivots, colcl_grid, order, G );
        auto  V = compute_V( colcl, k, pivots, rowcl_grid, order );
        auto  R = std::make_unique< matrix::tiled_lrmatrix< value_t > >( rowcl,
                                                                         colcl,
                                                                         std::move( U ),
                                                                         std::move( V ) );

        return R.release();
    }
        

    //
    // compute ACA pivots for approximating the generator function
    // in local block as defined by row- and column grid
    //
    
    pivot_arr_t
    comp_aca_pivots  ( const tensor_grid &  rowcl_grid,
                       const tensor_grid &  colcl_grid,
                       const real_t         eps,
                       const uint           ipol_order ) const
    {
        ///////////////////////////////////////////////////////////
        //
        // use ACA-Full
        //

        const size_t  max_rank = ipol_order * ipol_order * ipol_order;

        //
        // compute full tensor
        //

        blas::Matrix< value_t >  D( max_rank, max_rank );
        idx_t                    j = 0;

        for ( uint  jz = 0; jz < ipol_order; jz++ )
            for ( uint  jy = 0; jy < ipol_order; jy++ )
                for ( uint  jx = 0; jx < ipol_order; jx++, j++ )
                {
                    idx_t       i = 0;
                    const auto  y = colcl_grid( jx, jy, jz );
                
                    for ( uint  iz = 0; iz < ipol_order; iz++ )
                        for ( uint  iy = 0; iy < ipol_order; iy++ )
                            for ( uint  ix = 0; ix < ipol_order; ix++, i++ )
                            {
                                const auto  x = rowcl_grid( ix, iy, iz );

                                D( i, j ) = _generator_fn.eval( x, y );
                            }// for
                }// for
    
        //
        // perform ACA-Full on matrix, e.g. choosing maximal element of matrix
        // and compute next rank-1 matrix for low-rank approximation
        //
    
        size_t                   k           = 0;
        const auto               almost_zero = std::numeric_limits< real_t >::epsilon();
        real_t                   apr         = eps;
        blas::Vector< value_t >  row( D.nrows() );
        blas::Vector< value_t >  col( D.ncols() );
        pivot_arr_t              pivots;
                
        pivots.reserve( max_rank );
    
        while ( k < max_rank )
        {
            //
            // look for maximal element
            //

            idx_t  pivot_row, pivot_col;

            blas::max_idx( D, pivot_row, pivot_col );

            const value_t  pivot_val = D( pivot_row, pivot_col );

            // stop if maximal element is almost 0
            if ( std::abs( pivot_val ) < almost_zero )
                return pivots;
        
            //
            // copy row and column into A/B and update D
            //

            const auto  D_row = D.row( pivot_row );
            const auto  D_col = D.column( pivot_col );

            blas::copy( D_row, row );
            blas::copy( D_col, col );
        
            blas::conj( row );
            blas::scale( value_t(1) / conj(pivot_val), row );
        
            //
            // look at norm of residual
            //
            
            const auto  norm = blas::norm2( col ) * blas::norm2( row );
                
            pivots.push_back( { pivot_row, pivot_col } );
            ++k;
            
            if ( k == 1 )
            {
                // adjust stop criterion with norm-estimate of initial matrix
                apr *= norm;
            }// if
            else if ( norm < apr ) 
            {
                return pivots;
            }// else

            //
            // update dense matrix
            //
        
            blas::add_r1( value_t(-1), col, row, D );
        }// while

        return pivots;
    }

    //
    // compute collocation matrices
    //
    
    // blas::Matrix< value_t >
    // compute_U  ( const TIndexSet &    rowis,
    //              const size_t         rank,
    //              const pivot_arr_t &  pivots,
    //              const tensor_grid &  colcl_grid,
    //              const uint           ipol_order ) const
    // {
    //     //
    //     // set up collocation points and evaluate
    //     //
    
    //     std::vector< T3Point >  y_pts( rank );

    //     for ( size_t j = 0; j < rank; j++ )
    //         y_pts[j] = colcl_grid( unfold( pivots[j].second, ipol_order ) );
    
    //     blas::Matrix< value_t >  U( rowis.size(), rank );
            
    //     _generator_fn.integrate_dx( rowis, y_pts, U );

    //     return U;
    // }
    
    // blas::Matrix< value_t >
    // compute_V  ( const TIndexSet &    colis,
    //              const size_t         rank,
    //              const pivot_arr_t &  pivots,
    //              const tensor_grid &  rowcl_grid,
    //              const uint           ipol_order ) const
    // {
    //     //
    //     // set up collocation points and evaluate
    //     //
    
    //     std::vector< T3Point >  x_pts( rank );
    
    //     for ( size_t j = 0; j < rank; j++ )
    //         x_pts[j] = rowcl_grid( unfold( pivots[j].first, ipol_order ) );

    //     blas::Matrix< value_t >  V( colis.size(), rank );
        
    //     _generator_fn.integrate_dy( colis, x_pts, V );

    //     blas::conj( V );
        
    //     return V;
    // }

    matrix::tile_storage< value_t >
    compute_U  ( const TIndexSet &                rowis,
                 const size_t                     rank,
                 const pivot_arr_t &              pivots,
                 const tensor_grid &              colcl_grid,
                 const uint                       ipol_order,
                 const blas::Matrix< value_t > &  G ) const
    {
        //
        // set up collocation points and evaluate
        //

        matrix::tile_storage< value_t >  U;
        std::vector< T3Point >           y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = colcl_grid( unfold( pivots[j].second, ipol_order ) );

        HLR_ASSERT( _row_tile_map.contains( rowis ) );
        
        for ( auto  is : _row_tile_map.at( rowis ) )
        {
            blas::Matrix< value_t >  U_is( is.size(), rank );
            
            _generator_fn.integrate_dx( is, y_pts, U_is );

            U[ is ] = std::move( blas::prod( value_t(1), U_is, G ) );
        }// for

        return U;
    }
    
    matrix::tile_storage< value_t >
    compute_V  ( const TIndexSet &    colis,
                 const size_t         rank,
                 const pivot_arr_t &  pivots,
                 const tensor_grid &  rowcl_grid,
                 const uint           ipol_order ) const
    {
        //
        // set up collocation points and evaluate
        //
        
        matrix::tile_storage< value_t >  V;
        std::vector< T3Point >           x_pts( rank );
    
        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = rowcl_grid( unfold( pivots[j].first, ipol_order ) );

        HLR_ASSERT( _col_tile_map.contains( colis ) );
        
        for ( auto  is : _col_tile_map.at( colis ) )
        {
            blas::Matrix< value_t >  V_is( is.size(), rank );
            
            _generator_fn.integrate_dy( is, x_pts, V_is );
            blas::conj( V_is );

            V[ is ] = std::move( V_is );
        }// for
        
        return V;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_TILED_HCA_HH
