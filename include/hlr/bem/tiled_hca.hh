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

#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>

#include <hpro/cluster/TGeomCluster.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/blas/Algebra.hh>

#include <hlr/matrix/tiling.hh>
#include <hlr/matrix/tiled_lrmatrix.hh>

#include <hlr/bem/interpolation.hh>
#include <hlr/bem/tensor_grid.hh>
#include <hlr/bem/aca.hh>

#include <hlr/seq/arith_tiled_v2.hh>
#include <hlr/tbb/arith_tiled_v2.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

//////////////////////////////////////////////////////////////////////
//
// implement hybrid cross approximation but construct
// low-rank matrices in tiled format
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
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
    using  interpolation_fn_t = std::function< std::vector< double > ( const size_t ) >;

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
    interpolation_fn_t       _ipol_fn;
    
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
                matrix::tile_is_map_t &  arow_tile_map,
                matrix::tile_is_map_t &  acol_tile_map,
                interpolation_fn_t       aipol_fn = chebyshev_points )
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

        return approx( * cptrcast( bc->rowcl(), TGeomCluster ),
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

        const uint  order    = _ipol_order;

        // generate grid for local clusters for evaluation of kernel generator function
        const auto  row_grid = tensor_grid< real_t >( rowcl.bbox(), _ipol_fn( order ) );
        const auto  col_grid = tensor_grid< real_t >( colcl.bbox(), _ipol_fn( order ) );

        // determine ACA pivot elements for the kernel generator matrix
        const auto  pivots   = comp_aca_pivots( row_grid, col_grid, _aca_eps );
        const auto  k        = pivots.size();

        // immediately return empty matrix
        if ( k == 0 )
            return new TRkMatrix( rowcl, colcl );
    
        //
        // compute G = (S|_pivot_row,pivot_col)^-1
        //
    
        const auto  G = compute_G( pivots, row_grid, col_grid, order );

        //
        // compute low-rank matrix as (U·G) × V^H
        //

        matrix::tile_storage< value_t >  U, V;

        ::tbb::parallel_invoke(
            [&,k,order] ()
            {
                U = std::move( compute_U( rowcl, k, pivots, col_grid, order, G ) );
            },
            
            [&,k,order] ()
            {
                V = std::move( compute_V( colcl, k, pivots, row_grid, order ) );
            } );

        // recompression
        auto [ U_tr, V_tr ] = tbb::tiled2::truncate( rowcl, colcl, U, V, acc );
        
        // auto  U = compute_U( rowcl, k, pivots, col_grid, order, G );
        // auto  V = compute_V( colcl, k, pivots, row_grid, order );
        auto  R = std::make_unique< matrix::tiled_lrmatrix< value_t > >( rowcl,
                                                                         colcl,
                                                                         std::move( U_tr ),
                                                                         std::move( V_tr ) );

        return R.release();
    }
        

    //
    // compute ACA(-Full) pivots for approximating the generator function
    // in local block as defined by row- and column grid
    //
    
    pivot_arr_t
    comp_aca_pivots ( const tensor_grid< real_t > &  row_grid,
                      const tensor_grid< real_t > &  col_grid,
                      const real_t                   eps ) const
    {
        //
        // compute full tensor
        //

        const size_t             nrows    = row_grid.nvertices();
        const size_t             ncols    = col_grid.nvertices();
        blas::Matrix< value_t >  D( nrows, ncols );
        idx_t                    j = 0;

        for ( uint  jz = 0; jz < col_grid.order( 2 ); jz++ )
            for ( uint  jy = 0; jy < col_grid.order( 1 ); jy++ )
                for ( uint  jx = 0; jx < col_grid.order( 0 ); jx++, j++ )
                {
                    idx_t       i = 0;
                    const auto  y = col_grid( jx, jy, jz );
                
                    for ( uint  iz = 0; iz < row_grid.order( 2 ); iz++ )
                        for ( uint  iy = 0; iy < row_grid.order( 1 ); iy++ )
                            for ( uint  ix = 0; ix < row_grid.order( 0 ); ix++, i++ )
                            {
                                const auto  x = row_grid( ix, iy, iz );

                                D( i, j ) = _generator_fn.eval( x, y );
                            }// for
                }// for

        return aca_full_pivots( D, eps );
    }
    
    //
    // compute generator matrix G
    //

    blas::Matrix< value_t >
    compute_G ( const pivot_arr_t &            pivots,
                const tensor_grid< real_t > &  row_grid,
                const tensor_grid< real_t > &  col_grid,
                const size_t                   order ) const
    {
        const auto               k = pivots.size();
        blas::Matrix< value_t >  G( k, k );
    
        for ( idx_t  j = 0; j < idx_t(k); j++ )
        {
            const idx3_t   mj = unfold( pivots[j].second, order );
            const T3Point  y  = col_grid( mj );
        
            for ( idx_t  i = 0; i < idx_t(k); i++ )
            {
                const idx3_t   mi = unfold( pivots[i].first, order );
                const T3Point  x  = row_grid( mi );
            
                G(i,j) = _generator_fn.eval( x, y );
            }// for
        }// for

        blas::invert( G );

        return G;
    }
    
    //
    // compute collocation matrices
    //

    matrix::tile_storage< value_t >
    compute_U  ( const TIndexSet &                rowis,
                 const size_t                     rank,
                 const pivot_arr_t &              pivots,
                 const tensor_grid< real_t > &    col_grid,
                 const uint                       order,
                 const blas::Matrix< value_t > &  G ) const
    {
        //
        // set up collocation points and evaluate
        // - also multiply with G
        //

        matrix::tile_storage< value_t >  U;
        std::vector< T3Point >           y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( unfold( pivots[j].second, order ) );

        HLR_ASSERT( _row_tile_map.contains( rowis ) );

        const auto &  tiles = _row_tile_map.at( rowis );

        // for ( auto  is : tiles )
        ::tbb::parallel_for( size_t(0), tiles.size(),
            [&,rank] ( const auto  i )
            {
                const auto               is = tiles[ i ];
                blas::Matrix< value_t >  U_is( is.size(), rank );
            
                _generator_fn.integrate_dx( is, y_pts, U_is );

                U[ is ] = std::move( blas::prod( value_t(1), U_is, G ) );
            } );

        return U;
    }
    
    matrix::tile_storage< value_t >
    compute_V  ( const TIndexSet &              colis,
                 const size_t                   rank,
                 const pivot_arr_t &            pivots,
                 const tensor_grid< real_t > &  row_grid,
                 const uint                     order ) const
    {
        //
        // set up collocation points and evaluate
        //
        
        matrix::tile_storage< value_t >  V;
        std::vector< T3Point >           x_pts( rank );
    
        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( unfold( pivots[j].first, order ) );

        HLR_ASSERT( _col_tile_map.contains( colis ) );
        
        const auto &  tiles = _col_tile_map.at( colis );

        // for ( auto  is : _col_tile_map.at( colis ) )
        ::tbb::parallel_for( size_t(0), tiles.size(),
            [&,rank] ( const auto  i )
            {
                const auto               is = tiles[ i ];
                blas::Matrix< value_t >  V_is( is.size(), rank );
            
                _generator_fn.integrate_dy( is, x_pts, V_is );
                blas::conj( V_is );

               V[ is ] = std::move( V_is );
            } );
        
        return V;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_TILED_HCA_HH
