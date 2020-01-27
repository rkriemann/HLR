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

#include <hpro/cluster/TGeomCluster.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/blas/Algebra.hh>

#include <hlr/bem/interpolation.hh>
#include <hlr/bem/tensor_grid.hh>
#include <hlr/bem/aca.hh>

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
class hca : public hpro::TLowRankApx
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
    
public:
    //////////////////////////////////////
    //
    // constructor and destructor
    //

    hca ( const coeff_fn_t &       acoeff,
          const generator_fn_t &   agenerator,
          const real_t             aaca_eps,
          const uint               aipol_order,
          interpolation_fn_t       aipol_fn = chebyshev_points )
            : _coeff( acoeff )
            , _generator_fn( agenerator )
            , _aca_eps( aaca_eps )
            , _ipol_order( aipol_order )
            , _ipol_fn( aipol_fn )
    {}

    virtual ~hca () {}

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
        HLR_ERROR( "hca::build : block index set not supported" );
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
            HLR_ERROR( "hca::approx : unsupported dimension of cluster" );

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

        blas::Matrix< value_t >  U, V;

        ::tbb::parallel_invoke(
            [&,k,order] ()
            {
                U = std::move( compute_U( rowcl, k, pivots, col_grid, order, G ) );
            },
            
            [&,k,order] ()
            {
                V = std::move( compute_V( colcl, k, pivots, row_grid, order ) );
            } );

        // auto  U = compute_U( rowcl, k, pivots, col_grid, order, G );
        // auto  V = compute_V( colcl, k, pivots, row_grid, order );
        auto  R = std::make_unique< TRkMatrix >( rowcl, colcl, hpro::value_type< value_t >::value );

        // std::move not working above for TRkMatrix ???
        R->set_lrmat( U, V );
        R->truncate( acc );

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

    blas::Matrix< value_t >
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

        std::vector< T3Point >  y_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            y_pts[j] = col_grid( unfold( pivots[j].second, order ) );

        blas::Matrix< value_t >  U( rowis.size(), rank );

        _generator_fn.integrate_dx( rowis, y_pts, U );

        return blas::prod( value_t(1), U, G );
    }

    blas::Matrix< value_t >
    compute_V  ( const TIndexSet &              colis,
                 const size_t                   rank,
                 const pivot_arr_t &            pivots,
                 const tensor_grid< real_t > &  row_grid,
                 const uint                     order ) const
    {
        //
        // set up collocation points and evaluate
        //

        std::vector< T3Point >  x_pts( rank );

        for ( size_t j = 0; j < rank; j++ )
            x_pts[j] = row_grid( unfold( pivots[j].first, order ) );

        blas::Matrix< value_t >  V( colis.size(), rank );

        _generator_fn.integrate_dy( colis, x_pts, V );

        blas::conj( V );

        return V;
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_HCA_HH
