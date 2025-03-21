#ifndef __HLR_BEM_BASE_HCA_HH
#define __HLR_BEM_BASE_HCA_HH
//
// Project     : HLR
// Module      : base_hca.hh
// Description : base class for various HCA algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <vector>

#include <hpro/cluster/TGeomCluster.hh>
#include <hpro/algebra/TLowRankApx.hh>

#include <hlr/arith/blas.hh>
#include <hlr/bem/interpolation.hh>
#include <hlr/bem/tensor_grid.hh>
#include <hlr/bem/aca.hh>

#include <hlr/approx/rrqr.hh>
#include <hlr/approx/aca.hh>

namespace hlr { namespace bem {

//////////////////////////////////////////////////////////////////////
//
// base class for HCA algorithms
//
//////////////////////////////////////////////////////////////////////

template < typename T_coeff,
           typename T_generator_fn >
struct base_hca
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

    base_hca ( const coeff_fn_t &       acoeff,
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

    //
    // access local data
    //
    
    const coeff_fn_t &           coeff_fn     () const { return _coeff; }
    const generator_fn_t &       generator_fn () const { return _generator_fn; }
    real_t                       aca_eps      () const { return _aca_eps; }
    uint                         ipol_order   () const { return _ipol_order; }
    const interpolation_fn_t &   ipol_func    () const { return _ipol_fn; }
    const std::vector< double >  ipol_points  () const { return _ipol_fn( _ipol_order ); }
    
    //
    // compute ACA(-Full) pivots for approximating the generator function
    // in local block as defined by row- and column grid
    //
    std::tuple<
        std::vector< Hpro::idx_t >,
        std::vector< Hpro::idx_t >,
        blas::matrix< value_t >,
        blas::matrix< value_t > >
    comp_aca_pivots ( const tensor_grid< real_t > &  row_grid,
                      const tensor_grid< real_t > &  col_grid,
                      const real_t                   eps ) const
    {
        //
        // compute full tensor
        //

        const auto  nrows = row_grid.nvertices();
        const auto  ncols = col_grid.nvertices();
        auto        D     = blas::matrix< value_t >( nrows, ncols );

        for ( uint  i = 0; i < nrows; ++i )
        {
            const auto  x = row_grid( row_grid.fold( i ) );
            
            for ( uint  j = 0; j < ncols; ++j )
            {
                const auto  y = col_grid( col_grid.fold( j ) );

                D( i, j ) = _generator_fn.eval( x, y );
            }// for
        }// for

        //
        // ACA-Full
        //

        return aca_full_pivots( D, eps );

        //
        // standard ACA
        //
        
        // auto  pivots = approx::aca_pivots( D, fixed_prec( eps ) );

        // return pivot_arr_t( pivots.begin(), pivots.end() );

        //
        // RRQR
        //
        
        // // determine row pivots
        // auto  DT    = blas::copy( blas::adjoint( D ) );
        // auto  R     = blas::matrix< value_t >();
        // auto  P_row = std::vector< int >();

        // blas::qrp( DT, R, P_row );

        // const auto  k_row = approx::detail::trunc_rank( R, fixed_prec( eps ) );

        // // determine column pivots
        // auto  P_col = std::vector< int >();

        // blas::qrp( D, R, P_col );

        // const auto  k_col = approx::detail::trunc_rank( R, fixed_prec( eps ) );

        // // set up set of selected rows/columns
        // auto  pivots = pivot_arr_t( std::min( k_row, k_col ) );

        // for ( int  i = 0; i < std::min( k_row, k_col ); ++i )
        //     pivots[i] = { P_row[i], P_col[i] };

        // return pivots;
    }
    
    //
    // compute generator matrix G
    //
    blas::matrix< value_t >
    compute_G ( const std::vector< Hpro::idx_t > &  row_pivots,
                const std::vector< Hpro::idx_t > &  col_pivots,
                const tensor_grid< real_t > &       row_grid,
                const tensor_grid< real_t > &       col_grid ) const
    {
        const auto               k = row_pivots.size();
        blas::matrix< value_t >  G( k, k );
    
        for ( idx_t  j = 0; j < idx_t(k); j++ )
        {
            const auto  y  = col_grid( col_grid.fold( col_pivots[j] ) );
        
            for ( idx_t  i = 0; i < idx_t(k); i++ )
            {
                const auto  x  = row_grid( row_grid.fold( row_pivots[i] ) );
            
                G(i,j) = _generator_fn.eval( x, y );
            }// for
        }// for

        blas::invert( G );

        return G;
    }
};

//////////////////////////////////////////////////////////////////////
//
// lowrank approximation class for Hpro
//
//////////////////////////////////////////////////////////////////////

template < typename hca_impl_t >
class hca_lrapx : public Hpro::TLowRankApx< typename hca_impl_t::value_t >
{
public:
    using  value_t = typename hca_impl_t::value_t;
    
private:
    // HCA implementation to use
    const hca_impl_t &  _hca;
    
public:
    //
    // ctor
    //
    hca_lrapx ( const hca_impl_t &  hca )
            : _hca( hca )
    {}
        
    //////////////////////////////////////
    //
    // build low-rank matrix
    //

    // build low rank matrix for block cluster bct with
    // rank defined by accuracy acc
    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockCluster *  bc,
            const Hpro::TTruncAcc &      acc ) const
    {
        HLR_ASSERT( Hpro::is_geom_cluster( bc->rowcl() ) && Hpro::is_geom_cluster( bc->colcl() ) );
        
        return _hca.approx( * cptrcast( bc->rowcl(), Hpro::TGeomCluster ),
                            * cptrcast( bc->colcl(), Hpro::TGeomCluster ),
                            acc );
    }

    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockIndexSet & ,
            const Hpro::TTruncAcc & ) const
    {
        HLR_ERROR( "block index set not supported" );
    }
    
};
    
}}// namespace hlr::bem

#endif // __HLR_BEM_BASE_HCA_HH
