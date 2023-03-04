#ifndef __HLR_BEM_ACA_HH
#define __HLR_BEM_ACA_HH
//
// Project     : HLR
// Module      : aca.hh
// Description : various ACA algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <deque>
#include <limits>

#include <boost/math/constants/constants.hpp>

#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/blas/Algebra.hh>

#include <hlr/approx/aca.hh>

namespace hlr { namespace bem {

// represents array of pivot elements
using  pivot_arr_t    = std::vector< std::pair< Hpro::idx_t, Hpro::idx_t > >;

//////////////////////////////////////////////////////////////////////
//
// lowrank approximation class for Hpro
//
//////////////////////////////////////////////////////////////////////

template < typename coeff_fn_t >
class aca_lrapx : public Hpro::TLowRankApx< typename coeff_fn_t::value_t >
{
public:
    using  value_t = typename coeff_fn_t::value_t;
    
private:
    // coefficient function
    const coeff_fn_t &  _coeff_fn;
    
public:
    //
    // ctor
    //
    aca_lrapx ( const coeff_fn_t &  acoeff_fn )
            : _coeff_fn( acoeff_fn )
    {}
        
    //////////////////////////////////////
    //
    // build low-rank matrix
    //

    // build low rank matrix for block cluster bct with rank defined by accuracy acc
    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockCluster *   bc,
            const Hpro::TTruncAcc &       acc ) const
    {
        return build( bc->is(), acc );
    }

    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockIndexSet &  bis,
            const Hpro::TTruncAcc &       acc ) const
    {
        const auto  op           = coefffn_operator( bis, _coeff_fn );
        auto        pivot_search = approx::aca_pivot< decltype(op) >( op );
        

        auto [ U, V ] = approx::aca( op, pivot_search, acc, nullptr );

        // {
        //     auto  M = _coeff_fn.build( bis.row_is(), bis.col_is() );
        //     auto  UV = blas::prod( value_t(1), U, blas::adjoint(V) );

        //     blas::add( value_t(-1), Hpro::blas_mat< value_t >( cptrcast( M.get(), Hpro::TDenseMatrix< value_t > ) ), UV );

        //     std::cout << blas::norm_F( UV ) / blas::norm_F( Hpro::blas_mat< value_t >( cptrcast( M.get(), Hpro::TDenseMatrix< value_t > ) ) ) << std::endl;
            
        //     Hpro::DBG::write( U, "U.mat", "U" );
        //     Hpro::DBG::write( V, "V.mat", "V" );
        //     Hpro::DBG::write( M.get(), "M.mat", "M" );
        //     std::exit( 0 );
        // }
        
        auto  R = std::make_unique< Hpro::TRkMatrix< value_t > >( bis.row_is(), bis.col_is(), std::move( U ), std::move( V ) );

        R->truncate( acc );

        return R;
    }
};

//////////////////////////////////////////////////////////////////////
//
// ACA full functions (needed by HCA)
//
//////////////////////////////////////////////////////////////////////

//
// return pivot elements of ACA-Full applied to <M> with
// precision <eps>
//
template < typename value_t >
pivot_arr_t
aca_full_pivots  ( blas::matrix< value_t > &                    M,
                   const typename Hpro::real_type_t< value_t >  eps )
{
    using  real_t = typename Hpro::real_type_t< value_t >;

    //
    // perform ACA-Full on matrix, e.g. choosing maximal element of matrix
    // and compute next rank-1 matrix for low-rank approximation
    //
    
    const size_t             max_rank    = std::min( M.nrows(), M.ncols() );
    size_t                   k           = 0;
    const auto               almost_zero = std::numeric_limits< real_t >::epsilon();
    real_t                   apr         = eps;
    blas::vector< value_t >  row( M.ncols() );
    blas::vector< value_t >  col( M.nrows() );
    pivot_arr_t              pivots;
                
    pivots.reserve( max_rank );
    
    while ( k < max_rank )
    {
        //
        // look for maximal element
        //

        Hpro::idx_t  pivot_row, pivot_col;

        blas::max_idx( M, pivot_row, pivot_col );

        const value_t  pivot_val = M( pivot_row, pivot_col );

        // stop if maximal element is almost 0
        if ( std::abs( pivot_val ) < almost_zero )
            return pivots;
        
        //
        // copy row and column into A/B and update M
        //

        const auto  M_row = M.row( pivot_row );
        const auto  M_col = M.column( pivot_col );

        blas::copy( M_row, row );
        blas::copy( M_col, col );
        
        blas::conj( row );
        blas::scale( value_t(1) / math::conj(pivot_val), row );
        
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
        
        blas::add_r1( value_t(-1), col, row, M );
    }// while

    return pivots;
}

}}// namespace hlr::bem

#endif // __HLR_BEM_ACA_HH
