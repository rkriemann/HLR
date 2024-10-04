#ifndef __HLR_BEM_ACA_HH
#define __HLR_BEM_ACA_HH
//
// Project     : HLR
// Module      : aca.hh
// Description : ACA based lowrank block construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <limits>

#include <hpro/algebra/TLowRankApx.hh>

#include <hlr/arith/blas.hh>
#include <hlr/approx/aca.hh>
#include <hlr/matrix/lrmatrix.hh>

namespace hlr { namespace bem {

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
        auto  op           = coefffn_operator( bis, _coeff_fn );
        auto  pivot_search = approx::aca_pivot< decltype(op) >( op );
        auto  [ U, V ]     = approx::aca( op, pivot_search, acc, nullptr );
        auto  R            = std::make_unique< matrix::lrmatrix< value_t > >( bis.row_is(), bis.col_is(), std::move( U ), std::move( V ) );

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
std::tuple<
    std::vector< Hpro::idx_t >,
    std::vector< Hpro::idx_t >,
    blas::matrix< value_t >,
    blas::matrix< value_t > >
aca_full_pivots  ( blas::matrix< value_t > &                    M,
                   const typename Hpro::real_type_t< value_t >  eps )
{
    using  real_t = typename Hpro::real_type_t< value_t >;

    //
    // perform ACA-Full on matrix, e.g. choosing maximal element of matrix
    // and compute next rank-1 matrix for low-rank approximation
    //
    
    const size_t   max_rank    = std::min( M.nrows(), M.ncols() );
    size_t         k           = 0;
    const auto     almost_zero = std::numeric_limits< real_t >::epsilon();
    real_t         apr         = eps;
    auto           row         = blas::vector< value_t >( M.ncols() );
    auto           col         = blas::vector< value_t >( M.nrows() );
    auto           row_pivots  = std::vector< Hpro::idx_t >();
    auto           col_pivots  = std::vector< Hpro::idx_t >();
    auto           Us          = std::list< blas::vector< value_t > >();
    auto           Vs          = std::list< blas::vector< value_t > >();
                
    row_pivots.reserve( max_rank );
    col_pivots.reserve( max_rank );
    
    while ( k < max_rank )
    {
        //
        // look for maximal element
        //

        Hpro::idx_t  pivot_row, pivot_col;

        blas::max_idx( M, pivot_row, pivot_col );

        const auto  pivot_val = M( pivot_row, pivot_col );

        // stop if maximal element is almost 0
        if ( std::abs( pivot_val ) < almost_zero )
            break;
        
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
                
        row_pivots.push_back( pivot_row );
        col_pivots.push_back( pivot_col );
        Us.push_back( std::move( blas::copy( col ) ) );
        Vs.push_back( std::move( blas::copy( row ) ) );
        ++k;
            
        if ( k == 1 )
        {
            // adjust stop criterion with norm-estimate of initial matrix
            apr *= norm;
        }// if
        else if ( norm < apr ) 
        {
            break;
        }// else

        //
        // update dense matrix
        //
        
        blas::add_r1( value_t(-1), col, row, M );
    }// while

    //
    // copy vectors to matrix
    //
    
    auto  U   = blas::matrix< value_t >( M.nrows(), k );
    auto  V   = blas::matrix< value_t >( M.ncols(), k );
    uint  pos = 0;

    for ( auto  u : Us )
    {
        auto  Ui = U.column( pos++ );

        blas::copy( u, Ui );
    }// for

    pos = 0;
    for ( auto  v : Vs )
    {
        auto  Vi = V.column( pos++ );

        blas::copy( v, Vi );
    }// for
    blas::conj( V );
    
    return { std::move( row_pivots ), std::move( col_pivots ), std::move( U ), std::move( V ) };
}

}}// namespace hlr::bem

#endif // __HLR_BEM_ACA_HH
