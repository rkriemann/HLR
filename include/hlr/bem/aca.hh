#ifndef __HLR_BEM_ACA_HH
#define __HLR_BEM_ACA_HH
//
// Project     : HLR
// File        : aca.hh
// Description : various ACA algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>
#include <deque>
#include <limits>

#include <boost/math/constants/constants.hpp>

#include <hpro/blas/Algebra.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;
namespace math = hpro::Math;

using namespace hpro;

using  accuracy       = hpro::TTruncAcc;
using  block_indexset = hpro::TBlockIndexSet;

// represents array of pivot elements
using  pivot_arr_t = std::vector< std::pair< idx_t, idx_t > >;

//
// perform standard ACA to compute low-rank U·V' approximating given M
//
template < typename operator_t >
std::pair< blas::Matrix< typename operator_t::value_t >,
           blas::Matrix< typename operator_t::value_t > >
aca ( const operator_t &  M,
      const accuracy &    acc )
{
    using  value_t = typename operator_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;

    // operator data
    const auto  nrows_M  = nrows( M );
    const auto  ncols_M  = ncols( M );
    
    // maximal rank either defined by accuracy or dimension of matrix
    const auto  max_rank = ( acc.is_fixed_rank()
                             ? ( acc.has_max_rank()
                                 ? std::min( acc.rank(), acc.max_rank() )
                                 : acc.rank() )
                             : std::min( nrows_M, ncols_M ));
    
    // precision defined by accuracy or by machine precision
    // (to be corrected by operator norm)
    real_t      eps      = ( acc.is_fixed_prec()
                             ? acc.rel_eps()
                             : real_t(10 * std::numeric_limits< real_t >::epsilon() ));

    // defines absolute accuracy
    const auto  almost_zero = std::numeric_limits< real_t >::epsilon();

    // approximation of |M|
    real_t      norm_M      = real_t(0);

    // ACA data
    std::deque< blas::Vector< value_t > >  U, V;
    std::vector< bool >                    chosen( ncols_M, false );
    uint                                   next_col = 0;

    for ( uint  i = 0; i < max_rank; ++i )
    {
        //
        // choose pivots and compute current vector pair
        //
        
        // choose pivot column as i
        const uint  pivot_col = next_col;
        auto        column    = get_column( M, i );

        chosen[ pivot_col ] = true;
        
        // correct column by previously computed entries
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -conj( V[l]( pivot_col ) ), U[l], column );
        
        // determine row pivot
        const auto  pivot_row = blas::max_idx( column );

        if ( math::abs( column( pivot_row ) ) <= almost_zero )
            break;

        blas::scale( value_t(1) / column( pivot_row ), column );
        
        auto  row = get_row( M, pivot_row );

        // correct row by previously computed entries
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -conj( U[l]( pivot_row ) ), V[l], row );

        //
        // test convergence by comparing |u_i·v_i'| (approx. for remainder)
        // with |M| ≅ |U·V'|
        //
        
        const auto  norm_i = blas::norm2( column ) * blas::norm2( row );

        if ( norm_i < eps * norm_M )
        {
            U.push_back( std::move( column ) );
            V.push_back( std::move( row ) );
            break;
        }// if

        //
        // update approx. of |M|
        //
        //   |U(:,1:k)·V(:,1:k)'|² = ∑_r=1:k ∑_l=1:k u_r'·u_l  v_r'·v_l
        //                         = |U(:,1:k-1)·V(:,1:k-1)|²
        //                           + ∑_l=1:k-1 u_k'·u_l  v_k'·v_l
        //                           + ∑_l=1:k-1 u_l'·u_k  v_l'·v_k
        //                           + u_k·u_k v_k·v_k
        //

        value_t  upd = norm_i*norm_i;
        
        for ( uint  l = 0; l < U.size(); ++l )
            upd += ( blas::dot( U[l],   column ) * blas::dot( V[l], row  ) +
                     blas::dot( column, U[l]   ) * blas::dot( row,  V[l] ) );

        norm_M = std::sqrt( norm_M * norm_M + math::abs( upd ) );

        //
        // chose pivot column for next step
        // (for more advanced strategies)
        //

        next_col = i+1;

        //
        // add current vector pair
        //
        
        U.push_back( std::move( column ) );
        V.push_back( std::move( row ) );
    }// for
    
    //
    // copy to matrices and return
    //
    
    blas::Matrix< value_t >  MU( nrows_M, U.size() );
    blas::Matrix< value_t >  MV( ncols_M, V.size() );

    for ( uint  l = 0; l < U.size(); ++l )
    {
        auto  u_l = MU.column( l );
        auto  v_l = MV.column( l );

        blas::copy( U[l], u_l );
        blas::copy( V[l], v_l );
    }// for
    
    return { std::move( MU ), std::move( MV ) };
}

//////////////////////////////////////////////////////////////////////
//
// wrapper for TCoeffFn for operator to be used by ACA
//
//////////////////////////////////////////////////////////////////////

template < typename coeff_fn_t >
struct coefffn_operator
{
    using  value_t = typename coeff_fn_t::value_t;

    // block index set to be evaluated at
    block_indexset      bis;

    // coefficient function
    const coeff_fn_t &  func;

    coefffn_operator ( const block_indexset &  abis,
                       const coeff_fn_t &      afunc )
            : bis( abis )
            , func( afunc )
    {}
};

template < typename coeff_fn_t >
size_t
nrows ( const coefffn_operator< coeff_fn_t > &  op )
{
    return op.bis.row_is().size();
}

template < typename coeff_fn_t >
size_t
ncols ( const coefffn_operator< coeff_fn_t > &  op )
{
    return op.bis.col_is().size();
}

template < typename coeff_fn_t >
blas::Vector< typename coeff_fn_t::value_t >
get_row ( const coefffn_operator< coeff_fn_t > &  op,
          const size_t                            i )
{
    blas::Vector< typename coeff_fn_t::value_t >  v( ncols( op ) );
    const auto                                    ofs = i + op.bis.row_is().first();

    op.func.eval( is( ofs, ofs ), op.bis.col_is(), v.data() );
                      
    return v;
}

template < typename coeff_fn_t >
blas::Vector< typename coeff_fn_t::value_t >
get_column ( const coefffn_operator< coeff_fn_t > &  op,
             const size_t                            i )
{
    blas::Vector< typename coeff_fn_t::value_t >  v( nrows( op ) );
    const auto                                    ofs = i + op.bis.col_is().first();

    op.func.eval( op.bis.row_is(), is( ofs, ofs ), v.data() );
                      
    return v;
}

//////////////////////////////////////////////////////////////////////
//
// lowrank approximation class for Hpro
//
//////////////////////////////////////////////////////////////////////

template < typename coeff_fn_t >
class aca_lrapx : public TLowRankApx
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
    virtual TMatrix * build ( const TBlockCluster *   bc,
                              const TTruncAcc &       acc ) const
    {
        return build( bc->is(), acc );
    }

    virtual TMatrix * build ( const TBlockIndexSet &  bis,
                              const TTruncAcc &       acc ) const
    {
        coefffn_operator  op( bis, _coeff_fn );

        auto [ U, V ] = aca( op, acc );

        // {
        //     auto  M = _coeff_fn.build( bis.row_is(), bis.col_is() );
            
        //     hpro::DBG::write( U, "U.mat", "U" );
        //     hpro::DBG::write( V, "V.mat", "V" );
        //     hpro::DBG::write( M.get(), "M.mat", "M" );
        //     std::exit( 0 );
        // }
        
        auto  R = std::make_unique< TRkMatrix >( bis.row_is(), bis.col_is(), std::move( U ), std::move( V ) );

        R->truncate( acc );

        return R.release();
    }
};
    
//
// return pivot elements of ACA-Full applied to <M> with
// precision <eps>
//
template < typename value_t >
pivot_arr_t
aca_full_pivots  ( blas::Matrix< value_t > &                          M,
                   const typename hpro::real_type< value_t >::type_t  eps )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    //
    // perform ACA-Full on matrix, e.g. choosing maximal element of matrix
    // and compute next rank-1 matrix for low-rank approximation
    //
    
    const size_t             max_rank    = std::min( M.nrows(), M.ncols() );
    size_t                   k           = 0;
    const auto               almost_zero = std::numeric_limits< real_t >::epsilon();
    real_t                   apr         = eps;
    blas::Vector< value_t >  row( M.ncols() );
    blas::Vector< value_t >  col( M.nrows() );
    pivot_arr_t              pivots;
                
    pivots.reserve( max_rank );
    
    while ( k < max_rank )
    {
        //
        // look for maximal element
        //

        idx_t  pivot_row, pivot_col;

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
        
        blas::add_r1( value_t(-1), col, row, M );
    }// while

    return pivots;
}

}}// namespace hlr::bem

#endif // __HLR_BEM_ACA_HH
