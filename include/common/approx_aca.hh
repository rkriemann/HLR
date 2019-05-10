#ifndef __HLR_APPROX_ACA_HH
#define __HLR_APPROX_ACA_HH
//
// Project     : HLib
// File        : approx_aca.hh
// Description : low-rank approximation functions using ACA
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <vector>
#include <list>
#include <utility>

#include <blas/Vector.hh>
#include <blas/Matrix.hh>

namespace HLR
{

using namespace HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// auxiliary functions
//
////////////////////////////////////////////////////////////////////////////////

//
// compute  col = col - Σ_k U^k V^k_i
// (subtract i'th column of U·V^H)
//
template < typename value_t >
void
subtract ( const std::list< B::Vector< value_t > > &  U,
           const int                                  i,
           const std::list< B::Vector< value_t > > &  V,
           B::Vector< value_t > &                     col )
{
    auto  U_k = U.begin();
    auto  V_k = V.begin();
    
    for ( ; ( U_k != U.end() ) && ( V_k != V.end() ); ++U_k, ++V_k )
        B::add( -conj( (*V_k)( i ) ), (*U_k), col );
}

//
// compute  row = row - Σ_k U^k_i V^k
// (subtract i'th row of U·V^H)
//
template < typename value_t >
void
subtract ( const int                                  i,
           const std::list< B::Vector< value_t > > &  U,
           const std::list< B::Vector< value_t > > &  V,
           B::Vector< value_t > &                     row )
{
    auto  U_k = U.begin();
    auto  V_k = V.begin();
    
    for ( ; ( U_k != U.end() ) && ( V_k != V.end() ); ++U_k, ++V_k )
        B::add( -conj( (*U_k)( i ) ), (*V_k), row );
}

////////////////////////////////////////////////////////////////////////////////
//
// pivot search strategies
//
////////////////////////////////////////////////////////////////////////////////

//
// implements standard pivot search for ACA
//
template < typename T_operator >
struct ACAPivot
{
    using  operator_t = T_operator;
    
    int                  next_col = 0;
    std::vector< bool >  used_rows;
    std::vector< bool >  used_cols;

    //
    // initialise pivot search
    //
    void
    init ( const operator_t &  M )
    {
        next_col = 0;

        used_rows.resize( range_dim( M ),  false );
        used_cols.resize( domain_dim( M ), false );
    }
    
    //
    // return position of next pivot of (M - U·V^H)
    // - row/col hold the corresponding row/column data
    //
    template < typename value_t >
    std::pair< int, int >
    next ( const operator_t &                            M,
           const std::list< BLAS::Vector< value_t > > &  U,
           const std::list< BLAS::Vector< value_t > > &  V,
           BLAS::Vector< value_t > &                     row,
           BLAS::Vector< value_t > &                     col )
    {
        using real_t = typename real_type< value_t >::type_t;

        // value considered zero to avoid division by small values
        const real_t  zero_val  = Math::square( Limits::epsilon< real_t >() );
        
        // get "j"'th column
        const auto  piv_j = next_col;

        used_cols[ piv_j ] = true;
        col                = get_column< value_t >( piv_j, M );
        subtract( U, piv_j, V, col );
        
        const auto  piv_i = BLAS::max_idx( col );
        const auto  max_v = col( piv_i );

        // stop and signal no pivot found if remainder is "zero"
        if ( Math::abs( max_v ) <= zero_val )
            return { -1, -1 };

        // scale u by inverse of maximal element in u
        BLAS::scale( value_t(1) / max_v, col );
        
        used_rows[ piv_i ] = true;
        row                = get_row< value_t >( piv_i, M );
        subtract( piv_i, U, V, row );

        //
        // for next column, look for maximal element in computed row
        //
        
        // next_col++; // just use next column

        const auto  max_j = BLAS::max_idx( row );

        if ( ! used_cols[ max_j ] )
            next_col = max_j;
        else
        {
            for ( size_t  j = 0; j < used_cols.size(); ++j )
            {
                if ( ! used_cols[j] )
                {
                    next_col = int(j);
                    break;
                }// if
            }// for
        }// else
            
        return { piv_i, piv_j };
    }
};

////////////////////////////////////////////////////////////////////////////////
//
// ACA iteration
//
////////////////////////////////////////////////////////////////////////////////

//
// compute low-rank approximation of a given sum Σ_i M_i using ACA
// - only need matrix-vector evaluation of given operators
//
template < typename value_t,
           typename operator_t,
           typename pivotsearch_t >
std::pair< BLAS::Matrix< value_t >,
           BLAS::Matrix< value_t > >
approx_aca  ( const operator_t &                        M,
              const TTruncAcc &                         acc,
              std::list< std::pair< idx_t, idx_t > > *  pivots )
{
    using  real_t = typename real_type< value_t >::type_t;

    if ( M.empty() )
        return { BLAS::Matrix< value_t >(), BLAS::Matrix< value_t >() };
    
    std::list< BLAS::Vector< value_t > >  U, V;
    BLAS::Vector< value_t >               row, col;
    auto                                  sqnorm_M  = real_t(0);
    auto                                  norm_rest = real_t(0);
    uint                                  k         = 0;
    size_t                                nrows     = 0;
    size_t                                ncols     = 0;
    pivotsearch_t< operator_t >           pivot_search;

    pivot_search.init( M );
    
    while ( ! reached_approx( k, Math::sqrt( sqnorm_M ), norm_rest, acc ) )
    {
        // need to get i, j, row_i, col_j for next iteration
        auto [ piv_i, piv_j ] = pivot_search.next( M, U, V, row, col );

        if (( piv_i == -1 ) || ( piv_j == -1 ))
            break;
        
        // DBG::printf( "pivot : (%d,%d)", piv_i, piv_j );
        
        //
        // update norm of M as norm( U·V^H ) and
        // norm of ( M - U·V^H ) by last vector pair
        //

        const auto  sqnorm_rest = re( BLAS::dot( col, col ) * BLAS::dot( row, row ) );

        sqnorm_M += sqnorm_rest;

        auto  u_i = U.cbegin();
        auto  v_i = V.cbegin();
        
        for ( ; u_i != U.cend(); ++u_i, ++v_i )
        {
            sqnorm_M += re( BLAS::dot( col, *u_i ) * BLAS::dot( *v_i, row ) );
            sqnorm_M += re( BLAS::dot( *u_i, col ) * BLAS::dot( row, *v_i ) );
        }// for

        // DBG::printf( "|M| = %.4e", Math::sqrt( sqnorm_M ) );
        
        norm_rest = Math::sqrt( sqnorm_rest );
        
        // DBG::printf( "|M-U·V'| = %.4e", norm_rest );
        
        //
        // and store new vectors/pivots
        //

        nrows = col.length();
        ncols = row.length();
        
        U.push_back( std::move( col ) );
        V.push_back( std::move( row ) );

        if ( pivots != nullptr )
            pivots->push_back( { piv_i, piv_j } );

        ++k;
    }// while

    // DBG::printf( "rank = %d", k );
    
    //
    // copy vector pairs into low-rank matrix
    //

    BLAS::Matrix< value_t >  MU( nrows, k );
    BLAS::Matrix< value_t >  MV( ncols, k );

    auto  u_i = U.cbegin();
    auto  v_i = V.cbegin();
    
    k = 0;
    for ( ; u_i != U.cend() ; ++u_i, ++v_i, ++k )
    {
        auto  mu_i = MU.column( k );
        auto  mv_i = MV.column( k );
        
        BLAS::copy( *u_i, mu_i );
        BLAS::copy( *v_i, mv_i );
    }// for

    return { std::move( MU ), std::move( MV ) };
}

}// namespace HLR

#endif // __HLR_APPROX_ACA_HH
