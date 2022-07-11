#ifndef __HLR_APPROX_ACA_HH
#define __HLR_APPROX_ACA_HH
//
// Project     : HLib
// Module      : approx/aca
// Description : low-rank approximation functions using ACA
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>
#include <list>
#include <deque>
#include <utility>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>

namespace hlr { namespace approx {

using Hpro::idx_t;

////////////////////////////////////////////////////////////////////////////////
//
// pivot search strategies
//
////////////////////////////////////////////////////////////////////////////////

//
// using column i and max_row(column i) for i'th pivot element 
//
template < typename T_operator >
struct aca_pivot_next
{
    using  operator_t    = T_operator;
    using  value_t       = typename operator_t::value_t;
    using  real_t        = Hpro::real_type_t< value_t >;
    using  vector_list_t = std::deque< blas::vector< value_t > >;

    // value considered zero to avoid division by small values
    static constexpr real_t  zero_val = std::numeric_limits< real_t >::epsilon() * std::numeric_limits< real_t >::epsilon();
        
    
    int                  next_col = 0;
    std::vector< bool >  used_rows;
    std::vector< bool >  used_cols;

    //
    // initialise pivot search
    //
    aca_pivot_next ( const operator_t &  M )
    {
        next_col = 0;

        used_rows.resize( nrows( M ), false );
        used_cols.resize( ncols( M ), false );
    }
    
    //
    // return position of next pivot of (M - U·V^H)
    // - row/col hold the corresponding row/column data
    //
    std::tuple< int,
                int,
                blas::vector< value_t >,
                blas::vector< value_t > >
    next ( const operator_t &     M,
           const vector_list_t &  U,
           const vector_list_t &  V )
    {
        // get "j"'th column
        const auto  pivot_col  = next_col;

        used_cols[ pivot_col ] = true;

        auto  column = get_column( M, pivot_col );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -math::conj( V[l]( pivot_col ) ), U[l], column );
        
        const auto  pivot_row = blas::max_idx( column );
        const auto  max_val   = column( pivot_row );

        // stop and signal no pivot found if remainder is "zero"
        if ( Hpro::Math::abs( max_val ) <= zero_val )
            return { -1, -1, blas::vector< value_t >(), blas::vector< value_t >() };

        // scale <col> by inverse of maximal element in u
        blas::scale( value_t(1) / max_val, column );
        
        used_rows[ pivot_row ] = true;
        
        auto  row = get_row( M, pivot_row );
        
        // stored as column, hence conjugate
        blas::conj( row );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -math::conj( U[l]( pivot_row ) ), V[l], row );

        //
        // just use next column
        //

        next_col++;
            
        return { pivot_row, pivot_col, std::move( column ), std::move( row ) };
    }
};

//
// uses column with maximal entry in previous row for next pivot element
//
template < typename T_operator >
struct aca_pivot_max
{
    using  operator_t    = T_operator;
    using  value_t       = typename operator_t::value_t;
    using  real_t        = typename Hpro::real_type< value_t >::type_t;
    using  vector_list_t = std::deque< blas::vector< value_t > >;

    // value considered zero to avoid division by small values
    static constexpr real_t  zero_val = std::numeric_limits< real_t >::epsilon() * std::numeric_limits< real_t >::epsilon();
        
    
    int                  next_col = 0;
    std::vector< bool >  used_rows;
    std::vector< bool >  used_cols;

    //
    // initialise pivot search
    //
    aca_pivot_max ( const operator_t &  M )
    {
        next_col = 0;

        used_rows.resize( nrows( M ), false );
        used_cols.resize( ncols( M ), false );
    }
    
    //
    // return position of next pivot of (M - U·V^H)
    // - row/col hold the corresponding row/column data
    //
    std::tuple< int,
                int,
                blas::vector< value_t >,
                blas::vector< value_t > >
    next ( const operator_t &     M,
           const vector_list_t &  U,
           const vector_list_t &  V )
    {
        // get "j"'th column
        const auto  pivot_col  = next_col;

        used_cols[ pivot_col ] = true;

        auto  column = get_column( M, pivot_col );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -math::conj( V[l]( pivot_col ) ), U[l], column );
        
        const auto  pivot_row = blas::max_idx( column );
        const auto  max_val   = column( pivot_row );

        // stop and signal no pivot found if remainder is "zero"
        if ( Hpro::Math::abs( max_val ) <= zero_val )
            return { -1, -1, blas::vector< value_t >(), blas::vector< value_t >() };

        // scale <col> by inverse of maximal element in u
        blas::scale( value_t(1) / max_val, column );
        
        used_rows[ pivot_row ] = true;
        
        auto  row = get_row( M, pivot_row );

        // stored as column, hence conjugate
        blas::conj( row );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -math::conj( U[l]( pivot_row ) ), V[l], row );

        //
        // for next column, look for maximal element in computed row
        //

        real_t  max_v = real_t(0);
        int     max_j = -1;

        for ( size_t  j = 0; j < row.length(); ++j )
        {
            if ( ! used_cols[ j ] )
            {
                if ( std::abs( row(j) ) > max_v )
                {
                    max_v = std::abs( row(j) );
                    max_j = j;
                }// if
            }// if
        }// for
                
        next_col = max_j;
            
        return { pivot_row, pivot_col, std::move( column ), std::move( row ) };
    }
};

//
// default pivot search strategie
//
template < typename operator_t >
using aca_pivot = aca_pivot_max< operator_t >;

////////////////////////////////////////////////////////////////////////////////
//
// ACA iteration
//
////////////////////////////////////////////////////////////////////////////////

//
// compute low-rank approximation of a given sum Σ_i M_i using ACA
// - only need matrix-vector evaluation of given operators
//
template < typename pivotsearch_t >
std::pair< blas::matrix< typename pivotsearch_t::operator_t::value_t >,
           blas::matrix< typename pivotsearch_t::operator_t::value_t > >
aca  ( const typename pivotsearch_t::operator_t &  M,
       pivotsearch_t &                             pivot_search,
       const Hpro::TTruncAcc &                     acc,
       std::list< std::pair< idx_t, idx_t > > *    pivots )
{
    using  value_t = typename pivotsearch_t::operator_t::value_t;
    using  real_t  = typename Hpro::real_type< value_t >::type_t;

    // operator data
    const auto  nrows_M  = nrows( M );
    const auto  ncols_M  = ncols( M );
    const auto  min_dim  = std::min( nrows_M, ncols_M );
    
    // maximal rank either defined by accuracy or dimension of matrix
    const auto  max_rank = ( acc.is_fixed_rank()
                             ? ( acc.has_max_rank() ? std::min( acc.rank(), acc.max_rank() ) : acc.rank() )
                             : ( acc.has_max_rank() ? std::min( min_dim, acc.max_rank() ) : min_dim ) );
    
    // precision defined by accuracy or by machine precision
    // (to be corrected by operator norm)
    real_t      rel_eps  = ( acc.is_fixed_prec() ? acc.rel_eps() : real_t(10 * std::numeric_limits< real_t >::epsilon() ));
    real_t      abs_eps  = acc.abs_eps();
    
    // approximation of |M|
    real_t      norm_M   = real_t(0);

    // low-rank approximation
    std::deque< blas::vector< value_t > >  U, V;
    
    for ( uint  i = 0; i < max_rank; ++i )
    {
        // need to get i, j, row_i, col_j for next iteration
        auto [ pivot_row, pivot_col, column, row ] = pivot_search.next( M, U, V );

        if (( pivot_row == -1 ) || ( pivot_col == -1 ))
            break;
        
        //
        // test convergence by comparing |u_i·v_i'| (approx. for remainder)
        // with |M| ≅ |U·V'|
        //
        
        const auto  norm_i = blas::norm2( column ) * blas::norm2( row );

        if (( norm_i < rel_eps * norm_M ) || ( norm_i < abs_eps ))
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
        // and store new vectors/pivots
        //

        U.push_back( std::move( column ) );
        V.push_back( std::move( row ) );

        if ( pivots != nullptr )
            pivots->push_back( { pivot_row, pivot_col } );
    }// while
    
    //
    // copy to matrices and return
    //
    
    blas::matrix< value_t >  MU( nrows_M, U.size() );
    blas::matrix< value_t >  MV( ncols_M, V.size() );

    for ( uint  l = 0; l < U.size(); ++l )
    {
        auto  u_l = MU.column( l );
        auto  v_l = MV.column( l );

        blas::copy( U[l], u_l );
        blas::copy( V[l], v_l );
    }// for
    
    return { std::move( MU ), std::move( MV ) };
}

////////////////////////////////////////////////////////////////////////////////
//
// functions for dense approximation and low-rank truncation
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca ( blas::matrix< value_t > &  M,
      const Hpro::TTruncAcc &    acc )
{
    auto  pivot_search = aca_pivot< blas::matrix< value_t > >( M );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), M.ncols() ) );
    
    return std::move( aca( M, pivot_search, acc, nullptr ) );
}

template < typename value_t >
std::list< std::pair< idx_t, idx_t > >
aca_pivots ( blas::matrix< value_t > &  M,
             const Hpro::TTruncAcc &    acc )
{
    auto  pivot_search = aca_pivot< blas::matrix< value_t > >( M );

    // for update statistics
    HLR_APPROX_RANK_STAT( "full " << std::min( M.nrows(), M.ncols() ) );

    auto  pivots   = std::list< std::pair< idx_t, idx_t > >();
    auto  [ U, V ] = aca( M, pivot_search, acc, & pivots );

    return pivots;
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca ( const blas::matrix< value_t > &  U,
      const blas::matrix< value_t > &  V,
      const Hpro::TTruncAcc &          acc )
{
    HLR_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
    {
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };
    }// if

    if ( in_rank <= idx_t(acc.rank()) )
    {
        return { std::move( blas::copy( U ) ),
                 std::move( blas::copy( V ) ) };
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return aca( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  op           = operator_wrapper( U, V );
        auto  pivot_search = aca_pivot< decltype(op) >( op );
    
        return aca( op, pivot_search, acc, nullptr );
    }// else
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca ( const std::list< blas::matrix< value_t > > &  U,
      const std::list< blas::matrix< value_t > > &  V,
      const Hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows_U = U.front().nrows();
    const size_t  nrows_V = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  U_i : U )
        in_rank += U_i.ncols();

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        //
        // perform dense approximation
        //

        auto  M   = blas::matrix< value_t >( nrows_U, nrows_V );
        auto  u_i = U.cbegin();
        auto  v_i = V.cbegin();
        
        for ( ; u_i != U.cend(); ++u_i, ++v_i )
            blas::prod( value_t(1), *u_i, blas::adjoint( *v_i ), value_t(1), M );

        return aca( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  op           = operator_wrapper( U, V );
        auto  pivot_search = aca_pivot< decltype(op) >( op );
        
        return aca( op, pivot_search, acc, nullptr );
    }// else
}

template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca ( const std::list< blas::matrix< value_t > > &  U,
      const std::list< blas::matrix< value_t > > &  T,
      const std::list< blas::matrix< value_t > > &  V,
      const Hpro::TTruncAcc &                       acc )
{
    HLR_ASSERT( U.size() == T.size() );
    HLR_ASSERT( T.size() == V.size() );

    if ( U.empty() )
        return { std::move( blas::matrix< value_t >() ),
                 std::move( blas::matrix< value_t >() ) };
    
    //
    // determine maximal rank
    //

    const size_t  nrows_U = U.front().nrows();
    const size_t  nrows_V = V.front().nrows();
    uint          in_rank = 0;

    for ( auto &  T_i : T )
        in_rank += T_i.ncols();

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        //
        // perform dense approximation
        //

        auto  M   = blas::matrix< value_t >( nrows_U, nrows_V );
        auto  U_i = U.cbegin();
        auto  T_i = T.cbegin();
        auto  V_i = V.cbegin();
        
        for ( ; U_i != U.cend(); ++U_i, ++T_i, ++V_i )
        {
            const auto  UT_i = blas::prod( value_t(1), *U_i, *T_i );
            
            blas::prod( value_t(1), UT_i, blas::adjoint( *V_i ), value_t(1), M );
        }// for

        return aca( M, acc );
    }// if
    else
    {
        // for update statistics
        HLR_APPROX_RANK_STAT( "lowrank " << std::min( nrows_U, nrows_V ) << " " << in_rank );

        auto  op           = operator_wrapper( U, T, V );
        auto  pivot_search = aca_pivot< decltype(op) >( op );
        
        return aca( op, pivot_search, acc, nullptr );
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct ACA
{
    using  value_t = T_value;
    
    // signal support for general lin. operators
    static constexpr bool supports_general_operator = true;
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const Hpro::TTruncAcc &    acc ) const
    {
        return hlr::approx::aca( M, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const Hpro::TTruncAcc &          acc ) const 
    {
        return hlr::approx::aca( U, V, acc );
    }
    
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::aca( U, V, acc );
    }

    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const std::list< blas::matrix< value_t > > &  U,
                  const std::list< blas::matrix< value_t > > &  T,
                  const std::list< blas::matrix< value_t > > &  V,
                  const Hpro::TTruncAcc &                       acc ) const
    {
        return hlr::approx::aca( U, T, V, acc );
    }

    template < typename operator_t >
    std::pair< blas::matrix< typename operator_t::value_t >,
               blas::matrix< typename operator_t::value_t > >
    operator () ( const operator_t &       op,
                  const Hpro::TTruncAcc &  acc ) const
    {
        auto  pivot_search = aca_pivot< operator_t >( op );

        return std::move( aca( op, pivot_search, acc, nullptr ) );
    }
};

}}// namespace hlr::approx

#endif // __HLR_APPROX_ACA_HH
