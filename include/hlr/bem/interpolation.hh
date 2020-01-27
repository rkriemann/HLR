#ifndef __HLR_BEM_INTERPOLATION_HH
#define __HLR_BEM_INTERPOLATION_HH
//
// Project     : HLR
// File        : interpolation.hh
// Description : various interpolation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>
#include <cmath>

#include <boost/math/constants/constants.hpp>

namespace hlr { namespace bem {

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

}}// namespace hlr::bem

#endif // __HLR_BEM_INTERPOLATION_HH
