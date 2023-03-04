#ifndef __HLR_BEM_TENSOR_GRID_HH
#define __HLR_BEM_TENSOR_GRID_HH
//
// Project     : HLR
// Module      : tensor_grid_hca.hh
// Description : grid defined via tensor product X × Y × Z
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <array>
#include <vector>

#include <hpro/base/TPoint.hh>
#include <hpro/cluster/TBBox.hh>

namespace hlr { namespace bem {

using  point        = Hpro::T3Point;
using  bounding_box = Hpro::TBBox;

//
// multi-index of dimension three
//

using  idx3_t = std::array< idx_t, 3 >;

//
// represent axis aligned tensor grid X × Y × Z with
//
//     X = (x_0,...,x_n-1),
//     Y = (y_0,...,y_n-1) and
//     Z = (z_0,...,z_n-1).
//
template < typename T_real = double >
struct tensor_grid
{
    using  real_t = T_real;
    
    std::vector< real_t >  x, y, z;
    
    // setup grid for given bounding box and 1D points
    tensor_grid ( const bounding_box &           box,
                  const uint                     order,
                  const std::function< std::vector< double > ( const size_t ) > &  func )
    {
        const auto  middle = 0.5 * ( point( box.max() ) + point( box.min() ) );
        const auto  diam   = 0.5 * ( point( box.max() ) - point( box.min() ) );
        const auto  dmax   = std::max( std::max( diam.x(), diam.y() ), diam.z() );
        auto        nx     = order;
        auto        ny     = order;
        auto        nz     = order;

        auto  d = diam.x();
        
        while (( nx > 0 ) && ( d / dmax < 0.5 ))
        {
            d *= 2;
            nx--;
        }// while
        nx = std::max< uint >( 1, nx );

        d = diam.y();
        while (( ny > 0 ) &&  d / dmax < 0.5 )
        {
            d *= 2;
            ny--;
        }// while
        ny = std::max< uint >( 1, ny );
        
        d = diam.z();
        while (( nz > 0 ) &&  d / dmax < 0.5 )
        {
            d *= 2;
            nz--;
        }// while
        nz = std::max< uint >( 1, nz );
        
        x.reserve( nx );
        y.reserve( ny );
        z.reserve( nz );

        // std::cout << nx << ", " << ny << ", " << nz << std::endl;
        
        for( auto  p : func( nx ) )
            x.push_back( middle.x() + p * diam.x() );
        
        for( auto  p : func( ny ) )
            y.push_back( middle.y() + p * diam.y() );
        
        for( auto  p : func( nz ) )
            z.push_back( middle.z() + p * diam.z() );
    }

    // setup grid for given bounding box and 1D points
    tensor_grid ( const bounding_box &           box,
                  const std::vector< real_t > &  pts1d )
    {
        const auto  middle = 0.5 * ( point( box.max() ) + point( box.min() ) );
        const auto  diam   = 0.5 * ( point( box.max() ) - point( box.min() ) );
        const auto  n      = pts1d.size();

        x.reserve( n );
        y.reserve( n );
        z.reserve( n );
        
        for( auto  p : pts1d )
        {
            x.push_back( middle.x() + p * diam.x() );
            y.push_back( middle.y() + p * diam.y() );
            z.push_back( middle.z() + p * diam.z() );
        }// for
    }

    // return dimension per axis
    size_t  order ( const uint  d ) const
    {
        assert( d < 3 );
        
        switch ( d )
        {
            case 0 : return x.size();
            case 1 : return y.size();
            case 2 : return z.size();
        }// switch

        return 0;
    }
    
    // return total number of grid points
    size_t  nvertices () const
    {
        return x.size() * y.size() * z.size();
    }
    
    // return point at index (i,j,k)
    point
    operator () ( const idx_t  i,
                  const idx_t  j,
                  const idx_t  k ) const
    {
        return point( x[i], y[j], z[k] );
    }
    
    // return point at (multi-) index midx = (i,j,k)
    point
    operator () ( const idx3_t  midx ) const
    {
        return point( x[ midx[0] ],
                      y[ midx[1] ],
                      z[ midx[2] ] );
    }

    // unfold given multi index (i₀,i₁,i₂) into i
    idx3_t
    unfold ( const idx3_t  mi ) const
    {
        // 0: xz plane → 1: z fibre → 2: entry
        return (mi[2] * x.size() + mi[1]) * y.size() + mi[0];
    }

    // unfold given index i into (i₀,i₁,i₂) 
    idx3_t
    fold ( const idx_t  i ) const
    {
        const idx_t  i2 = i  / x.size();
        const idx_t  i3 = i2 / y.size();

        return { idx_t( i  % x.size() ),
                 idx_t( i2 % y.size() ),
                 idx_t( i3 % z.size() ) };
    }
};

}}// namespace hlr::bem

#endif // __HLR_BEM_TENSOR_GRID_HH

