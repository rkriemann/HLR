#ifndef __HLR_BEM_TENSOR_GRID_HH
#define __HLR_BEM_TENSOR_GRID_HH
//
// Project     : HLR
// File        : tensor_grid_hca.hh
// Description : grid defined via tensor product X × Y × Z
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <array>
#include <vector>

#include <hpro/base/TPoint.hh>
#include <hpro/cluster/TBBox.hh>

namespace hlr { namespace bem {

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using namespace hpro;

using  point        = hpro::T3Point;
using  bounding_box = hpro::TBBox;

//
// multi-index of dimension three
//

using  idx3_t = std::array< idx_t, 3 >;

//
// unfold given index i into (i₀,i₁,i₂) with
// constant dimension <dim> per index dimension
//
idx3_t
unfold ( const idx_t   i,
         const size_t  dim )
{
    idx_t   idx = i;
    idx3_t  midx;
        
    midx[0] = idx % dim; idx /= dim;
    midx[1] = idx % dim; idx /= dim;
    midx[2] = idx % dim;

    return  midx;
};

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
};

}}// namespace hlr::bem

#endif // __HLR_BEM_TENSOR_GRID_HH
