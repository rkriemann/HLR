#ifndef __HLIB_TENSOR_HH
#define __HLIB_TENSOR_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : tensor containers based on std::vector
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include <vector>

namespace HLIB
{

template < typename T >
struct tensor2
{
public:
    using  value_type      = T;
    using  pointer         = value_type *;
    using  const_pointer   = const pointer;
    using  reference       = value_type &;
    using  const_reference = const reference;
    using  size_type       = size_t;
    using  difference_type = ptrdiff_t;

private:
    std::vector< value_type >  data;
    const size_type      dim0;

public:
    tensor2 ( const size_type  adim0,
              const size_type  adim1 )
            : data( adim0 * adim1 )
            , dim0( adim0 )
    {}

    tensor2 ( const size_type  adim0,
              const size_type  adim1,
              const value_type       adefault )
            : data( adim0 * adim1, adefault )
            , dim0( adim0 )
    {}

    tensor2 ( const tensor2 &  t )
            : data( t.data )
            , dim0( t.dim0 )
    {}

    tensor2 ( tensor2 &&  t )
            : data( std::move( t.data ) )
            , dim0( t.dim0 )
    {}

    tensor2 &
    operator = ( tensor2 &&  t )
    {
        data = std::move( t.data );
        dim0 = t.dim0;

        return *this;
    }
    
    const value_type  operator ()  ( const size_type  i, const size_type  j ) const { return data[ j*dim0 + i ]; }
    value_type &      operator ()  ( const size_type  i, const size_type  j )       { return data[ j*dim0 + i ]; }
};
         
template < typename T >
struct tensor3
{
public:
    using  value_type      = T;
    using  pointer         = value_type *;
    using  const_pointer   = const pointer;
    using  reference       = value_type &;
    using  const_reference = const reference;
    using  size_type       = size_t;
    using  difference_type = ptrdiff_t;

private:
    std::vector< value_type >  data;
    const size_type            dim0;
    const size_type            dim1;

public:
    tensor3 ( const size_type  adim0,
              const size_type  adim1,
              const size_type  adim2 )
            : data( adim0 * adim1 * adim2 )
            , dim0( adim0 )
            , dim1( adim1 )
    {}

    tensor3 ( const size_type   adim0,
              const size_type   adim1,
              const size_type   adim2,
              const value_type  adefault )
            : data( adim0 * adim1 * adim2, adefault )
            , dim0( adim0 )
            , dim1( adim1 )
    {}

    tensor3 ( const tensor3 &  t )
            : data( t.data )
            , dim0( t.dim0 )
            , dim1( t.dim1 )
    {}

    tensor3 ( tensor3 &&  t )
            : data( std::move( t.data ) )
            , dim0( t.dim0 )
            , dim1( t.dim1 )
    {}

    tensor3 &
    operator = ( tensor3 &&  t )
    {
        data = std::move( t.data );
        dim0 = t.dim0;
        dim1 = t.dim1;

        return *this;
    }
    
    const value_type  operator ()  ( const size_type  i, const size_type  j, const size_type  k ) const { return data[ (k*dim1 + j)*dim0 + i ]; }
    value_type &      operator ()  ( const size_type  i, const size_type  j, const size_type  k )       { return data[ (k*dim1 + j)*dim0 + i ]; }
};
         
}// namespace HLIB

#endif   // __HLIB_TENSOR_HH
