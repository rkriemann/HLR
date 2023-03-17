#ifndef __HLR_TENSOR_DENSE_TENSOR_HH
#define __HLR_TENSOR_DENSE_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/dense_tensor
// Description : dense (full) tensor with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <array>

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;
using Hpro::is;

namespace tensor
{

//
// implements dense (full) 3D tensor
// - storage layout is column-major
//
template < typename T_value >
class dense_tensor3
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;

    static constexpr uint  dimension = 3;

private:
    // globally unique id
    int                        _id;

    // index sets per dimensions
    std::array< indexset, 3 >  _indexsets;

    // tensor data
    blas::tensor3< value_t >   _tensor;

public:
    //
    // ctors
    //

    dense_tensor3 ()
            : _id(-1)
    {}

    dense_tensor3 ( const indexset &  is0,
                    const indexset &  is1,
                    const indexset &  is2 )
            : _id(-1)
            , _indexsets{ is0, is1, is2 }
            , _tensor( is0.size(), is1.size(), is2.size() )
    {}
    
    dense_tensor3 ( const std::array< indexset, 3 > &  ais )
            : _id(-1)
            , _indexsets{ ais[0], ais[1], ais[2] }
            , _tensor( ais[0].size(), ais[1].size(), ais[2].size() )
    {}
    
    dense_tensor3 ( const indexset &             is0,
                    const indexset &             is1,
                    const indexset &             is2,
                    blas::tensor3< value_t > &&  t )
            : _id(-1)
            , _indexsets{ is0, is1, is2 }
            , _tensor( std::move( t ) )
    {}
    
    dense_tensor3 ( const dense_tensor3 &  t )
            : _id( t._id )
            , _indexsets( t._indexsets )
            , _tensor( t._tensor.size() )
    {
        blas::copy( t._tensor, _tensor );
    }

    dense_tensor3 ( dense_tensor3 &&  t )
    {
        std::swap( _id,        t._id );
        std::swap( _indexsets, t._indexsets );
        std::swap( _tensor,    t._tensor );
    }

    // dtor
    virtual ~dense_tensor3 ()
    {}

    // assignment
    dense_tensor3 &  operator = ( const dense_tensor3 &  t )
    {
        _id        = t._id;
        _indexsets = t._indexsets;
        _tensor    = blas::copy( t._tensor );
    }

    dense_tensor3 &  operator = ( dense_tensor3 &&  t )
    {
        _indexsets = std::move( t._indexsets );
        _tensor    = std::move( t._tensor );

        t._indexsets.fill( is( 0, -1 ) );
    }

    //
    // access internal data
    //

    int              id   () const { return _id; }
    
    blas::tensor3< value_t > &        tensor ()       { return _tensor; }
    const blas::tensor3< value_t > &  tensor () const { return _tensor; }

    uint             rank ()                const { return dimension; }
    size_t           dim  ( const uint  d ) const { HLR_DBG_ASSERT( d < dimension ); return _indexsets[d].size(); }
    indexset         is   ( const uint  d ) const { HLR_DBG_ASSERT( d < dimension ); return _indexsets[d]; }

    value_t          coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const { return this->_tensor(i,j,l); }
    value_t &        coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l )       { return this->_tensor(i,j,l); }
    
    value_t          operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const { return coeff( i, j, l ); }
    value_t &        operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l )       { return coeff( i, j, l ); }
    
    //
    // misc
    //
    
    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return _tensor.byte_size() + sizeof(_indexsets) + sizeof(_id);
    }
};

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_DENSE_TENSOR3_HH
