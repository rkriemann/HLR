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
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

namespace tensor
{

//
// implements dense (full) tensor
// - storage layout is column-major
//
template < typename T_value, uint C_dim >
class dense_tensor
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;

    using  multiindex = std::array< size_t, C_dim >;
    
    static constexpr uint  dimension = C_dim;

private:
    // globally unique id
    int                                _id;

    // index sets per dimensions
    std::array< indexset, dimension >  _indexsets;

    // tensor data
    std::vector< value_t >             _data;

public:
    //
    // ctors
    //

    dense_tensor ()
            : _id(-1)
    {}

    template < typename index_t >
    requires std::integral< index_t >
    dense_tensor ( std::initializer_list< index_t >  adims )
            : _id(-1)
    {
        HLR_ASSERT( adims.size() == dimension );
        
        std::array< size_t, dimension >  vdims;

        std::copy( adims.begin(), adims.end(), vdims.begin() );
        
        size_t  dim_prod = 1;

        for ( uint  i = 0; i < dimension; ++i )
        {
            _indexsets[i] = indexset( 0, vdims[i]-1 );
            dim_prod     *= vdims[i];
        }// for
        
        _data.resize( dim_prod );
    }
    
    dense_tensor ( std::initializer_list< indexset >  ais )
            : _id(-1)
            , _indexsets( ais )
    {
        size_t  dim_prod = 1;

        for ( auto  is : _indexsets )
            dim_prod *= is.size();

        _data.resize( dim_prod );
    }
    
    dense_tensor ( std::array< indexset, dimension > &  ais )
            : _id(-1)
            , _indexsets( ais )
    {
        size_t  dim_prod = 1;

        for ( auto  is : _indexsets )
            dim_prod *= is.size();

        _data.resize( dim_prod );
    }

    dense_tensor ( const dense_tensor &  t )
            : _id( t._id )
            , _indexsets( t._indexsets )
            , _data( t._data.size() )
    {
        std::copy( t._data.begin(), t._data.end(), _data.begin() );
    }

    dense_tensor ( dense_tensor &&  t )
    {
        std::swap( _id,        t._id );
        std::swap( _indexsets, t._indexsets );
        std::swap( _data,      t._data );
    }

    // dtor
    virtual ~dense_tensor ()
    {}

    // assignment
    dense_tensor &  operator = ( const dense_tensor &  t )
    {
        _indexsets = t._indexsets;
        _data.resize( t._data.size() );
        std::copy( t._data.begin(), t._data.end(), _data.begin() );
    }

    dense_tensor &  operator = ( dense_tensor &&  t )
    {
        _indexsets = std::move( t._indexsets );
        _data      = std::move( t._data );

        t._indexsets.fill( 0 );
    }

    //
    // access internal data
    //

    int              id   () const { return _id; }
    
    value_t *        data ()       { return _data.data(); }
    const value_t *  data () const { return _data.data(); }

    uint             rank () const { return dimension; }

    size_t           dim  ( const uint  d ) const
    {
        HLR_DBG_ASSERT( d < dimension );
        
        return  _indexsets[d].size();
    }

    indexset         is   ( const uint  d ) const
    {
        HLR_DBG_ASSERT( d < dimension );
        return _indexsets[d];
    }

    value_t          coeff       ( const multiindex &  idx ) const;
    value_t &        coeff       ( const multiindex &  idx );
    
    value_t          operator () ( const multiindex &  idx ) const { return coeff( idx ); }
    value_t &        operator () ( const multiindex &  idx )       { return coeff( idx ); }
    
    //
    // misc
    //
    
    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof(_data) + sizeof(value_t) * _data.size() + sizeof(_indexsets);
    }
};

template < typename value_t, uint dim >
value_t
dense_tensor< value_t, dim >::coeff ( const multiindex &  idx ) const
{
    size_t  pos = idx[dimension-1];

    for ( int  d = dimension-2; d >= 0; --d )
        pos = pos * this->_indexsets[d].size() + idx[d];
            
    return this->_data[pos];
}

template < typename value_t, uint dim >
value_t &
dense_tensor< value_t, dim >::coeff ( const multiindex &  idx )
{
    size_t  pos = idx[dimension-1];

    for ( int  d = dimension-2; d >= 0; --d )
        pos = pos * this->_indexsets[d].size() + idx[d];
            
    return this->_data[pos];
}

// template < typename value_t >
// struct dense_tensor< value_t, 3 >
// {
//     value_t
//     coeff ( const std::array< size_t, 3 > &  idx ) const
//     {
//         return this->_data[ this->_dims[0] * ( this->_dims[1] * idx[2] + idx[1] ) + idx[0] ];
//     }

//     value_t &
//     coeff ( const std::array< size_t, 3 > &  idx )
//     {
//         return this->_data[ this->_dims[0] * ( this->_dims[1] * idx[2] + idx[1] ) + idx[0] ];
//     }
// };

}} // namespace hlr::tensor

#endif // __HLR_TENSOR_DENSE_TENSOR_HH
