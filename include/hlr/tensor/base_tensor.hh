#ifndef __HLR_TENSOR_BASE_TENSOR_HH
#define __HLR_TENSOR_BASE_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/base_tensor
// Description : base class for tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>

#include <hpro/base/traits.hh>
#include <hpro/cluster/TIndexSet.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;
using Hpro::is;

namespace tensor
{

//
// defines basic interfaces and handles index sets
//
template < typename T_value >
class base_tensor3
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

public:
    //
    // ctors
    //

    base_tensor3 ()
            : _id(-1)
    {}

    base_tensor3 ( const base_tensor3 &  t )
            : _id( t._id )
            , _indexsets( t._indexsets )
    {}

    base_tensor3 ( base_tensor3 &&  t )
            : _id( std::move( t._id ) )
            , _indexsets( std::move( t._indexsets ) )
    {}

    base_tensor3 ( const indexset &  is0,
                   const indexset &  is1,
                   const indexset &  is2 )
            : _id(-1)
            , _indexsets{ is0, is1, is2 }
    {}
    
    base_tensor3 ( const std::array< indexset, 3 > &  ais )
            : _id(-1)
            , _indexsets{ ais[0], ais[1], ais[2] }
    {}
    
    // dtor
    virtual ~base_tensor3 ()
    {}

    // assignment
    base_tensor3 &  operator = ( const base_tensor3 &  t )
    {
        _id        = t._id;
        _indexsets = t._indexsets;
    }

    base_tensor3 &  operator = ( base_tensor3 &&  t )
    {
        _indexsets = std::move( t._indexsets );

        t._indexsets.fill( is( 0, -1 ) );
    }

    //
    // access internal data
    //

    int       id   () const { return _id; }
    
    uint      rank ()                const { return dimension; }
    size_t    dim  ( const uint  d ) const { HLR_DBG_ASSERT( d < dimension ); return _indexsets[d].size(); }
    indexset  is   ( const uint  d ) const { HLR_DBG_ASSERT( d < dimension ); return _indexsets[d]; }
    
    //
    // misc
    //
    
    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof(_indexsets) + sizeof(_id);
    }
};

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_BASE_TENSOR3_HH
