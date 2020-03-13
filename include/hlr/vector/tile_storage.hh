#ifndef __HLR_VECTOR_TILE_STORAGE_HH
#define __HLR_VECTOR_TILE_STORAGE_HH
//
// Project     : HLib
// File        : arith.hh
// Description : tile-based arithmetic functions v2
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <unordered_map>
#include <mutex>

#include <hpro/cluster/TIndexSet.hh>
#include <hpro/blas/Vector.hh>

#include <hlr/utils/hash.hh>
#include <hlr/arith/blas.hh>
#include <hlr/matrix/tiling.hh>

namespace hlr { namespace vector {

namespace hpro = HLIB;

// map HLIB types to HLR 
using  indexset = hpro::TIndexSet;

using hlr::matrix::tile_is_map_t;

// tile type
template < typename value_t >
using  tile     = blas::vector< value_t >;

// tile mapping type
template < typename value_t >
using  tilemap  = std::unordered_map< indexset, tile< value_t >, indexset_hash >;

//
// represents storage for consecutive tiles
//
template < typename T_value >
class tile_storage
{
public:
    //
    // export local types
    //

    using value_t        = T_value;
    using size_type      = typename tilemap< value_t >::size_type;
    using iterator       = typename tilemap< value_t >::iterator;
    using const_iterator = typename tilemap< value_t >::const_iterator;
 
private:
    // the map of tiles
    tilemap< value_t >  _tiles;

    // lock for concurrent access protection
    mutable std::mutex  _mtx;

public:
    // ctors
    tile_storage () {}

    tile_storage ( tile_storage &&  ts )
            : _tiles( std::move( ts._tiles ) )
    {}
    
    //
    // copy operators
    //

    tile_storage &
    operator =  ( tile_storage &&  ts )
    {
        _tiles = std::move( ts._tiles );

        return *this;
    }

    //
    // index operators
    //
    
    tile< value_t > &
    operator [] ( const indexset  is )
    {
        std::scoped_lock  lock( _mtx );

        return _tiles[is];
    }

    tile< value_t > &
    at ( const indexset  is )
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.at( is );
    }

    const tile< value_t > &
    at ( const indexset  is ) const
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.at( is );
    }

    //
    // iterators
    //

    iterator
    begin () noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.begin();
    }

    const_iterator
    begin () const noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.begin();
    }

    const_iterator
    cbegin () const noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.cbegin();
    }

    iterator
    end () noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.end();
    }

    const_iterator
    end () const noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.cend();
    }

    const_iterator
    cend () const noexcept
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.cend();
    }

    //
    // size information
    //

    size_type
    size () const noexcept
    {
        return _tiles.size();
    }

    bool
    empty () const noexcept
    {
        return _tiles.empty();
    }

    //
    // misc.
    //

    bool
    contains ( const indexset &  is ) const
    {
        std::scoped_lock  lock( _mtx );

        return _tiles.contains( is );
    }
};

//
// convert tile_storage to blas::vector
//
template < typename value_t >
blas::vector< value_t >
to_dense ( const tile_storage< value_t > &  ts )
{
    //
    // determine length
    //

    bool      first = true;
    indexset  row_is;

    for ( auto & [ is, ts_i ] : ts )
    {
        if ( first )
        {
            row_is = is;
            first  = false;
        }// if
        else
            row_is = join( row_is, is );
    }// for

    blas::vector< value_t >  v( row_is.size() );

    for ( auto & [ is, ts_i ] : ts )
    {
        blas::vector< value_t >  v_i( v, is - row_is.first() );

        blas::copy( ts_i, v_i );
    }// for

    return std::move( v );
}

}} // namespace hlr::vector

#endif  // __HLR_VECTOR_TILE_STORAGE_HH
