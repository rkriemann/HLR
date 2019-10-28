#ifndef __HLR_MATRIX_TILE_STORAGE_HH
#define __HLR_MATRIX_TILE_STORAGE_HH
//
// Project     : HLib
// File        : arith.hh
// Description : tile-based arithmetic functions v2
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <map>
#include <mutex>

#include <cluster/TIndexSet.hh>
#include <blas/Matrix.hh>

namespace hlr { namespace matrix {

// map HLIB types to HLR 
using  indexset = HLIB::TIndexSet;
using  range    = HLIB::BLAS::Range;

// tile type
template < typename value_t >
using  tile     = HLIB::BLAS::Matrix< value_t >;

// tile mapping type
template < typename value_t >
using  tilemap  = std::map< indexset, tile< value_t > >;

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
    using size_type      = tilemap< value_t >::size_type;
    using iterator       = tilemap< value_t >::iterator;
    using const_iterator = tilemap< value_t >::const_iterator;
 
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

}} // namespace hlr::matrix

#endif  // __HLR_MATRIX_TILE_STORAGE_HH
