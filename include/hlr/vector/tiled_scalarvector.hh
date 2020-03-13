#ifndef __HLR_VECTOR_TILED_SCALARVECTOR_HH
#define __HLR_VECTOR_TILED_SCALARVECTOR_HH
//
// Project     : HLR
// File        : tiled_scalarvector.hh
// Description : scalar vector using tiled storage
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hlr/matrix/tiling.hh>
#include <hlr/vector/tile_storage.hh>
#include <hlr/utils/checks.hh>

namespace hlr
{ 

namespace hpro = HLIB;
namespace math = hpro::Math;

using hpro::real;
using hpro::complex;
using hpro::idx_t;

// local vector type
DECLARE_TYPE( tiled_scalarvector );

namespace vector
{

//
// Represents a scalar vector stored using tiles, e.g.
//
//       ⎡v_1⎤
//   v = ⎢...⎥ 
//       ⎣v_r⎦
//
// with v_i of size ntile and r = #size / ntile.
//
template < typename T_value >
class tiled_scalarvector : public hpro::TVector
{
public:
    //
    // export local types
    //

    // value type
    using  value_t = T_value;

    using  mutex_map_t = std::unordered_map< indexset, std::mutex, indexset_hash >;

private:
    // local index set of vector
    indexset                 _is;
    
    // parts of the vector stored as tiles
    // mapping of (sub-) index set to tile
    tile_storage< value_t >  _tiles;

    // mapping of indexsets to mutices
    mutex_map_t              _tile_mutices;
    
    // tile size
    size_t                   _ntile;

public:
    //
    // ctors
    //

    tiled_scalarvector ()
            : TVector( 0, hpro::value_type< value_t >::value )
            , _is( 0, 0 )
            , _ntile( 0 )
    {}
    
    tiled_scalarvector ( const indexset &  ais,
                         tile_is_map_t &   atile_is_map )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _ntile( 0 )
    {
        init_tiles( atile_is_map );
    }

    tiled_scalarvector ( const indexset  ais,
                         const size_t    antile )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _ntile( antile )
    {
        init_tiles();
    }

    tiled_scalarvector ( const indexset                   ais,
                         const size_t                     antile,
                         const blas::Vector< value_t > &  av )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _ntile( antile )
    {
        copy_tiles( av );
    }

    // dtor
    virtual ~tiled_scalarvector ()
    {}
    
    //
    // access internal data
    //

    tile< value_t > &                at ( const indexset &  is )       { return _tiles.at( is ); }
    const tile< value_t > &          at ( const indexset &  is ) const { return _tiles.at( is ); }
    
    tile_storage< value_t > &        tiles ()       { return _tiles; }
    const tile_storage< value_t > &  tiles () const { return _tiles; }

    std::mutex &                     tile_mtx ( const indexset & is ) { return _tile_mutices.at( is ); }
    
    void
    set_tiles ( tile_storage< value_t > &&  atiles )
    {
        _tiles = std::move( atiles );
    }
    
    //
    // vector data
    //
    
    virtual size_t  size  () const { return _is.size(); }
    
    //
    // tile management
    //

    // allocate storage for all tiles
    void  init_tiles ();

    // allocate storage for all tiles
    void  init_tiles ( tile_is_map_t &  tile_is_map );

    // copy data from given factors to local tiles (allocate if needed)
    void  copy_tiles ( const blas::Vector< value_t > &  atiles );
    
    ////////////////////////////////////////////////
    //
    // BLAS functionality (real valued)
    //

    // fill with constant
    virtual void fill ( const real  a )
    {
        for ( auto [ is, v_is ] : _tiles )
            blas::fill( value_t(a), v_is );
    }

    // fill with random numbers
    virtual void fill_rand ( const uint  /* seed */ )
    {
        assert( false );
    }

    // scale vector by constant factor
    virtual void scale ( const real  a )
    {
        for ( auto [ is, v_is ] : _tiles )
            blas::scale( a, v_is );
    }

    // this ≔ a · vector
    virtual void assign ( const real             /* a */,
                          const hpro::TVector *  /* v */ )
    {
        assert( false );
    }

    // copy operator for all vectors
    hpro::TVector &
    operator = ( const hpro::TVector &  v )
    {
        assign( real(1), & v );
        return *this;
    }
    
    // return euclidean norm
    virtual real norm2 () const
    {
        value_t  f = 0;
        
        for ( auto [ is, v_is ] : _tiles )
            f += blas::dot( v_is, v_is );
        
        return hpro::re( math::sqrt( f ) );
    }

    // return infimum norm
    virtual real norm_inf () const
    {
        assert( false );
        return real(0);
    }
    
    // this ≔ this + α·x
    virtual void axpy ( const real             /* a */,
                        const hpro::TVector *  /* x */ )
    {
        assert( false );
    }
    
    ////////////////////////////////////////////////
    //
    // BLAS functionality (complex valued)
    //

    // conjugate entries
    virtual void conjugate ()
    {
        for ( auto [ is, v_is ] : _tiles )
            blas::conj( v_is );
    }
        
    // fill with constant
    virtual void cfill ( const complex & )
    { assert( false ); }

    // scale vector by constant factor
    virtual void cscale ( const complex & )
    { assert( false ); }

    // this ≔ f · vector
    virtual void cassign ( const complex &, const TVector * )
    { assert( false ); }

    // return dot-product, <x,y> = x^H · y, where x = this
    virtual complex dot  ( const TVector *  ) const
    {
        assert( false );
        return complex(0);
    }

    // return dot-product, <x,y> = x^T · y, where x = this
    virtual complex dotu ( const TVector *  ) const
    {
        assert( false );
        return complex(0);
    }

    // this ≔ this + α·x
    virtual void caxpy ( const complex &, const TVector * )
    { assert( false ); }

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( tiled_scalarvector, TVector )

    //
    // virtual constructor
    //

    // return vector of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< hpro::TVector > { return std::make_unique< tiled_scalarvector >(); }

    // return copy of vector
    virtual auto   copy         () const -> std::unique_ptr< hpro::TVector >;

    // copy vector data to A
    virtual void   copy_to      ( hpro::TVector *  v ) const;
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const;

protected:
    //
    // change field type 
    //
    
    virtual void  to_real     () { assert( false ); }
    virtual void  to_complex  () { assert( false ); }
};

//
// allocate storage for all tiles
//
template < typename value_t >
void
tiled_scalarvector< value_t >::init_tiles ()
{
    for ( idx_t  i = _is.first(); i < _is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::min< idx_t >( i + _ntile - 1, _is.last() ) );

        _tiles[ is_i ] = tile< value_t >( is_i.size() );
        _tile_mutices[ is_i ].lock();  // ensure mutex exists in map
        _tile_mutices[ is_i ].unlock();
    }// for
}

//
// allocate storage for all tiles as defined by given map
//
template < typename value_t >
void
tiled_scalarvector< value_t >::init_tiles ( tile_is_map_t &  tile_is_map )
{
    const auto &  tiles = tile_is_map[ _is ];

    for ( const auto  is : tiles )
    {
        _tiles[ is ] = tile< value_t >( is.size() );
        _tile_mutices[ is ].lock();  // ensure mutex exists in map
        _tile_mutices[ is ].unlock();
    }// for
}

//
// copy data from given vector to local tiles (allocate if needed)
//
template < typename value_t >
void
tiled_scalarvector< value_t >::copy_tiles ( const blas::Vector< value_t > &  v )
{
    assert( v.length() == _is.size() );
    
    for ( idx_t  i = _is.first(); i < _is.last(); i += _ntile )
    {
        const indexset         is_i( i, std::min< idx_t >( i + _ntile - 1, _is.last() ) );
        const tile< value_t >  v_i( v, is_i - _is.first() );

        _tiles[ is_i ] = tile< value_t >( v_i, hpro::copy_value );
        _tile_mutices[ is_i ].lock();
        _tile_mutices[ is_i ].unlock();
    }// for
}

//
// return copy of vector
//
template < typename value_t >
std::unique_ptr< hpro::TVector >
tiled_scalarvector< value_t >::copy () const
{
    auto  v = std::make_unique< tiled_scalarvector >( _is, _ntile );

    for ( const auto & [ is, v_i ] : _tiles )
        v->_tiles[ is ] = std::move( blas::copy( v_i ) );

    return v;
}

//
// copy vector data to v
//
template < typename value_t >
void
tiled_scalarvector< value_t >::copy_to ( hpro::TVector *  v ) const
{
    hpro::TVector::copy_to( v );
    
    assert( IS_TYPE( v, tiled_scalarvector ) );

    auto  sv = ptrcast( v, tiled_scalarvector );

    sv->_is    = _is;
    sv->_ntile = _ntile;

    // assuming no other tiles present
    for ( const auto & [ is, v_i ] : _tiles )
        sv->_tiles[ is ] = std::move( blas::copy( v_i ) );
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
tiled_scalarvector< value_t >::byte_size () const
{
    size_t  size = hpro::TVector::byte_size();

    size += sizeof(_is);
    size += sizeof(_tiles);
    size += sizeof(_ntile);

    for ( const auto & [ is, v_i ] : _tiles )
        size += sizeof(is) + sizeof(value_t) * v_i.length();

    return size;
}

//
// type test
inline
bool
is_tiled_scalar ( const hpro::TVector *  M )
{
    return ! is_null( M ) && IS_TYPE( M, tiled_scalarvector );
}

}} // namespace hlr::vector

#endif // __HLR_VECTOR_TILED_SCALARVECTOR_HH
