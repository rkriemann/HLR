#ifndef __HLR_MATRIX_TILED_LRMATRIX_HH
#define __HLR_MATRIX_TILED_LRMATRIX_HH
//
// Project     : HLR
// File        : tiled_lrmatrix.hh
// Description : low-rank matrix with tiled storage
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <matrix/TMatrix.hh>

namespace std
{

// (partial) ordering of index sets
bool
operator < ( const HLIB::TIndexSet  is1,
             const HLIB::TIndexSet  is2 )
{
    return is1.is_strictly_left_of( is2 );
}

}// namespace std

namespace hlr { namespace matrix {

using namespace HLIB;
namespace hpro = HLIB;

// local matrix type
DECLARE_TYPE( tiled_lrmatrix );

//
// Represents a low-rank matrix in factorised form: U·V^H
// The matrices U and V are stored using tiles, e.g.
//
//       ⎡U_1⎤      ⎡V_1⎤
//   U = ⎢...⎥, V = ⎢...⎥ 
//       ⎣U_r⎦      ⎣V_s⎦
//
// with U_i, V_i of size ntile × k and r = #rows / ntile
// and s = #cols / ntile.
//
template < typename T_value >
class tiled_lrmatrix : public TMatrix
{
public:
    //
    // export local types
    //

    // value type
    using  value_t   = T_value;

    // map HLIB type to HLR 
    using  indexset  = TIndexSet;
    
    // tile type
    using  tile_t    = BLAS::Matrix< value_t >;

    // tile mapping type
    using  tilemap_t = std::map< indexset, tile_t >;
        
private:
    // local index set of matrix
    indexset   _row_is, _col_is;
    
    // low-rank factors in tiled storage:
    // mapping of (sub-) index set to tile
    tilemap_t  _U, _V;

    // numerical rank
    uint       _rank;

    // tile size
    size_t     _ntile;

public:
    //
    // ctors
    //

    tiled_lrmatrix ()
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _rank( 0 )
            , _ntile( 0 )
    {
    }
    
    tiled_lrmatrix ( const indexset  arow_is,
                     const indexset  acol_is,
                     const size_t    antile )
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
            , _ntile(  antile )
    {
        set_ofs( _row_is.first(), _col_is.first() );
        init_tiles();
    }

    tiled_lrmatrix ( const indexset                   arow_is,
                     const indexset                   acol_is,
                     const size_t                     antile,
                     const BLAS::Matrix< value_t > &  aU,
                     const BLAS::Matrix< value_t > &  aV )
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
            , _ntile(  antile )
    {
        set_ofs( _row_is.first(), _col_is.first() );
        copy_tiles( aU, aV );
    }

    // dtor
    virtual ~tiled_lrmatrix ()
    {}
    
    //
    // access internal data
    //

    tile_t &           tile_U ( const indexset &  is )       { return _U[ is ]; }
    const tile_t &     tile_U ( const indexset &  is ) const { return _U[ is ]; }
    
    tile_t &           tile_V ( const indexset &  is )       { return _V[ is ]; }
    const tile_t &     tile_V ( const indexset &  is ) const { return _V[ is ]; }

    tilemap_t &        U      ()       { return U; }
    const tilemap_t &  U      () const { return U; }

    tilemap_t &        V      ()       { return V; }
    const tilemap_t &  V      () const { return V; }
    
    uint               rank   () const { return _rank; }
    
    //
    // matrix data
    //
    
    virtual size_t  nrows     () const { return _row_is.size(); }
    virtual size_t  ncols     () const { return _col_is.size(); }

    virtual size_t  rows      () const { return nrows(); }
    virtual size_t  cols      () const { return ncols(); }

    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( _rank == 0 ); }
    
    virtual void    set_size  ( const size_t  nrows,
                                const size_t  ncols ) { assert( false ); }
    
    //
    // tile management
    //

    // allocate storage for all tiles
    void  init_tiles ();

    // copy data from given factors to local tiles (allocate if needed)
    void  copy_tiles ( const BLAS::Matrix< value_t > &  U,
                       const BLAS::Matrix< value_t > &  V );
    
    //
    // change field type 
    //
    
    virtual void  to_real     () { assert( false ); }
    virtual void  to_complex  () { assert( false ); }

    //
    // algebra routines
    //

    // truncate matrix to accuracy \a acc
    virtual void truncate ( const TTruncAcc & acc );

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( TMatrix, tiled_lrmatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto  create  () const -> std::unique_ptr< TMatrix > { return std::make_unique< tiled_lrmatrix >(); }
};

//
// allocate storage for all tiles
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::init_tiles ()
{
    for ( idx_t  i = _row_is.first(); i < _row_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::max< idx_t >( i + _ntile - 1, _row_is.last() ) );

        _U[ is_i ] = BLAS::Matrix< value_t >( is_i.size(), _rank );
    }// for

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::max< idx_t >( i + _ntile - 1, _col_is.last() ) );

        _V[ is_i ] = BLAS::Matrix< value_t >( is_i.size(), _rank );
    }// for
}

//
// copy data from given factors to local tiles (allocate if needed)
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::copy_tiles ( const BLAS::Matrix< value_t > &  U,
                                        const BLAS::Matrix< value_t > &  V )
{
    assert( U.ncols() == V.ncols() );

    _rank = U.ncols();
    
    for ( idx_t  i = _row_is.first(); i < _row_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::max< idx_t >( i + _ntile - 1, _row_is.last() ) );
        const tile_t    U_i( U, is_i - _row_is.first(), BLAS::Range::all );

        _U[ is_i ] = BLAS::Matrix< value_t >( U_i, copy_value );
    }// for

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::max< idx_t >( i + _ntile - 1, _col_is.last() ) );
        const tile_t    V_i( V, is_i - _col_is.first(), BLAS::Range::all );

        _V[ is_i ] = BLAS::Matrix< value_t >( V_i, copy_value );
    }// for
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::truncate ( const TTruncAcc & acc )
{
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_TILED_LRMATRIX_HH
