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

// local matrix type
DECLARE_TYPE( TRkMatrix );

namespace hlr { namespace matrix {

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
class tiled_lrmatrix : public HLIB::TMatrix
{
public:
    //
    // export local types
    //

    // value type
    using  value_t   = T_value;

    // map HLIB type to HLR 
    using  indexset  = HLIB::TIndexSet;
    
    // tile type
    using  tile_t    = HLIB::BLAS::Matrix< value_t >;

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
    {
        set_complex( HLIB::is_complex_type< value_t >::value );
    }
    
    tiled_lrmatrix ( const indexset  arow_is,
                     const indexset  acol_is,
                     const size_t    antile )
            : tiled_lrmatrix()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _ntile(  antile )
    {
        set_ofs( _row_is.first(), _col_is.first() );
        init_tiles();
    }

    tiled_lrmatrix ( const indexset                         arow_is,
                     const indexset                         acol_is,
                     const size_t                           antile,
                     const HLIB::BLAS::Matrix< value_t > &  aU,
                     const HLIB::BLAS::Matrix< value_t > &  aV )
            : tiled_lrmatrix()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _ntile(  antile )
    {
        set_ofs( _row_is.first(), _col_is.first() );
        copy_tiles( aU, aV );
    }

    // dtor
    virtual tiled_lrmatrix ()
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

    //! return true, if matrix is zero
    virtual bool    is_zero   () const { return ( _rank == 0 ); }
    
    virtual void    set_size  ( const size_t  nrows,
                                const size_t  ncols ) { assert( false ); }
    
    //
    // tile management
    //

    // allocate storage for all tiles
    void  init_tiles ();

    // copy data from given factors to local tiles (allocate if needed)
    void  copy_tiles ( const HLIB::BLAS::Matrix< value_t > &  U,
                       const HLIB::BLAS::Matrix< value_t > &  V );
    
    //
    // change field type 
    //
    
    virtual void  to_real     () { assert( false ); }
    virtual void  to_complex  () { assert( false ); }

    //
    // algebra routines
    //

    //! truncate matrix to accuracy \a acc
    virtual void truncate ( const HLIB::TTruncAcc & acc );

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( HLIB::TMatrix, tiled_lrmatrix )

    //
    // virtual constructor
    //

    //! return matrix of same class (but no content)
    virtual auto  create  () const -> std::unique_ptr< HLIB::TMatrix > { return std::make_unique< tiled_lrmatrix >(); }
};

//
// allocate storage for all tiles
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::init_tiles ()
{
}

//
// copy data from given factors to local tiles (allocate if needed)
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::copy_tiles ( const HLIB::BLAS::Matrix< value_t > &  U,
                                        const HLIB::BLAS::Matrix< value_t > &  V )
{
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::truncate ( const HLIB::TTruncAcc & acc )
{
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_TILED_LRMATRIX_HH
