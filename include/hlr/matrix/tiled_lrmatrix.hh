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

#include <hlr/matrix/tile_storage.hh>
#include <hlr/utils/checks.hh>

namespace std
{

// (partial) ordering of index sets
inline
bool
operator < ( const HLIB::TIndexSet  is1,
             const HLIB::TIndexSet  is2 )
{
    return is1.is_strictly_left_of( is2 );
}

}// namespace std

namespace hlr
{ 

using namespace HLIB;
namespace hpro = HLIB;

// local matrix type
DECLARE_TYPE( tiled_lrmatrix );

namespace matrix
{

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
    using  value_t = T_value;

private:
    // local index set of matrix
    indexset                 _row_is, _col_is;
    
    // low-rank factors in tiled storage:
    // mapping of (sub-) index set to tile
    tile_storage< value_t >  _U, _V;

    // numerical rank
    uint                     _rank;

    // tile size
    size_t                   _ntile;

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

    tile< value_t > &                tile_U ( const indexset &  is )       { return _U.at( is ); }
    const tile< value_t > &          tile_U ( const indexset &  is ) const { return _U.at( is ); }
    
    tile< value_t > &                tile_V ( const indexset &  is )       { return _V.at( is ); }
    const tile< value_t > &          tile_V ( const indexset &  is ) const { return _V.at( is ); }

    tile_storage< value_t > &        U ()       { return _U; }
    const tile_storage< value_t > &  U () const { return _U; }

    tile_storage< value_t > &        V ()       { return _V; }
    const tile_storage< value_t > &  V () const { return _V; }
    
    uint                  rank      () const { return _rank; }

    void                  set_lrmat ( tile_storage< value_t > &&  U,
                                      tile_storage< value_t > &&  V )
    {
        _U = std::move( U );
        _V = std::move( V );

        // adjust rank (assuming all tiles have same rank)
        for ( const auto & [ is, U_i ] : _U )
        {
            _rank = U_i.ncols();
            break;
        }// for
    }
    
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

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec ( const real       alpha,
                           const TVector *  x,
                           const real       beta,
                           TVector       *  y,
                           const matop_t    op = apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const TTruncAcc & acc );

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( tiled_lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto  create  () const -> std::unique_ptr< TMatrix > { return std::make_unique< tiled_lrmatrix >(); }

    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const;
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
        const indexset  is_i( i, std::min< idx_t >( i + _ntile - 1, _row_is.last() ) );

        _U[ is_i ] = BLAS::Matrix< value_t >( is_i.size(), _rank );
    }// for

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::min< idx_t >( i + _ntile - 1, _col_is.last() ) );

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
        const indexset         is_i( i, std::min< idx_t >( i + _ntile - 1, _row_is.last() ) );
        const tile< value_t >  U_i( U, is_i - _row_is.first(), BLAS::Range::all );

        _U[ is_i ] = BLAS::Matrix< value_t >( U_i, copy_value );
    }// for

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset         is_i( i, std::min< idx_t >( i + _ntile - 1, _col_is.last() ) );
        const tile< value_t >  V_i( V, is_i - _col_is.first(), BLAS::Range::all );

        _V[ is_i ] = BLAS::Matrix< value_t >( V_i, copy_value );
    }// for
}

template < typename value_t >
void
tiled_lrmatrix< value_t >::mul_vec ( const real       alpha,
                                     const TVector *  ax,
                                     const real       beta,
                                     TVector       *  ay,
                                     const matop_t    op ) const
{
    using  vector = BLAS::Vector< value_t >;
        
    assert( ax->is_complex() == this->is_complex() );
    assert( ay->is_complex() == this->is_complex() );
    assert( ax->is() == this->col_is( op ) );
    assert( ay->is() == this->row_is( op ) );
    assert( is_scalar_all( ax, ay ) );

    // exclude complex value and transposed operation for now
    assert( (  op == apply_normal     ) ||
            (  op == apply_adjoint    ) ||
            (( op == apply_transposed ) && ! is_complex_type< value_t >::value ) );

    const auto  x = cptrcast( ax, TScalarVector );
    const auto  y = ptrcast(  ay, TScalarVector );

    // y := β·y
    if ( beta != value_t(1) )
        BLAS::scale( beta, blas_vec< value_t >( y ) );
                     
    vector  t( _rank );
            
    if ( op == apply_normal )
    {
        // t := Σ V_i^H x_i
        for ( const auto & [ is, V_i ] : _V )
        {
            const auto  x_i = vector( blas_vec< value_t >( x ), is - _col_is.first() );

            BLAS::mulvec( value_t(1), BLAS::adjoint( V_i ), x_i, value_t(1), t );
        }// for

        // y_i := y_i + α U_i t
        for ( const auto & [ is, U_i ] : _U )
        {
            auto  y_i = vector( blas_vec< value_t >( y ), is - _row_is.first() );

            BLAS::mulvec( value_t(alpha), U_i, t, value_t(1), y_i );
        }// for
    }// if
    else
    {
        // t := Σ U_i^H x_i
        for ( const auto & [ is, U_i ] : _U )
        {
            const auto  x_i = vector( blas_vec< value_t >( x ), is - _row_is.first() );

            BLAS::mulvec( value_t(1), BLAS::adjoint( U_i ), x_i, value_t(1), t );
        }// for

        // y_i := y_i + α V_i t
        for ( const auto & [ is, V_i ] : _V )
        {
            auto  y_i = vector( blas_vec< value_t >( y ), is - _col_is.first() );

            BLAS::mulvec( value_t(alpha), V_i, t, value_t(1), y_i );
        }// for
    }// if
}


//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::truncate ( const TTruncAcc & acc )
{
}

    // return size in bytes used by this object
template < typename value_t >
size_t
tiled_lrmatrix< value_t >::byte_size () const
{
    size_t  size = TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_U) + sizeof(_V);
    size += sizeof(_rank) + sizeof(_ntile);

    for ( const auto & [ is, U_i ] : _U )
        size += sizeof(is) + sizeof(value_t) * U_i.nrows() * U_i.ncols();

    for ( const auto & [ is, V_i ] : _V )
        size += sizeof(is) + sizeof(value_t) * V_i.nrows() * V_i.ncols();

    return size;
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_TILED_LRMATRIX_HH
