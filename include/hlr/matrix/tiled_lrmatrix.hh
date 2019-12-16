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

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/tile_storage.hh>
#include <hlr/vector/tiled_scalarvector.hh>
#include <hlr/utils/checks.hh>

namespace hlr
{ 

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using hpro::real;
using hpro::complex;
using hpro::idx_t;

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
class tiled_lrmatrix : public hpro::TMatrix
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
                     const blas::Matrix< value_t > &  aU,
                     const blas::Matrix< value_t > &  aV )
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

    // use "op" versions from TMatrix
    using TMatrix::nrows;
    using TMatrix::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( _rank == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // tile management
    //

    // allocate storage for all tiles
    void  init_tiles ();

    // copy data from given factors to local tiles (allocate if needed)
    void  copy_tiles ( const blas::Matrix< value_t > &  U,
                       const blas::Matrix< value_t > &  V );
    
    //
    // change field type 
    //
    
    virtual void  to_real     () { assert( false ); }
    virtual void  to_complex  () { assert( false ); }

    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec ( const real             alpha,
                           const hpro::TVector *  x,
                           const real             beta,
                           hpro::TVector       *  y,
                           const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const hpro::TTruncAcc & acc );

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( tiled_lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto  create  () const -> std::unique_ptr< hpro::TMatrix > { return std::make_unique< tiled_lrmatrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< hpro::TMatrix >;

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const -> std::unique_ptr< hpro::TMatrix >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< hpro::TMatrix >;

    // copy matrix data to \a A
    virtual void   copy_to      ( hpro::TMatrix *          A ) const;

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( hpro::TMatrix *          A,
                                  const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const;
    
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
    _U.tile_is().reserve( _row_is.size() / _ntile + 1 );
    
    for ( idx_t  i = _row_is.first(); i < _row_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::min< idx_t >( i + _ntile - 1, _row_is.last() ) );

        _U[ is_i ] = blas::Matrix< value_t >( is_i.size(), _rank );
        _U.tile_is().push_back( is_i );
    }// for

    _V.tile_is().reserve( _col_is.size() / _ntile + 1 );

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset  is_i( i, std::min< idx_t >( i + _ntile - 1, _col_is.last() ) );

        _V[ is_i ] = blas::Matrix< value_t >( is_i.size(), _rank );
        _V.tile_is().push_back( is_i );
    }// for
}

//
// copy data from given factors to local tiles (allocate if needed)
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::copy_tiles ( const blas::Matrix< value_t > &  U,
                                        const blas::Matrix< value_t > &  V )
{
    assert( U.ncols() == V.ncols() );

    _rank = U.ncols();
    
    _U.tile_is().reserve( _row_is.size() / _ntile + 1 );
    
    for ( idx_t  i = _row_is.first(); i < _row_is.last(); i += _ntile )
    {
        const indexset         is_i( i, std::min< idx_t >( i + _ntile - 1, _row_is.last() ) );
        const tile< value_t >  U_i( U, is_i - _row_is.first(), blas::Range::all );

        _U[ is_i ] = blas::Matrix< value_t >( U_i, hpro::copy_value );
        _U.tile_is().push_back( is_i );
    }// for

    _V.tile_is().reserve( _col_is.size() / _ntile + 1 );

    for ( idx_t  i = _col_is.first(); i < _col_is.last(); i += _ntile )
    {
        const indexset         is_i( i, std::min< idx_t >( i + _ntile - 1, _col_is.last() ) );
        const tile< value_t >  V_i( V, is_i - _col_is.first(), blas::Range::all );

        _V[ is_i ] = blas::Matrix< value_t >( V_i, hpro::copy_value );
        _V.tile_is().push_back( is_i );
    }// for
}

template < typename value_t >
void
tiled_lrmatrix< value_t >::mul_vec ( const real             alpha,
                                     const hpro::TVector *  ax,
                                     const real             beta,
                                     hpro::TVector       *  ay,
                                     const hpro::matop_t    op ) const
{
    using  vector = blas::Vector< value_t >;
        
    assert( ax->is_complex() == this->is_complex() );
    assert( ay->is_complex() == this->is_complex() );
    assert( ax->is() == this->col_is( op ) );
    assert( ay->is() == this->row_is( op ) );
    assert( is_scalar_all( ax, ay ) );

    // exclude complex value and transposed operation for now
    assert( (  op == hpro::apply_normal     ) ||
            (  op == hpro::apply_adjoint    ) ||
            (( op == hpro::apply_transposed ) && ! hpro::is_complex_type< value_t >::value ) );

    const auto  x = cptrcast( ax, hpro::TScalarVector );
    const auto  y = ptrcast(  ay, hpro::TScalarVector );

    // y := β·y
    if ( beta != value_t(1) )
        blas::scale( beta, hpro::blas_vec< value_t >( y ) );
                     
    vector  t( _rank );
            
    if ( op == hpro::apply_normal )
    {
        // t := Σ V_i^H x_i
        for ( const auto & [ is, V_i ] : _V )
        {
            const auto  x_i = vector( hpro::blas_vec< value_t >( x ), is - _col_is.first() );

            blas::mulvec( value_t(1), blas::adjoint( V_i ), x_i, value_t(1), t );
        }// for

        // y_i := y_i + α U_i t
        for ( const auto & [ is, U_i ] : _U )
        {
            auto  y_i = vector( hpro::blas_vec< value_t >( y ), is - _row_is.first() );

            blas::mulvec( value_t(alpha), U_i, t, value_t(1), y_i );
        }// for
    }// if
    else
    {
        // t := Σ U_i^H x_i
        for ( const auto & [ is, U_i ] : _U )
        {
            const auto  x_i = vector( hpro::blas_vec< value_t >( x ), is - _row_is.first() );

            blas::mulvec( value_t(1), blas::adjoint( U_i ), x_i, value_t(1), t );
        }// for

        // y_i := y_i + α V_i t
        for ( const auto & [ is, V_i ] : _V )
        {
            auto  y_i = vector( hpro::blas_vec< value_t >( y ), is - _col_is.first() );

            blas::mulvec( value_t(alpha), V_i, t, value_t(1), y_i );
        }// for
    }// if
}


//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::truncate ( const hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
tiled_lrmatrix< value_t >::copy () const
{
    auto  M = std::make_unique< tiled_lrmatrix >( _row_is, _col_is, _ntile );

    M->copy_struct_from( this );

    M->_rank = _rank;
    
    for ( const auto & [ is, U_i ] : _U )
        M->_U[ is ] = std::move( blas::copy( U_i ) );

    for ( const auto & [ is, V_i ] : _V )
        M->_V[ is ] = std::move( blas::copy( V_i ) );

    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
tiled_lrmatrix< value_t >::copy ( const hpro::TTruncAcc &,
                                  const bool       ) const
{
    auto  M = std::make_unique< tiled_lrmatrix >( _row_is, _col_is, _ntile );

    M->_rank = _rank;
    
    for ( const auto & [ is, U_i ] : _U )
        M->_U[ is ] = std::move( blas::copy( U_i ) );

    for ( const auto & [ is, V_i ] : _V )
        M->_V[ is ] = std::move( blas::copy( V_i ) );

    return M;
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
tiled_lrmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< tiled_lrmatrix >( _row_is, _col_is, _ntile );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::copy_to ( hpro::TMatrix *  A ) const
{
    hpro::TMatrix::copy_to( A );
    
    assert( IS_TYPE( A, tiled_lrmatrix ) );

    auto  R = ptrcast( A, tiled_lrmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_ntile  = _ntile;
    R->_rank   = _rank;

    // assuming no other tiles present
    
    for ( const auto & [ is, U_i ] : _U )
        R->_U[ is ] = std::move( blas::copy( U_i ) );

    for ( const auto & [ is, V_i ] : _V )
        R->_V[ is ] = std::move( blas::copy( V_i ) );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
tiled_lrmatrix< value_t >::copy_to ( hpro::TMatrix *          A,
                                     const hpro::TTruncAcc &,
                                     const bool          ) const
{
    hpro::TMatrix::copy_to( A );
    
    assert( IS_TYPE( A, tiled_lrmatrix ) );

    auto  R = ptrcast( A, tiled_lrmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_ntile  = _ntile;
    R->_rank   = _rank;

    // assuming no other tiles present
    
    for ( const auto & [ is, U_i ] : _U )
        R->_U[ is ] = std::move( blas::copy( U_i ) );

    for ( const auto & [ is, V_i ] : _V )
        R->_V[ is ] = std::move( blas::copy( V_i ) );
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
tiled_lrmatrix< value_t >::byte_size () const
{
    size_t  size = hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_U) + sizeof(_V);
    size += sizeof(_rank) + sizeof(_ntile);

    for ( const auto & [ is, U_i ] : _U )
        size += sizeof(is) + sizeof(value_t) * U_i.nrows() * U_i.ncols();

    for ( const auto & [ is, V_i ] : _V )
        size += sizeof(is) + sizeof(value_t) * V_i.nrows() * V_i.ncols();

    return size;
}

//
// type test
//
inline
bool
is_tiled_lowrank ( const hpro::TMatrix &  M )
{
    return ! IS_TYPE( &M, tiled_lrmatrix );
}

inline
bool
is_tiled_lowrank ( const hpro::TMatrix *  M )
{
    return ! is_null( M ) && IS_TYPE( M, tiled_lrmatrix );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_TILED_LRMATRIX_HH
