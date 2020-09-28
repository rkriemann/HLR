#ifndef __HLR_MATRIX_UNIFORM_LRMATRIX_HH
#define __HLR_MATRIX_UNIFORM_LRMATRIX_HH
//
// Project     : HLR
// File        : uniform_lrmatrix.hh
// Description : low-rank matrix with (joined) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

namespace hpro = HLIB;

using indexset = hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( uniform_lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class uniform_lrmatrix : public hpro::TMatrix
{
public:
    //
    // export local types
    //

    // value type
    using  value_t = T_value;
    
private:
    // local index set of matrix
    indexset                          _row_is, _col_is;
    
    // low-rank factors in uniform storage:
    // mapping of (sub-) index set to tile
    const cluster_basis< value_t > *  _row_cb;
    const cluster_basis< value_t > *  _col_cb;

    // local coefficient matrix
    blas::Matrix< value_t >           _S;
    
public:
    //
    // ctors
    //

    uniform_lrmatrix ()
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
    {
    }
    
    uniform_lrmatrix ( const indexset                arow_is,
                       const indexset                acol_is )
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( arow_is )
            , _col_is( acol_is )
    {
        set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lrmatrix ( const indexset                    arow_is,
                       const indexset                    acol_is,
                       const cluster_basis< value_t > &  arow_cb,
                       const cluster_basis< value_t > &  acol_cb,
                       hlr::blas::matrix< value_t > &&   aS )
            : TMatrix( hpro::value_type< value_t >::value )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S( std::move( aS ) )
    {
        set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~uniform_lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint                              rank     () const { return std::min( _S.nrows(), _S.ncols() ); }

    uint                              row_rank () const { return _S.nrows(); }
    uint                              col_rank () const { return _S.ncols(); }

    uint                              row_rank ( const hpro::matop_t  op )       { return op == hpro::apply_normal ? row_rank() : col_rank(); }
    uint                              col_rank ( const hpro::matop_t  op )       { return op == hpro::apply_normal ? col_rank() : row_rank(); }

    // cluster_basis< value_t > &        row_cb   ()       { return *_row_cb; }
    const cluster_basis< value_t > &  row_cb   () const { return *_row_cb; }

    // cluster_basis< value_t > &        col_cb   ()       { return *_col_cb; }
    const cluster_basis< value_t > &  col_cb   () const { return *_col_cb; }

    // cluster_basis< value_t > &        row_cb   ( const hpro::matop_t  op )       { return op == hpro::apply_normal ? row_cb() : col_cb(); }
    const cluster_basis< value_t > &  row_cb   ( const hpro::matop_t  op ) const { return op == hpro::apply_normal ? row_cb() : col_cb(); }

    // cluster_basis< value_t > &        col_cb   ( const hpro::matop_t  op )       { return op == hpro::apply_normal ? col_cb() : row_cb(); }
    const cluster_basis< value_t > &  col_cb   ( const hpro::matop_t  op ) const { return op == hpro::apply_normal ? col_cb() : row_cb(); }

    void
    set_cluster_bases ( const cluster_basis< value_t > &  arow_cb,
                        const cluster_basis< value_t > &  acol_cb )
    {
        _row_cb = arow_cb;
        _col_cb = acol_cb;

        if (( _S.nrows() != _row_cb->rank() ) ||
            ( _S.ncols() != _col_cb->rank() ))
            _S = std::move( blas::Matrix< value_t >( _row_cb->rank(), _col_cb->rank() ) );
    }

    blas::matrix< value_t > &        coeff ()       { return _S; }
    const blas::matrix< value_t > &  coeff () const { return _S; }
    
    void
    set_coeff ( const blas::Matrix< value_t > &  aS )
    {
        HLR_ASSERT(( aS.nrows() == _row_cb->rank() ) && ( aS.ncols() == _col_cb->rank() ));

        blas::copy( aS, _S );
    }
    
    void
    set_coeff ( blas::Matrix< value_t > &&  aS )
    {
        HLR_ASSERT(( aS.nrows() == _row_cb->rank() ) && ( aS.ncols() == _col_cb->rank() ));

        _S = std::move( aS );
    }

    // set coefficient matrix without checking dimensions
    // (because cluster bases need to be adjusted later)
    void
    set_coeff_unsafe ( const blas::Matrix< value_t > &  aS )
    {
        blas::copy( aS, _S );
    }
    
    void
    set_coeff_unsafe ( blas::Matrix< value_t > &&  aS )
    {
        _S = std::move( aS );
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
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // change field type 
    //
    
    virtual void  to_real     () { HLR_ASSERT( false ); }
    virtual void  to_complex  () { HLR_ASSERT( false ); }

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

    HLIB_RTTI_DERIVED( uniform_lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< hpro::TMatrix > { return std::make_unique< uniform_lrmatrix >(); }

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
// matrix vector multiplication
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::mul_vec ( const real             alpha,
                                       const hpro::TVector *  vx,
                                       const real             beta,
                                       hpro::TVector *        vy,
                                       const hpro::matop_t    op ) const
{
    assert( vx->is_complex() == this->is_complex() );
    assert( vy->is_complex() == this->is_complex() );
    assert( vx->is() == this->col_is( op ) );
    assert( vy->is() == this->row_is( op ) );
    assert( is_scalar_all( vx, vy ) );

    // exclude complex value and transposed operation for now
    assert( (  op == hpro::apply_normal     ) ||
            (  op == hpro::apply_adjoint    ) ||
            (( op == hpro::apply_transposed ) && ! hpro::is_complex_type< value_t >::value ) );

    const auto  x = cptrcast( vx, hpro::TScalarVector );
    const auto  y = ptrcast(  vy, hpro::TScalarVector );

    // y := β·y
    if ( beta != real(1) )
        blas::scale( value_t(beta), hpro::blas_vec< value_t >( y ) );
                     
    if ( op == hpro::apply_normal )
    {
        //
        // y = y + U·S·V^H x
        //
        
        // t := V^H x
        auto  t = _col_cb->transform_forward( hpro::blas_vec< value_t >( x ) );

        // s := S t
        auto  s = blas::mulvec( value_t(1), _S, t );
        
        // r := U s
        auto  r = _row_cb->transform_backward( s );

        // y = y + r
        blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    }// if
    else if ( op == hpro::apply_transposed )
    {
        //
        // y = y + (U·S·V^H)^T x
        //   = y + conj(V)·S^T·U^T x
        //
        
        // t := U^T x = conj( conj(U^T) conj(x) ) = conj( U^H conj(x) )
        auto  cx = blas::copy( hpro::blas_vec< value_t >( x ) );

        blas::conj( cx );
        
        auto  t  = _row_cb->transform_forward( cx );

        blas::conj( t );
        
        // s := S^T t
        auto  s = blas::mulvec( value_t(1), blas::transposed(_S), t );
        
        // r := conj(V) s
        auto  r = _col_cb->transform_backward( s );

        // y = y + r
        blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    }// if
    else if ( op == hpro::apply_adjoint )
    {
        //
        // y = y + (U·S·V^H)^H x
        //   = y + V·S^H·U^H x
        //
        
        // t := U^H x
        auto  t = _row_cb->transform_forward( hpro::blas_vec< value_t >( x ) );

        // s := S t
        auto  s = blas::mulvec( value_t(1), blas::adjoint(_S), t );
        
        // r := V s
        auto  r = _col_cb->transform_backward( s );

        // y = y + r
        blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    }// if
}


//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::truncate ( const hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
uniform_lrmatrix< value_t >::copy () const
{
    auto  M = std::make_unique< uniform_lrmatrix >( _row_is, _col_is, *_row_cb, *_col_cb, std::move( blas::copy( _S ) ) );

    M->copy_struct_from( this );
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
uniform_lrmatrix< value_t >::copy ( const hpro::TTruncAcc &,
                                    const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
uniform_lrmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< uniform_lrmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::copy_to ( hpro::TMatrix *  A ) const
{
    hpro::TMatrix::copy_to( A );
    
    assert( IS_TYPE( A, uniform_lrmatrix ) );

    auto  R = ptrcast( A, uniform_lrmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_row_cb = _row_cb;
    R->_col_cb = _col_cb;
    R->_S      = std::move( blas::copy( _S ) );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::copy_to ( hpro::TMatrix *          A,
                                       const hpro::TTruncAcc &,
                                       const bool          ) const
{
    return copy_to( A );
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
uniform_lrmatrix< value_t >::byte_size () const
{
    size_t  size = hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_row_cb) + sizeof(_col_cb);
    size += sizeof(_S) + sizeof(value_t) * _S.nrows() * _S.ncols();

    return size;
}

//
// type test
//
inline
bool
is_uniform_lowrank ( const hpro::TMatrix &  M )
{
    return IS_TYPE( &M, uniform_lrmatrix );
}

inline
bool
is_uniform_lowrank ( const hpro::TMatrix *  M )
{
    return ! is_null( M ) && IS_TYPE( M, uniform_lrmatrix );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_UNIFORM_LRMATRIX_HH
