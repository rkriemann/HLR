#ifndef __HLR_MATRIX_LRMATRIX_HH
#define __HLR_MATRIX_LRMATRIX_HH
//
// Project     : HLR
// File        : lrmatrix.hh
// Description : low-rank matrix with (joined) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

namespace hpro = HLIB;

using indexset = hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
class lrmatrix : public hpro::TMatrix
{
private:
    // local index set of matrix
    indexset              _row_is, _col_is;
    
    // lowrank factors
    blas::generic_matrix  _U, _V;

    // indicates internal value type
    blas::value_type      _vtype;
    
public:
    //
    // ctors
    //

    lrmatrix ()
            : TMatrix()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _vtype( blas::value_type::undefined )
    {
    }
    
    lrmatrix ( const indexset                arow_is,
               const indexset                acol_is )
            : TMatrix()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _vtype( blas::value_type::undefined )
    {
        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &   aU,
               hlr::blas::matrix< value_t > &   aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( blas::copy( aU ) )
            , _V( blas::copy( aV ) )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _U ).nrows() ) &&
                   ( _col_is.size() == std::get< blas::matrix< value_t > >( _V ).nrows() ) &&
                   ( std::get< blas::matrix< value_t > >( _U ).ncols() == std::get< blas::matrix< value_t > >( _V ).ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &&  aU,
               hlr::blas::matrix< value_t > &&  aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( std::move( aU ) )
            , _V( std::move( aV ) )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _U ).nrows() ) &&
                   ( _col_is.size() == std::get< blas::matrix< value_t > >( _V ).nrows() ) &&
                   ( std::get< blas::matrix< value_t > >( _U ).ncols() == std::get< blas::matrix< value_t > >( _V ).ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint
    rank () const
    {
        return std::visit( [] ( auto &&  M ) { return M.ncols(); }, _U );
    }

    // return value type of matrix
    blas::value_type  value_type () const { return _vtype; }

    template < typename value_t > blas::matrix< value_t > &        U ()       { return std::get< blas::matrix< value_t > >( _U ); }
    template < typename value_t > blas::matrix< value_t > &        V ()       { return std::get< blas::matrix< value_t > >( _V ); }
    
    template < typename value_t > const blas::matrix< value_t > &  U () const { return std::get< blas::matrix< value_t > >( _U ); }
    template < typename value_t > const blas::matrix< value_t > &  V () const { return std::get< blas::matrix< value_t > >( _V ); }
    
    template < typename value_t >
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        if (( blas::value_type_v< value_t > == _vtype ) && ( aU.ncols() == U< value_t >().ncols() ))
        {
            blas::copy( aU, U< value_t >() );
            blas::copy( aV, V< value_t >() );
        }// if
        else
        {
            _U = blas::copy( aU );
            _V = blas::copy( aV );

            _vtype = blas::value_type_v< value_t >;
        }// else
    }
    
    template < typename value_t >
    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        _U     = std::move( aU );
        _V     = std::move( aV );
        _vtype = blas::value_type_v< value_t >;
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
    // change value type 
    //
    
    virtual void  to_real     () { HLR_ASSERT( false ); }
    virtual void  to_complex  () { HLR_ASSERT( false ); }

    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec  ( const real             alpha,
                            const hpro::TVector *  x,
                            const real             beta,
                            hpro::TVector       *  y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void cmul_vec ( const complex          alpha,
                            const hpro::TVector *  x,
                            const complex          beta,
                            hpro::TVector       *  y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const hpro::TTruncAcc & acc );

    // scale matrix by alpha
    virtual void scale    ( const real  alpha )
    {
        std::visit(
            [alpha] ( auto &&  M )
            {
                using  value_t  = typename std::decay_t< decltype(M) >::value_t;
                
                blas::scale( value_t(alpha), M );
            },
            _U );
    }

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< hpro::TMatrix > { return std::make_unique< lrmatrix >(); }

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
inline
void
lrmatrix::mul_vec ( const real             alpha,
                    const hpro::TVector *  vx,
                    const real             beta,
                    hpro::TVector *        vy,
                    const hpro::matop_t    op ) const
{
    // HLR_ASSERT( vx->is_complex() == this->is_complex() );
    // HLR_ASSERT( vy->is_complex() == this->is_complex() );
    // HLR_ASSERT( vx->is() == this->col_is( op ) );
    // HLR_ASSERT( vy->is() == this->row_is( op ) );
    // HLR_ASSERT( is_scalar_all( vx, vy ) );

    // // exclude complex value and transposed operation for now
    // HLR_ASSERT( (  op == hpro::apply_normal     ) ||
    //             (  op == hpro::apply_adjoint    ) ||
    //             (( op == hpro::apply_transposed ) && ! hpro::is_complex_type< value_t >::value ) );

    // const auto  x = cptrcast( vx, hpro::TScalarVector );
    // const auto  y = ptrcast(  vy, hpro::TScalarVector );

    // // y := β·y
    // if ( beta != real(1) )
    //     blas::scale( value_t(beta), hpro::blas_vec< value_t >( y ) );
                     
    // if ( op == hpro::apply_normal )
    // {
    //     //
    //     // y = y + U·V^H x
    //     //
        
    //     // t := V^H x
    //     auto  t = std::visit( [=] ( auto &&  V ) { return blas::mulvec( blas::adjoint( V ), hpro::blas_vec< value_t >( x ) ); }, _V );

    //     // r := U t
    //     auto  r = std::visit( [=] ( auto &&  U ) { return blas::mulvec( U, t ); }, _U );

    //     // y = y + r
    //     blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    // }// if
    // else if ( op == hpro::apply_transposed )
    // {
    //     // //
    //     // // y = y + (U·V^H)^T x
    //     // //   = y + conj(V)·U^T x
    //     // //
        
    //     // // t := U^T x
    //     // auto  t = blas::mulvec( blas::transposed( U() ), hpro::blas_vec< value_t >( x ) );
        
    //     // // r := conj(V) t
    //     // blas::conj( t );
            
    //     // auto  r = blas::mulvec( V(), t );

    //     // blas::conj( r );

    //     // // y = y + r
    //     // blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    // }// if
    // else if ( op == hpro::apply_adjoint )
    // {
    //     //
    //     // y = y + (U·V^H)^H x
    //     //   = y + V·U^H x
    //     //
        
    //     // t := U^H x
    //     auto  t = blas::mulvec( blas::adjoint( U() ), hpro::blas_vec< value_t >( x ) );

    //     // r := V t
    //     auto  r = blas::mulvec( V(), t );

    //     // y = y + r
    //     blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    // }// if
}

inline
void
lrmatrix::cmul_vec ( const complex          alpha,
                     const hpro::TVector *  vx,
                     const complex          beta,
                     hpro::TVector *        vy,
                     const hpro::matop_t    op ) const
{
    // HLR_ASSERT( vx->is_complex() == this->is_complex() );
    // HLR_ASSERT( vy->is_complex() == this->is_complex() );
    // HLR_ASSERT( vx->is() == this->col_is( op ) );
    // HLR_ASSERT( vy->is() == this->row_is( op ) );
    // HLR_ASSERT( is_scalar_all( vx, vy ) );

    // if constexpr( std::is_same_v< value_t, complex > )
    // {
    //     const auto  x = cptrcast( vx, hpro::TScalarVector );
    //     const auto  y = ptrcast(  vy, hpro::TScalarVector );
        
    //     // y := β·y
    //     if ( beta != complex(1) )
    //         blas::scale( value_t(beta), hpro::blas_vec< value_t >( y ) );
                     
    //     if ( op == hpro::apply_normal )
    //     {
    //         //
    //         // y = y + U·S·V^H x
    //         //
            
    //         // t := V^H x
    //         auto  t = blas::mulvec( blas::adjoint( col_basis() ), hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( _S, t );
        
    //         // r := U s
    //         auto  r = blas::mulvec( row_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == hpro::apply_transposed )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^T x
    //         //   = y + conj(V)·S^T·U^T x
    //         //
        
    //         // t := U^T x
    //         auto  t = blas::mulvec( blas::transposed( row_basis() ), hpro::blas_vec< value_t >( x ) );
        
    //         // s := S^T t
    //         auto  s = blas::mulvec( blas::transposed(_S), t );

    //         // r := conj(V) s
    //         blas::conj( s );
            
    //         auto  r = blas::mulvec( col_basis(), s );

    //         blas::conj( r );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == hpro::apply_adjoint )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^H x
    //         //   = y + V·S^H·U^H x
    //         //
        
    //         // t := U^H x
    //         auto  t = blas::mulvec( blas::adjoint( row_basis() ), hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( blas::adjoint(_S), t );
        
    //         // r := V s
    //         auto  r = blas::mulvec( col_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    // }// if
    // else
    //     HLR_ERROR( "todo" );
}


//
// truncate matrix to accuracy <acc>
//
inline
void
lrmatrix::truncate ( const hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy () const
{
    // auto  M = std::make_unique< lrmatrix >( _row_is, _col_is, *_row_cb, *_col_cb, std::move( blas::copy( _S ) ) );

    // M->copy_struct_from( this );
    
    // return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
inline
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy ( const hpro::TTruncAcc &,
                 const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy_struct  () const
{
    return std::make_unique< lrmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
inline
void
lrmatrix::copy_to ( hpro::TMatrix *  A ) const
{
    // hpro::TMatrix::copy_to( A );
    
    // HLR_ASSERT( IS_TYPE( A, lrmatrix ) );

    // auto  R = ptrcast( A, lrmatrix );

    // R->_row_is = _row_is;
    // R->_col_is = _col_is;
    // R->_row_cb = _row_cb;
    // R->_col_cb = _col_cb;
    // R->_S      = std::move( blas::copy( _S ) );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
inline
void
lrmatrix::copy_to ( hpro::TMatrix *          A,
                    const hpro::TTruncAcc &,
                    const bool          ) const
{
    return copy_to( A );
}

//
// return size in bytes used by this object
//
inline
size_t
lrmatrix::byte_size () const
{
    size_t  size = hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    // size += sizeof(_row_cb) + sizeof(_col_cb);
    // size += sizeof(_S) + sizeof(value_t) * _S.nrows() * _S.ncols();

    return size;
}

//
// type test
//
// inline
// bool
// is_lowrank ( const hpro::TMatrix &  M )
// {
//     return IS_TYPE( &M, lrmatrix );
// }

// inline
// bool
// is_lowrank ( const hpro::TMatrix *  M )
// {
//     return ! is_null( M ) && IS_TYPE( M, lrmatrix );
// }

// HLR_TEST_ALL( is_lowrank, hpro::TMatrix )
// HLR_TEST_ANY( is_lowrank, hpro::TMatrix )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRMATRIX_HH
