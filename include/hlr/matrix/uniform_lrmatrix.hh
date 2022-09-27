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
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/compressible.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

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
class uniform_lrmatrix : public Hpro::TMatrix< T_value >, public compressible
{
public:
    //
    // export local types
    //

    // value type
    using  value_t = T_value;
    
private:
    // local index set of matrix
    indexset                    _row_is, _col_is;
    
    // low-rank factors in uniform storage
    cluster_basis< value_t > *  _row_cb;
    cluster_basis< value_t > *  _col_cb;

    // local coupling matrix
    blas::matrix< value_t >     _S;
    
    #if HLR_HAS_COMPRESSION == 1
    // stores compressed data
    compress::zarray            _zS;
    #endif
    
public:
    //
    // ctors
    //

    uniform_lrmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {
    }
    
    uniform_lrmatrix ( const indexset  arow_is,
                       const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lrmatrix ( const indexset                   arow_is,
                       const indexset                   acol_is,
                       cluster_basis< value_t > &       arow_cb,
                       cluster_basis< value_t > &       acol_cb,
                       hlr::blas::matrix< value_t > &   aS )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S( blas::copy( aS ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lrmatrix ( const indexset                   arow_is,
                       const indexset                   acol_is,
                       cluster_basis< value_t > &       arow_cb,
                       cluster_basis< value_t > &       acol_cb,
                       hlr::blas::matrix< value_t > &&  aS )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S( std::move( aS ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~uniform_lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint                              rank     () const { return std::min( row_rank(), col_rank() ); }

    uint                              row_rank () const { HLR_ASSERT( ! is_null( _row_cb ) ); return _row_cb->rank(); }
    uint                              col_rank () const { HLR_ASSERT( ! is_null( _col_cb ) ); return _col_cb->rank(); }

    uint                              row_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_rank() : col_rank(); }
    uint                              col_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_rank() : row_rank(); }

    cluster_basis< value_t > &        row_cb   () const { return *_row_cb; }
    cluster_basis< value_t > &        col_cb   () const { return *_col_cb; }

    cluster_basis< value_t > &        row_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_cb() : col_cb(); }
    cluster_basis< value_t > &        col_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_cb() : row_cb(); }

    const blas::matrix< value_t > &   row_basis () const { return _row_cb->basis(); }
    const blas::matrix< value_t > &   col_basis () const { return _col_cb->basis(); }
    
    const blas::matrix< value_t > &   row_basis ( const matop_t  op ) const { return op == Hpro::apply_normal ? row_basis() : col_basis(); }
    const blas::matrix< value_t > &   col_basis ( const matop_t  op ) const { return op == Hpro::apply_normal ? col_basis() : row_basis(); }
    
    void
    set_cluster_bases ( cluster_basis< value_t > &  arow_cb,
                        cluster_basis< value_t > &  acol_cb )
    {
        HLR_ASSERT(( _S.nrows() == arow_cb.rank() ) &&
                   ( _S.ncols() == acol_cb.rank() ));
            
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
    }

    blas::matrix< value_t > &        coeff ()       { return _S; }
    const blas::matrix< value_t > &  coeff () const { return _S; }
    
    void
    set_coeff ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT(( aS.nrows() == _row_cb->rank() ) && ( aS.ncols() == _col_cb->rank() ));

        blas::copy( aS, _S );
    }
    
    void
    set_coeff ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT(( aS.nrows() == _row_cb->rank() ) && ( aS.ncols() == _col_cb->rank() ));

        _S = std::move( aS );
    }

    // set coupling matrix without bases consistency check
    // (because cluster bases need to be adjusted later)
    void
    set_coeff_unsafe ( const blas::matrix< value_t > &  aS )
    {
        blas::copy( aS, _S );
    }
    
    void
    set_coeff_unsafe ( blas::matrix< value_t > &&  aS )
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
    using Hpro::TMatrix< value_t >::nrows;
    using Hpro::TMatrix< value_t >::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec  ( const value_t                     alpha,
                            const Hpro::TVector< value_t > *  x,
                            const value_t                     beta,
                            Hpro::TVector< value_t > *        y,
                            const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const Hpro::TTruncAcc & acc );

    // scale matrix by alpha
    virtual void scale    ( const value_t  alpha )
    {
        blas::scale( alpha, _S );
    }

    //
    // compression
    //

    // compress internal data based on given configuration
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // compress internal data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        return ! is_null( _zS.data() );
        #else
        return false;
        #endif
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( uniform_lrmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< uniform_lrmatrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const Hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // copy matrix data to \a A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const;

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const Hpro::TTruncAcc &     acc,
                                  const bool                  do_coarsen = false ) const;
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const;

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        _zS = compress::zarray();
        #endif
    }
};

//
// matrix vector multiplication
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::mul_vec ( const value_t                     alpha,
                                       const Hpro::TVector< value_t > *  vx,
                                       const value_t                     beta,
                                       Hpro::TVector< value_t > *        vy,
                                       const Hpro::matop_t               op ) const
{
    HLR_ASSERT( vx->is_complex() == this->is_complex() );
    HLR_ASSERT( vy->is_complex() == this->is_complex() );
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    const auto  x = cptrcast( vx, Hpro::TScalarVector< value_t > );
    const auto  y = ptrcast(  vy, Hpro::TScalarVector< value_t > );
        
    // y := β·y
    if ( beta != value_t(1) )
        blas::scale( value_t(beta), Hpro::blas_vec< value_t >( y ) );
                     
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·S·V^H x
        //
            
        // t := V^H x
        auto  t = blas::mulvec( blas::adjoint( col_basis() ), Hpro::blas_vec< value_t >( x ) );

        // s := S t
        auto  s = blas::mulvec( _S, t );
        
        // r := U s
        auto  r = blas::mulvec( row_basis(), s );

        // y = y + r
        blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·S·V^H)^T x
        //   = y + conj(V)·S^T·U^T x
        //
        
        // t := U^T x
        auto  t = blas::mulvec( blas::transposed( row_basis() ), Hpro::blas_vec< value_t >( x ) );
        
        // s := S^T t
        auto  s = blas::mulvec( blas::transposed(_S), t );

        // r := conj(V) s
        blas::conj( s );
            
        auto  r = blas::mulvec( col_basis(), s );

        blas::conj( r );

        // y = y + r
        blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·S·V^H)^H x
        //   = y + V·S^H·U^H x
        //
        
        // t := U^H x
        auto  t = blas::mulvec( blas::adjoint( row_basis() ), Hpro::blas_vec< value_t >( x ) );

        // s := S t
        auto  s = blas::mulvec( blas::adjoint(_S), t );
        
        // r := V s
        auto  r = blas::mulvec( col_basis(), s );

        // y = y + r
        blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    }// if
}


//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::truncate ( const Hpro::TTruncAcc & )
{
}

//
// compress internal data
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;

    const size_t  mem_dense = sizeof(value_t) * _S.nrows() * _S.ncols();
    auto          zS        = compress::compress< value_t >( zconfig, _S.data(), _S.nrows(), _S.ncols() );

    if ( compress::byte_size( zS ) < mem_dense )
    {
        _zS = std::move( zS );
        _S  = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if

    #endif
}

template < typename value_t >
void
uniform_lrmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    if ( this->nrows() * this->ncols() == 0 )
        return;
        
    compress( compress::get_config( acc( this->row_is(), this->col_is() ).rel_eps() ) );
}

//
// decompress internal data
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    auto  M = blas::matrix< value_t >( row_rank(), col_rank() );
    
    compress::decompress< value_t >( _zS, M.data(), M.nrows(), M.ncols() );
        
    _S = std::move( M );

    remove_compressed();

    #endif
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
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
std::unique_ptr< Hpro::TMatrix< value_t > >
uniform_lrmatrix< value_t >::copy ( const Hpro::TTruncAcc &,
                                    const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
uniform_lrmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< uniform_lrmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
uniform_lrmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    Hpro::TMatrix< value_t >::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, uniform_lrmatrix ) );

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
uniform_lrmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A,
                                       const Hpro::TTruncAcc &,
                                       const bool  ) const
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
    size_t  size = Hpro::TMatrix< value_t >::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_row_cb) + sizeof(_col_cb);
    size += sizeof(_S) + sizeof(value_t) * _S.nrows() * _S.ncols();

    return size;
}

//
// type test
//
template < typename value_t >
bool
is_uniform_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, uniform_lrmatrix );
}

template < typename value_t >
bool
is_uniform_lowrank ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, uniform_lrmatrix );
}

HLR_TEST_ALL( is_uniform_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_uniform_lowrank, Hpro::TMatrix< value_t > )

//
// replace current cluster basis by given cluster bases
// ASSUMPTION: bases have identical tree structure
//
template < typename value_t >
void
replace_cluster_basis ( Hpro::TMatrix< value_t > &  M,
                        cluster_basis< value_t > &  rowcb,
                        cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( & M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );

                if ( ! is_null( B->block( i, j ) ) )
                    replace_cluster_basis( *B->block( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R = ptrcast( & M, uniform_lrmatrix< value_t > );

        R->set_cluster_bases( rowcb, colcb );
    }// if
}
    
}} // namespace hlr::matrix

#endif // __HLR_MATRIX_UNIFORM_LRMATRIX_HH
