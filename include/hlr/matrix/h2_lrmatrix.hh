#ifndef __HLR_MATRIX_H2_LRMATRIX_HH
#define __HLR_MATRIX_H2_LRMATRIX_HH
//
// Project     : HLR
// Module      : h2_lrmatrix.hh
// Description : low-rank matrix with shared and nested cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( h2_lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class h2_lrmatrix : public Hpro::TMatrix< T_value >, public compress::compressible
{
public:
    //
    // export local types
    //

    // value type
    using  value_t = T_value;
    
private:
    // local index set of matrix
    indexset                           _row_is, _col_is;
    
    // low-rank factors in uniform storage
    nested_cluster_basis< value_t > *  _row_cb;
    nested_cluster_basis< value_t > *  _col_cb;

    // local coupling matrix
    blas::matrix< value_t >            _S;
    
    // #if HLR_HAS_COMPRESSION == 1
    // // stores compressed data
    // compress::zarray                   _zS;
    // #endif
    
public:
    //
    // ctors
    //

    h2_lrmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {}
    
    h2_lrmatrix ( const indexset  arow_is,
                  const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    h2_lrmatrix ( const indexset                     arow_is,
                  const indexset                     acol_is,
                  nested_cluster_basis< value_t > &  arow_cb,
                  nested_cluster_basis< value_t > &  acol_cb,
                  hlr::blas::matrix< value_t > &     aS )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S( blas::copy( aS ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    h2_lrmatrix ( const indexset                     arow_is,
                  const indexset                     acol_is,
                  nested_cluster_basis< value_t > &  arow_cb,
                  nested_cluster_basis< value_t > &  acol_cb,
                  hlr::blas::matrix< value_t > &&    aS )
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
    virtual ~h2_lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint                               rank     () const { return std::min( row_rank(), col_rank() ); }

    uint                               row_rank () const { HLR_ASSERT( ! is_null( _row_cb ) ); return _row_cb->rank(); }
    uint                               col_rank () const { HLR_ASSERT( ! is_null( _col_cb ) ); return _col_cb->rank(); }

    uint                               row_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_rank() : col_rank(); }
    uint                               col_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_rank() : row_rank(); }

    nested_cluster_basis< value_t > &  row_cb   () const { return *_row_cb; }
    nested_cluster_basis< value_t > &  col_cb   () const { return *_col_cb; }

    nested_cluster_basis< value_t > &  row_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_cb() : col_cb(); }
    nested_cluster_basis< value_t > &  col_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_cb() : row_cb(); }

    void
    set_cluster_bases ( nested_cluster_basis< value_t > &  arow_cb,
                        nested_cluster_basis< value_t > &  acol_cb )
    {
        HLR_ASSERT(( row_rank() == arow_cb.rank() ) &&
                   ( col_rank() == acol_cb.rank() ));
            
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
    }

    // return decompressed local coupling matrix
    hlr::blas::matrix< value_t >  coupling () const
    {
        // #if HLR_HAS_COMPRESSION == 1

        // if ( is_compressed() )
        // {
        //     auto  S = blas::matrix< value_t >( row_rank(), col_rank() );
    
        //     compress::decompress< value_t >( _zS, S );

        //     return S;
        // }// if

        // #endif

        return _S;
    }
    
    void
    set_coupling ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == row_rank() ) && ( aS.ncols() == col_rank() ));

        blas::copy( aS, _S );
    }
    
    void
    set_coupling ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == row_rank() ) && ( aS.ncols() == col_rank() ));

        _S = std::move( aS );
    }

    // set coupling matrix without bases consistency check
    // (because cluster bases need to be adjusted later)
    void
    set_coupling_unsafe ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        blas::copy( aS, _S );
    }
    
    void
    set_coupling_unsafe ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        _S = std::move( aS );
    }
    
    void
    set_matrix_data ( nested_cluster_basis< value_t > &  arow_cb,
                      const blas::matrix< value_t > &    aS,
                      nested_cluster_basis< value_t > &  acol_cb )
    {
        HLR_ASSERT(( aS.nrows() == arow_cb.rank() ) &&
                   ( aS.ncols() == acol_cb.rank() ));
            
        blas::copy( aS, _S );
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
    }

    void
    set_matrix_data ( nested_cluster_basis< value_t > &  arow_cb,
                      blas::matrix< value_t > &&         aS,
                      nested_cluster_basis< value_t > &  acol_cb )
    {
        HLR_ASSERT(( aS.nrows() == arow_cb.rank() ) &&
                   ( aS.ncols() == acol_cb.rank() ));
            
        _S = std::move( aS );
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
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
    
    // same as above but for blas::vector
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;
    using Hpro::TMatrix< value_t >::apply_add;
    
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
        // #if HLR_HAS_COMPRESSION == 1
        // return ! is_null( _zS.data() );
        // #else
        // return false;
        // #endif
        return false;
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( h2_lrmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< h2_lrmatrix >(); }

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
        // #if HLR_HAS_COMPRESSION == 1
        // _zS = compress::zarray();
        // #endif
    }
};

//
// matrix vector multiplication
//
template < typename value_t >
void
h2_lrmatrix< value_t >::mul_vec ( const value_t                     alpha,
                                  const Hpro::TVector< value_t > *  vx,
                                  const value_t                     beta,
                                  Hpro::TVector< value_t > *        vy,
                                  const Hpro::matop_t               op ) const
{
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    const auto  x = cptrcast( vx, Hpro::TScalarVector< value_t > );
    const auto  y = ptrcast(  vy, Hpro::TScalarVector< value_t > );
        
    // y := β·y
    if ( beta != value_t(1) )
        blas::scale( value_t(beta), blas::vec( y ) );

    apply_add( alpha, blas::vec( *x ), blas::vec( *y ), op );
}

template < typename value_t >
void
h2_lrmatrix< value_t >::apply_add ( const value_t                    alpha,
                                    const blas::vector< value_t > &  x,
                                    blas::vector< value_t > &        y,
                                    const Hpro::matop_t              op ) const
{
    HLR_ASSERT( x.length() == this->col_is( op ).size() );
    HLR_ASSERT( y.length() == this->row_is( op ).size() );

    const auto  S = coupling();
    
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·S·V^H x
        //
        
        auto  t = col_cb().transform_forward( x );    // t := V^H x
        auto  s = blas::mulvec( S, t );               // s := S t
        auto  r = row_cb().transform_backward( s );   // r := U s

        blas::add( value_t(alpha), r, y );            // y = y + r
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        HLR_ERROR( "TODO" );
        // //
        // // y = y + (U·S·V^H)^T x
        // //   = y + conj(V)·S^T·U^T x
        // //
        
        // auto  t = blas::mulvec( blas::transposed( U ), x ); // t := U^T x
        // auto  s = blas::mulvec( blas::transposed( S ), t ); // s := S^T t
        
        // blas::conj( s );                                    // r := conj(V) s
            
        // auto  r = blas::mulvec( V, s );

        // blas::conj( r );
        // blas::add( value_t(alpha), r, y );                  // y = y + r
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·S·V^H)^H x
        //   = y + V·S^H·U^H x
        //
        
        auto  t = row_cb().transform_forward( x );          // t := U^H x
        auto  s = blas::mulvec( blas::adjoint( S ), t );    // s := S t
        auto  r = col_cb().transform_backward( s );         // r := V s

        blas::add( value_t(alpha), r, y );                  // y = y + r
    }// if
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
h2_lrmatrix< value_t >::truncate ( const Hpro::TTruncAcc & )
{
}

//
// compress internal data
//
template < typename value_t >
void
h2_lrmatrix< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    // #if HLR_HAS_COMPRESSION == 1
        
    // if ( is_compressed() )
    //     return;

    // const size_t  mem_dense = sizeof(value_t) * _S.nrows() * _S.ncols();
    // auto          zS        = compress::compress( zconfig, _S );

    // if ( compress::byte_size( zS ) < mem_dense )
    // {
    //     _zS = std::move( zS );
    //     _S  = std::move( blas::matrix< value_t >( 0, 0 ) );
    // }// if

    // #endif
}

template < typename value_t >
void
h2_lrmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    // HLR_ASSERT( acc.rel_eps() == 0 );

    // if ( this->nrows() * this->ncols() == 0 )
    //     return;
        
    // compress( compress::get_config( acc( this->row_is(), this->col_is() ).abs_eps() ) );
}

//
// decompress internal data
//
template < typename value_t >
void
h2_lrmatrix< value_t >::decompress ()
{
    // #if HLR_HAS_COMPRESSION == 1
        
    // if ( ! is_compressed() )
    //     return;

    // _S = std::move( coupling() );

    // remove_compressed();

    // #endif
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
h2_lrmatrix< value_t >::copy () const
{
    auto  M = std::make_unique< h2_lrmatrix >( _row_is, _col_is, *_row_cb, *_col_cb, std::move( blas::copy( _S ) ) );

    M->copy_struct_from( this );

    // #if HLR_HAS_COMPRESSION == 1

    // if ( is_compressed() )
    // {
    //     M->_zS = compress::zarray( _zS.size() );
    //     std::copy( _zS.begin(), _zS.end(), M->_zS.begin() );
    // }// if

    // #endif
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
h2_lrmatrix< value_t >::copy ( const Hpro::TTruncAcc &,
                               const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
h2_lrmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< h2_lrmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
h2_lrmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    Hpro::TMatrix< value_t >::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, h2_lrmatrix ) );

    auto  R = ptrcast( A, h2_lrmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_row_cb = _row_cb;
    R->_col_cb = _col_cb;
    R->_S      = std::move( blas::copy( _S ) );

    // #if HLR_HAS_COMPRESSION == 1

    // R->_zS = compress::zarray( _zS.size() );
    // std::copy( _zS.begin(), _zS.end(), R->_zS.begin() );

    // #endif
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
h2_lrmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A,
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
h2_lrmatrix< value_t >::byte_size () const
{
    size_t  size = Hpro::TMatrix< value_t >::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_row_cb) + sizeof(_col_cb);
    size += sizeof(_S) + sizeof(value_t) * _S.nrows() * _S.ncols();

    // #if HLR_HAS_COMPRESSION == 1

    // size += compress::byte_size( _zS );

    // #endif
    
    return size;
}

//
// type test
//
template < typename value_t >
bool
is_h2_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, h2_lrmatrix );
}

template < typename value_t >
bool
is_h2_lowrank ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, h2_lrmatrix );
}

HLR_TEST_ALL( is_h2_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_h2_lowrank, Hpro::TMatrix< value_t > )

//
// replace current cluster basis by given cluster bases
// ASSUMPTION: bases have identical tree structure
//
template < typename value_t >
void
replace_cluster_basis ( Hpro::TMatrix< value_t > &         M,
                        nested_cluster_basis< value_t > &  rowcb,
                        nested_cluster_basis< value_t > &  colcb )
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
    else if ( is_h2_lowrank( M ) )
    {
        auto  R = ptrcast( & M, h2_lrmatrix< value_t > );

        R->set_cluster_bases( rowcb, colcb );
    }// if
}
    
}} // namespace hlr::matrix

#endif // __HLR_MATRIX_H2_LRMATRIX_HH
