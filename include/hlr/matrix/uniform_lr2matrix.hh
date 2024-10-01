#ifndef __HLR_MATRIX_UNIFORM_LR2MATRIX_HH
#define __HLR_MATRIX_UNIFORM_LR2MATRIX_HH
//
// Project     : HLR
// Module      : uniform_lr2matrix.hh
// Description : low-rank matrix with shared cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/shared_cluster_basis.hh>
#include <hlr/compress/direct.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( uniform_lr2matrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class uniform_lr2matrix : public Hpro::TMatrix< T_value >, public compress::compressible
{
public:
    //
    // export local types
    //

    using  value_t         = T_value;
    using  cluster_basis_t = shared_cluster_basis< value_t >;
    
private:
    // local index set of matrix
    indexset                 _row_is, _col_is;

    // block rank
    uint                     _rank;
    
    // low-rank factors in uniform storage
    cluster_basis_t *        _row_cb;
    cluster_basis_t *        _col_cb;

    // row/column coupling matrices
    blas::matrix< value_t >  _S_row;
    blas::matrix< value_t >  _S_col;
    
    // stores compressed data
    compress::zarray         _zS_row;
    compress::zarray         _zS_col;
    
public:
    //
    // ctors
    //

    uniform_lr2matrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _rank( 0 )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {
    }
    
    uniform_lr2matrix ( const indexset  arow_is,
                        const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
            , _row_cb( nullptr )
            , _col_cb( nullptr )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lr2matrix ( const indexset            arow_is,
                       const indexset             acol_is,
                       cluster_basis_t &          arow_cb,
                       cluster_basis_t &          acol_cb,
                       blas::matrix< value_t > &  aSrow,
                       blas::matrix< value_t > &  aScol )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( aSrow.ncols() )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S_row( blas::copy( aSrow ) )
            , _S_col( blas::copy( aScol ) )
    {
        HLR_ASSERT( _row_cb->rank() == _S_row.nrows() );
        HLR_ASSERT( _col_cb->rank() == _S_col.nrows() );
        HLR_ASSERT( _S_row.ncols()  == _S_col.ncols() );
        
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lr2matrix ( const indexset              arow_is,
                        const indexset              acol_is,
                        cluster_basis_t &           arow_cb,
                        cluster_basis_t &           acol_cb,
                        blas::matrix< value_t > &&  aSrow,
                        blas::matrix< value_t > &&  aScol )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( aSrow.ncols() )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _S_row( std::move( aSrow ) )
            , _S_col( std::move( aScol ) )
    {
        HLR_ASSERT( _row_cb->rank() == _S_row.nrows() );
        HLR_ASSERT( _col_cb->rank() == _S_col.nrows() );
        HLR_ASSERT( _S_row.ncols()  == _S_col.ncols() );
        
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    uniform_lr2matrix ( const indexset       arow_is,
                        const indexset       acol_is,
                        cluster_basis_t &    arow_cb,
                        cluster_basis_t &    acol_cb,
                        const uint           arank,
                        compress::zarray &&  azSrow,
                        compress::zarray &&  azScol )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( arank )
            , _row_cb( &arow_cb )
            , _col_cb( &acol_cb )
            , _zS_row( std::move( azSrow ) )
            , _zS_col( std::move( azScol ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~uniform_lr2matrix ()
    {}
    
    //
    // access internal data
    //

    uint                           rank     () const { return _rank; }

    uint                           row_rank () const { HLR_ASSERT( ! is_null( _row_cb ) ); return _row_cb->rank(); }
    uint                           col_rank () const { HLR_ASSERT( ! is_null( _col_cb ) ); return _col_cb->rank(); }

    uint                           row_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_rank() : col_rank(); }
    uint                           col_rank ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_rank() : row_rank(); }

    cluster_basis_t &              row_cb   () const { return *_row_cb; }
    cluster_basis_t &              col_cb   () const { return *_col_cb; }

    cluster_basis_t &              row_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? row_cb() : col_cb(); }
    cluster_basis_t &              col_cb   ( const Hpro::matop_t  op ) const { return op == Hpro::apply_normal ? col_cb() : row_cb(); }

    const blas::matrix< value_t >  row_basis () const { return _row_cb->basis(); }
    const blas::matrix< value_t >  col_basis () const { return _col_cb->basis(); }
    
    const blas::matrix< value_t >  row_basis ( const matop_t  op ) const { return op == Hpro::apply_normal ? row_basis() : col_basis(); }
    const blas::matrix< value_t >  col_basis ( const matop_t  op ) const { return op == Hpro::apply_normal ? col_basis() : row_basis(); }
    
    void
    set_cluster_bases ( cluster_basis_t &  arow_cb,
                        cluster_basis_t &  acol_cb )
    {
        HLR_ASSERT( ( is_null( _row_cb ) || ( row_rank() == arow_cb.rank() )) &&
                    ( is_null( _col_cb ) || ( col_rank() == acol_cb.rank() )) );
            
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
    }

    // unsafe (without checks) versions
    void set_row_basis_unsafe ( cluster_basis_t &  acb ) { _row_cb = & acb; }
    void set_col_basis_unsafe ( cluster_basis_t &  acb ) { _col_cb = & acb; }

    // return true if row/column couplings are present
    bool  has_row_coupling () const { return _S_row.nrows() != 0; }
    bool  has_col_coupling () const { return _S_col.nrows() != 0; }
            
    // return coupling matrices
    blas::matrix< value_t >
    row_coupling () const
    {
        if ( is_compressed() )
        {
            auto  S = blas::matrix< value_t >( row_rank(), rank() );
    
            compress::decompress< value_t >( _zS_row, S );

            return S;
        }// if

        return _S_row;
    }
    
    blas::matrix< value_t >
    col_coupling () const
    {
        if ( is_compressed() )
        {
            auto  S = blas::matrix< value_t >( col_rank(), rank() );
    
            compress::decompress< value_t >( _zS_col, S );

            return S;
        }// if

        return _S_col;
    }
    
    hlr::blas::matrix< value_t >
    row_coupling ( const matop_t  op ) const
    {
        if ( op == apply_normal ) return row_coupling();
        else                      return col_coupling();
    }
    
    hlr::blas::matrix< value_t >
    col_coupling ( const matop_t  op ) const
    {
        if ( op == apply_normal ) return col_coupling();
        else                      return row_coupling();
    }
    
    void
    set_row_coupling ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == row_rank() ) && ( aS.ncols() == rank() ));

        blas::copy( aS, _S_row );
    }
    
    void
    set_col_coupling ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == col_rank() ) && ( aS.ncols() == rank() ));

        blas::copy( aS, _S_col );
    }
    
    void
    set_row_coupling ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == row_rank() ) && ( aS.ncols() == rank() ));

        _S_row = std::move( aS );
    }

    void
    set_col_coupling ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        HLR_ASSERT(( aS.nrows() == col_rank() ) && ( aS.ncols() == rank() ));

        _S_col = std::move( aS );
    }

    compress::zarray  zrow_coupling () const { return _zS_row; }
    compress::zarray  zcol_coupling () const { return _zS_col; }

    // unsafe (without checks) versions
    void
    set_row_coupling_unsafe ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        blas::copy( aS, _S_row );
        _rank  = _S_row.ncols();
    }
    
    void
    set_col_coupling_unsafe ( const blas::matrix< value_t > &  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        blas::copy( aS, _S_col );
        _rank  = _S_col.ncols();
    }
    
    void
    set_row_coupling_unsafe ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        _S_row = std::move( aS );
        _rank  = _S_row.ncols();
    }
    void
    set_col_coupling_unsafe ( blas::matrix< value_t > &&  aS )
    {
        HLR_ASSERT( ! is_compressed() );
        _S_col = std::move( aS );
        _rank  = _S_col.ncols();
    }

    void
    set_matrix_data ( cluster_basis_t &                arow_cb,
                      const blas::matrix< value_t > &  aS_row,
                      const blas::matrix< value_t > &  aS_col,
                      cluster_basis_t &                acol_cb )
    {
        HLR_ASSERT(( aS_row.nrows() == arow_cb.rank() ) &&
                   ( aS_col.nrows() == acol_cb.rank() ) &&
                   ( aS_row.ncols() == aS_col.ncols() ));
            
        blas::copy( aS_row, _S_row );
        blas::copy( aS_col, _S_col );
        _row_cb = & arow_cb;
        _col_cb = & acol_cb;
    }

    void
    set_matrix_data ( cluster_basis_t &           arow_cb,
                      blas::matrix< value_t > &&  aS_row,
                      blas::matrix< value_t > &&  aS_col,
                      cluster_basis_t &           acol_cb )
    {
        HLR_ASSERT(( aS_row.nrows() == arow_cb.rank() ) &&
                   ( aS_col.nrows() == acol_cb.rank() ) &&
                   ( aS_row.ncols() == aS_col.ncols() ));
            
        _S_row  = std::move( aS_row );
        _S_col  = std::move( aS_col );
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
    virtual void mul_vec        ( const value_t                     alpha,
                                  const Hpro::TVector< value_t > *  x,
                                  const value_t                     beta,
                                  Hpro::TVector< value_t > *        y,
                                  const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // same as above but for blas::vector
    virtual void  apply_add     ( const value_t                     alpha,
                                  const blas::vector< value_t > &   x,
                                  blas::vector< value_t > &         y,
                                  const matop_t                     op = apply_normal ) const;
    using Hpro::TMatrix< value_t >::apply_add;
    
    // matrix vector product in uniform format
    void         uni_apply_add  ( const value_t                     alpha,
                                  const blas::vector< value_t > &   ux, // uniform coefficients
                                  blas::vector< value_t > &         uy,
                                  const matop_t                     op = apply_normal ) const;

    // truncate matrix to accuracy \a acc
    virtual void truncate ( const accuracy & acc );

    // scale matrix by alpha
    virtual void scale    ( const value_t  alpha )
    {
        if ( is_compressed() ) { HLR_ERROR( "TODO" ); }
        else
        {
            if ( row_rank() < col_rank() )
                blas::scale( alpha, _S_row );
            else
                blas::scale( alpha, _S_col );
        }// else
    }

    //
    // compression
    //

    // compress internal data based on given accuracy
    virtual void   compress      ( const accuracy &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _zS_row.empty();
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( uniform_lr2matrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< uniform_lr2matrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const accuracy &  acc,
                                  const bool        do_coarsen = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // copy matrix data to \a A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const;

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const accuracy &            acc,
                                  const bool                  do_coarsen = false ) const;
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size      () const;

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const;
    
protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        _zS_row = compress::zarray();
        _zS_col = compress::zarray();
    }
};

//
// matrix vector multiplication
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::mul_vec ( const value_t                     alpha,
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
uniform_lr2matrix< value_t >::apply_add ( const value_t                    alpha,
                                          const blas::vector< value_t > &  x,
                                          blas::vector< value_t > &        y,
                                          const Hpro::matop_t              op ) const
{
    HLR_ASSERT( x.length() == this->col_is( op ).size() );
    HLR_ASSERT( y.length() == this->row_is( op ).size() );

    const auto  U     = row_basis();
    const auto  V     = col_basis();
    const auto  S_row = row_coupling();
    const auto  S_col = col_coupling();
    
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·S_r·S_c^H·V^H x
        //
        
        auto  t  = blas::mulvec( blas::adjoint( V ), x );     // t  := V^H x
        auto  s1 = blas::mulvec( blas::adjoint( S_col ), t ); // s₁ := S_c t
        auto  s2 = blas::mulvec( S_row, s1 );                 // s₂ := S_r s₁
        auto  r  = blas::mulvec( U, s2 );                     // r  := U s₂

        blas::add( value_t(alpha), r, y );                    // y = y + r
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·S_r·S_c^H·V^H)^T x
        //   = y + conj(V·S_c)·S^T·U^T x
        //
        
        auto  t  = blas::mulvec( blas::transposed( U ), x );      // t := U^T x
        auto  s1 = blas::mulvec( blas::transposed( S_row ), t );  // s₁ := S^T t
        
        blas::conj( s1 );                                         // r := conj(V) s₁
            
        auto  s2 = blas::mulvec( blas::transposed( S_row ), s1 ); // s := S^T t
        auto  r  = blas::mulvec( V, s2 );

        blas::conj( r );
        blas::add( value_t(alpha), r, y );                        // y = y + r
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·S_r·S_c^H·V^H)^H x
        //   = y + V·S_c·S_r^H·U^H x
        //
        
        auto  t = blas::mulvec( blas::adjoint( U ), x );      // t := U^H x
        auto  s1 = blas::mulvec( blas::adjoint( S_row ), t ); // s₁ := S_r t
        auto  s2 = blas::mulvec( S_col, s1 );                 // s₂ := S_c s₁
        auto  r = blas::mulvec( V, s2 );                      // r := V s₂

        blas::add( value_t(alpha), r, y );                    // y = y + r
    }// if
}

template < typename value_t >
void
uniform_lr2matrix< value_t >::uni_apply_add ( const value_t                    alpha,
                                              const blas::vector< value_t > &  ux,
                                              blas::vector< value_t > &        uy,
                                              const Hpro::matop_t              op ) const
{
    auto  t = blas::vector< value_t >( _rank );
    
    if ( is_compressed() )
    {
        #if defined(HLR_HAS_ZBLAS_DIRECT)
        
        switch ( op )
        {
            case apply_normal     :
                compress::zblas::mulvec( col_rank(), _rank, apply_adjoint, value_t(1), _zS_col, ux.data(), t.data() );
                compress::zblas::mulvec( row_rank(), _rank, apply_normal,       alpha, _zS_row, t.data(),  uy.data() );
                break;
                    
            case apply_conjugate  : { HLR_ASSERT( false ); }
            case apply_transposed : { HLR_ASSERT( false ); }
                    
            case apply_adjoint    :
                compress::zblas::mulvec( row_rank(), _rank, apply_adjoint, value_t(1), _zS_row, ux.data(), t.data() );
                compress::zblas::mulvec( col_rank(), _rank, apply_normal,       alpha, _zS_col, t.data(),  uy.data() );
                break;
                    
            default :
                HLR_ERROR( "unsupported matrix operator" );
        }// switch
        
        #else
        
        auto  Sr = row_coupling();
        auto  Sc = col_coupling();
        
        switch ( op )
        {
            case apply_normal     :
                blas::mulvec( value_t(1), blas::adjoint( Sc ), ux, value_t(1), t );
                blas::mulvec( alpha, Sr, t, value_t(1), uy );
                break;
                    
            case apply_conjugate  : { HLR_ASSERT( false ); }
            case apply_transposed : { HLR_ASSERT( false ); }
                    
            case apply_adjoint    :
                blas::mulvec( value_t(1), blas::adjoint( Sr ), ux, value_t(1), t );
                blas::mulvec( alpha, Sc, t, value_t(1), uy );
                break;
                    
            default               :
                HLR_ERROR( "unsupported matrix operator" );
        }// switch
        
        #endif
    }// if
    else
    {
        switch ( op )
        {
            case apply_normal     :
                blas::mulvec( value_t(1), blas::adjoint( _S_col ), ux, value_t(1), t );
                blas::mulvec( alpha, _S_row, t, value_t(1), uy );
                break;
                    
            case apply_conjugate  : { HLR_ASSERT( false ); }
            case apply_transposed : { HLR_ASSERT( false ); }
                    
            case apply_adjoint    :
                blas::mulvec( value_t(1), blas::adjoint( _S_row ), ux, value_t(1), t );
                blas::mulvec( alpha, _S_col, t, value_t(1), uy );
                break;
                    
            default               :
                HLR_ERROR( "unsupported matrix operator" );
        }// switch
    }// else
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::truncate ( const accuracy & )
{
}

//
// compress internal data
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::compress ( const accuracy &  acc )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    if ( is_compressed() )
        return;

    if ( this->nrows() * this->ncols() == 0 )
        return;

    const auto  lacc     = acc( this->row_is(), this->col_is() );
    auto        tol      = lacc.abs_eps();
    real_t      norm_row = real_t(1);
    real_t      norm_col = real_t(1);

    if ( lacc.abs_eps() != 0 )
    {
        if ( lacc.norm_mode() == Hpro::spectral_norm  )
        {
            norm_row = blas::norm_2( _S_row );
            norm_col = blas::norm_2( _S_col );
        }// if
        else if ( lacc.norm_mode() == Hpro::frobenius_norm )
        {
            norm_row = blas::norm_F( _S_row );
            norm_col = blas::norm_F( _S_col );
        }// if
        else
            HLR_ERROR( "unsupported norm mode" );
    }// if
    else if ( lacc.rel_eps() != 0 )
    {
        tol = lacc.rel_eps();
        // no norm required
    }// if
    else
        HLR_ERROR( "zero error" );

    const auto    zconf_row = compress::get_config( tol / norm_row );
    const auto    zconf_col = compress::get_config( tol / norm_col );
    const size_t  mem_dense = sizeof(value_t) * ( _S_row.nrows() * _S_row.ncols() + _S_col.nrows() * _S_col.ncols() );
    auto          zS_row    = compress::compress( zconf_row, _S_row );
    auto          zS_col    = compress::compress( zconf_col, _S_col );
    const auto    zmem      = compress::compressed_size( zS_row ) + compress::compressed_size( zS_col );

    // // DEBUG
    // {
    //     io::matlab::write( _S_row, "Sr" );
    //     io::matlab::write( _S_col, "Sc" );

    //     auto  Zr = blas::matrix< value_t >( _S_row.nrows(), _S_row.ncols() );
    //     auto  Zc = blas::matrix< value_t >( _S_col.nrows(), _S_col.ncols() );

    //     compress::decompress( zS_row, Zr );
    //     compress::decompress( zS_col, Zc );
        
    //     io::matlab::write( Zr, "Zr" );
    //     io::matlab::write( Zc, "Zc" );

    //     auto  S = blas::prod( _S_row, blas::adjoint( _S_col ) );
    //     auto  Z = blas::prod( Zr, blas::adjoint( Zc ) );

    //     blas::add( value_t(-1), S, Z );
        
    //     std::cout << blas::norm_F( Z ) / blas::norm_F( S ) << std::endl;
    // }
    // // DEBUG
    
    if (( zmem > 0 ) && ( zmem < mem_dense ))
    {
        HLR_ASSERT( (( zS_row.size() == 0 ) || ( zS_row.data() != nullptr )) &&
                    (( zS_col.size() == 0 ) || ( zS_col.data() != nullptr )) );

        _zS_row = std::move( zS_row );
        _zS_col = std::move( zS_col );
        _S_row  = std::move( blas::matrix< value_t >( 0, 0 ) );
        _S_col  = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    _S_row = std::move( row_coupling() );
    _S_col = std::move( col_coupling() );

    remove_compressed();
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
uniform_lr2matrix< value_t >::copy () const
{
    if ( is_compressed() )
    {
        auto  zS_row = compress::zarray( _zS_row.size() );
        auto  zS_col = compress::zarray( _zS_col.size() );
        
        std::copy( _zS_row.begin(), _zS_row.end(), zS_row.begin() );
        std::copy( _zS_col.begin(), _zS_col.end(), zS_col.begin() );

        auto  M = std::make_unique< uniform_lr2matrix >( _row_is, _col_is, *_row_cb, *_col_cb, rank(), std::move( zS_row ), std::move( zS_col ) );

        M->copy_struct_from( this );
        return M;
    }// if
    else
    {
        auto  M = std::make_unique< uniform_lr2matrix >( _row_is, _col_is, *_row_cb, *_col_cb, std::move( blas::copy( _S_row ) ), std::move( blas::copy( _S_col ) ) );

        M->copy_struct_from( this );
        return M;
    }// else
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
uniform_lr2matrix< value_t >::copy ( const accuracy &,
                                     const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
uniform_lr2matrix< value_t >::copy_struct  () const
{
    return std::make_unique< uniform_lr2matrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    Hpro::TMatrix< value_t >::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, uniform_lr2matrix ) );

    auto  R = ptrcast( A, uniform_lr2matrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_row_cb = _row_cb;
    R->_col_cb = _col_cb;
    R->_S_row  = std::move( blas::copy( _S_row ) );
    R->_S_col  = std::move( blas::copy( _S_col ) );

    R->_zS_row = compress::zarray( _zS_row.size() );
    std::copy( _zS_row.begin(), _zS_row.end(), R->_zS_row.begin() );
    R->_zS_col = compress::zarray( _zS_col.size() );
    std::copy( _zS_col.begin(), _zS_col.end(), R->_zS_col.begin() );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
uniform_lr2matrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A,
                                       const accuracy &,
                                       const bool  ) const
{
    return copy_to( A );
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
uniform_lr2matrix< value_t >::byte_size () const
{
    size_t  size = Hpro::TMatrix< value_t >::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += sizeof(_row_cb) + sizeof(_col_cb);
    size += sizeof(_rank);
    size += _S_row.byte_size() + _S_col.byte_size();
    size += compress::byte_size( _zS_row ) + compress::byte_size( _zS_col );

    return size;
}

//
// return size of (floating point) data in bytes handled by this object
//
template < typename value_t >
size_t
uniform_lr2matrix< value_t >::data_byte_size () const
{
    if ( is_compressed() )
        return hlr::compress::byte_size( _zS_row ) + hlr::compress::byte_size( _zS_col );
    else
        return sizeof( value_t ) * rank() * ( row_rank() + col_rank() );
}

//
// type test
//
template < typename value_t >
bool
is_uniform_lowrank2 ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, uniform_lr2matrix );
}

template < typename value_t >
bool
is_uniform_lowrank2 ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, uniform_lr2matrix );
}

HLR_TEST_ALL( is_uniform_lowrank2, hlr::matrix::is_uniform_lowrank2, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_uniform_lowrank2, hlr::matrix::is_uniform_lowrank2, Hpro::TMatrix< value_t > )

// //
// // replace current cluster basis by given cluster bases
// // ASSUMPTION: bases have identical tree structure
// //
// template < typename value_t >
// void
// replace_cluster_basis ( Hpro::TMatrix< value_t > &         M,
//                         shared_cluster_basis< value_t > &  rowcb,
//                         shared_cluster_basis< value_t > &  colcb )
// {
//     if ( is_blocked( M ) )
//     {
//         auto  B = ptrcast( & M, Hpro::TBlockMatrix< value_t > );

//         HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
//         HLR_ASSERT( B->nblock_cols() == colcb.nsons() );

//         for ( uint  i = 0; i < B->nblock_rows(); ++i )
//         {
//             auto  rowcb_i = rowcb.son( i );
            
//             for ( uint  j = 0; j < B->nblock_cols(); ++j )
//             {
//                 auto  colcb_j = colcb.son( j );

//                 if ( ! is_null( B->block( i, j ) ) )
//                     replace_cluster_basis( *B->block( i, j ), *rowcb_i, *colcb_j );
//             }// for
//         }// for
//     }// if
//     else if ( is_uniform_lowrank( M ) )
//     {
//         auto  R = ptrcast( & M, uniform_lr2matrix< value_t > );

//         R->set_cluster_bases( rowcb, colcb );
//     }// if
// }
    
}} // namespace hlr::matrix

#endif // __HLR_MATRIX_UNIFORM_LR2MATRIX_HH
