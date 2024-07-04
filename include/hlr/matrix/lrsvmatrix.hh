#ifndef __HLR_MATRIX_LRSVMATRIX_HH
#define __HLR_MATRIX_LRSVMATRIX_HH
//
// Project     : HLR
// Module      : lrsvmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/compress/compressible.hh>
#include <hlr/compress/direct.hh>
#include <hlr/compress/aplr.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

#include <hlr/utils/io.hh> // DEBUG

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrsvmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with orthogonal U/V and diagonal S, i.e., its singular
// values.
//
template < typename T_value >
class lrsvmatrix : public Hpro::TMatrix< T_value >, public compress::compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    // local index set of matrix
    indexset                 _row_is, _col_is;
    
    // low-rank factors
    blas::matrix< value_t >  _U, _V;

    // singular values
    blas::vector< real_t >   _S;

    // compressed storage
    compress::aplr::zarray   _zU, _zV;

    // rank for simplified access and for compressed factors
    uint                     _rank;
    
public:
    //
    // ctors
    //

    lrsvmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _rank( 0 )
    {}
    
    lrsvmatrix ( const indexset  arow_is,
                 const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( _row_is.size(), 0 ) // to avoid issues with nrows/ncols
            , _V( _col_is.size(), 0 )
            , _rank( 0 )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrsvmatrix ( const indexset              arow_is,
                 const indexset              acol_is,
                 blas::matrix< value_t > &   aU,
                 blas::vector< real_t > &    aS,
                 blas::matrix< value_t > &   aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
    {
        HLR_ASSERT(( arow_is.size() == aU.nrows() ) &&
                   ( acol_is.size() == aV.nrows() ) &&
                   ( aU.ncols()     == aV.ncols() ) &&
                   ( aU.ncols()     == aS.length() ));
        
        this->set_ofs( _row_is.first(), _col_is.first() );
        set_lrmat( aU, aS, aV );
    }

    lrsvmatrix ( const indexset              arow_is,
                 const indexset              acol_is,
                 blas::matrix< value_t > &&  aU,
                 blas::vector< real_t > &&   aS,
                 blas::matrix< value_t > &&  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
    {
        HLR_ASSERT(( arow_is.size() == aU.nrows() ) &&
                   ( acol_is.size() == aV.nrows() ) &&
                   ( aU.ncols()     == aV.ncols() ) &&
                   ( aU.ncols()     == aS.length() ));

        this->set_ofs( _row_is.first(), _col_is.first() );
        set_lrmat( std::move( aU ), std::move( aS ), std::move( aV ) );
    }

    lrsvmatrix ( const indexset              arow_is,
                 const indexset              acol_is,
                 blas::matrix< value_t > &   aU,
                 blas::matrix< value_t > &   aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
    {
        HLR_ASSERT(( arow_is.size() == aU.nrows() ) &&
                   ( acol_is.size() == aV.nrows() ) &&
                   ( aU.ncols()     == aV.ncols() ));

        this->set_ofs( _row_is.first(), _col_is.first() );
        set_lrmat( aU, aV );
    }

    lrsvmatrix ( const indexset              arow_is,
                 const indexset              acol_is,
                 blas::matrix< value_t > &&  aU,
                 blas::matrix< value_t > &&  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _rank( 0 )
    {
        HLR_ASSERT(( arow_is.size() == aU.nrows() ) &&
                   ( acol_is.size() == aV.nrows() ) &&
                   ( aU.ncols()     == aV.ncols() ));

        this->set_ofs( _row_is.first(), _col_is.first() );
        set_lrmat( std::move( aU ), std::move( aV ) );
    }

    // dtor
    virtual ~lrsvmatrix ()
    {}
    
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
    
    uint  rank  () const { return _rank; }
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  anrows,
                                const size_t  ancols )
    {
        // change of dimensions not supported
        HLR_ASSERT(( anrows == nrows() ) && ( ancols == ncols() ));
    }
    
    //
    // access low-rank factors
    //

    blas::matrix< value_t >  U () const;
    blas::matrix< value_t >  V () const;

    blas::matrix< value_t >  U ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U() : V() ); }
    blas::matrix< value_t >  V ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V() : U() ); }

    blas::vector< real_t > &        S ()       { return _S; }
    const blas::vector< real_t > &  S () const { return _S; }
    
    //
    // directly set low-rank factors
    //
    
    void  set_lrmat  ( const blas::matrix< value_t > &  aU,
                       const blas::matrix< value_t > &  aV );
    
    void  set_lrmat  ( blas::matrix< value_t > &&       aU,
                       blas::matrix< value_t > &&       aV );

    // assuming U/V are orthogonal!!!
    void  set_lrmat  ( const blas::matrix< value_t > &  aU,
                       const blas::vector< real_t > &   aS,
                       const blas::matrix< value_t > &  aV );
    
    void  set_lrmat  ( blas::matrix< value_t > &&       aU,
                       blas::vector< real_t > &&        aS,
                       blas::matrix< value_t > &&       aV );

    // same but recompress if compressed before
    void  set_lrmat  ( const blas::matrix< value_t > &  aU,
                       const blas::vector< real_t > &   aS,
                       const blas::matrix< value_t > &  aV,
                       const accuracy &                 acc );
    
    void  set_lrmat  ( blas::matrix< value_t > &&       aU,
                       blas::vector< real_t > &&        aS,
                       blas::matrix< value_t > &&       aV,
                       const accuracy &                 acc );

    // set U/V and recompress if currently compressed
    //  - neither U nor V are assumed to be orthogonal, so also reorthogonalize 
    void  set_U      ( blas::matrix< value_t > &&       aU,
                       const accuracy &                 acc );
    void  set_V      ( blas::matrix< value_t > &&       aV,
                       const accuracy &                 acc );
        
    //
    // algebra routines
    //

    //! compute y ≔ α·op(this)·x + β·y
    virtual void  mul_vec    ( const value_t                     alpha,
                               const Hpro::TVector< value_t > *  x,
                               const value_t                     beta,
                               Hpro::TVector< value_t > *        y,
                               const matop_t                     op = Hpro::apply_normal ) const;
    using Hpro::TMatrix< value_t >::mul_vec;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;
    using Hpro::TMatrix< value_t >::apply_add;
    
    // truncate matrix to accuracy acc
    virtual void truncate ( const Hpro::TTruncAcc & acc )
    {
        HLR_ERROR( "todo" );
    }

    // scale matrix by alpha
    virtual void scale    ( const value_t  alpha )
    {
        // TODO: may lead to loss in accuracy during compression (!!!)
        std::cout << "TODO" << std::endl;
        blas::scale( alpha, _S );
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( lrsvmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< lrsvmatrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return copy matrix wrt. given accuracy; if do_coarsen is set, perform coarsening
    virtual auto   copy         ( const Hpro::TTruncAcc &  /* acc */,
                                  const bool               /* do_coarsen */ = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return copy();
    }

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return std::make_unique< lrsvmatrix< value_t > >( _row_is, _col_is );
    }

    // copy matrix data to A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const;

    // copy matrix data to A and truncate w.r.t. acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const Hpro::TTruncAcc &     /* acc */,
                                  const bool                  /* do_coarsen */ = false ) const
    {
        return copy_to( A );
    }
    
    //
    // compression
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig ) { HLR_ERROR( "unsupported" ); }
    virtual void   compress      ( const Hpro::TTruncAcc &      acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return _zU.size() > 0;
    }

    // access multiprecision data
    const compress::aplr::zarray  zU () const { return _zU; }
    const compress::aplr::zarray  zV () const { return _zV; }
    
    void  set_zlrmat  ( compress::aplr::zarray &&  azU,
                        compress::aplr::zarray &&  azV )
    {
        // ASSUMPTION: compatible with S!!!
        _zU = std::move( azU );
        _zV = std::move( azV );
    }
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  size = Hpro::TMatrix< value_t >::byte_size();

        size += sizeof(_row_is) + sizeof(_col_is) + sizeof(_rank);
        
        size += _U.byte_size();
        size += _V.byte_size();
        size += _S.byte_size();
        size += compress::aplr::byte_size( _zU );
        size += compress::aplr::byte_size( _zV );
        
        return size;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        if ( is_compressed() )
            return sizeof(value_t) * _rank + compress::aplr::byte_size( _zU ) + compress::aplr::byte_size( _zV );
        else
            return sizeof(value_t) * ( _rank + _rank * ( _row_is.size() + _col_is.size() ) );
    }
    
    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        if ( is_compressed() )
        {
            auto  RU = U();
            auto  RV = V();

            RU.check_data();
            RV.check_data();
        }// if
        else
        {
            U().check_data();
            V().check_data();
        }// else
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        _zU = compress::aplr::zarray();
        _zV = compress::aplr::zarray();
    }
};

//
// type test
//
template < typename value_t >
inline
bool
is_lowrank_sv ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, lrsvmatrix );
}

template < typename value_t >
bool
is_lowrank_sv ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrsvmatrix );
}

HLR_TEST_ALL( is_lowrank_sv, hlr::matrix::is_lowrank_sv, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_lowrank_sv, hlr::matrix::is_lowrank_sv, Hpro::TMatrix< value_t > )

//
// access low-rank factors
//
template < typename value_t >
blas::matrix< value_t >
lrsvmatrix< value_t >::U () const
{
    if ( is_compressed() )
    {
        auto  dU = blas::matrix< value_t >( this->nrows(), _rank );
        uint  k  = 0;

        compress::aplr::decompress_lr( _zU, dU );

        // for ( uint  l = 0; l < dU.ncols(); ++l )
        // {
        //     auto  u_l = dU.column( l );

        //     blas::scale( _S(l), u_l );
        // }// for
            
        return dU;
    }// if
    else
    {
        return _U;
    }// else
}
    
template < typename value_t >
blas::matrix< value_t >
lrsvmatrix< value_t >::V () const
{
    if ( is_compressed() )
    {
        auto  dV = blas::matrix< value_t >( this->ncols(), _rank );

        compress::aplr::decompress_lr( _zV, dV );
            
        return dV;
    }// if
    else
    {
        return _V;
    }// else
}

//
// directly set low-rank factors
//
template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( const blas::matrix< value_t > &  aU,
                                   const blas::matrix< value_t > &  aV )
{
    HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
               ( this->ncols() == aV.nrows() ) &&
               ( aU.ncols()    == aV.ncols() ));

    if ( is_compressed() )
        remove_compressed();

    //
    // orthogonalise
    //

    auto  [ QU, RU ]    = blas::qr( aU );
    auto  [ QV, RV ]    = blas::qr( aV );
    auto  R             = blas::prod( RU, blas::adjoint(RV) );
    auto  [ Us, S, Vs ] = blas::svd( R );

    _U = std::move( blas::prod( QU, Us ) );
    _V = std::move( blas::prod( QV, Vs ) );
    _S = std::move( S );

    this->_rank = QU.ncols();
}
    
template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( blas::matrix< value_t > &&  aU,
                                   blas::matrix< value_t > &&  aV )
{
    HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
               ( this->ncols() == aV.nrows() ) &&
               ( aU.ncols()    == aV.ncols() ));

    if ( is_compressed() )
        remove_compressed();
        
    //
    // orthogonalise
    //

    auto  k  = aU.ncols();
    auto  RU = blas::matrix< value_t >( k, k );
    auto  RV = blas::matrix< value_t >( k, k );

    blas::qr( aU, RU );
    blas::qr( aV, RV );

    auto  R             = blas::prod( RU, blas::adjoint(RV) );
    auto  [ Us, S, Vs ] = blas::svd( R );

    _U = std::move( blas::prod( aU, Us ) );
    _V = std::move( blas::prod( aV, Vs ) );
    _S = std::move( S );

    this->_rank = aU.ncols();
}

template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( const blas::matrix< value_t > &  aU,
                                   const blas::vector< real_t > &   aS,
                                   const blas::matrix< value_t > &  aV )
{
    HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
               ( this->ncols() == aV.nrows() ) &&
               ( aU.ncols()    == aV.ncols() ) &&
               ( aU.ncols()    == aS.length() ));

    if ( is_compressed() )
        remove_compressed();

    _U    = std::move( blas::copy( aU ) );
    _S    = std::move( blas::copy( aS ) );
    _V    = std::move( blas::copy( aV ) );
    _rank = _S.length();
}
    
template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( blas::matrix< value_t > &&  aU,
                                   blas::vector< real_t > &&   aS,
                                   blas::matrix< value_t > &&  aV )
{
    HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
               ( this->ncols() == aV.nrows() ) &&
               ( aU.ncols()    == aV.ncols() ) &&
               ( aU.ncols()    == aS.length() ));

    if ( is_compressed() )
        remove_compressed();

    _U    = std::move( aU );
    _S    = std::move( aS );
    _V    = std::move( aV );
    _rank = _S.length();
}

template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( const blas::matrix< value_t > &  aU,
                                   const blas::vector< real_t > &   aS,
                                   const blas::matrix< value_t > &  aV,
                                   const accuracy &                 acc )
{
    const auto  was_compressed = is_compressed();

    set_lrmat( aU, aS, aV );

    if ( was_compressed )
        compress( acc );
}
    
template < typename value_t >
void
lrsvmatrix< value_t >::set_lrmat ( blas::matrix< value_t > &&  aU,
                                   blas::vector< real_t > &&   aS,
                                   blas::matrix< value_t > &&  aV,
                                   const accuracy &            acc )
{
    const auto  was_compressed = is_compressed();

    set_lrmat( std::move( aU ), std::move( aS ), std::move( aV ) );

    if ( was_compressed )
        compress( acc );
}

// set U/V and recompress if currently compressed
//  - neither U nor V are assumed to be orthogonal, so also reorthogonalize 
template < typename value_t >
void
lrsvmatrix< value_t >::set_U ( blas::matrix< value_t > &&  aU,
                               const accuracy &            acc )
{
    HLR_ASSERT( aU.ncols() == _S.length() );
    
    //
    // orthogonalise aU S V^H
    //

    auto  RU = blas::matrix< value_t >( aU.ncols(), aU.ncols() );

    blas::qr( aU, RU );

    auto  R             = blas::prod_diag( RU, _S );
    auto  [ Us, S, Vs ] = blas::svd( R );

    auto  TU = blas::prod( aU, Us );
    auto  TV = blas::prod( V(), Vs );

    const bool  was_compressed = is_compressed();
    
    set_lrmat( std::move( TU ),
               std::move(  S ),
               std::move( TV ) );

    if ( was_compressed )
        compress( acc );
}

template < typename value_t >
void
lrsvmatrix< value_t >::set_V ( blas::matrix< value_t > && aV,
                               const accuracy &           acc )
{
    HLR_ASSERT( aV.ncols() == _S.length() );
    
    //
    // orthogonalise U S aV^H
    //

    auto  RV = blas::matrix< value_t >( aV.ncols(), aV.ncols() );

    blas::qr( aV, RV );

    auto  R             = blas::prod_diag( _S, blas::adjoint( RV ) );
    auto  [ Us, S, Vs ] = blas::svd( R );

    auto  TU = blas::prod( U(), Us );
    auto  TV = blas::prod(  aV, Vs );

    const bool  was_compressed = is_compressed();
    
    set_lrmat( std::move( TU ),
               std::move(  S ),
               std::move( TV ) );

    if ( was_compressed )
        compress( acc );
}

//
// matrix vector multiplication
//
template < typename value_t >
void
lrsvmatrix< value_t >::mul_vec  ( const value_t                     alpha,
                                  const Hpro::TVector< value_t > *  vx,
                                  const value_t                     beta,
                                  Hpro::TVector< value_t > *        vy,
                                  const matop_t                     op ) const
{
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    const auto  sx = cptrcast( vx, Hpro::TScalarVector< value_t > );
    const auto  sy = ptrcast(  vy, Hpro::TScalarVector< value_t > );

    // y := β·y
    if ( beta != value_t(1) )
        vy->scale( beta );
    
    apply_add( alpha, blas::vec( *sx ), blas::vec( *sy ), op );
}
    
template < typename value_t >
void
lrsvmatrix< value_t >::apply_add ( const value_t                    alpha,
                                   const blas::vector< value_t > &  x,
                                   blas::vector< value_t > &        y,
                                   const matop_t                    op ) const
{
    HLR_ASSERT( x.length() == this->ncols( op ) );
    HLR_ASSERT( y.length() == this->nrows( op ) );

    #if defined(HLR_HAS_ZBLAS_APLR)
    if ( is_compressed() )
    {
        const auto  nrows = this->nrows();
        const auto  ncols = this->ncols();
        const auto  k     = this->rank();
        auto        t     = blas::vector< value_t >( k );
        
        if ( op == Hpro::apply_normal )
        {
            // t := V^H x
            compress::aplr::zblas::mulvec( ncols, k, apply_adjoint, value_t(1), _zV, x.data(), t.data() );

            // t := α·t
            for ( uint  i = 0; i < k; ++i )
                t(i) *= alpha * _S(i);
        
            // y := y + U t
            compress::aplr::zblas::mulvec( nrows, k, apply_normal, value_t(1), _zU, t.data(), y.data() );
        }// if
        else if ( op == Hpro::apply_transposed )
        {
            HLR_ERROR( "TODO" );
        }// if
        else if ( op == Hpro::apply_adjoint )
        {
            // t := U^H x
            compress::aplr::zblas::mulvec( nrows, k, apply_adjoint, value_t(1), _zU, x.data(), t.data() );

            // t := α·t
            for ( uint  i = 0; i < k; ++i )
                t(i) *= alpha * _S(i);
        
            // y := t + V t
            compress::aplr::zblas::mulvec( ncols, k, apply_normal, value_t(1), _zV, t.data(), y.data() );
        }// if
    }// if
    else
    #endif
    {
        const auto  uU = U();
        const auto  uS = S();
        const auto  uV = V();
    
        blas::mulvec_lr( alpha, U(), S(), V(), op, x, y );
    }// else
}

//
// virtual constructor
//

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsvmatrix< value_t >::copy () const
{
    auto  R = std::make_unique< lrsvmatrix< value_t > >( _row_is, _col_is );

    R->copy_struct_from( this );
    R->_rank = _rank;

    R->_S = std::move( blas::copy( _S ) );

    if ( is_compressed() )
    {
        R->_zU = compress::aplr::zarray( _zU.size() );
        R->_zV = compress::aplr::zarray( _zV.size() );

        std::copy( _zU.begin(), _zU.end(), R->_zU.begin() );
        std::copy( _zV.begin(), _zV.end(), R->_zV.begin() );
    }// if
    else
    {
        R->_U = std::move( blas::copy( _U ) );
        R->_V = std::move( blas::copy( _V ) );
    }// else

    return R;
}

template < typename value_t >
void
lrsvmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    HLR_ASSERT( IS_TYPE( A, lrsvmatrix ) );

    Hpro::TMatrix< value_t >::copy_to( A );
    
    auto  R = ptrcast( A, lrsvmatrix< value_t > );

    R->_row_is = this->_row_is;
    R->_col_is = this->_col_is;
    R->_rank   = this->_rank;
    R->_U      = std::move( blas::copy( _U ) );
    R->_V      = std::move( blas::copy( _V ) );
    R->_S      = std::move( blas::copy( _S ) );
            
    if ( is_compressed() )
    {
        R->_zU = compress::aplr::zarray( _zU.size() );
        R->_zV = compress::aplr::zarray( _zV.size() );
            
        std::copy( _zU.begin(), _zU.end(), R->_zU.begin() );
        std::copy( _zV.begin(), _zV.end(), R->_zV.begin() );
    }// if
}

//
// compression
//

// compress internal data
// - may result in non-compression if storage does not decrease
template < typename value_t >
void
lrsvmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    if ( this->nrows() * this->ncols() == 0 )
        return;

    if ( is_compressed() )
        return;

    // // DEBUG
    // auto  R1  = this->copy();
    // auto  US1 = blas::prod_diag( ptrcast( R1.get(), lrsvmatrix< value_t > )->U(),
    //                              ptrcast( R1.get(), lrsvmatrix< value_t > )->S() );
    // auto  M1  = blas::prod( US1, blas::adjoint( ptrcast( R1.get(), lrsvmatrix< value_t > )->V() ) );
    
    auto  oU = _U;
    auto  oV = _V;
    
    //
    // compute Frobenius norm and set tolerance
    //

    // defaults to absolute error: δ = ε
    auto  lacc = acc( this->row_is(), this->col_is() );
    auto  tol  = lacc.abs_eps();

    if ( lacc.rel_eps() != 0 )
    {
        // use relative error: δ = ε |M|
        real_t  norm = real_t(0);

        for ( uint  i = 0; i < _S.length(); ++i )
            norm += math::square( _S(i) );

        norm = math::sqrt( norm );
    
        tol = lacc.rel_eps() * norm;
    }// if
        
    const auto  k = this->rank();

    //
    // we aim for σ_i ≈ δ u_i and hence choose u_i = δ / σ_i
    //
    
    auto  S_tol = blas::copy( _S );

    for ( uint  l = 0; l < k; ++l )
        S_tol(l) = tol / _S(l);

    auto          zU     = compress::aplr::compress_lr( oU, S_tol );
    auto          zV     = compress::aplr::compress_lr( oV, S_tol );
    const size_t  mem_lr = sizeof(value_t) * k * ( oU.nrows() + oV.nrows() );
    const size_t  mem_zU = compress::aplr::compressed_size( zU );
    const size_t  mem_zV = compress::aplr::compressed_size( zV );

    // // DEBUG
    // {
    //     auto  tU = blas::copy( oU );
    //     auto  tV = blas::copy( oV );
    //     auto  dU = blas::matrix< value_t >( oU.nrows(), oU.ncols() );
    //     auto  dV = blas::matrix< value_t >( oV.nrows(), oV.ncols() );

    //     compress::aplr::decompress_lr( zU, dU );
    //     compress::aplr::decompress_lr( zV, dV );

    //     for ( uint  l = 0; l < dU.ncols(); ++l )
    //     {
    //         auto  u1 = tU.column( l );
    //         auto  u2 = dU.column( l );

    //         blas::scale( _S(l), u1 );
    //         blas::scale( _S(l), u2 );
    //     }// for

    //     auto  M1 = blas::prod( tU, blas::adjoint( tV ) );
    //     auto  M2 = blas::prod( dU, blas::adjoint( dV ) );
        
    //     // io::matlab::write( tU, "U1" );
    //     // io::matlab::write( dU, "U2" );

    //     blas::add( value_t(-1), tU, dU );
    //     std::cout << this->block_is().to_string() << " : " << "U " << this->block_is().to_string() << " : "
    //               << boost::format( "%.4e" ) % ( blas::norm_F( dU ) / blas::norm_F(tU) )
    //               << " / "
    //               << boost::format( "%.4e" ) % blas::max_abs_val( dU )
    //               << std::endl;

    //     blas::add( value_t(-1), tV, dV );
    //     std::cout << this->block_is().to_string() << " : " << "V " << this->block_is().to_string() << " : "
    //               << boost::format( "%.4e" ) % ( blas::norm_F( dV ) / blas::norm_F(tV) )
    //               << " / "
    //               << boost::format( "%.4e" ) % blas::max_abs_val( dV )
    //               << std::endl;

    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << this->block_is().to_string() << " : " << "M " << this->block_is().to_string() << " : "
    //               << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F(M1) )
    //               << " / "
    //               << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //               << std::endl;
    // }
    
    if (( mem_zU != 0 ) && ( mem_zV != 0 ) && ( mem_zU + mem_zV < mem_lr ))
    {
        _zU = std::move( zU );
        _zV = std::move( zV );
    
        _U = std::move( blas::matrix< value_t >( 0, 0 ) );
        _V = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if

    // // DEBUG
    // auto  R2  = this->copy();
    // auto  US2 = blas::prod_diag( ptrcast( R2.get(), lrsvmatrix< value_t > )->U(),
    //                              ptrcast( R2.get(), lrsvmatrix< value_t > )->S() );
    // auto  M2  = blas::prod( US2, blas::adjoint( ptrcast( R2.get(), lrsvmatrix< value_t > )->V() ) );

    // blas::add( -1.0, M1, M2 );

    // auto  n1 = blas::norm_F( M1 );
    // auto  n2 = blas::norm_F( M2 );

    // std::cout << "R: " << boost::format( "%.4e" ) % n1 << " / " << boost::format( "%.4e" ) % n2 << " / " << boost::format( "%.4e" ) % ( n2 / n1 ) << std::endl;
}

// decompress internal data
template < typename value_t >
void
lrsvmatrix< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    _U = std::move( U() );
    _V = std::move( V() );

    remove_compressed();
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRSVMATRIX_HH
