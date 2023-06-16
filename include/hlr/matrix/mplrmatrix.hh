#ifndef __HLR_MATRIX_MPLRMATRIX_HH
#define __HLR_MATRIX_MPLRMATRIX_HH
//
// Project     : HLR
// Module      : mplrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <boost/format.hpp>

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

#include <hlr/utils/io.hh> // DEBUG

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( mplrmatrix );

namespace matrix
{

#define HLR_USE_APCOMPRESSION  1

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with orthogonal U/V and diagonal S.
//
template < typename T_value >
class mplrmatrix : public Hpro::TRkMatrix< T_value >, public compress::compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    //
    // compressed storage based on underlying floating point type
    //
    struct mp_storage
    {
        #if HLR_USE_APCOMPRESSION == 1
        compress::ap::zarray      zU, zV;
        #endif
    };

private:
    // singular values
    blas::vector< real_t >   _S;

    // compressed storage
    mp_storage               _mpdata;

public:
    //
    // ctors
    //

    mplrmatrix ()
            : Hpro::TRkMatrix< value_t >()
    {}
    
    mplrmatrix ( const indexset  arow_is,
                 const indexset  acol_is )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is )
    {}

    mplrmatrix ( const indexset                   arow_is,
                 const indexset                   acol_is,
                 hlr::blas::matrix< value_t > &   aU,
                 hlr::blas::matrix< value_t > &   aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is )
    {
        set_lrmat( aU, aV );
    }

    mplrmatrix ( const indexset                   arow_is,
                 const indexset                   acol_is,
                 hlr::blas::matrix< value_t > &&  aU,
                 hlr::blas::matrix< value_t > &&  aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is )
    {
        set_lrmat( std::move( aU ), std::move( aV ) );
    }

    // dtor
    virtual ~mplrmatrix ()
    {}
    
    //
    // access low-rank factors
    //

    blas::matrix< value_t >  U () const;
    blas::matrix< value_t >  V () const;

    blas::matrix< value_t >  U ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U() : V() ); }
    blas::matrix< value_t >  V ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V() : U() ); }

    //
    // directly set low-rank factors
    //
    
    virtual void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV );
    
    virtual void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV );

    //
    // matrix data
    //
    
    virtual void  set_size  ( const size_t  anrows,
                              const size_t  ancols )
    {
        if ( is_compressed() )
        {
            HLR_ERROR( "TODO" );
        }// if
        else
        {
            Hpro::TRkMatrix< value_t >::set_size( anrows, ancols );
        }// else
    }
    
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
        if ( is_compressed() )
        {
            HLR_ERROR( "todo" );
        }// if
        else
        {
            Hpro::TRkMatrix< value_t >::scale( alpha );
        }// else
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( mplrmatrix, Hpro::TRkMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< mplrmatrix< value_t > >(); }

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
        return std::make_unique< mplrmatrix< value_t > >( this->row_is(), this->col_is() );
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
        #if HLR_USE_APCOMPRESSION == 1
        return _mpdata.zU.size() > 0;
        #else
        return false;
        #endif
    }

    // access multiprecision data
    const mp_storage &  mp_data () const { return _mpdata; }
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  size = Hpro::TRkMatrix< value_t >::byte_size();

        size += _S.byte_size();

        #if HLR_USE_APCOMPRESSION == 1

        size += compress::ap::byte_size( _mpdata.zU );
        size += compress::ap::byte_size( _mpdata.zV );
        
        #endif
        
        return size;
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
        #if HLR_USE_APCOMPRESSION == 1

        _mpdata.zU = compress::ap::zarray();
        _mpdata.zV = compress::ap::zarray();
        
        #endif
    }
};

//
// type test
//
template < typename value_t >
inline
bool
is_mixedprec_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, mplrmatrix );
}

template < typename value_t >
bool
is_mixedprec_lowrank ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, mplrmatrix );
}

HLR_TEST_ALL( is_mixedprec_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_mixedprec_lowrank, Hpro::TMatrix< value_t > )

//
// access low-rank factors
//
template < typename value_t >
blas::matrix< value_t >
mplrmatrix< value_t >::U () const
{
    if ( is_compressed() )
    {
        auto  dU = blas::matrix< value_t >( this->nrows(), this->rank() );
        uint  k  = 0;

        #if HLR_USE_APCOMPRESSION == 1
            
        compress::ap::decompress_lr( _mpdata.zU, dU );

        for ( uint  l = 0; l < dU.ncols(); ++l )
        {
            auto  u_l = dU.column( l );

            blas::scale( _S(l), u_l );
        }// for
            
        #endif
            
        return dU;
    }// if
    else
        return this->blas_mat_A();
}
    
template < typename value_t >
blas::matrix< value_t >
mplrmatrix< value_t >::V () const
{
    if ( is_compressed() )
    {
        auto        dV  = blas::matrix< value_t >( this->ncols(), this->rank() );

        #if HLR_USE_APCOMPRESSION == 1
            
        compress::ap::decompress_lr( _mpdata.zV, dV );
            
        #endif
            
        return dV;
    }// if
    else
        return this->blas_mat_B();
}

//
// directly set low-rank factors
//
    
template < typename value_t >
void
mplrmatrix< value_t >::set_lrmat ( const blas::matrix< value_t > &  aU,
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

    this->_mat_A = std::move( blas::prod( QU, Us ) );
    this->_mat_B = std::move( blas::prod( QV, Vs ) );

    _S = std::move( S );

    this->_rank = QU.ncols();
}
    
template < typename value_t >
void
mplrmatrix< value_t >::set_lrmat ( blas::matrix< value_t > &&  aU,
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

    this->_mat_A = std::move( blas::prod( aU, Us ) );
    this->_mat_B = std::move( blas::prod( aV, Vs ) );

    _S = std::move( S );

    this->_rank = aU.ncols();
}

//
// matrix vector multiplication
//
template < typename value_t >
void
mplrmatrix< value_t >::mul_vec  ( const value_t                     alpha,
                                  const Hpro::TVector< value_t > *  vx,
                                  const value_t                     beta,
                                  Hpro::TVector< value_t > *        vy,
                                  const matop_t                     op ) const
{
    if ( is_compressed() )
    {
        HLR_ASSERT( vx->is() == this->col_is( op ) );
        HLR_ASSERT( vy->is() == this->row_is( op ) );
        HLR_ASSERT( is_scalar_all( vx, vy ) );

        const auto  sx = cptrcast( vx, Hpro::TScalarVector< value_t > );
        const auto  sy = ptrcast(  vy, Hpro::TScalarVector< value_t > );

        // y := β·y
        if ( beta != value_t(1) )
            vy->scale( beta );

        auto        x  = blas::vec( *sx );
        const auto  uU = U();
        const auto  uV = V();
    
        blas::mulvec_lr( alpha, uU, uV, op, x, blas::vec( *sy ) );
    }// if
    else
    {
        Hpro::TRkMatrix< value_t >::mul_vec( alpha, vx, beta, vy, op );
    }// else
}
    
template < typename value_t >
void
mplrmatrix< value_t >::apply_add ( const value_t                    alpha,
                                   const blas::vector< value_t > &  x,
                                   blas::vector< value_t > &        y,
                                   const matop_t                    op ) const
{
    if ( is_compressed() )
    {
        HLR_ASSERT( x.length() == this->ncols( op ) );
        HLR_ASSERT( y.length() == this->nrows( op ) );

        const auto  uU = U();
        const auto  uV = V();
        
        blas::mulvec_lr( alpha, uU, uV, op, x, y );
    }// if
    else
    {
        Hpro::TRkMatrix< value_t >::apply_add( alpha, x, y, op );
    }// else
}

//
// virtual constructor
//

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
mplrmatrix< value_t >::copy () const
{
    auto  M = Hpro::TMatrix< value_t >::copy();
    
    HLR_ASSERT( IS_TYPE( M.get(), mplrmatrix ) );

    auto  R = ptrcast( M.get(), mplrmatrix< value_t > );

    if ( this->cluster() != nullptr )
        R->set_cluster( this->cluster() );

    R->_rank  = this->_rank;
    R->_mat_A = std::move( blas::copy( this->_mat_A ) );
    R->_mat_B = std::move( blas::copy( this->_mat_B ) );
    R->_S     = std::move( blas::copy( _S ) );

    if ( is_compressed() )
    {
        #if HLR_USE_APCOMPRESSION == 1

        R->_mpdata.zU = compress::ap::zarray( _mpdata.zU.size() );
        R->_mpdata.zV = compress::ap::zarray( _mpdata.zV.size() );
            
        std::copy( _mpdata.zU.begin(), _mpdata.zU.end(), R->_mpdata.zU.begin() );
        std::copy( _mpdata.zV.begin(), _mpdata.zV.end(), R->_mpdata.zV.begin() );
            
        #endif
    }// if

    return M;
}

template < typename value_t >
void
mplrmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    HLR_ASSERT( IS_TYPE( A, mplrmatrix ) );

    Hpro::TMatrix< value_t >::copy_to( A );
    
    auto  R = ptrcast( A, mplrmatrix< value_t > );

    R->_rows  = this->_rows;
    R->_cols  = this->_cols;
    R->_rank  = this->_rank;
    R->_mat_A = std::move( blas::copy( this->blas_mat_A() ) );
    R->_mat_B = std::move( blas::copy( this->blas_mat_B() ) );
    R->_S     = std::move( blas::copy( _S ) );
            
    if ( is_compressed() )
    {
        #if HLR_USE_APCOMPRESSION == 1

        R->_mpdata.zU = compress::ap::zarray( _mpdata.zU.size() );
        R->_mpdata.zV = compress::ap::zarray( _mpdata.zV.size() );
            
        std::copy( _mpdata.zU.begin(), _mpdata.zU.end(), R->_mpdata.zU.begin() );
        std::copy( _mpdata.zV.begin(), _mpdata.zV.end(), R->_mpdata.zV.begin() );
            
        #endif
    }// if
}

//
// compression
//

// compress internal data
// - may result in non-compression if storage does not decrease
template < typename value_t >
void
mplrmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    if ( this->nrows() * this->ncols() == 0 )
        return;

    if ( is_compressed() )
        return;

    auto  oU = this->_mat_A;
    auto  oV = this->_mat_B;
    
    //
    // compute Frobenius norm and set toleranc
    //

    real_t  norm = real_t(0);

    for ( uint  i = 0; i < _S.length(); ++i )
        norm += math::square( _S(i) );

    norm = math::sqrt( norm );
    
    const auto  tol = acc( this->row_is(), this->col_is() ).abs_eps() * norm;
    const auto  k   = this->rank();

    #if HLR_USE_APCOMPRESSION == 1

    // need tolerance divided by singular values for accuracy
    auto  S_tol = blas::copy( _S );

    for ( uint  l = 0; l < k; ++l )
        S_tol(l) = tol / _S(l);

    auto          zU     = compress::ap::compress_lr( oU, S_tol );
    auto          zV     = compress::ap::compress_lr( oV, S_tol );
    const size_t  mem_lr = sizeof(value_t) * k * ( oU.nrows() + oV.nrows() );
    const size_t  mem_mp = compress::ap::byte_size( zU ) + compress::ap::byte_size( zV ) + sizeof(real_t) * k;

    // // DEBUG
    // {
    //     auto  tU = blas::copy( oU );
    //     auto  tV = blas::copy( oV );
    //     auto  dU = blas::matrix< value_t >( oU.nrows(), oU.ncols() );
    //     auto  dV = blas::matrix< value_t >( oV.nrows(), oV.ncols() );

    //     compress::ap::decompress_lr( zU, dU );
    //     compress::ap::decompress_lr( zV, dV );

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
    
    if ( mem_mp < mem_lr )
    {
        _mpdata.zU = std::move( zU );
        _mpdata.zV = std::move( zV );
    
        this->_mat_A = std::move( blas::matrix< value_t >( 0, 0 ) );
        this->_mat_B = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if

    #endif
}

// decompress internal data
template < typename value_t >
void
mplrmatrix< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    this->_mat_A = std::move( U() );
    this->_mat_B = std::move( V() );

    remove_compressed();
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_MPLRMATRIX_HH
