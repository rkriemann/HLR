#ifndef __HLR_MATRIX_MPLRMATRIX_HH
#define __HLR_MATRIX_MPLRMATRIX_HH
//
// Project     : HLR
// Module      : mplrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <boost/format.hpp>

#include <hpro/matrix/TMatrix.hh>

#include <hlr/matrix/compressible.hh>
#include <hlr/arith/blas.hh>
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

//
// GCC >= 12 and Clang >= 15 support _Float16
// check some of the default defines for this
//
#if defined(__FLT16_EPSILON__)
#  define HAS_FLOAT16  1
#else
#  define HAS_FLOAT16  0
#endif

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class mplrmatrix : public Hpro::TRkMatrix< T_value >, public compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    //
    // compressed storage based on underlying floating point type
    //
    struct compressed_storage
    {
        std::vector< double >   U1, V1;
        std::vector< float >    U2, V2;

        #if HAS_FLOAT16
        std::vector< _Float16 > U3, V3;
    };

private:
    compressed_storage    _mpdata;

public:
    //
    // ctors
    //

    mplrmatrix ()
            : Hpro::TRkMatrix< value_t >()
    {}
    
    mplrmatrix ( const indexset                arow_is,
                 const indexset                acol_is )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is )
    {}

    mplrmatrix ( const indexset                   arow_is,
                 const indexset                   acol_is,
                 hlr::blas::matrix< value_t > &   aU,
                 hlr::blas::matrix< value_t > &   aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is, aU, aV )
    {}

    mplrmatrix ( const indexset                   arow_is,
                 const indexset                   acol_is,
                 hlr::blas::matrix< value_t > &&  aU,
                 hlr::blas::matrix< value_t > &&  aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is, std::move( aU ), std::move( aV ) )
    {}

    // dtor
    virtual ~mplrmatrix ()
    {}
    
    //
    // access low-rank factors
    //

    blas::matrix< value_t > &        U ()       { return this->blas_mat_A(); }
    blas::matrix< value_t > &        V ()       { return this->blas_mat_B(); }
    
    const blas::matrix< value_t > &  U () const { return this->blas_mat_A(); }
    const blas::matrix< value_t > &  V () const { return this->blas_mat_B(); }

    blas::matrix< value_t >  U_decompressed () const
    {
        if ( is_compressed() )
        {
            auto  dU = blas::matrix< value_t >( this->nrows(), this->rank() );
            uint  k  = 0;

            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "TODO" );
            }// if
            else
            {
                size_t      pos    = 0;
                const uint  n_fp64 = _mpdata.U1.size() / dU.nrows();
                
                for ( uint  k1 = 0; k1 < n_fp64; ++k1, ++k )
                {
                    for ( uint  i = 0; i < dU.nrows(); ++i )
                        dU(i,k) = value_t( _mpdata.U1[ pos++ ] );
                }// for

                const uint  n_fp32 = _mpdata.U2.size() / dU.nrows();

                pos = 0;
                for ( uint  k2 = 0; k2 < n_fp32; ++k2, ++k )
                {
                    for ( uint  i = 0; i < dU.nrows(); ++i )
                        dU(i,k) = value_t( _mpdata.U2[ pos++ ] );
                }// for
                
                #if HAS_FLOAT16
                const uint  n_fp16 = _mpdata.U3.size() / dU.nrows();

                pos = 0;
                for ( uint  k3 = 0; k3 < n_fp16; ++k3, ++k )
                {
                    for ( uint  i = 0; i < dU.nrows(); ++i )
                        dU(i,k) = value_t( _mpdata.U3[ pos++ ] );
                }// for
                #endif
            }// else
            
            return dU;
        }// if
        else
            return U();
    }
    
    blas::matrix< value_t >  V_decompressed () const
    {
        if ( is_compressed() )
        {
            auto  dV = blas::matrix< value_t >( this->ncols(), this->rank() );
            uint  k  = 0;

            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "TODO" );
            }// if
            else
            {
                size_t      pos    = 0;
                const uint  n_fp64 = _mpdata.V1.size() / dV.nrows();
                
                for ( uint  k1 = 0; k1 < n_fp64; ++k1, ++k )
                {
                    for ( uint  i = 0; i < dV.nrows(); ++i )
                        dV(i,k) = value_t( _mpdata.V1[ pos++ ] );
                }// for

                const uint  n_fp32 = _mpdata.V2.size() / dV.nrows();

                pos = 0;
                for ( uint  k2 = 0; k2 < n_fp32; ++k2, ++k )
                {
                    for ( uint  i = 0; i < dV.nrows(); ++i )
                        dV(i,k) = value_t( _mpdata.V2[ pos++ ] );
                }// for
                
                #if HAS_FLOAT16
                const uint  n_fp16 = _mpdata.V3.size() / dV.nrows();

                pos = 0;
                for ( uint  k3 = 0; k3 < n_fp16; ++k3, ++k )
                {
                    for ( uint  i = 0; i < dV.nrows(); ++i )
                        dV(i,k) = value_t( _mpdata.V3[ pos++ ] );
                }// for
                #endif
            }// else
            
            return dV;
        }// if
        else
            return V();
    }

    //
    // access low-rank factors with matrix operator
    //
    
    blas::matrix< value_t > &        U ( const Hpro::matop_t  op ) { return ( op == apply_normal ? U() : V() ); }
    blas::matrix< value_t > &        V ( const Hpro::matop_t  op ) { return ( op == apply_normal ? V() : U() ); }
    
    const blas::matrix< value_t > &  U ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U() : V() ); }
    const blas::matrix< value_t > &  V ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V() : U() ); }

    blas::matrix< value_t >          U_decompressed ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U_decompressed() : V_decompressed() ); }
    blas::matrix< value_t >          V_decompressed ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V_decompressed() : U_decompressed() ); }

    //
    // directly set low-rank factors
    //
    
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
                   ( this->ncols() == aV.nrows() ) &&
                   ( aU.ncols()    == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        blas::copy( aU, U() );
        blas::copy( aV, V() );
    }
    
    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
                   ( this->ncols() == aV.nrows() ) &&
                   ( aU.ncols()    == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        U() = std::move( aU );
        V() = std::move( aV );

        this->_rank = U().ncols();
    }

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
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = Hpro::TMatrix< value_t >::copy();
    
        HLR_ASSERT( IS_TYPE( M.get(), mplrmatrix ) );

        auto  R = ptrcast( M.get(), mplrmatrix< value_t > );

        if ( this->cluster() != nullptr )
            R->set_cluster( this->cluster() );

        R->_rank  = this->_rank;
        R->_mat_A = std::move( blas::copy( this->_mat_A ) );
        R->_mat_B = std::move( blas::copy( this->_mat_B ) );
        
        if ( is_compressed() )
        {
            R->_mpdata.U1 = std::vector< double >( _mpdata.U1.size() );
            R->_mpdata.V1 = std::vector< double >( _mpdata.V1.size() );

            std::copy( _mpdata.U1.begin(), _mpdata.U1.end(), R->_mpdata.U1.begin() );
            std::copy( _mpdata.V1.begin(), _mpdata.V1.end(), R->_mpdata.V1.begin() );

            R->_mpdata.U2 = std::vector< float >( _mpdata.U2.size() );
            R->_mpdata.V2 = std::vector< float >( _mpdata.V2.size() );

            std::copy( _mpdata.U2.begin(), _mpdata.U2.end(), R->_mpdata.U2.begin() );
            std::copy( _mpdata.V2.begin(), _mpdata.V2.end(), R->_mpdata.V2.begin() );

            #if HAS_FLOAT16
            
            R->_mpdata.U3 = std::vector< _Float16 >( _mpdata.U3.size() );
            R->_mpdata.V3 = std::vector< _Float16 >( _mpdata.V3.size() );

            std::copy( _mpdata.U3.begin(), _mpdata.U3.end(), R->_mpdata.U3.begin() );
            std::copy( _mpdata.V3.begin(), _mpdata.V3.end(), R->_mpdata.V3.begin() );

            #endif
        }// if

        return M;
    }

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
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const
    {
        HLR_ASSERT( IS_TYPE( A, mplrmatrix ) );

        Hpro::TMatrix< value_t >::copy_to( A );
    
        auto  R = ptrcast( A, mplrmatrix< value_t > );

        R->_rows  = this->_rows;
        R->_cols  = this->_cols;
        R->_rank  = this->_rank;
        R->_mat_A = std::move( blas::copy( this->blas_mat_A() ) );
        R->_mat_B = std::move( blas::copy( this->blas_mat_B() ) );
            
        if ( is_compressed() )
        {
            R->_mpdata.U1 = std::vector< double >( _mpdata.U1.size() );
            R->_mpdata.V1 = std::vector< double >( _mpdata.V1.size() );

            std::copy( _mpdata.U1.begin(), _mpdata.U1.end(), R->_mpdata.U1.begin() );
            std::copy( _mpdata.V1.begin(), _mpdata.V1.end(), R->_mpdata.V1.begin() );

            R->_mpdata.U2 = std::vector< float >( _mpdata.U2.size() );
            R->_mpdata.V2 = std::vector< float >( _mpdata.V2.size() );

            std::copy( _mpdata.U2.begin(), _mpdata.U2.end(), R->_mpdata.U2.begin() );
            std::copy( _mpdata.V2.begin(), _mpdata.V2.end(), R->_mpdata.V2.begin() );

            #if HAS_FLOAT16
            
            R->_mpdata.U3 = std::vector< _Float16 >( _mpdata.U3.size() );
            R->_mpdata.V3 = std::vector< _Float16 >( _mpdata.V3.size() );

            std::copy( _mpdata.U3.begin(), _mpdata.U3.end(), R->_mpdata.U3.begin() );
            std::copy( _mpdata.V3.begin(), _mpdata.V3.end(), R->_mpdata.V3.begin() );

            #endif
        }// if
    }
        

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
        #if HAS_FLOAT16
        return _mpdata.U1.size() + _mpdata.U2.size() + _mpdata.U3.size() > 0;
        #else
        return _mpdata.U1.size() + _mpdata.U2.size()                     > 0;
        #endif   
    }

    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  size = Hpro::TRkMatrix< value_t >::byte_size();

        size += sizeof(double)   * ( _mpdata.U1.size() + _mpdata.V1.size() );
        size += sizeof(float)    * ( _mpdata.U2.size() + _mpdata.V2.size() );
        #if HAS_FLOAT16
        size += sizeof(_Float16) * ( _mpdata.U3.size() + _mpdata.V3.size() );
        #endif
        
        return size;
    }

    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        if ( is_compressed() )
        {
            auto  RU = U_decompressed();
            auto  RV = V_decompressed();

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
        _mpdata.U1 = std::vector< double >();
        _mpdata.V1 = std::vector< double >();
        _mpdata.U2 = std::vector< float >();
        _mpdata.V2 = std::vector< float >();
        #if HAS_FLOAT16
        _mpdata.U3 = std::vector< _Float16 >();
        _mpdata.V3 = std::vector< _Float16 >();
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
        const auto  uU = U_decompressed();
        const auto  uV = V_decompressed();
    
        blas::mulvec_lr( alpha, uU, uV, op, x, blas::vec( *sy ) );
    }// if
    else
    {
        Hpro::TRkMatrix< value_t >::mul_vec( alpha, vx, beta, vy, op );
    }// else
}
    
template < typename value_t >
void
mplrmatrix< value_t >::apply_add ( const value_t                   alpha,
                                   const blas::vector< value_t > &  x,
                                   blas::vector< value_t > &        y,
                                   const matop_t                   op ) const
{
    if ( is_compressed() )
    {
        HLR_ASSERT( x.length() == this->ncols( op ) );
        HLR_ASSERT( y.length() == this->nrows( op ) );

        const auto  uU = U_decompressed();
        const auto  uV = V_decompressed();
        
        blas::mulvec_lr( alpha, uU, uV, op, x, y );
    }// if
    else
    {
        Hpro::TRkMatrix< value_t >::apply_add( alpha, x, y, op );
    }// else
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

    //
    // determine singular values and W S X' representation with orthogonal W,X
    //
    
    double      tol   = acc( this->row_is(), this->col_is() ).rel_eps();
    auto        oU    = blas::copy( this->U() );
    auto        oV    = blas::copy( this->V() );
    const auto  orank = oU.ncols();
    auto        S     = blas::vector< real_t >( std::min( this->nrows(), this->ncols() ) );

    auto  QU = oU;
    auto  RU = blas::matrix< value_t >( orank, orank );
        
    blas::qr( QU, RU );
        
    auto  QV = oV;
    auto  RV = std::move( blas::matrix< value_t >( orank, orank ) );
        
    blas::qr( QV, RV );

    auto  R = blas::prod( value_t(1), RU, adjoint(RV) );

    auto  Ss = blas::vector< real_t >( orank );
    auto  Us = R;
    auto  Vs = RV;
    
    blas::svd( Us, Ss, Vs );

    //
    // determine corresponding parts for FP64, FP32 and FP16
    //
    
    int  i = orank-1;

    auto  test_prec = [&i,&S,tol] ( double  u )
    {
        uint  nprec = 0;
            
        while ( i >= 0 )
        {
            if ( S(i) <= tol / u ) nprec++;
            else                   break;
            --i;
        }// while

        return nprec;
    };

    #if HAS_FLOAT16
    const uint  n_fp16 = test_prec( 4.9e-4 );
    #endif
    const uint  n_fp32 = test_prec( 6.0e-8 );
    const uint  n_fp64 = i+1;
    size_t      s      = 0;

    HLR_ASSERT( n_fp64 >= 0 );
    HLR_ASSERT( n_fp16 + n_fp32 + n_fp64 == orank );
    
    if ( n_fp64 < orank )
    {
        // reset as changed during blas::sv
        oU = this->U();
        oV = this->V();
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
        {
            HLR_ERROR( "TODO" );
        }// if
        else
        {
            uint    k     = 0;
            size_t  pos_U = 0;
            size_t  pos_V = 0;

            _mpdata.U1 = std::vector< double >( n_fp64 * oU.nrows() );
            _mpdata.V1 = std::vector< double >( n_fp64 * oV.nrows() );
            
            pos_U = pos_V = 0;
            for ( uint  k1 = 0; k1 < n_fp64; ++k1, ++k )
            {
                for ( uint  i = 0; i < oU.nrows(); ++i, ++pos_U ) _mpdata.U1[pos_U] = double( oU(i,k) );
                for ( uint  i = 0; i < oV.nrows(); ++i, ++pos_V ) _mpdata.V1[pos_V] = double( oV(i,k) );
            }// for

            _mpdata.U2 = std::vector< float >( n_fp32 * oU.nrows() );
            _mpdata.V2 = std::vector< float >( n_fp32 * oV.nrows() );

            pos_U = pos_V = 0;
            for ( uint  k2 = 0; k2 < n_fp32; ++k2, ++k )
            {
                for ( uint  i = 0; i < oU.nrows(); ++i, ++pos_U ) _mpdata.U2[pos_U] = float( oU(i,k) );
                for ( uint  i = 0; i < oV.nrows(); ++i, ++pos_V ) _mpdata.V2[pos_V] = float( oV(i,k) );
            }// for

            #if HAS_FLOAT16
            _mpdata.U3 = std::vector< _Float16 >( n_fp16 * oU.nrows() );
            _mpdata.V3 = std::vector< _Float16 >( n_fp16 * oV.nrows() );
            
            pos_U = pos_V = 0;
            for ( uint  k3 = 0; k3 < n_fp16; ++k3, ++k )
            {
                for ( uint  i = 0; i < oU.nrows(); ++i, ++pos_U ) _mpdata.U3[pos_U] = _Float16( oU(i,k) );
                for ( uint  i = 0; i < oV.nrows(); ++i, ++pos_V ) _mpdata.V3[pos_V] = _Float16( oV(i,k) );
            }// for
            #endif
        }// else

        {
            auto  dU = U_decompressed();

            io::matlab::write( oU, "U1" );
            io::matlab::write( dU, "U2" );
            
            blas::add( value_t(-1), oU, dU );
            std::cout << "U " << this->block_is().to_string() << " : "
                      << boost::format( "%.4e" ) % ( blas::norm_F( dU ) / blas::norm_F(oU) )
                      << " / "
                      << boost::format( "%.4e" ) % blas::max_abs_val( dU )
                      << std::endl;

            // for ( size_t  i = 0; i < oU.nrows() * oU.ncols(); ++i )
            // {
            //     const auto  error = std::abs( (oU.data()[i] - dU.data()[i]) / oU.data()[i] );

            //     if ( error > 1e-6 )
            //         std::cout << "U " << i << " : "
            //                   << oU.data()[i] << " / "
            //                   << dU.data()[i] << " / "
            //                   << std::abs( error )
            //                   << std::endl;
            // }// for
        }
    
        {
            auto  dV = V_decompressed();

            io::matlab::write( oV, "V1" );
            io::matlab::write( dV, "V2" );
            
            blas::add( value_t(-1), oV, dV );
            std::cout << "V " << this->block_is().to_string() << " : "
                      << boost::format( "%.4e" ) % ( blas::norm_F( dV ) / blas::norm_F(oV) )
                      << " / "
                      << boost::format( "%.4e" ) % blas::max_abs_val( dV )
                      << std::endl;

            // for ( size_t  i = 0; i < oV.nrows() * oV.ncols(); ++i )
            // {
            //     const auto  error = std::abs( (oV.data()[i] - dV.data()[i]) / oV.data()[i] );

            //     if ( error > 1e-6 )
            //         std::cout << "V " << i << " : "
            //                   << oV.data()[i] << " / "
            //                   << dV.data()[i] << " / "
            //                   << std::abs( error )
            //                   << std::endl;
            // }// for
        }

        this->U() = std::move( blas::matrix< value_t >( 0, 0 ) );
        this->V() = std::move( blas::matrix< value_t >( 0, 0 ) );

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

    this->U() = std::move( U_decompressed() );
    this->V() = std::move( V_decompressed() );

    remove_compressed();
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_MPLRMATRIX_HH
