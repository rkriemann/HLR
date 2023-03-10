#ifndef __HLR_MATRIX_LRMATRIX_HH
#define __HLR_MATRIX_LRMATRIX_HH
//
// Project     : HLR
// Module      : lrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

// #include <boost/format.hpp>

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
DECLARE_TYPE( lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class lrmatrix : public Hpro::TRkMatrix< T_value >, public compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    //
    // compressed storage based on underlying floating point type
    //
    #if HLR_HAS_COMPRESSION == 1

    struct compressed_factors
    {
        compress::zarray  U, V;
    };

    using  compressed_storage = compressed_factors;
    
    #endif

private:
    #if HLR_HAS_COMPRESSION == 1
    // optional: stores compressed data
    compressed_storage    _zdata;
    #endif

public:
    //
    // ctors
    //

    lrmatrix ()
            : Hpro::TRkMatrix< value_t >()
    {}
    
    lrmatrix ( const indexset                arow_is,
               const indexset                acol_is )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is )
    {}

    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &   aU,
               hlr::blas::matrix< value_t > &   aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is, aU, aV )
    {}

    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &&  aU,
               hlr::blas::matrix< value_t > &&  aV )
            : Hpro::TRkMatrix< value_t >( arow_is, acol_is, std::move( aU ), std::move( aV ) )
    {}

    // dtor
    virtual ~lrmatrix ()
    {}
    
    //
    // access low-rank factors
    //

    blas::matrix< value_t > &        U ()       { return this->blas_mat_A(); }
    blas::matrix< value_t > &        V ()       { return this->blas_mat_B(); }
    
    const blas::matrix< value_t > &  U () const { return this->blas_mat_A(); }
    const blas::matrix< value_t > &  V () const { return this->blas_mat_B(); }

    blas::matrix< value_t >          U_decompressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  dU = blas::matrix< value_t >( this->nrows(), this->rank() );
    
            compress::decompress< value_t >( _zdata.U, dU );

            return dU;
        }// if
        else
            return U();
        
        #else

        return U();

        #endif
    }
    
    blas::matrix< value_t >          V_decompressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  dV = blas::matrix< value_t >( this->ncols(), this->rank() );
    
            compress::decompress< value_t >( _zdata.V, dV );

            return dV;
        }// if
        else
            return V();
        
        #else

        return V();

        #endif
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
    
    virtual void    set_size  ( const size_t  anrows,
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

    HPRO_RTTI_DERIVED( lrmatrix, Hpro::TRkMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< lrmatrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = Hpro::TMatrix< value_t >::copy();
    
        HLR_ASSERT( IS_TYPE( M.get(), lrmatrix ) );

        auto  R = ptrcast( M.get(), lrmatrix< value_t > );

        if ( this->cluster() != nullptr )
            R->set_cluster( this->cluster() );

        R->_rank  = this->_rank;
        R->_mat_A = std::move( blas::copy( this->_mat_A ) );
        R->_mat_B = std::move( blas::copy( this->_mat_B ) );
        
        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            R->_zdata.U = compress::zarray( _zdata.U.size() );
            R->_zdata.V = compress::zarray( _zdata.V.size() );

            std::copy( _zdata.U.begin(), _zdata.U.end(), R->_zdata.U.begin() );
            std::copy( _zdata.V.begin(), _zdata.V.end(), R->_zdata.V.begin() );
        }// if

        #endif

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
        return std::make_unique< lrmatrix< value_t > >( this->row_is(), this->col_is() );
    }

    // copy matrix data to A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const
    {
        HLR_ASSERT( IS_TYPE( A, lrmatrix ) );

        Hpro::TMatrix< value_t >::copy_to( A );
    
        auto  R = ptrcast( A, lrmatrix< value_t > );

        R->_rows  = this->_rows;
        R->_cols  = this->_cols;
        R->_rank  = this->_rank;
        R->_mat_A = std::move( blas::copy( this->blas_mat_A() ) );
        R->_mat_B = std::move( blas::copy( this->blas_mat_B() ) );
            
        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            R->_zdata.U = compress::zarray( _zdata.U.size() );
            R->_zdata.V = compress::zarray( _zdata.V.size() );

            std::copy( _zdata.U.begin(), _zdata.U.end(), R->_zdata.U.begin() );
            std::copy( _zdata.V.begin(), _zdata.V.end(), R->_zdata.V.begin() );
        }// if

        #endif
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
    virtual void   compress      ( const compress::zconfig_t &  zconfig );
    virtual void   compress      ( const Hpro::TTruncAcc &      acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        return ! is_null( _zdata.U.data() );
        #else
        return false;
        #endif
    }

    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  size = Hpro::TRkMatrix< value_t >::byte_size();

        #if HLR_HAS_COMPRESSION == 1

        size += hlr::compress::byte_size( _zdata.U );
        size += hlr::compress::byte_size( _zdata.V );

        // if ( is_compressed() )
        //     size -= this->rank() * ( this->nrows() + this->ncols() ) * sizeof(value_t);
    
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
        #if HLR_HAS_COMPRESSION == 1
        _zdata.U = compress::zarray();
        _zdata.V = compress::zarray();
        #endif
    }
};

//
// type test
//
template < typename value_t >
inline
bool
is_compressible_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, lrmatrix );
}

template < typename value_t >
bool
is_compressible_lowrank ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrmatrix );
}

HLR_TEST_ALL( is_compressible_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_compressible_lowrank, Hpro::TMatrix< value_t > )

//
// matrix vector multiplication
//
template < typename value_t >
void
lrmatrix< value_t >::mul_vec  ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  vx,
                                const value_t                     beta,
                                Hpro::TVector< value_t > *        vy,
                                const matop_t                     op ) const
{
    #if HLR_HAS_COMPRESSION == 1

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

        auto  x = blas::vec( *sx );
        // auto  y = blas::vector< value_t >( sy->size() );

        auto  uU = blas::matrix< value_t >( this->nrows(), this->rank() );
        auto  uV = blas::matrix< value_t >( this->ncols(), this->rank() );
    
        compress::decompress< value_t >( _zdata.U, uU );
        compress::decompress< value_t >( _zdata.V, uV );
        
        blas::mulvec_lr( alpha, uU, uV, op, x, blas::vec( *sy ) );
        // blas::add( value_t(1), y, blas::vec( *sy ) );
    }// if
    else

    #endif
    {
        Hpro::TRkMatrix< value_t >::mul_vec( alpha, vx, beta, vy, op );
    }// else
}
    
template < typename value_t >
void
lrmatrix< value_t >::apply_add ( const value_t                   alpha,
                                 const blas::vector< value_t > &  x,
                                 blas::vector< value_t > &        y,
                                 const matop_t                   op ) const
{
    #if HLR_HAS_COMPRESSION == 1

    if ( is_compressed() )
    {
        HLR_ASSERT( x.length() == this->ncols( op ) );
        HLR_ASSERT( y.length() == this->nrows( op ) );

        // auto  ty = blas::vector< value_t >( y.length() );
        auto  uU = blas::matrix< value_t >( this->nrows(), this->rank() );
        auto  uV = blas::matrix< value_t >( this->ncols(), this->rank() );
    
        compress::decompress< value_t >( _zdata.U, uU );
        compress::decompress< value_t >( _zdata.V, uV );
        
        blas::mulvec_lr( alpha, uU, uV, op, x, y );
        // blas::add( value_t(1), ty, y );
    }// if
    else

    #endif
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
lrmatrix< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;
                 
    // if ( this->block_is() == Hpro::bis( Hpro::is( 192, 255 ), Hpro::is( 256, 319 ) ) )
    //     std::cout << std::endl;

    auto          oU      = this->U();
    auto          oV      = this->V();
    const auto    orank   = oU.ncols();
    const size_t  mem_lr  = sizeof(value_t) * orank * ( oU.nrows() + oV.nrows() );
    auto          zU      = compress::compress< value_t >( zconfig, oU );
    auto          zV      = compress::compress< value_t >( zconfig, oV );

    // {
    //     auto  dU = blas::matrix< value_t >( oU.nrows(), oU.ncols() );

    //     compress::decompress( zU, dU );

    //     // io::matlab::write( oU, "U1" );
    //     // io::matlab::write( dU, "U2" );
            
    //     blas::add( value_t(-1), oU, dU );
    //     std::cout << "U " << this->block_is().to_string() << " : "
    //               << boost::format( "%.4e" ) % ( blas::norm_F( dU ) / blas::norm_F(oU) )
    //               << " / "
    //               << boost::format( "%.4e" ) % blas::max_abs_val( dU )
    //               << std::endl;

    //     // for ( size_t  i = 0; i < oU.nrows() * oU.ncols(); ++i )
    //     // {
    //     //     const auto  error = std::abs( (oU.data()[i] - dU.data()[i]) / oU.data()[i] );

    //     //     if ( error > 1e-6 )
    //     //         std::cout << "U " << i << " : "
    //     //                   << oU.data()[i] << " / "
    //     //                   << dU.data()[i] << " / "
    //     //                   << std::abs( error )
    //     //                   << std::endl;
    //     // }// for
    // }
    
    // {
    //     auto  dV = blas::matrix< value_t >( oV.nrows(), oV.ncols() );

    //     compress::decompress( zV, dV );

    //     // io::matlab::write( oV, "V1" );
    //     // io::matlab::write( dV, "V2" );
            
    //     blas::add( value_t(-1), oV, dV );
    //     std::cout << "V " << this->block_is().to_string() << " : "
    //               << boost::format( "%.4e" ) % ( blas::norm_F( dV ) / blas::norm_F(oV) )
    //               << " / "
    //               << boost::format( "%.4e" ) % blas::max_abs_val( dV )
    //               << std::endl;

    //     // for ( size_t  i = 0; i < oV.nrows() * oV.ncols(); ++i )
    //     // {
    //     //     const auto  error = std::abs( (oV.data()[i] - dV.data()[i]) / oV.data()[i] );

    //     //     if ( error > 1e-6 )
    //     //         std::cout << "V " << i << " : "
    //     //                   << oV.data()[i] << " / "
    //     //                   << dV.data()[i] << " / "
    //     //                   << std::abs( error )
    //     //                   << std::endl;
    //     // }// for
    // }
    
    if ( compress::byte_size( zU ) + compress::byte_size( zV ) < mem_lr )
    {
        _zdata.U  = std::move( zU );
        _zdata.V  = std::move( zV );
        this->U() = std::move( blas::matrix< value_t >( 0, 0 ) );
        this->V() = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if

    #endif
}

template < typename value_t >
void
lrmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    if ( this->nrows() * this->ncols() == 0 )
        return;
        
    compress( compress::get_config( acc( this->row_is(), this->col_is() ).rel_eps() ) );
}

// decompress internal data
template < typename value_t >
void
lrmatrix< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    this->U() = std::move( U_decompressed() );
    this->V() = std::move( V_decompressed() );

    remove_compressed();
        
    #endif
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRMATRIX_HH
