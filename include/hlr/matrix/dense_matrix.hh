#ifndef __HLR_MATRIX_DENSE_MATRIX_HH
#define __HLR_MATRIX_DENSE_MATRIX_HH
//
// Project     : HLR
// Module      : dense_matrix
// Description : dense matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>

#include <hpro/matrix/TDenseMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/io.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( dense_matrix );

namespace matrix
{

//
// implements compressable dense matrix
//
template < typename T_value >
class dense_matrix : public Hpro::TDenseMatrix< T_value >, public compress::compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    #if HLR_HAS_COMPRESSION == 1
    // stores compressed data
    compress::zarray  _zM;
    #endif
    
public:
    //
    // ctors
    //

    dense_matrix ()
            : Hpro::TDenseMatrix< value_t >()
    {}
    
    dense_matrix ( const indexset  arow_is,
                   const indexset  acol_is )
            : Hpro::TDenseMatrix< value_t >( arow_is, acol_is )
    {}

    dense_matrix ( const indexset             arow_is,
                   const indexset             acol_is,
                   blas::matrix< value_t > &  aM )
            : Hpro::TDenseMatrix< value_t >( arow_is, acol_is, aM )
    {}

    dense_matrix ( const indexset              arow_is,
                   const indexset              acol_is,
                   blas::matrix< value_t > &&  aM )
            : Hpro::TDenseMatrix< value_t >( arow_is, acol_is, std::move( aM ) )
    {}

    // dtor
    virtual ~dense_matrix ()
    {}
    
    //
    // access internal data
    //

    blas::matrix< value_t > &        mat ()       { return this->blas_mat(); }
    const blas::matrix< value_t > &  mat () const { return this->blas_mat(); }
    
    blas::matrix< value_t >          mat_decompressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  dM = blas::matrix< value_t >( this->nrows(), this->ncols() );
    
            compress::decompress< value_t >( _zM, dM );
            
            return dM;
        }// if
        else
            return mat();

        #else

        return mat();

        #endif
    }

    void
    set_matrix ( const blas::matrix< value_t > &  aM )
    {
        HLR_ASSERT(( this->nrows() == aM.nrows() ) && ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        blas::copy( aM, this->blas_mat() );
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
            Hpro::TDenseMatrix< value_t >::set_size( anrows, ancols );
        }// else
    }
    
    //
    // algebra routines
    //

    // scale matrix by constant factor \a f
    virtual void  scale      ( const value_t  f )
    {
        if ( is_compressed() )
        {
            HLR_ERROR( "to do" );
        }// if
        else
        {
            Hpro::TDenseMatrix< value_t >::scale( f );
        }// else
    }
    
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
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const Hpro::TTruncAcc & ) {}

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
        return ! is_null( _zM.data() );
        #else
        return false;
        #endif
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( dense_matrix, Hpro::TDenseMatrix< value_t > )

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< dense_matrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = Hpro::TMatrix< value_t >::copy();
    
        HLR_ASSERT( IS_TYPE( M.get(), dense_matrix ) );

        auto  D = ptrcast( M.get(), dense_matrix< value_t > );

        D->_rows = this->_rows;
        D->_cols = this->_cols;
        D->_mat  = std::move( blas::copy( this->_mat ) );
        
        HLR_ASSERT( ( D->nrows() == this->nrows() ) &&
                    ( D->ncols() == this->ncols() ) );
        
        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            D->_zM = compress::zarray( _zM.size() );

            std::copy( _zM.begin(), _zM.end(), D->_zM.begin() );
        }// if

        #endif
    
        return M;
    }

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const Hpro::TTruncAcc &  /* acc */,
                                  const bool               /* do_coarsen */ = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return copy();
    }

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return std::make_unique< dense_matrix< value_t > >( this->row_is(), this->col_is() );
    }

    // copy matrix data to \a A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const
    {
        Hpro::TMatrix< value_t >::copy_to( A );
    
        HLR_ASSERT( IS_TYPE( A, dense_matrix ) );

        auto  D = ptrcast( A, dense_matrix< value_t > );

        D->_rows = this->_rows;
        D->_cols = this->_cols;
        D->_mat  = std::move( blas::copy( this->_mat ) );

        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            D->_zM = compress::zarray( _zM.size() );

            std::copy( _zM.begin(), _zM.end(), D->_zM.begin() );
        }// if

        #endif
    }

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const Hpro::TTruncAcc &     /* acc */,
                                  const bool                  /* do_coarsen */ = false ) const
    {
        copy_to( A );
    }
    
    //
    // misc
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size () const
    {
        size_t  size = Hpro::TDenseMatrix< value_t >::byte_size();

        #if HLR_HAS_COMPRESSION == 1

        size += hlr::compress::byte_size( _zM );

        // if ( is_compressed() )
        //     size -= this->nrows() * this->ncols() * sizeof(value_t);
    
        #endif
        
        return size;
    }

    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        if ( is_compressed() )
        {
            auto  M = mat_decompressed();

            M.check_data();
        }// if
        else
        {
            mat().check_data();
        }// else
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        _zM = compress::zarray();
        #endif
    }
    
};

//
// type test
//
template < typename value_t >
bool
is_compressible_dense ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, dense_matrix );
}

template < typename value_t >
bool
is_compressible_dense ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, dense_matrix );
}

HLR_TEST_ALL( is_compressible_dense, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_compressible_dense, Hpro::TMatrix< value_t > )

//
// matrix vector multiplication
//
template < typename value_t >
void
dense_matrix< value_t >::mul_vec    ( const value_t                     alpha,
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

        apply_add( alpha, blas::vec( *sx ), blas::vec( sy ), op );
    }// if
    else

    #endif
    {
        Hpro::TDenseMatrix< value_t >::mul_vec( alpha, vx, beta, vy, op );
    }// else
    
}

template < typename value_t >
void
dense_matrix< value_t >::apply_add   ( const value_t                    alpha,
                                       const blas::vector< value_t > &  x,
                                       blas::vector< value_t > &        y,
                                       const matop_t                    op ) const
{
    #if HLR_HAS_COMPRESSION == 1

    if ( is_compressed() )
    {
        HLR_ASSERT( x.length() == this->ncols( op ) );
        HLR_ASSERT( y.length() == this->nrows( op ) );

        // #if defined(HAS_UNIVERSAL)

        // // perform mat-vec directly with posits
        // compress::posits::mulvec( this->nrows(), this->ncols(), op, alpha, _zM, x.data(), value_t(1), y.data() );

        // #else
        
        auto  M = mat_decompressed();
        
        blas::mulvec( alpha, blas::mat_view( op, M ), x, value_t(1), y );

        // #endif
    }// if
    else

    #endif
    {
        Hpro::TDenseMatrix< value_t >::apply_add( alpha, x, y, op );
    }// else
}

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
dense_matrix< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;

    // if ( this->block_is() == Hpro::bis( Hpro::is( 96, 111 ), Hpro::is( 225, 240 ) ) )
    //     std::cout << std::endl;

    auto          M         = this->blas_mat();
    const size_t  mem_dense = sizeof(value_t) * M.nrows() * M.ncols();
    auto          zM        = compress::compress< value_t >( zconfig, M );

    // // DEBUG
    // {
    //     auto  dM = blas::matrix< value_t >( M.nrows(), M.ncols() );

    //     compress::decompress( zM, dM );

    //     io::matlab::write( M, "M1" );
    //     io::matlab::write( dM, "M2" );
        
    //     blas::add( value_t(-1), M, dM );

    //     std::cout << "D " << this->block_is().to_string() << " : "
    //               << blas::norm_F( dM ) / blas::norm_F(M)
    //               << " / "
    //               << blas::max_abs_val( dM )
    //               << std::endl;
            
    //     // for ( size_t  i = 0; i < M.nrows() * M.ncols(); ++i )
    //     // {
    //     //     const auto  error = std::abs( (M.data()[i] - dM.data()[i]) / M.data()[i] );

    //     //     if ( error > 1e-6 )
    //     //         std::cout << "D " << i << " : "
    //     //                   << M.data()[i] << " / "
    //     //                   << dM.data()[i] << " / "
    //     //                   << std::abs( error )
    //     //                   << std::endl;
    //     // }// for
    // }
    
    if ( compress::byte_size( zM ) < mem_dense )
    {
        _zM           = std::move( zM );
        this->blas_mat() = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if

    #endif
}

template < typename value_t >
void
dense_matrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    if ( this->nrows() * this->ncols() == 0 )
        return;
        
    const auto  eps   = acc( this->row_is(), this->col_is() ).rel_eps();
    // const auto  normF = blas::norm_F( this->blas_mat() );
    // const auto  delta = eps * normF / std::sqrt( double( this->nrows() * this->ncols() ) );
    // const auto  lacc  = acc( this->row_is(), this->col_is() );
    // const auto  peps  = std::ceil( std::log2( lacc.rel_eps() ) ) + 7; // see ZFP documentation; FAQ Q20
    // const auto  vmin = blas::min_val( this->blas_mat() );
    // const auto  vmax = blas::max_val( this->blas_mat() );

    // std::cout << vmin << " / " << vmax << std::endl;
        
    // compress( compress::get_config( eps * normF / double(std::min( this->nrows(), this->ncols() )) ) );
    compress( compress::get_config( eps ) );
}

//
// decompress internal data
//
template < typename value_t >
void
dense_matrix< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    this->blas_mat() = std::move( mat_decompressed() );

    remove_compressed();

    #endif
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_DENSE_MATRIX_HH
