#ifndef __HLR_MATRIX_DENSE_MATRIX_HH
#define __HLR_MATRIX_DENSE_MATRIX_HH
//
// Project     : HLR
// Module      : dense_matrix
// Description : dense matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <vector>
#include <variant>

#include <hpro/matrix/TDenseMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

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
class dense_matrix : public Hpro::TDenseMatrix< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    //
    // compressed storage based on underlying floating point type
    //
    #if defined(HAS_ZFP)
    
    using  compressed_storage = hlr::zfp::zarray;
    
    #endif

private:
    #if defined(HAS_ZFP)
    // optional: stores compressed data
    compressed_storage    _zdata;
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
    virtual void  mul_vec    ( const value_t               alpha,
                               const Hpro::TVector< value_t > *  x,
                               const value_t               beta,
                               Hpro::TVector< value_t > *        y,
                               const matop_t               op = Hpro::apply_normal ) const;
    using Hpro::TMatrix< value_t >::mul_vec;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const Hpro::TTruncAcc & ) {}

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( dense_matrix, Hpro::TDenseMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< dense_matrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = Hpro::TMatrix< value_t >::copy();
    
        HLR_ASSERT( IS_TYPE( M.get(), dense_matrix ) );

        auto  D = ptrcast( M.get(), dense_matrix< value_t > );

        HLR_ASSERT( ( D->nrows() == this->nrows() ) &&
                    ( D->ncols() == this->ncols() ) );
        
        #if defined( HAS_ZFP )

        if ( is_compressed() )
        {
            D->_zdata = zfp::zarray( _zdata.size() );

            std::copy( _zdata.begin(), _zdata.end(), D->_zdata.begin() );
        }// if

        #endif

        else
        {
            D->blas_mat() = std::move( blas::copy( this->blas_mat() ) );
        }// else
    
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
        Hpro::TDenseMatrix< value_t >::copy_to( A );
    
        HLR_ASSERT( IS_TYPE( A, dense_matrix ) );

        #if defined( HAS_ZFP )

        if ( is_compressed() )
        {
            // auto  D = ptrcast( A, dense_matrix< value_t > );

            HLR_ERROR( "TODO" );
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
    // misc.
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const Hpro::TTruncAcc &  acc )
    {
        #if defined(HAS_ZFP)
        
        if ( is_compressed() )
            return;

        HLR_ASSERT( acc.is_fixed_prec() );
        
        auto          M         = this->blas_mat();
        const size_t  mem_dense = sizeof(value_t) * M.nrows() * M.ncols();
        const auto    zconfig   = zfp::fixed_accuracy( acc.rel_eps() );
        auto          v         = zfp::compress< value_t >( zconfig, M.data(), M.nrows(), M.ncols() );

        if ( v.size() < mem_dense )
        {
            _zdata           = std::move( v );
            this->blas_mat() = std::move( blas::matrix< value_t >( 0, 0 ) );
        }// if

        #endif
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        #if defined(HAS_ZFP)
        
        if ( ! is_compressed() )
            return;

        auto  M = blas::matrix< value_t >( this->nrows(), this->ncols() );
    
        zfp::uncompress< value_t >( _zdata, M.data(), this->nrows(), this->ncols() );
        
        this->blas_mat() = std::move( M );

        #endif
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if defined(HAS_ZFP)
        return ! is_null( _zdata.data() );
        #else
        return false;
        #endif
    }
    
    // return size in bytes used by this object
    virtual size_t byte_size () const
    {
        size_t  size = Hpro::TDenseMatrix< value_t >::byte_size();

        #if defined(HAS_ZFP)

        size += sizeof(_zdata) + _zdata.size();

        if ( is_compressed() )
            size -= this->nrows() * this->ncols() * sizeof(value_t);
    
        #endif
        
        return size;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if defined(HAS_ZFP)
        _zdata = zfp::zarray();
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
    #if defined(HAS_ZFP)

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
        auto  y = blas::vector< value_t >( sy->size() );

        auto  M = blas::matrix< value_t >( this->nrows(), this->ncols() );
    
        zfp::uncompress< value_t >( _zdata, M.data(), this->nrows(), this->ncols() );
        
        blas::mulvec( value_t(alpha), blas::mat_view( op, M ), x, value_t(0), y );
        
        blas::add( value_t(1), y, blas::vec( sy ) );
    }// if
    else

    #endif
    {
        Hpro::TDenseMatrix< value_t >::mul_vec( alpha, vx, beta, vy, op );
    }// else
    
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_DENSE_MATRIX_HH
