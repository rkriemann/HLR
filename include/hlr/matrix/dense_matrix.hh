#ifndef __HLR_MATRIX_DENSE_MATRIX_HH
#define __HLR_MATRIX_DENSE_MATRIX_HH
//
// Project     : HLR
// Module      : dense_matrix
// Description : dense matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/compress/compressible.hh>
#include <hlr/compress/direct.hh>
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
class dense_matrix : public Hpro::TMatrix< T_value >, public compress::compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    // local index set of matrix
    indexset                 _row_is, _col_is;
    
    // low-rank factors
    blas::matrix< value_t >  _mat;

    // stores compressed data
    compress::zarray         _zM;
    
public:
    //
    // ctors
    //

    dense_matrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
    {}
    
    dense_matrix ( const indexset  arow_is,
                   const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _mat( _row_is.size(), _col_is.size() )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    dense_matrix ( const indexset             arow_is,
                   const indexset             acol_is,
                   blas::matrix< value_t > &  aM )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _mat( aM )
    {
        HLR_ASSERT(( _row_is.size() == _mat.nrows() ) &&
                   ( _col_is.size() == _mat.ncols() ));

        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    dense_matrix ( const indexset              arow_is,
                   const indexset              acol_is,
                   blas::matrix< value_t > &&  aM )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _mat( std::move( aM ) )
    {
        HLR_ASSERT(( _row_is.size() == _mat.nrows() ) &&
                   ( _col_is.size() == _mat.ncols() ));

        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    dense_matrix ( const indexset       arow_is,
                   const indexset       acol_is,
                   compress::zarray &&  azM )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _zM( std::move( azM ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    dense_matrix ( dense_matrix< value_t > &&  aM )
            : Hpro::TMatrix< value_t >()
            , _row_is( aM.row_is() )
            , _col_is( aM.col_is() )
            , _mat( std::move( aM._mat ) )
            , _zM( std::move( aM._zM ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }
    
    // dtor
    virtual ~dense_matrix ()
    {}
    
    //
    // access internal data
    //

    blas::matrix< value_t >  mat () const
    {
        if ( is_compressed() )
        {
            auto  dM = blas::matrix< value_t >( this->nrows(), this->ncols() );
    
            compress::decompress< value_t >( _zM, dM );
            
            return dM;
        }// if

        return _mat;
    }

    blas::matrix< value_t > &        mat_direct  ()       { HLR_ASSERT( ! is_compressed() ); return _mat; }
    const blas::matrix< value_t > &  mat_direct  () const { HLR_ASSERT( ! is_compressed() ); return _mat; }
    
    void
    set_matrix ( const blas::matrix< value_t > &  aM )
    {
        HLR_ASSERT(( this->nrows() == aM.nrows() ) && ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        blas::copy( aM, _mat );
    }
    
    void
    set_matrix ( const blas::matrix< value_t > &  aM,
                 const accuracy &                 acc )
    {
        HLR_ASSERT(( this->nrows() == aM.nrows() ) && ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
        {
            remove_compressed();
            _mat = blas::copy( aM );
            compress( acc );
        }// if
        else
            blas::copy( aM, _mat );
    }

    void
    set_matrix ( blas::matrix< value_t > &&  aM )
    {
        HLR_ASSERT(( this->nrows() == aM.nrows() ) && ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        _mat = std::move( aM );
    }
    
    void
    set_matrix ( blas::matrix< value_t > &&  aM,
                 const accuracy &            acc )
    {
        HLR_ASSERT(( this->nrows() == aM.nrows() ) && ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
        {
            remove_compressed();
            _mat = std::move( aM );
            compress( acc );
        }// if
        else
            _mat = std::move( aM );
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
    
    // return true, if matrix is zero (just trivial test)
    virtual bool    is_zero   () const { return ( nrows() * ncols() == 0 ); }
    
    virtual void    set_size  ( const size_t  anrows,
                                const size_t  ancols )
    {
        // change of dimensions not supported
        HLR_ASSERT(( anrows == nrows() ) && ( ancols == ncols() ));
    }
    
    //
    // algebra routines
    //

    // scale matrix by constant factor \a f
    virtual void  scale  ( const value_t  f )
    {
        if ( is_compressed() )
        {
            HLR_ERROR( "to do" );
        }// if
        else
        {
            blas::scale( f, _mat );
        }// else
    }
    
    //! compute y ≔ α·op(this)·x + β·y
    virtual void  mul_vec    ( const value_t                     alpha,
                               const Hpro::TVector< value_t > *  x,
                               const value_t                     beta,
                               Hpro::TVector< value_t > *        y,
                               const matop_t                     op = Hpro::apply_normal ) const;

    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add  ( const value_t                    alpha,
                               const blas::vector< value_t > &  x,
                               blas::vector< value_t > &        y,
                               const matop_t                    op = apply_normal ) const;

    virtual void  apply_add  ( const value_t                    alpha,
                               const blas::matrix< value_t > &  X,
                               blas::matrix< value_t > &        Y,
                               const matop_t                    op = apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void  truncate   ( const accuracy & ) {}

    //
    // compression
    //

    // // compress internal data based on given configuration
    // // - may result in non-compression if storage does not decrease
    // virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // compress internal data based on given accuracy
    virtual void   compress      ( const accuracy &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _zM.empty();
    }

    // access compression data
    const compress::zarray  zmat () const { return _zM; }
    
    void  set_zmatrix  ( compress::zarray &&  azM )
    {
        // ASSUMPTION: correct size!!!
        _zM = std::move( azM );
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( dense_matrix, Hpro::TMatrix< value_t > )

    // return matrix of same class (but no content)
    virtual auto   create  () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< dense_matrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy    () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = std::unique_ptr< dense_matrix< value_t > >();

        if ( is_compressed() )
        {
            auto  zM = compress::zarray( _zM.size() );

            std::copy( _zM.begin(), _zM.end(), zM.begin() );
            
            M = std::make_unique< dense_matrix< value_t > >( _row_is, _col_is, std::move( zM ) );
        }// if
        else
        {
            M = std::make_unique< dense_matrix< value_t > >( _row_is, _col_is, std::move( blas::copy( this->_mat ) ) );
        }// else

        M->copy_struct_from( this );
        
        return M;
    }

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy  ( const accuracy &  /* acc */,
                           const bool        /* do_coarsen */ = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return copy();
    }

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return std::make_unique< dense_matrix< value_t > >( this->row_is(), this->col_is() );
    }

    // copy matrix data to \a A
    virtual void   copy_to  ( Hpro::TMatrix< value_t > *  A ) const
    {
        Hpro::TMatrix< value_t >::copy_to( A );
    
        HLR_ASSERT( IS_TYPE( A, dense_matrix ) );

        auto  D = ptrcast( A, dense_matrix< value_t > );

        D->_row_is = _row_is;
        D->_col_is = _col_is;
        D->_mat    = std::move( blas::copy( this->_mat ) );

        if ( is_compressed() )
        {
            D->_zM = compress::zarray( _zM.size() );

            std::copy( _zM.begin(), _zM.end(), D->_zM.begin() );
        }// if
        else
        {
            D->_zM = compress::zarray();
        }// else
    }

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const accuracy &            /* acc */,
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
        size_t  size = Hpro::TMatrix< value_t >::byte_size();

        size += sizeof(_row_is) + sizeof(_col_is);
        size += _mat.byte_size();
        size += hlr::compress::byte_size( _zM );
        
        return size;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        if ( is_compressed() )
            return hlr::compress::byte_size( _zM );
        else
            return sizeof( value_t ) * _row_is.size() * _col_is.size();
    }
    
    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        auto  M = mat();

        M.check_data();
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        _zM = compress::zarray();
    }
    
};

//
// type test
//
template < typename value_t >
bool
is_dense ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, dense_matrix );
}

template < typename value_t >
bool
is_dense ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, dense_matrix );
}

HLR_TEST_ALL( is_dense, hlr::matrix::is_dense, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_dense, hlr::matrix::is_dense, Hpro::TMatrix< value_t > )

//
// matrix vector multiplication
//
template < typename value_t >
void
dense_matrix< value_t >::mul_vec ( const value_t                     alpha,
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
    
    apply_add( alpha, blas::vec( *sx ), blas::vec( sy ), op );
}

template < typename value_t >
void
dense_matrix< value_t >::apply_add   ( const value_t                    alpha,
                                       const blas::vector< value_t > &  x,
                                       blas::vector< value_t > &        y,
                                       const matop_t                    op ) const
{
    HLR_ASSERT( x.length() == this->ncols( op ) );
    HLR_ASSERT( y.length() == this->nrows( op ) );

    #if defined(HLR_HAS_ZBLAS_DIRECT)
    if ( is_compressed() )
    {
        compress::zblas::mulvec( nrows(), ncols(), op, alpha, _zM, x.data(), y.data() );
    }// if
    else
    #endif
    {
        auto  M = mat();
        
        blas::mulvec( alpha, blas::mat_view( op, M ), x, value_t(1), y );
    }// else
}

template < typename value_t >
void
dense_matrix< value_t >::apply_add   ( const value_t                    alpha,
                                       const blas::matrix< value_t > &  X,
                                       blas::matrix< value_t > &        Y,
                                       const matop_t                    op ) const
{
    HLR_ASSERT( X.nrows() == this->ncols( op ) );
    HLR_ASSERT( Y.nrows() == this->nrows( op ) );

    auto  M = mat();
        
    blas::prod( alpha, blas::mat_view( op, M ), X, value_t(1), Y );
}

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
dense_matrix< value_t >::compress ( const accuracy &  acc )
{
    if ( this->nrows() * this->ncols() == 0 )
        return;

    //
    // compress
    //

    auto          M       = _mat;
    const auto    zconfig = compress::get_config( acc( this->row_is(), this->col_is() ), M );
    auto          zM      = compress::compress< value_t >( zconfig, M );
    
    // // DEBUG
    // {
    //     auto  M1 = blas::copy( M );
    //     auto  M2 = blas::matrix< value_t >( M.nrows(), M.ncols() );

    //     compress::decompress( zM, M2 );

    //     // io::matlab::write( M1, "M1" );
    //     // io::matlab::write( M2, "M2" );
        
    //     blas::add( -1.0, M1, M2 );

    //     if ( acc.norm_mode() == Hpro::spectral_norm )
    //         std::cout << "D"
    //                   << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_2( M1 ) )
    //                   << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) )
    //                   << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2(M1) )
    //                   << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                   << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                   << std::endl;
    //     else
    //         std::cout << "D"
    //                   << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) )
    //                   << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) )
    //                   << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F(M1) )
    //                   << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                   << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                   << std::endl;
    // }
    // // DEBUG

    //
    // finish
    //

    const size_t  mem  = sizeof(value_t) * M.nrows() * M.ncols();
    const auto    zmem = compress::compressed_size( zM );
    
    // std::cout << "D: " << double(mem) / double(zmem) << std::endl;

    if (( zmem != 0 ) && ( zmem < mem ))
    {
        _zM  = std::move( zM );
        _mat = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
dense_matrix< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    _mat = std::move( mat() );

    remove_compressed();
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_DENSE_MATRIX_HH
