#ifndef __HLR_MATRIX_LRMATRIX_HH
#define __HLR_MATRIX_LRMATRIX_HH
//
// Project     : HLR
// Module      : lrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <variant>

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

namespace hpro = HLIB;

using indexset = hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
class lrmatrix : public hpro::TMatrix
{
private:
    //
    // compressed storage based on underlying floating point type
    //
    #if defined(HAS_SZ)
    
    struct compressed_factors
    {
        sz::carray_view  U, V;
    };

    using  compressed_storage = compressed_factors;
    
    #elif defined(HAS_ZFP)

    template < typename T_value >
    struct compressed_factors
    {
        using value_t = T_value;

        zfp::const_array2< value_t >  U, V;
    };

    using  compressed_storage = std::variant< std::unique_ptr< compressed_factors< float > >,
                                              std::unique_ptr< compressed_factors< double > > >;
    
    #endif

public:
    template < typename T_value >
    struct lrfactors
    {
        using value_t = T_value;
        
        blas::matrix< value_t >  U, V;
    };

    using  generic_lrfactors = std::variant<
        lrfactors< float >,
        lrfactors< std::complex< float > >,
        lrfactors< double >,
        lrfactors< std::complex< double > >
        >;
    
private:
    // local index set of matrix
    indexset              _row_is, _col_is;
    
    // lowrank factors
    generic_lrfactors     _UV;

    // indicates internal value type
    // - after initialization identical to _M.index()
    blas::value_type      _vtype;
    
    #if defined(HAS_SZ) || defined(HAS_ZFP)
    // optional: stores compressed data
    compressed_storage    _zdata;
    #endif

public:
    //
    // ctors
    //

    lrmatrix ()
            : TMatrix()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _vtype( blas::value_type::undefined )
    {
    }
    
    lrmatrix ( const indexset                arow_is,
               const indexset                acol_is )
            : TMatrix()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _vtype( blas::value_type::undefined )
    {
        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &   aU,
               hlr::blas::matrix< value_t > &   aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _UV( lrfactors< value_t >{ blas::copy( aU ), blas::copy( aV ) } )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< lrfactors< value_t > >( _UV ).U.nrows() ) &&
                   ( _col_is.size() == std::get< lrfactors< value_t > >( _UV ).V.nrows() ) &&
                   ( std::get< lrfactors< value_t > >( _UV ).U.ncols() == std::get< lrfactors< value_t > >( _UV ).V.ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &&  aU,
               hlr::blas::matrix< value_t > &&  aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _UV( lrfactors< value_t >{ std::move( aU ), std::move( aV ) } )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< lrfactors< value_t > >( _UV ).U.nrows() ) &&
                   ( _col_is.size() == std::get< lrfactors< value_t > >( _UV ).V.nrows() ) &&
                   ( std::get< lrfactors< value_t > >( _UV ).U.ncols() == std::get< lrfactors< value_t > >( _UV ).V.ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint
    rank () const
    {
        return std::visit( [] ( auto &&  M ) { return M.U.ncols(); }, _UV );
    }

    // return value type of matrix
    blas::value_type  value_type () const { return _vtype; }

    generic_lrfactors        factors ()       { return _UV; }
    const generic_lrfactors  factors () const { return _UV; }
    
    template < typename value_t > blas::matrix< value_t > &        U ()       { return std::get< lrfactors< value_t > >( _UV ).U; }
    template < typename value_t > blas::matrix< value_t > &        V ()       { return std::get< lrfactors< value_t > >( _UV ).V; }
    
    template < typename value_t > const blas::matrix< value_t > &  U () const { return std::get< lrfactors< value_t > >( _UV ).U; }
    template < typename value_t > const blas::matrix< value_t > &  V () const { return std::get< lrfactors< value_t > >( _UV ).V; }
    
    template < typename value_t >
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        if (( blas::value_type_v< value_t > == _vtype ) && ( aU.ncols() == U< value_t >().ncols() ))
        {
            blas::copy( aU, U< value_t >() );
            blas::copy( aV, V< value_t >() );
        }// if
        else
        {
            _UV    = lrfactors< value_t >{ blas::copy( aU ), blas::copy( aV ) };
            _vtype = blas::value_type_v< value_t >;
        }// else
    }
    
    template < typename value_t >
    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        _UV    = lrfactors< value_t >{ std::move( aU ), std::move( aV ) };
        _vtype = blas::value_type_v< value_t >;
    }

    //
    // matrix data
    //
    
    virtual size_t  nrows     () const { return _row_is.size(); }
    virtual size_t  ncols     () const { return _col_is.size(); }

    virtual size_t  rows      () const { return nrows(); }
    virtual size_t  cols      () const { return ncols(); }

    // use "op" versions from TMatrix
    using TMatrix::nrows;
    using TMatrix::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // change value type 
    //
    
    virtual void  to_real     () { HLR_ASSERT( false ); }
    virtual void  to_complex  () { HLR_ASSERT( false ); }

    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec  ( const hpro::real       alpha,
                            const hpro::TVector *  x,
                            const hpro::real       beta,
                            hpro::TVector *        y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    using hpro::TMatrix::mul_vec;
    
    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void cmul_vec ( const hpro::complex    alpha,
                            const hpro::TVector *  x,
                            const hpro::complex    beta,
                            hpro::TVector *        y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // truncate matrix to accuracy acc
    virtual void truncate ( const hpro::TTruncAcc & acc );

    // scale matrix by alpha
    virtual void scale    ( const hpro::real  alpha )
    {
        if ( is_compressed() )
        {
            HLR_ERROR( "todo" );
        }// if
        else
        {
            std::visit(
                [alpha] ( auto &&  M )
                {
                    using  value_t  = typename std::decay_t< decltype(M) >::value_t;
                    
                    if ( M.U.nrows() < M.V.ncols() )
                        blas::scale( value_t(alpha), M.U );
                    else
                        blas::scale( value_t(alpha), M.V );
                },
                _UV );
        }// else
    }

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< hpro::TMatrix > { return std::make_unique< lrmatrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< hpro::TMatrix >;

    // return copy matrix wrt. given accuracy; if do_coarsen is set, perform coarsening
    virtual auto   copy         ( const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const -> std::unique_ptr< hpro::TMatrix >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< hpro::TMatrix >;

    // copy matrix data to A
    virtual void   copy_to      ( hpro::TMatrix *          A ) const;

    // copy matrix data to A and truncate w.r.t. acc with optional coarsening
    virtual void   copy_to      ( hpro::TMatrix *          A,
                                  const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const;
    
    //
    // misc.
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const zconfig_t &  config );

    // uncompress internal data
    virtual void   uncompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if defined(HAS_SZ)
        return ! is_null( _zdata.U.data() );
        #elif defined(HAS_ZFP)
        return ! std::visit( [] ( auto && d ) { return is_null( d ); }, _zdata );
        #else
        return false;
        #endif
    }
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const;

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if defined(HAS_SZ)
        _zdata.U.free();
        _zdata.V.free();
        #elif defined(HAS_ZFP)
        std::visit( [] ( auto && d ) { d.reset( nullptr ); }, _zdata );
        #endif
    }
};

//
// type test
//
inline
bool
is_generic_lowrank ( const hpro::TMatrix &  M )
{
    return IS_TYPE( &M, lrmatrix );
}

inline
bool
is_generic_lowrank ( const hpro::TMatrix *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrmatrix );
}

HLR_TEST_ALL( is_generic_lowrank, hpro::TMatrix )
HLR_TEST_ANY( is_generic_lowrank, hpro::TMatrix )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRMATRIX_HH
