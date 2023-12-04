#ifndef __HLR_MATRIX_LRSMATRIX_HH
#define __HLR_MATRIX_LRSMATRIX_HH
//
// Project     : HLR
// Module      : matrix/lrsmatrix
// Description : low-rank matrix with U·S·V' representation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

// activate to also compress S
// #define COMPRESS_S

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrsmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V'.
// - U and V are expected to be orthogonal (CHECK!) and not shared
// - compression for U, S and V is implemented
//
template < typename T_value >
class lrsmatrix : public Hpro::TMatrix< T_value >, public compress::compressible
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
        #if defined(COMPRESS_S)
        compress::zarray  S;
        #endif
    };

    using  compressed_storage = compressed_factors;
    
    #endif

private:
    // local index set of matrix
    indexset                 _row_is, _col_is;
    
    // low-rank factors
    blas::matrix< value_t >  _U, _S, _V;

    #if HLR_HAS_COMPRESSION == 1
    // optional: stores compressed data
    compressed_storage       _zdata;
    #endif
    
public:
    //
    // ctors
    //

    lrsmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
    {}
    
    lrsmatrix ( const indexset  arow_is,
                const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrsmatrix ( const indexset                   arow_is,
                const indexset                   acol_is,
                const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aS,
                const blas::matrix< value_t > &  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( aU )
            , _S( aS )
            , _V( aV )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrsmatrix ( const indexset              arow_is,
                const indexset              acol_is,
                blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aS,
                blas::matrix< value_t > &&  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( std::move( aU ) )
            , _S( std::move( aS ) )
            , _V( std::move( aV ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~lrsmatrix ()
    {}
    
    //
    // access internal data
    //

    uint  rank     () const { return std::min( row_rank(), col_rank() ); }

    uint  row_rank () const { return _U.ncols(); }
    uint  col_rank () const { return _V.ncols(); }

    uint  row_rank ( const Hpro::matop_t  op )  { return op == Hpro::apply_normal ? row_rank() : col_rank(); }
    uint  col_rank ( const Hpro::matop_t  op )  { return op == Hpro::apply_normal ? col_rank() : row_rank(); }

    const blas::matrix< value_t > &  U () const { return _U; }
    const blas::matrix< value_t > &  S () const { return _S; }
    const blas::matrix< value_t > &  V () const { return _V; }
    
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aS,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT( aU.ncols() == aS.nrows() );
        HLR_ASSERT( aV.ncols() == aS.ncols() );
        
        if ( is_compressed() )
            remove_compressed();
        
        _U = aU;
        _S = aS;
        _V = aV;
    }

    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aS,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT( aU.ncols() == aS.nrows() );
        HLR_ASSERT( aV.ncols() == aS.ncols() );
        
        if ( is_compressed() )
            remove_compressed();
        
        _U = std::move( aU );
        _S = std::move( aS );
        _V = std::move( aV );
    }

    // // modify coefficients S even if not consistent with U/V
    // void
    // set_coeff_unsafe ( const blas::matrix< value_t > &  T )
    // {
    //     blas::copy( T, _S );
    // }
    
    // void
    // set_coeff_unsafe ( blas::matrix< value_t > &&  T )
    // {
    //     if (( _S.nrows() == T.nrows() ) && ( _S.ncols() == T.ncols() ))
    //         blas::copy( T, _S );
    //     else
    //         _S = std::move( T );
    // }

    // // clear row/column "bases" (HACK for parallel handling!!!)
    // void clear_row_basis () { _U = blas::matrix< value_t >(); }
    // void clear_col_basis () { _V = blas::matrix< value_t >(); }
    
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
    
    virtual void    set_size  ( const size_t  anrows,
                                const size_t  ancols )
    {
        // change of dimensions not supported
        HLR_ASSERT(( anrows == nrows() ) && ( ancols == ncols() ));
    }
    
    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec ( const value_t                     alpha,
                           const Hpro::TVector< value_t > *  x,
                           const value_t                     beta,
                           Hpro::TVector< value_t > *        y,
                           const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;

    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::matrix< value_t > &  X,
                                blas::matrix< value_t > &        Y,
                                const matop_t                    op = apply_normal ) const;

    // truncate matrix to accuracy \a acc
    virtual void truncate ( const Hpro::TTruncAcc & acc );
    
    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( lrsmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< lrsmatrix< value_t > >(); }

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
    // compression
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // compress data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

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
    virtual size_t byte_size  () const;

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        _zdata.U = compress::zarray();
        _zdata.V = compress::zarray();
        #if defined(COMPRESS_S)
        _zdata.S = compress::zarray();
        #endif
        #endif
    }
};

//
// matrix vector multiplication
//
template < typename value_t >
void
lrsmatrix< value_t >::mul_vec ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  vx,
                                const value_t                     beta,
                                Hpro::TVector< value_t > *        vy,
                                const Hpro::matop_t               op ) const
{
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    // exclude complex value and transposed operation for now
    HLR_ASSERT( (  op == Hpro::apply_normal     ) ||
                (  op == Hpro::apply_adjoint    ) ||
                (( op == Hpro::apply_transposed ) && ! Hpro::is_complex_type_v< value_t > ) );

    const auto  x = cptrcast( vx, Hpro::TScalarVector< value_t > );
    const auto  y = ptrcast(  vy, Hpro::TScalarVector< value_t > );

    // y := β·y
    if ( beta != value_t(1) )
        blas::scale( value_t(beta), blas::vec( y ) );

    apply_add( alpha, blas::vec( *x ), blas::vec( *y ), op );
}

template < typename value_t >
void
lrsmatrix< value_t >::apply_add  ( const value_t                    alpha,
                                   const blas::vector< value_t > &  x,
                                   blas::vector< value_t > &        y,
                                   const matop_t                    op ) const
{
    // exclude complex value and transposed operation for now
    HLR_ASSERT( (  op == Hpro::apply_normal     ) ||
                (  op == Hpro::apply_adjoint    ) ||
                (( op == Hpro::apply_transposed ) && ! Hpro::is_complex_type_v< value_t > ) );

    #if HLR_HAS_COMPRESSION == 1

    if ( is_compressed() )
    {
        auto  uU = blas::matrix< value_t >( this->nrows(), this->row_rank() );
        auto  uV = blas::matrix< value_t >( this->ncols(), this->col_rank() );

        compress::decompress< value_t >( _zdata.U, uU );
        compress::decompress< value_t >( _zdata.V, uV );
        
        #if defined(COMPRESS_S)
        
        auto  uS = blas::matrix< value_t >( this->row_rank(), this->col_rank() );

        compress::decompress< value_t >( _zdata.S, uS );

        blas::mulvec_lr( alpha, uU, uS, uV, op, x, y );

        #else
        
        blas::mulvec_lr( alpha, uU, S(), uV, op, x, y );
        
        #endif
    }// if
    else

    #endif
    {
        blas::mulvec_lr( alpha, U(), S(), V(), op, x, y );
    }// else
}

template < typename value_t >
void
lrsmatrix< value_t >::apply_add ( const value_t                    alpha,
                                  const blas::matrix< value_t > &  X,
                                  blas::matrix< value_t > &        Y,
                                  const matop_t                    op ) const
{
    if ( is_compressed() )
        HLR_ERROR( "TODO" );
    
    switch ( op )
    {
        case apply_normal :
        {
            // Y = Y + U·(S·(V'·X))
            auto  T1 = blas::prod( blas::adjoint( _V ), X );
            auto  T2 = blas::prod( _S, T1 );

            blas::prod( alpha, _U, T2, value_t(1), Y );
        }
        break;

        case apply_conjugate :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // Y = Y + U·(S·(V'·X))
                auto  T1 = blas::prod( blas::adjoint( _V ), X );
                auto  T2 = blas::prod( _S, T1 );
                
                blas::prod( alpha, _U, T2, value_t(1), Y );
            }// else
        }
        break;
        
        case apply_transposed :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // Y = Y + V·(S'·(U'·X))
                auto  T1 = blas::prod( blas::adjoint( _U ), X );
                auto  T2 = blas::prod( blas::adjoint( _S ), T1 );
                
                blas::prod( alpha, _V, T2, value_t(1), Y );
            }// else
        }
        break;
        
        case apply_adjoint :
        {
            // Y = Y + V·(S'·(U'·X))
            auto  T1 = blas::prod( blas::adjoint( _U ), X );
            auto  T2 = blas::prod( blas::adjoint( _S ), T1 );

            blas::prod( alpha, _V, T2, value_t(1), Y );
        }
        break;
    }// switch
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
lrsmatrix< value_t >::truncate ( const Hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy () const
{
    auto  M = std::make_unique< lrsmatrix< value_t > >( _row_is, _col_is,
                                                        std::move( blas::copy( _U ) ),
                                                        std::move( blas::copy( _S ) ),
                                                        std::move( blas::copy( _V ) ) );

    M->copy_struct_from( this );
    
    HLR_ASSERT( IS_TYPE( M.get(), lrsmatrix ) );

    auto  R = ptrcast( M.get(), lrsmatrix< value_t > );

    #if HLR_HAS_COMPRESSION == 1

    if ( is_compressed() )
    {
        R->_zdata.U = compress::zarray( _zdata.U.size() );
        R->_zdata.V = compress::zarray( _zdata.V.size() );

        std::copy( _zdata.U.begin(), _zdata.U.end(), R->_zdata.U.begin() );
        std::copy( _zdata.V.begin(), _zdata.V.end(), R->_zdata.V.begin() );

        #if defined(COMPRESS_S)
        R->_zdata.S = compress::zarray( _zdata.S.size() );

        std::copy( _zdata.S.begin(), _zdata.S.end(), R->_zdata.S.begin() );
        #endif
    }// if

    #endif
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy ( const Hpro::TTruncAcc &,
                             const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< lrsmatrix< value_t > >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
lrsmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    Hpro::TMatrix< value_t >::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, lrsmatrix ) );

    auto  R = ptrcast( A, lrsmatrix< value_t > );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_U      = std::move( blas::copy( _U ) );
    R->_S      = std::move( blas::copy( _S ) );
    R->_V      = std::move( blas::copy( _V ) );

    HLR_ASSERT( IS_TYPE( A, lrsmatrix ) );

    #if HLR_HAS_COMPRESSION == 1

    if ( is_compressed() )
    {
        HLR_ERROR( "TODO" );
    }// if

    #endif
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
lrsmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A,
                                const Hpro::TTruncAcc &,
                                const bool          ) const
{
    return copy_to( A );
}

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
lrsmatrix< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;

    auto          oU       = this->U();
    auto          oV       = this->V();
    const auto    orowrank = this->row_rank();
    const auto    ocolrank = this->row_rank();
    auto          zU       = compress::compress< value_t >( zconfig, oU.data(), oU.nrows(), oU.ncols() );
    auto          zV       = compress::compress< value_t >( zconfig, oV.data(), oV.nrows(), oV.ncols() );

    #if defined(COMPRESS_S)
    auto          oS       = this->S();
    auto          zS       = compress::compress< value_t >( zconfig, oS.data(), oS.nrows(), oS.ncols() );
    const size_t  mem_lr   = sizeof(value_t) * ( orowrank * oU.nrows() + ocolrank * oV.nrows() + orowrank * ocolrank );
    const size_t  mem_z    = compress::compressed_size( zU ) + compress::compressed_size( zV ) + compress::compressed_size( zS );
    #else
    const size_t  mem_lr   = sizeof(value_t) * ( orowrank * oU.nrows() + ocolrank * oV.nrows() );
    const size_t  mem_z    = compress::compressed_size( zU ) + compress::compressed_size( zV );
    #endif

    // const auto  vmin = blas::min_abs_val( oU );
    // const auto  vmax = blas::max_abs_val( oU );

    // std::cout << vmin << " / " << vmax << " / " << vmax / vmin << std::endl;
        
    if ( mem_z < mem_lr )
    {
        _zdata.U  = std::move( zU );
        _U        = std::move( blas::matrix< value_t >( 0, orowrank ) ); // remember rank !!!

        _zdata.V  = std::move( zV );
        _V        = std::move( blas::matrix< value_t >( 0, ocolrank ) );
        
        #if defined(COMPRESS_S)
        _S        = std::move( blas::matrix< value_t >( 0, 0 ) );
        _zdata.S  = std::move( zS );
        #endif
    }// if

    #endif
}

template < typename value_t >
void
lrsmatrix< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    if ( this->nrows() * this->ncols() == 0 )
        return;
        
    const auto  eps   = acc( this->row_is(), this->col_is() ).rel_eps();
    const auto  normF = blas::norm_F( _S ); // assuming U/V being orthogonal
        
    // compress( compress::get_config( eps * normF / double(std::min( this->nrows(), this->ncols() )) ) );
    compress( compress::get_config( eps ) );
}

//
// decompress internal data
//
template < typename value_t >
void
lrsmatrix< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    auto  uU = blas::matrix< value_t >( this->nrows(), this->row_rank() );
    auto  uV = blas::matrix< value_t >( this->ncols(), this->col_rank() );

    compress::decompress< value_t >( _zdata.U, uU.data(), uU.nrows(), uU.ncols() );
    compress::decompress< value_t >( _zdata.V, uV.data(), uV.nrows(), uV.ncols() );

    _U = std::move( uU );
    _V = std::move( uV );
    
    #if defined(COMPRESS_S)
    auto  uS = blas::matrix< value_t >( this->row_rank(), this->col_rank() );

    compress::decompress< value_t >( _zdata.S, uS.data(), uS.nrows(), uS.ncols() );
    _S = std::move( uS );
    #endif

    remove_compressed();
        
    #endif
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
lrsmatrix< value_t >::byte_size () const
{
    size_t  size = Hpro::TMatrix< value_t >::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += _U.byte_size() + _S.byte_size() + _V.byte_size();

    #if HLR_HAS_COMPRESSION == 1

    size += hlr::compress::byte_size( _zdata.U );
    size += hlr::compress::byte_size( _zdata.V );

    #if defined(COMPRESS_S)
    size += hlr::compress::byte_size( _zdata.S );
    #endif
    
    #endif
    
    return size;
}

//
// type test
//
template < typename value_t >
bool
is_lowrankS ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, lrsmatrix );
}

template < typename value_t >
bool
is_lowrankS ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrsmatrix );
}

HLR_TEST_ALL( is_lowrankS, hlr::matrix::is_lowrankS, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_lowrankS, hlr::matrix::is_lowrankS, Hpro::TMatrix< value_t > )

// template < typename value_t >
// bool
// is_compressible_lowrankS ( const Hpro::TMatrix< value_t > &  M )
// {
//     return IS_TYPE( &M, lrsmatrix );
// }

// template < typename value_t >
// bool
// is_compressible_lowrankS ( const Hpro::TMatrix< value_t > *  M )
// {
//     return ! is_null( M ) && IS_TYPE( M, lrsmatrix );
// }

// HLR_TEST_ALL( is_compressible_lowrankS, Hpro::TMatrix< value_t > )
// HLR_TEST_ANY( is_compressible_lowrankS, Hpro::TMatrix< value_t > )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRSMATRIX_HH
