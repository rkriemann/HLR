#ifndef __HLR_MATRIX_LRMATRIX_HH
#define __HLR_MATRIX_LRMATRIX_HH
//
// Project     : HLR
// Module      : lrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <boost/format.hpp> // DEBUG

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/approx/svd.hh>
#include <hlr/compress/compressible.hh>
#include <hlr/compress/direct.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

// #include <hlr/utils/io.hh> // DEBUG

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·V^H
// with optional compression of U/V
//
template < typename T_value >
class lrmatrix : public Hpro::TMatrix< T_value >, public compress::compressible
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    
private:
    //
    // compressed storage based on underlying floating point type
    //
    struct zstorage
    {
        compress::zarray  U, V;
    };

private:
    // local index set of matrix
    indexset                 _row_is, _col_is;
    
    // low-rank factors
    blas::matrix< value_t >  _U, _V;

    // compressed factors
    compress::zarray         _zU, _zV;

    // rank for simplified access and for compressed factors
    uint                     _rank;

public:
    //
    // ctors
    //

    lrmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _rank( 0 )
    {}
    
    lrmatrix ( const indexset              arow_is,
               const indexset              acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( _row_is.size(), 0 ) // to avoid issues with nrows/ncols
            , _V( _col_is.size(), 0 )
            , _rank( 0 )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrmatrix ( const indexset              arow_is,
               const indexset              acol_is,
               blas::matrix< value_t > &   aU,
               blas::matrix< value_t > &   aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( aU )
            , _V( aV )
            , _rank( _U.ncols() )
    {
        HLR_ASSERT(( _row_is.size() == _U.nrows() ) &&
                   ( _col_is.size() == _V.nrows() ) &&
                   ( _U.ncols()     == _V.ncols() ));
        
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrmatrix ( const indexset              arow_is,
               const indexset              acol_is,
               blas::matrix< value_t > &&  aU,
               blas::matrix< value_t > &&  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( std::move( aU ) )
            , _V( std::move( aV ) )
            , _rank( _U.ncols() )
    {
        HLR_ASSERT(( _row_is.size() == _U.nrows() ) &&
                   ( _col_is.size() == _V.nrows() ) &&
                   ( _U.ncols()     == _V.ncols() ));
        
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrmatrix ( const lrmatrix< value_t > &  M )
            : Hpro::TMatrix< value_t >()
            , _row_is( M._row_is )
            , _col_is( M._col_is )
            , _U( blas::copy( M._U ) )
            , _V( blas::copy( M._V ) )
            , _rank( _U.ncols() )
    {}

    // dtor
    virtual ~lrmatrix ()
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
    // access low-rank data
    //

    blas::matrix< value_t >  U  () const
    {
        if ( is_compressed() )
        {
            auto  dU = blas::matrix< value_t >( this->nrows(), this->rank() );
    
            compress::decompress< value_t >( _zU, dU );

            return dU;
        }// if
        
        return _U;
    }
    
    blas::matrix< value_t >  V  () const
    {
        if ( is_compressed() )
        {
            auto  dV = blas::matrix< value_t >( this->ncols(), this->rank() );
    
            compress::decompress< value_t >( _zV, dV );

            return dV;
        }// if

        return _V;
    }

    blas::matrix< value_t >  U  ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U() : V() ); }
    blas::matrix< value_t >  V  ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V() : U() ); }

    // for direct access of lowrank factors assuming no compression
    // - also to be used as fail-safe for algorithms without compression support
    blas::matrix< value_t > &        U_direct  ()       { HLR_ASSERT( ! is_compressed() ); return _U; }
    blas::matrix< value_t > &        V_direct  ()       { HLR_ASSERT( ! is_compressed() ); return _V; }

    const blas::matrix< value_t > &  U_direct  () const { HLR_ASSERT( ! is_compressed() ); return _U; }
    const blas::matrix< value_t > &  V_direct  () const { HLR_ASSERT( ! is_compressed() ); return _V; }
    
    blas::matrix< value_t > &        U_direct  ( const Hpro::matop_t  op )       { return ( op == apply_normal ? U_direct() : V_direct() ); }
    blas::matrix< value_t > &        V_direct  ( const Hpro::matop_t  op )       { return ( op == apply_normal ? V_direct() : U_direct() ); }

    const blas::matrix< value_t > &  U_direct  ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? U_direct() : V_direct() ); }
    const blas::matrix< value_t > &  V_direct  ( const Hpro::matop_t  op ) const { return ( op == apply_normal ? V_direct() : U_direct() ); }

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
        
        _U = blas::copy( aU );
        _V = blas::copy( aV );

        _rank = _U.ncols();
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
        
        _U = std::move( aU );
        _V = std::move( aV );

        _rank = _U.ncols();
    }

    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV,
                const accuracy &            acc )
    {
        HLR_ASSERT(( this->nrows() == aU.nrows() ) &&
                   ( this->ncols() == aV.nrows() ) &&
                   ( aU.ncols()    == aV.ncols() ));

        auto  was_compressed = is_compressed();
        
        if ( was_compressed )
            remove_compressed();
        
        _U = std::move( aU );
        _V = std::move( aV );

        _rank = _U.ncols();

        if ( was_compressed )
            compress( acc );
    }

    // update U and recompress if needed
    void
    set_U ( const blas::matrix< value_t > &  aU,
            const accuracy &                 acc )
    {
        HLR_ASSERT(( this->nrows() == aU.nrows() ) && ( aU.ncols() == this->rank() ));

        if ( is_compressed() )
        {
            // as this is just an update, compress without memory size test
            auto  zconfig = compress::get_config( acc( this->row_is(), this->col_is() ), aU );
            
            _zU = std::move( compress::compress< value_t >( zconfig, aU ) );
        }// if
        else
        {
            blas::copy( aU, _U );
        }// else
    }

    void
    set_U ( blas::matrix< value_t > &&  aU,
            const accuracy &            acc )
    {
        HLR_ASSERT(( this->nrows() == aU.nrows() ) && ( aU.ncols() == this->rank() ));

        if ( is_compressed() )
        {
            // as this is just an update, compress without memory size test
            auto  zconfig = compress::get_config( acc( this->row_is(), this->col_is() ), aU );
            
            _zU = std::move( compress::compress< value_t >( zconfig, aU ) );
        }// if
        else
        {
            _U = std::move( aU );
        }// else
    }

    // update V and recompress if needed
    void
    set_V ( const blas::matrix< value_t > &  aV,
            const accuracy &                 acc )
    {
        HLR_ASSERT(( this->ncols() == aV.nrows() ) && ( aV.ncols() == this->rank() ));

        if ( is_compressed() )
        {
            // as this is just an update, compress without memory size test
            auto  zconfig = compress::get_config( acc( this->row_is(), this->col_is() ), aV );
            
            _zV = std::move( compress::compress< value_t >( zconfig, aV ) );
        }// if
        else
        {
            blas::copy( aV, _V );
        }// else
    }

    void
    set_V ( blas::matrix< value_t > &&  aV,
            const accuracy &            acc )
    {
        HLR_ASSERT(( this->ncols() == aV.nrows() ) && ( aV.ncols() == this->rank() ));

        if ( is_compressed() )
        {
            // as this is just an update, compress without memory size test
            auto  zconfig = compress::get_config( acc( this->row_is(), this->col_is() ), aV );
            
            _zV = std::move( compress::compress< value_t >( zconfig, aV ) );
        }// if
        else
        {
            _V = std::move( aV );
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
    
    // truncate matrix to accuracy acc
    virtual void truncate    ( const accuracy & acc )
    {
        if ( is_compressed() )
        {
            HLR_ERROR( "todo" );
        }// if
        else
        {
            auto  apx      = approx::SVD< value_t >();
            auto  [ W, X ] = apx( _U, _V, acc );

            set_lrmat( std::move( W ), std::move( X ), acc );
        }// else
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
            if ( nrows() < ncols()  ) blas::scale( alpha, _U );
            else                      blas::scale( alpha, _V );
        }// else
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( lrmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< lrmatrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  R = std::make_unique< lrmatrix< value_t > >( _row_is, _col_is );
        
        R->copy_struct_from( this );
        R->_rank = _rank;
        
        if ( is_compressed() )
        {
            R->_zU = compress::zarray( _zU.size() );
            R->_zV = compress::zarray( _zV.size() );

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

    // return copy matrix wrt. given accuracy; if do_coarsen is set, perform coarsening
    virtual auto   copy         ( const accuracy &  /* acc */,
                                  const bool        /* do_coarsen */ = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >
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

        R->_row_is = _row_is;
        R->_row_is = _col_is;
        R->_U      = std::move( blas::copy( _U ) );
        R->_V      = std::move( blas::copy( _V ) );
        R->_rank   = _rank;
            
        if ( is_compressed() )
        {
            R->_zU = compress::zarray( _zU.size() );
            R->_zV = compress::zarray( _zV.size() );

            std::copy( _zU.begin(), _zU.end(), R->_zU.begin() );
            std::copy( _zV.begin(), _zV.end(), R->_zV.begin() );
        }// if
        else
        {
            R->_zU = compress::zarray();
            R->_zV = compress::zarray();
        }// else
    }
        

    // copy matrix data to A and truncate w.r.t. acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const accuracy &            /* acc */,
                                  const bool                  /* do_coarsen */ = false ) const
    {
        return copy_to( A );
    }
    
    //
    // compression
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const accuracy &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! is_null( _zU.data() );
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
        
        size += hlr::compress::byte_size( _zU );
        size += hlr::compress::byte_size( _zV );
        
        return size;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        if ( is_compressed() )
            return hlr::compress::byte_size( _zU ) + hlr::compress::byte_size( _zV );
        else
            return sizeof( value_t ) * _rank * ( _row_is.size() + _col_is.size() );
    }
    
    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        auto  RU = U();
        auto  RV = V();

        RU.check_data();
        RV.check_data();
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        _zU = compress::zarray();
        _zV = compress::zarray();
    }
};

//
// type test
//
template < typename value_t >
inline
bool
is_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, lrmatrix );
}

template < typename value_t >
bool
is_lowrank ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrmatrix );
}

HLR_TEST_ALL( is_lowrank, hlr::matrix::is_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_lowrank, hlr::matrix::is_lowrank, Hpro::TMatrix< value_t > )

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
lrmatrix< value_t >::apply_add ( const value_t                    alpha,
                                 const blas::vector< value_t > &  x,
                                 blas::vector< value_t > &        y,
                                 const matop_t                    op ) const
{
    HLR_ASSERT( x.length() == this->ncols( op ) );
    HLR_ASSERT( y.length() == this->nrows( op ) );
    
    #if defined(HLR_HAS_ZBLAS_DIRECT)
    if ( is_compressed() )
    {
        const auto  nrows = this->nrows();
        const auto  ncols = this->ncols();
        auto        t     = blas::vector< value_t >( _rank );
        
        if ( op == Hpro::apply_normal )
        {
            // t := V^H x
            compress::zblas::mulvec( ncols, _rank, apply_adjoint, value_t(1), _zV, x.data(), t.data() );

            // t := α·t
            for ( uint  i = 0; i < _rank; ++i )
                t(i) *= value_t(alpha);
        
            // y := y + U t
            compress::zblas::mulvec( nrows, _rank, apply_normal, value_t(1), _zU, t.data(), y.data() );
        }// if
        else if ( op == Hpro::apply_transposed )
        {
            HLR_ERROR( "TODO" );
        }// if
        else if ( op == Hpro::apply_adjoint )
        {
            // t := U^H x
            compress::zblas::mulvec( nrows, _rank, apply_adjoint, value_t(1), _zU, x.data(), t.data() );

            // t := α·t
            for ( uint  i = 0; i < _rank; ++i )
                t(i) *= value_t(alpha);
        
            // y := t + V t
            compress::zblas::mulvec( ncols, _rank, apply_normal, value_t(1), _zV, t.data(), y.data() );
        }// if
    }// if
    else
    #endif
    {
        auto  dU = U();
        auto  dV = V();

        blas::mulvec_lr( alpha, dU, dV, op, x, y );
    }// else
}

template < typename value_t >
void
lrmatrix< value_t >::apply_add ( const value_t                    alpha,
                                 const blas::matrix< value_t > &  X,
                                 blas::matrix< value_t > &        Y,
                                 const matop_t                    op ) const
{
    HLR_ASSERT( X.nrows() == this->ncols( op ) );
    HLR_ASSERT( Y.nrows() == this->nrows( op ) );
    
    auto  dU = U();
    auto  dV = V();
    
    switch ( op )
    {
        case apply_normal :
        {
            // Y = Y + U·(V'·X)
            auto  T = blas::prod( blas::adjoint( dV ), X );

            blas::prod( alpha, dU, T, value_t(1), Y );
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
                // Y = Y + U·(V'·X)
                auto  T = blas::prod( blas::adjoint( dV ), X );
                
                blas::prod( alpha, dU, T, value_t(1), Y );
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
                // Y = Y + V·(U'·X)
                auto  T = blas::prod( blas::adjoint( dU ), X );
                
                blas::prod( alpha, dV, T, value_t(1), Y );
            }// else
        }
        break;
        
        case apply_adjoint :
        {
            // Y = Y + V·(U'·X)
            auto  T = blas::prod( blas::adjoint( dU ), X );

            blas::prod( alpha, dV, T, value_t(1), Y );
        }
        break;
    }// switch
}

//
// compression
//

// compress internal data
// - may result in non-compression if storage does not decrease
template < typename value_t >
void
lrmatrix< value_t >::compress ( const accuracy &  acc )
{
    if ( this->nrows() * this->ncols() == 0 )
        return;

    if ( is_compressed() )
        return;
                 
    //
    // compress
    //
    
    auto        oU     = _U;
    auto        oV     = _V;
    const auto  zconfU = compress::get_config( acc( this->row_is(), this->col_is() ), oU );
    const auto  zconfV = compress::get_config( acc( this->row_is(), this->col_is() ), oV );
    auto        zU     = compress::compress< value_t >( zconfU, oU );
    auto        zV     = compress::compress< value_t >( zconfV, oV );

    // // DEBUG
    // {
    //     {
    //         auto  dU = blas::matrix< value_t >( oU.nrows(), oU.ncols() );

    //         compress::decompress( zU, dU );

    //         // io::matlab::write( oU, "U1" );
    //         // io::matlab::write( dU, "U2" );

    //         auto  M1 = oU;
    //         auto  M2 = dU;
            
    //         blas::add( value_t(-1), M1, M2 );

    //         if ( acc.norm_mode() == Hpro::spectral_norm )
    //             std::cout << "U" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_2( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //         else
    //             std::cout << "U" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //     }
    
    //     {
    //         auto  dV = blas::matrix< value_t >( oV.nrows(), oV.ncols() );

    //         compress::decompress( zV, dV );

    //         // io::matlab::write( oV, "V1" );
    //         // io::matlab::write( dV, "V2" );

    //         auto  M1 = oV;
    //         auto  M2 = dV;
            
    //         blas::add( value_t(-1), M1, M2 );

    //         if ( acc.norm_mode() == Hpro::spectral_norm )
    //             std::cout << "V" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_2( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //         else
    //             std::cout << "V" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //     }
    
    //     {
    //         auto  M1 = blas::prod( oU, blas::adjoint( oV ) );
    //         auto  dU = blas::matrix< value_t >( oU.nrows(), oU.ncols() );
    //         auto  dV = blas::matrix< value_t >( oV.nrows(), oV.ncols() );

    //         compress::decompress( zU, dU );
    //         compress::decompress( zV, dV );

    //         auto  M2 = blas::prod( dU, blas::adjoint( dV ) );
        
    //         blas::add( value_t(-1), M1, M2 );
        
    //         if ( acc.norm_mode() == Hpro::spectral_norm )
    //             std::cout << "R" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_2( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_2( M2 ) / blas::norm_2(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //         else
    //             std::cout << "R" // << this->block_is().to_string()
    //                       << " / nrm = " << boost::format( "%.4e" ) % ( blas::norm_F( M1 ) )
    //                       << " / abs = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) )
    //                       << " / rel = " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F(M1) )
    //                       << " / max val = " << boost::format( "%.4e" ) % blas::max_abs_val( M2 )
    //                       << " / max rat = " << boost::format( "%.4e" ) % ( blas::max_abs_val( M2 ) / blas::max_abs_val( M1 ) )
    //                       << std::endl;
    //     }
    // }
    // // DEBUG
    
    const size_t  mem_lr = sizeof(value_t) * oU.ncols() * ( oU.nrows() + oV.nrows() );
    const auto    zmem_U = compress::compressed_size( zU );
    const auto    zmem_V = compress::compressed_size( zV );

    // std::cout << this->block_is().to_string() << " / " << this->nrows() << " / " << this->ncols() << " : " << double(mem_lr) / double(zmem_U + zmem_V) << std::endl;

    // if ( double(mem_lr) / double(zmem_U + zmem_V) < 1.1 )
    // {
    //     std::cout << this->block_is().to_string() << " / " << this->nrows() << " / " << this->ncols() << std::endl;
    // }// if
    
    if (( zmem_U != 0 ) && ( zmem_V != 0 ) && ( zmem_U + zmem_V < mem_lr ))
    {
        _zU = std::move( zU );
        _zV = std::move( zV );
        _U  = std::move( blas::matrix< value_t >( 0, 0 ) );
        _V  = std::move( blas::matrix< value_t >( 0, 0 ) );
    }// if
}

// decompress internal data
template < typename value_t >
void
lrmatrix< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    _U = std::move( U() );
    _V = std::move( V() );

    remove_compressed();
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRMATRIX_HH
