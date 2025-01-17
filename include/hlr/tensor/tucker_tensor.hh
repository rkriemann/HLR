#ifndef __HLR_TENSOR_TUCKER_TENSOR_HH
#define __HLR_TENSOR_TUCKER_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/tucker_tensor
// Description : tensor using tucker decomposition
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <array>

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/compress/direct.hh>

#include <hlr/tensor/base_tensor.hh>

namespace hlr { namespace tensor {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// implements tensor using Tucker decomposition (d=3)
//
template < typename T_value >
class tucker_tensor3 : public base_tensor3< T_value >
{
public:
    static constexpr uint  dimension = 3;

    using  value_t     = T_value;
    using  real_t      = Hpro::real_type_t< value_t >;
    using  super_t     = base_tensor3< value_t >;
    using  mat_array_t = std::array< blas::matrix< value_t >, dimension >;

private:
    // core tensor
    blas::tensor3< value_t >         _G;

    // per mode matrices
    mat_array_t                      _X;

    // ranks of G (and X)
    std::array< size_t, dimension >  _rank;

    // compressed data
    compress::zarray                           _zG;
    std::array< compress::zarray, dimension >  _zX;

public:
    //
    // ctors
    //

    tucker_tensor3 ()
    {}

    tucker_tensor3 ( const tucker_tensor3 &  t )
            : super_t( t )
            , _G( t._G )
            , _X( t._X )
            , _rank{ 0, 0, 0 }
    {}

    tucker_tensor3 ( tucker_tensor3 &&  t )
            : super_t( std::forward< base_tensor3< value_t > >( t ) )
            , _G( std::move( t._G ) )
            , _X( std::move( t._X ) )
            , _rank( std::move( t._rank ) )
    {}

    tucker_tensor3 ( const indexset &  is0,
                     const indexset &  is1,
                     const indexset &  is2,
                     blas::tensor3< value_t > &&  aG,
                     blas::matrix< value_t > &&   aX0,
                     blas::matrix< value_t > &&   aX1,
                     blas::matrix< value_t > &&   aX2 )
            : super_t( is0, is1, is2 )
            , _G( std::move( aG ) )
            , _X{ std::move( aX0 ), std::move( aX1 ), std::move( aX2 ) }
            , _rank{ _G.size(0), _G.size(1), _G.size(2) }
    {
        HLR_ASSERT( ( is0.size() == _X[0].nrows() ) &&
                    ( is1.size() == _X[1].nrows() ) &&
                    ( is2.size() == _X[2].nrows() ) &&
                    ( _G.size(0) == _X[0].ncols() ) &&
                    ( _G.size(1) == _X[1].ncols() ) &&
                    ( _G.size(2) == _X[2].ncols() ) );
    }
    
    // dtor
    virtual ~tucker_tensor3 ()
    {}

    // assignment
    tucker_tensor3 &  operator = ( const tucker_tensor3 &  t )
    {
        super_t::operator = ( t );

        _G  = std::move( blas::copy( t._G ) );

        for ( uint  i = 0; i < dimension; ++i )
            _X[i] = std::move( blas::copy( t._X[i] ) );

        _rank = t._rank;
    }

    tucker_tensor3 &  operator = ( tucker_tensor3 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _G    = std::move( t._G );
        _X    = std::move( t._X );
        _rank = std::move( t._rank );
    }

    //
    // access Tucker data
    //

    uint  rank ( const uint  d ) const { HLR_ASSERT( d < dimension ); return _rank[ d ]; }

    // TODO: replace by set_* functions to update rank data
    // blas::tensor3< value_t > &        G ()                     { return _G; }
    // blas::matrix< value_t > &         X ( const uint d )       { HLR_ASSERT(( d < dimension ) && ! is_compressed()); return _X[d]; }
    
    // const blas::tensor3< value_t > &  G () const               { return _G; }
    // const blas::matrix< value_t > &   X ( const uint d ) const { HLR_ASSERT(( d < dimension ) && ! is_compressed()); return _X[d]; }
    
    blas::tensor3< value_t >
    G () const
    {
        if ( is_compressed() )
        {
            auto  dG = blas::tensor3< value_t >( _rank[0], _rank[1], _rank[2] );
    
            compress::decompress< value_t >( _zG, dG );
            
            return dG;
        }// if
            
        return _G;
    }

    blas::matrix< value_t >
    X ( const uint  d ) const
    {
        HLR_ASSERT( d < dimension );
                   
        if ( is_compressed() )
        {
            auto  dX = blas::matrix< value_t >( this->dim(d), _rank[d] );

            compress::decompress< value_t >( _zX[d], dX );
            
            return dX;
        }// if
            
        return _X[d];
    }

    //
    // compression
    //

    // same but compress based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _zG.empty();
    }

    //
    // misc
    //
    
    // return copy of local object
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    copy () const
    {
        auto  T = super_t::copy();
        auto  X = ptrcast( T.get(), tucker_tensor3< value_t > );

        X->_G  = blas::copy( _G );

        for ( uint  i = 0; i < dimension; ++i )
            X->_X[i] = blas::copy( _X[i] );
        
        if ( is_compressed() )
        {
            X->_zG = compress::zarray( _zG.size() );
            std::copy( _zG.begin(), _zG.end(), X->_zG.begin() );
        
            for ( uint  i = 0; i < dimension; ++i )
            {
                X->_zX[i] = compress::zarray( _zX[i].size() );
                std::copy( _zX[i].begin(), _zX[i].end(), X->_zX[i].begin() );
            }// if
        }// if

        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    create () const
    {
        return std::make_unique< tucker_tensor3< value_t > >();
    }
    
    // return size in bytes used by this object
    virtual size_t  byte_size () const
    {
        size_t  s = super_t::byte_size() + _G.byte_size() + sizeof(_X);
        
        for ( uint  i = 0; i < dimension; ++i )
            s += _X[i].byte_size();

        s += sizeof( _rank );
        s += hlr::compress::byte_size( _zG );

        for ( uint  i = 0; i < dimension; ++i )
            s += hlr::compress::byte_size( _zX[i] );

        return s;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        size_t  s = 0;
        
        if ( is_compressed() )
        {
            s += hlr::compress::byte_size( _zG );
            
            for ( uint  i = 0; i < dimension; ++i )
                s += hlr::compress::byte_size( _zX[i] );
        }// if
        else
        {
            s += _G.data_byte_size();

            for ( uint  i = 0; i < dimension; ++i )
                s += _X[i].data_byte_size();
        }// else

        return  s;
    }

    // return name of type
    virtual std::string  typestr () const { return "tucker_tensor3"; }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void  remove_compressed ()
    {
        _zG = compress::zarray();
        
        for ( uint  i = 0; i < dimension; ++i )
            _zX[i] = compress::zarray();
    }
};

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
tucker_tensor3< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    if ( is_compressed() )
        return;

    auto  tol_G = acc.abs_eps();
    auto  tol_X = acc.abs_eps();

    if ( acc.abs_eps() != 0 )
    {
        if      ( acc.norm_mode() == Hpro::spectral_norm  ) tol_G = acc.abs_eps() / blas::norm_F( _G ); // TODO
        else if ( acc.norm_mode() == Hpro::frobenius_norm ) tol_G = acc.abs_eps() / blas::norm_F( _G );
        else
            HLR_ERROR( "unsupported norm mode" );
    }// if
    else if ( acc.rel_eps() != 0 )
    {
        tol_G = acc.rel_eps();
        tol_X = acc.rel_eps();
    }// if
    else
        HLR_ERROR( "zero error" );

    auto  zconf_G = compress::get_config( tol_G );
    auto  zconf_X = compress::get_config( tol_X );
    auto  zG      = compress::compress< value_t >( zconf_G, _G );
    auto  zX0     = compress::compress< value_t >( zconf_X, _X[0] );
    auto  zX1     = compress::compress< value_t >( zconf_X, _X[1] );
    auto  zX2     = compress::compress< value_t >( zconf_X, _X[2] );
    
    // // DEBUG
    // {
    //     auto  dG  = blas::tensor3< value_t >( _G.size(0), _G.size(1), _G.size(2) );
    //     auto  dX0 = blas::matrix< value_t >( _X[0].nrows(), _X[0].ncols() );
    //     auto  dX1 = blas::matrix< value_t >( _X[1].nrows(), _X[1].ncols() );
    //     auto  dX2 = blas::matrix< value_t >( _X[2].nrows(), _X[2].ncols() );

    //     compress::decompress( zG, dG );
    //     compress::decompress( zX[0], dX0 );
    //     compress::decompress( zX[1], dX1 );
    //     compress::decompress( zX[2], dX2 );

    //     Hpro::DBG::write( _X[0], "X0.mat", "X0" );
    //     Hpro::DBG::write( _X[1], "Y0.mat", "Y0" );
    //     Hpro::DBG::write( _X[2], "Z0.mat", "Z0" );
        
    //     Hpro::DBG::write( dX0, "X1.mat", "X1" );
    //     Hpro::DBG::write( dX1, "Y1.mat", "Y1" );
    //     Hpro::DBG::write( dX2, "Z1.mat", "Z1" );
        
    //     // io::hdf5::write( _G, "G1" );
    //     // io::hdf5::write( dG, "G2" );
    //     // io::hdf5::write( _X[0], "X1" );
    //     // io::hdf5::write( dX0,   "X2" );
    //     // io::hdf5::write( _X[1], "Y1" );
    //     // io::hdf5::write( dX1,   "Y2" );
    //     // io::hdf5::write( _X[2], "Z1" );
    //     // io::hdf5::write( dX2,   "Z2" );
        
    //     blas::add( -1, _G, dG );
    //     blas::add( -1, _X[0], dX0 );
    //     blas::add( -1, _X[1], dX1 );
    //     blas::add( -1, _X[2], dX2 );

    //     std::cout << "R "
    //               << this->is(0) << " × " 
    //               << this->is(1) << " × " 
    //               << this->is(2)
    //               << " : " 
    //               << blas::norm_F( dG ) / blas::norm_F( _G )
    //               << " / "
    //               << blas::norm_F( dX0 ) / blas::norm_F( _X[0] )
    //               << " / "
    //               << blas::norm_F( dX1 ) / blas::norm_F( _X[1] )
    //               << " / "
    //               << blas::norm_F( dX2 ) / blas::norm_F( _X[2] )
    //               << std::endl;
            
    //     // auto  T0 = tensor_product( _G, _X[0], 0 );
    //     // auto  T1 = tensor_product( T0, _X[1], 1 );
    //     // auto  M  = tensor_product( T1, _X[2], 2 );
        
    //     // auto  R0 = tensor_product( dG, dX0, 0 );
    //     // auto  R1 = tensor_product( R0, dX1, 1 );
    //     // auto  dM = tensor_product( R1, dX2, 2 );
        
    //     // for ( size_t  i = 0; i < M.size(0) * M.size(1) * M.size(2); ++i )
    //     // {
    //     //     const auto  error = std::abs( (M.data()[i] - dM.data()[i]) / M.data()[i] );

    //     //     std::cout << "D " << i << " : "
    //     //               << M.data()[i] << " / "
    //     //               << dM.data()[i] << " / "
    //     //               << std::abs( error )
    //     //               << std::endl;
    //     // }// for
    // }

    const size_t  mem  = _G.data_byte_size() + _X[0].data_byte_size() + _X[1].data_byte_size() + _X[2].data_byte_size();
    const size_t  zmem = compress::byte_size( zG ) + compress::byte_size( zX0 ) + compress::byte_size( zX1 ) + compress::byte_size( zX2 );

    if ( zmem < mem )
    {
        _zG    = std::move( zG );
        _zX[0] = std::move( zX0 );
        _zX[1] = std::move( zX1 );
        _zX[2] = std::move( zX2 );

        _G     = std::move( blas::tensor3< value_t >() );
        _X[0]  = std::move( blas::matrix< value_t >() );
        _X[1]  = std::move( blas::matrix< value_t >() );
        _X[2]  = std::move( blas::matrix< value_t >() );
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
tucker_tensor3< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    this->_G = std::move( G() );

    for ( uint  i = 0; i < dimension; ++i )
        this->_X[i] = std::move( X(i) );

    remove_compressed();
}

//
// type tests
//
bool
is_tucker_tensor3 ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor3< value_t > * >( &t ) != nullptr;
}

bool
is_tucker_tensor3 ( const has_value_type auto *  t )
{
    using type_t  = std::remove_cv_t< std::remove_pointer_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor3< value_t > * >( t ) != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// implements tensor using Tucker decomposition (d=4)
//
template < typename T_value >
class tucker_tensor4 : public base_tensor4< T_value >
{
public:
    static constexpr uint  dimension = 4;

    using  value_t     = T_value;
    using  real_t      = Hpro::real_type_t< value_t >;
    using  super_t     = base_tensor4< value_t >;
    using  mat_array_t = std::array< blas::matrix< value_t >, dimension >;

private:
    // core tensor
    blas::tensor4< value_t >         _G;

    // per mode matrices
    mat_array_t                      _X;

    // ranks of G (and X)
    std::array< size_t, dimension >  _rank;

    // compressed data
    compress::zarray                           _zG;
    std::array< compress::zarray, dimension >  _zX;

public:
    //
    // ctors
    //

    tucker_tensor4 ()
    {}

    tucker_tensor4 ( const tucker_tensor4 &  t )
            : super_t( t )
            , _G( t._G )
            , _X( t._X )
            , _rank{ 0, 0, 0, 0 }
    {}

    tucker_tensor4 ( tucker_tensor4 &&  t )
            : super_t( std::forward< base_tensor4< value_t > >( t ) )
            , _G( std::move( t._G ) )
            , _X( std::move( t._X ) )
            , _rank( std::move( t._rank ) )
    {}

    tucker_tensor4 ( const indexset &  is0,
                     const indexset &  is1,
                     const indexset &  is2,
                     const indexset &  is3,
                     blas::tensor4< value_t > &&  aG,
                     blas::matrix< value_t > &&   aX0,
                     blas::matrix< value_t > &&   aX1,
                     blas::matrix< value_t > &&   aX2,
                     blas::matrix< value_t > &&   aX3 )
            : super_t( is0, is1, is2, is3 )
            , _G( std::move( aG ) )
            , _X{ std::move( aX0 ), std::move( aX1 ), std::move( aX2 ), std::move( aX3 ) }
            , _rank{ _G.size(0), _G.size(1), _G.size(2), _G.size(3) }
    {
        HLR_ASSERT( ( is0.size() == _X[0].nrows() ) &&
                    ( is1.size() == _X[1].nrows() ) &&
                    ( is2.size() == _X[2].nrows() ) &&
                    ( is3.size() == _X[3].nrows() ) &&
                    ( _G.size(0) == _X[0].ncols() ) &&
                    ( _G.size(1) == _X[1].ncols() ) &&
                    ( _G.size(2) == _X[2].ncols() ) &&
                    ( _G.size(3) == _X[3].ncols() ) );
    }
    
    // dtor
    virtual ~tucker_tensor4 ()
    {}

    // assignment
    tucker_tensor4 &  operator = ( const tucker_tensor4 &  t )
    {
        super_t::operator = ( t );

        _G  = std::move( blas::copy( t._G ) );

        for ( uint  i = 0; i < dimension; ++i )
            _X[i] = std::move( blas::copy( t._X[i] ) );

        _rank = t._rank;
    }

    tucker_tensor4 &  operator = ( tucker_tensor4 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _G    = std::move( t._G );
        _X    = std::move( t._X );
        _rank = std::move( t._rank );
    }

    //
    // access Tucker data
    //

    uint  rank ( const uint  d ) const { HLR_ASSERT( d < dimension ); return _rank[ d ]; }

    blas::tensor4< value_t >
    G () const
    {
        if ( is_compressed() )
        {
            auto  dG = blas::tensor4< value_t >( _rank[0], _rank[1], _rank[2], _rank[3] );
    
            compress::decompress< value_t >( _zG, dG );
            
            return dG;
        }// if
            
        return _G;
    }

    blas::matrix< value_t >
    X ( const uint  d ) const
    {
        HLR_ASSERT( d < dimension );
                   
        if ( is_compressed() )
        {
            auto  dX = blas::matrix< value_t >( this->dim(d), _rank[d] );

            compress::decompress< value_t >( _zX[d], dX );
            
            return dX;
        }// if
            
        return _X[d];
    }

    //
    // compression
    //

    // same but compress based on given accuracy
    virtual void   compress      ( const accuracy &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _zG.empty();
    }

    //
    // misc
    //
    
    // return copy of local object
    virtual
    std::unique_ptr< base_tensor4< value_t > >
    copy () const
    {
        auto  T = super_t::copy();
        auto  X = ptrcast( T.get(), tucker_tensor4< value_t > );

        X->_G  = blas::copy( _G );

        for ( uint  i = 0; i < dimension; ++i )
            X->_X[i] = blas::copy( _X[i] );
        
        if ( is_compressed() )
        {
            X->_zG = compress::zarray( _zG.size() );
            std::copy( _zG.begin(), _zG.end(), X->_zG.begin() );
        
            for ( uint  i = 0; i < dimension; ++i )
            {
                X->_zX[i] = compress::zarray( _zX[i].size() );
                std::copy( _zX[i].begin(), _zX[i].end(), X->_zX[i].begin() );
            }// if
        }// if

        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor4< value_t > >
    create () const
    {
        return std::make_unique< tucker_tensor4< value_t > >();
    }
    
    // return size in bytes used by this object
    virtual size_t  byte_size () const
    {
        size_t  s = super_t::byte_size() + _G.byte_size() + sizeof(_X);
        
        for ( uint  i = 0; i < dimension; ++i )
            s += _X[i].byte_size();

        s += sizeof( _rank );
        s += hlr::compress::byte_size( _zG );

        for ( uint  i = 0; i < dimension; ++i )
            s += hlr::compress::byte_size( _zX[i] );

        return s;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        size_t  s = 0;
        
        if ( is_compressed() )
        {
            s += hlr::compress::byte_size( _zG );
            
            for ( uint  i = 0; i < dimension; ++i )
                s += hlr::compress::byte_size( _zX[i] );
        }// if
        else
        {
            s += _G.data_byte_size();

            for ( uint  i = 0; i < dimension; ++i )
                s += _X[i].data_byte_size();
        }// else

        return  s;
    }

    // return name of type
    virtual std::string  typestr () const { return "tucker_tensor4"; }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void  remove_compressed ()
    {
        _zG = compress::zarray();
        
        for ( uint  i = 0; i < dimension; ++i )
            _zX[i] = compress::zarray();
    }
};

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
tucker_tensor4< value_t >::compress ( const accuracy &  acc )
{
    if ( is_compressed() )
        return;

    auto  tol_G = acc.abs_eps();
    auto  tol_X = acc.abs_eps();

    if ( acc.abs_eps() != 0 )
    {
        if      ( acc.norm_mode() == Hpro::spectral_norm  ) tol_G = acc.abs_eps() / blas::norm_F( _G ); // TODO
        else if ( acc.norm_mode() == Hpro::frobenius_norm ) tol_G = acc.abs_eps() / blas::norm_F( _G );
        else
            HLR_ERROR( "unsupported norm mode" );
    }// if
    else if ( acc.rel_eps() != 0 )
    {
        tol_G = acc.rel_eps();
        tol_X = acc.rel_eps();
    }// if
    else
        HLR_ERROR( "zero error" );

    auto  zconf_G = compress::get_config( tol_G );
    auto  zconf_X = compress::get_config( tol_X );
    auto  zG      = compress::compress< value_t >( zconf_G, _G );
    auto  zX0     = compress::compress< value_t >( zconf_X, _X[0] );
    auto  zX1     = compress::compress< value_t >( zconf_X, _X[1] );
    auto  zX2     = compress::compress< value_t >( zconf_X, _X[2] );
    auto  zX3     = compress::compress< value_t >( zconf_X, _X[3] );
    
    const size_t  mem  = _G.data_byte_size() + _X[0].data_byte_size() + _X[1].data_byte_size() + _X[2].data_byte_size() + _X[3].data_byte_size();
    const size_t  zmem = ( compress::byte_size( zG )
                           + compress::byte_size( zX0 )
                           + compress::byte_size( zX1 )
                           + compress::byte_size( zX2 )
                           + compress::byte_size( zX3 ) );

    // std::cout << _G.data_byte_size()
    //           << " / " << _X[0].data_byte_size()
    //           << " / " << _X[1].data_byte_size()
    //           << " / " << _X[2].data_byte_size()
    //           << " / " << _X[3].data_byte_size() << std::endl;

    // std::cout << compress::byte_size( zG )
    //           << " / " << compress::byte_size( zX0 )
    //           << " / " << compress::byte_size( zX1 )
    //           << " / " << compress::byte_size( zX2 )
    //           << " / " << compress::byte_size( zX3 ) << std::endl;
    
    if ( zmem < mem )
    {
        _zG    = std::move( zG );
        _zX[0] = std::move( zX0 );
        _zX[1] = std::move( zX1 );
        _zX[2] = std::move( zX2 );
        _zX[3] = std::move( zX3 );

        _G     = std::move( blas::tensor4< value_t >() );
        _X[0]  = std::move( blas::matrix< value_t >() );
        _X[1]  = std::move( blas::matrix< value_t >() );
        _X[2]  = std::move( blas::matrix< value_t >() );
        _X[3]  = std::move( blas::matrix< value_t >() );
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
tucker_tensor4< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    this->_G = std::move( G() );

    for ( uint  i = 0; i < dimension; ++i )
        this->_X[i] = std::move( X(i) );

    remove_compressed();
}

//
// type tests
//
bool
is_tucker_tensor4 ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor4< value_t > * >( &t ) != nullptr;
}

bool
is_tucker_tensor4 ( const has_value_type auto *  t )
{
    using type_t  = std::remove_cv_t< std::remove_pointer_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor4< value_t > * >( t ) != nullptr;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_TUCKER_TENSOR_HH
