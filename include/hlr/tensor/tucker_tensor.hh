#ifndef __HLR_TENSOR_TUCKER_TENSOR_HH
#define __HLR_TENSOR_TUCKER_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/tucker_tensor
// Description : tensor using tucker decomposition
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <array>

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/compress/direct.hh>

#include <hlr/tensor/base_tensor.hh>

namespace hlr { namespace tensor {

//
// implements tensor using Tucker decomposition
//
template < typename T_value >
class tucker_tensor3 : public base_tensor3< T_value >, public compress::compressible
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
    
    #if HLR_HAS_COMPRESSION == 1
    // compressed data
    compress::zarray                           _zG;
    std::array< compress::zarray, dimension >  _zX;
    #endif

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

        _G  = blas::copy( t._G );

        for ( uint  i = 0; i < dimension; ++i )
            _X[i] = blas::copy( t._X[i] );

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
    blas::tensor3< value_t > &        G ()                     { return _G; }
    blas::matrix< value_t > &         X ( const uint d )       { HLR_ASSERT(( d < dimension ) && ! is_compressed()); return _X[d]; }
    
    const blas::tensor3< value_t > &  G () const               { return _G; }
    const blas::matrix< value_t > &   X ( const uint d ) const { HLR_ASSERT(( d < dimension ) && ! is_compressed()); return _X[d]; }
    
    blas::tensor3< value_t >
    G_decompressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  dG = blas::tensor3< value_t >( _rank[0], _rank[1], _rank[2] );
    
            compress::decompress< value_t >( _zG, dG );
            
            return dG;
        }// if

        #endif
            
        return _G;
    }

    blas::matrix< value_t >
    X_decompressed ( const uint  d ) const
    {
        HLR_ASSERT( d < dimension );
                   
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  dX = blas::matrix< value_t >( this->dim(d), _rank[d] );
    
            compress::decompress< value_t >( _zX[d], dX );
            
            return dX;
        }// if

        #endif
            
        return _X[d];
    }

    //
    // compression
    //

    // compress internal data based on given configuration
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // same but compress based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        return ! is_null( _zG.data() );
        #else
        return false;
        #endif
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
        
        #if HLR_HAS_COMPRESSION == 1

        X->_zG = compress::zarray( _zG.size() );
        std::copy( _zG.begin(), _zG.end(), X->_zG.begin() );
        
        for ( uint  i = 0; i < dimension; ++i )
        {
            X->_zX[i] = compress::zarray( _zX[i].size() );
            std::copy( _zX[i].begin(), _zX[i].end(), X->_zX[i].begin() );
        }// for
        
        #endif
        
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

        #if HLR_HAS_COMPRESSION == 1

        s += hlr::compress::byte_size( _zG );
        
        for ( uint  i = 0; i < dimension; ++i )
            s += hlr::compress::byte_size( _zX[i] );

        s += sizeof( _rank );
        
        #endif

        return s;
    }

    // return name of type
    virtual std::string  typestr () const { return "tucker_tensor3"; }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void  remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        _zG = compress::zarray();

        for ( uint  i = 0; i < dimension; ++i )
            _zX[i] = compress::zarray();
        #endif
    }
};

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
tucker_tensor3< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;

    const size_t  mem = sizeof(value_t) * ( _G.size(0) * _G.size(1) * _G.size(2) +
                                            _X[0].nrows() + _X[0].ncols() +
                                            _X[1].nrows() + _X[1].ncols() +
                                            _X[2].nrows() + _X[2].ncols() );
    auto          zG  = compress::compress< value_t >( zconfig, _G );
    auto          zX0 = compress::compress< value_t >( zconfig, _X[0] );
    auto          zX1 = compress::compress< value_t >( zconfig, _X[1] );
    auto          zX2 = compress::compress< value_t >( zconfig, _X[2] );
    
    // // DEBUG
    // {
    //     auto  dG  = blas::tensor3< value_t >( _G.size(0), _G.size(1), _G.size(2) );
    //     auto  dX0 = blas::matrix< value_t >( _X[0].nrows(), _X[0].ncols() );
    //     auto  dX1 = blas::matrix< value_t >( _X[1].nrows(), _X[1].ncols() );
    //     auto  dX2 = blas::matrix< value_t >( _X[2].nrows(), _X[2].ncols() );

    //     compress::decompress( zG, dG );
    //     compress::decompress( zX0, dX0 );
    //     compress::decompress( zX1, dX1 );
    //     compress::decompress( zX2, dX2 );

    //     // io::hdf5::write( T, "T1" );
    //     // io::hdf5::write( dT, "T2" );
        
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

    #endif
}

template < typename value_t >
void
tucker_tensor3< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    const auto  eps   = acc.rel_eps();
    // const auto  eps   = acc( this->is(0), this->is(1), this->is(2) ).rel_eps();
        
    compress( compress::get_config( eps ) );
}

//
// decompress internal data
//
template < typename value_t >
void
tucker_tensor3< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    this->_G = std::move( G_decompressed() );

    for ( uint  i = 0; i < dimension; ++i )
        this->_X[i] = std::move( X_decompressed(i) );

    remove_compressed();

    #endif
}

//
// type tests
//
bool
is_tucker ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor3< value_t > * >( &t ) != nullptr;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_TUCKER_TENSOR3_HH
