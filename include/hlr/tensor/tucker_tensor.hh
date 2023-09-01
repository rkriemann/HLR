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
#include <hlr/utils/compression.hh>

#include <hlr/tensor/base_tensor.hh>

#include <hlr/utils/io.hh>

namespace hlr { namespace tensor {

#define HLR_USE_APLR  0

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
    //
    // tensor data
    //
    
    // core tensor
    blas::tensor3< value_t >         _G;

    // per mode matrices
    mat_array_t                      _X;

    // ranks of G (and X)
    std::array< size_t, dimension >  _rank;

    //
    // compression related data
    //
    
    #if HLR_HAS_COMPRESSION == 1

    // compressed data for core tensor
    compress::zarray                                 _zG;

    #if HLR_USE_APLR == 1

    // compressed mode matrices
    std::array< compress::aplr::zarray, dimension >  _zX;

    // // singular values per mode (CHECK: extracted from G?)
    // std::array< blas::vector< real_t >, dimension >  _S;
    
    #else
    
    // compressed mode matrices
    std::array< compress::zarray, dimension >        _zX;
    
    #endif // HLR_USE_APLR
    #endif // HLR_HAS_COMPRESSION

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
    
    // tucker_tensor3 ( const indexset &             is0,
    //                  const indexset &             is1,
    //                  const indexset &             is2,
    //                  blas::tensor3< value_t > &&  aG,
    //                  blas::vector< value_t > &&   aS0,
    //                  blas::matrix< value_t > &&   aX0,
    //                  blas::vector< value_t > &&   aS1,
    //                  blas::matrix< value_t > &&   aX1,
    //                  blas::vector< value_t > &&   aS2,
    //                  blas::matrix< value_t > &&   aX2 )
    //         : super_t( is0, is1, is2 )
    //         , _G( std::move( aG ) )
    //         , _X{ std::move( aX0 ), std::move( aX1 ), std::move( aX2 ) }
    //         , _rank{ _G.size(0), _G.size(1), _G.size(2) }
    //         #if HLR_USE_APLR == 1
    //         , _S{ std::move( aS0 ), std::move( aS1 ), std::move( aS2 ) }
    //         #endif
    // {
    //     HLR_ASSERT( ( is0.size() == _X[0].nrows() ) &&
    //                 ( is1.size() == _X[1].nrows() ) &&
    //                 ( is2.size() == _X[2].nrows() ) &&
    //                 ( _G.size(0) == _X[0].ncols() ) &&
    //                 ( _G.size(1) == _X[1].ncols() ) &&
    //                 ( _G.size(2) == _X[2].ncols() ) );
    //     #if HLR_USE_APLR == 1
    //     HLR_ASSERT( ( _X[0].ncols() == _S[0].length() ) &&
    //                 ( _X[1].ncols() == _S[1].length() ) &&
    //                 ( _X[2].ncols() == _S[2].length() ) );
    //     #endif
    // }
    
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

            #if HLR_USE_APLR == 1
            compress::aplr::decompress_lr< value_t >( _zX[d], dX );
            #else
            compress::decompress< value_t >( _zX[d], dX );
            #endif
            
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
        
        if ( is_compressed() )
        {
            #if HLR_HAS_COMPRESSION == 1

            X->_zG = compress::zarray( _zG.size() );
            std::copy( _zG.begin(), _zG.end(), X->_zG.begin() );

            # if HLR_USE_APLR == 1
            
            for ( uint  i = 0; i < dimension; ++i )
            {
                X->_zX[i] = compress::aplr::zarray( _zX[i].size() );
                std::copy( _zX[i].begin(), _zX[i].end(), X->_zX[i].begin() );

                // X->_S[i] = std::move( blas::copy( _S[i] ) );
            }// for
            
            #else
            
            for ( uint  i = 0; i < dimension; ++i )
            {
                X->_zX[i] = compress::zarray( _zX[i].size() );
                std::copy( _zX[i].begin(), _zX[i].end(), X->_zX[i].begin() );
            }// for
        
            #endif // HLR_USE_APLR
            #endif // HLR_HAS_COMPRESSION
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
        
        #if HLR_HAS_COMPRESSION == 1

        s += hlr::compress::byte_size( _zG );

        #if HLR_USE_APLR == 1

        // s += sizeof(_S);  // TODO: check size compared to sizeof(blas::vector)
        
        for ( uint  i = 0; i < dimension; ++i )
        {
            s += hlr::compress::aplr::byte_size( _zX[i] );
            // s += _S[i].byte_size();
        }// for
        
        #else
        
        for ( uint  i = 0; i < dimension; ++i )
            s += hlr::compress::byte_size( _zX[i] );

        #endif // HLR_USE_APLR
        #endif // HLR_HAS_COMPRESSION

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

        #if HLR_USE_APLR == 1

        for ( uint  i = 0; i < dimension; ++i )
            _zX[i] = compress::aplr::zarray();
        
        #else
        
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

    const size_t  mem  = sizeof(value_t) * ( _G.size(0) * _G.size(1) * _G.size(2) +
                                             _X[0].nrows() + _X[0].ncols() +
                                             _X[1].nrows() + _X[1].ncols() +
                                             _X[2].nrows() + _X[2].ncols() );
    size_t        zmem = 0;
    
    #if HLR_USE_APLR == 1

    HLR_ERROR( "APLR only supported with given tolerance" );
    
    #else
    
    auto  zG = compress::compress< value_t >( zconfig, _G );

    zmem += compress::byte_size( zG );

    // compress the mode matrices via APLR
    auto  zX = std::array< compress::aplr::zarray, dimension >();
    
    for ( uint  i = 0; i < dimension; ++i )
    {
        zX[i] = std::move( compress::compress( zconfig, _X[i] ) );
        zmem += compress::byte_size( zX[i] );
    }// for
    
    // DEBUG
    {
        auto  dG  = blas::tensor3< value_t >( _G.size(0), _G.size(1), _G.size(2) );
        auto  dX0 = blas::matrix< value_t >( _X[0].nrows(), _X[0].ncols() );
        auto  dX1 = blas::matrix< value_t >( _X[1].nrows(), _X[1].ncols() );
        auto  dX2 = blas::matrix< value_t >( _X[2].nrows(), _X[2].ncols() );

        compress::decompress( zG, dG );
        compress::decompress( zX[0], dX0 );
        compress::decompress( zX[1], dX1 );
        compress::decompress( zX[2], dX2 );

        Hpro::DBG::write( _X[0], "X0.mat", "X0" );
        Hpro::DBG::write( _X[1], "Y0.mat", "Y0" );
        Hpro::DBG::write( _X[2], "Z0.mat", "Z0" );
        
        Hpro::DBG::write( dX0, "X1.mat", "X1" );
        Hpro::DBG::write( dX1, "Y1.mat", "Y1" );
        Hpro::DBG::write( dX2, "Z1.mat", "Z1" );
        
        // io::hdf5::write( _G, "G1" );
        // io::hdf5::write( dG, "G2" );
        // io::hdf5::write( _X[0], "X1" );
        // io::hdf5::write( dX0,   "X2" );
        // io::hdf5::write( _X[1], "Y1" );
        // io::hdf5::write( dX1,   "Y2" );
        // io::hdf5::write( _X[2], "Z1" );
        // io::hdf5::write( dX2,   "Z2" );
        
        blas::add( -1, _G, dG );
        blas::add( -1, _X[0], dX0 );
        blas::add( -1, _X[1], dX1 );
        blas::add( -1, _X[2], dX2 );

        std::cout << "R "
                  << this->is(0) << " × " 
                  << this->is(1) << " × " 
                  << this->is(2)
                  << " : " 
                  << blas::norm_F( dG ) / blas::norm_F( _G )
                  << " / "
                  << blas::norm_F( dX0 ) / blas::norm_F( _X[0] )
                  << " / "
                  << blas::norm_F( dX1 ) / blas::norm_F( _X[1] )
                  << " / "
                  << blas::norm_F( dX2 ) / blas::norm_F( _X[2] )
                  << std::endl;
            
        // auto  T0 = tensor_product( _G, _X[0], 0 );
        // auto  T1 = tensor_product( T0, _X[1], 1 );
        // auto  M  = tensor_product( T1, _X[2], 2 );
        
        // auto  R0 = tensor_product( dG, dX0, 0 );
        // auto  R1 = tensor_product( R0, dX1, 1 );
        // auto  dM = tensor_product( R1, dX2, 2 );
        
        // for ( size_t  i = 0; i < M.size(0) * M.size(1) * M.size(2); ++i )
        // {
        //     const auto  error = std::abs( (M.data()[i] - dM.data()[i]) / M.data()[i] );

        //     std::cout << "D " << i << " : "
        //               << M.data()[i] << " / "
        //               << dM.data()[i] << " / "
        //               << std::abs( error )
        //               << std::endl;
        // }// for
    }

    if ( zmem < mem )
    {
        _zG = std::move( zG );
        _G  = std::move( blas::tensor3< value_t >() );

        for ( uint i = 0; i < dimension; ++i )
        {
            _zX[i] = std::move( zX[i] );
            _X[i]  = std::move( blas::matrix< value_t >() );
        }// for
    }// if

    #endif // HLR_USE_APLR
    #endif // HLR_HAS_COMPRESSION
}

template < typename value_t >
void
tucker_tensor3< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( acc.is_fixed_prec() );

    // defaults to absolute error: δ = ε
    auto  lacc = acc; // ( this->is(0), this->is(1), this->is(2) );
    auto  tol  = lacc.abs_eps();

    #if HLR_USE_APLR == 1

    real_t  norm_G = 1;
    
    if ( lacc.rel_eps() != 0 )
    {
        // use relative error: δ = ε |M|
        norm_G = blas::norm_F( _G );
        tol    = lacc.rel_eps();
    }// if

    const size_t  mem = sizeof(value_t) * ( _G.size(0) * _G.size(1) * _G.size(2) +
                                            _X[0].nrows() + _X[0].ncols() +
                                            _X[1].nrows() + _X[1].ncols() +
                                            _X[2].nrows() + _X[2].ncols() );

    // compress G directly
    size_t  zmem = 0;

    // compress the mode matrices via APLR
    auto  zX = std::array< compress::aplr::zarray, dimension >();
    
    for ( uint  i = 0; i < dimension; ++i )
    {
        auto  Gi    = _G.unfold( i );
        auto  S_tol = blas::sv( Gi );

        for ( uint  l = 0; l < _rank[i]; ++l )
            S_tol(l) = ( tol * norm_G ) / S_tol(l);

        zX[i] = std::move( compress::aplr::compress_lr( _X[i], S_tol ) );
        zmem += compress::aplr::byte_size( zX[i] );
    }// for
    
    // compress G directly (without |G| applied)
    auto  zG = compress::compress< value_t >( compress::get_config( tol ), _G );

    //
    // debug
    //

    // DEBUG
    {
        auto  dG  = blas::tensor3< value_t >( _G.size(0), _G.size(1), _G.size(2) );
        auto  dX0 = blas::matrix< value_t >( _X[0].nrows(), _X[0].ncols() );
        auto  dX1 = blas::matrix< value_t >( _X[1].nrows(), _X[1].ncols() );
        auto  dX2 = blas::matrix< value_t >( _X[2].nrows(), _X[2].ncols() );

        compress::decompress( zG, dG );
        compress::aplr::decompress_lr( zX[0], dX0 );
        compress::aplr::decompress_lr( zX[1], dX1 );
        compress::aplr::decompress_lr( zX[2], dX2 );

        Hpro::DBG::write( _X[0], "X0.mat", "X0" );
        Hpro::DBG::write( _X[1], "Y0.mat", "Y0" );
        Hpro::DBG::write( _X[2], "Z0.mat", "Z0" );
        
        Hpro::DBG::write( dX0, "X1.mat", "X1" );
        Hpro::DBG::write( dX1, "Y1.mat", "Y1" );
        Hpro::DBG::write( dX2, "Z1.mat", "Z1" );
        
        blas::add( -1, _G, dG );
        blas::add( -1, _X[0], dX0 );
        blas::add( -1, _X[1], dX1 );
        blas::add( -1, _X[2], dX2 );

        std::cout << "R "
                  << this->is(0) << " × " 
                  << this->is(1) << " × " 
                  << this->is(2)
                  << " : " 
                  << blas::norm_F( dG ) / blas::norm_F( _G )
                  << " / "
                  << blas::norm_F( dX0 ) / blas::norm_F( _X[0] )
                  << " / "
                  << blas::norm_F( dX1 ) / blas::norm_F( _X[1] )
                  << " / "
                  << blas::norm_F( dX2 ) / blas::norm_F( _X[2] )
                  << std::endl;
            
        // auto  T0 = tensor_product( _G, _X[0], 0 );
        // auto  T1 = tensor_product( T0, _X[1], 1 );
        // auto  M  = tensor_product( T1, _X[2], 2 );
        
        // auto  R0 = tensor_product( dG, dX0, 0 );
        // auto  R1 = tensor_product( R0, dX1, 1 );
        // auto  dM = tensor_product( R1, dX2, 2 );
        
        // for ( size_t  i = 0; i < M.size(0) * M.size(1) * M.size(2); ++i )
        // {
        //     const auto  error = std::abs( (M.data()[i] - dM.data()[i]) / M.data()[i] );

        //     std::cout << "D " << i << " : "
        //               << M.data()[i] << " / "
        //               << dM.data()[i] << " / "
        //               << std::abs( error )
        //               << std::endl;
        // }// for
    }
    
    zmem += compress::byte_size( zG );

    if ( zmem < mem )
    {
        _zG = std::move( zG );
        _G  = std::move( blas::tensor3< value_t >() );

        for ( uint  i = 0; i < dimension; ++i )
        {
            _zX[i] = std::move( zX[i] );
            _X[i]  = std::move( blas::matrix< value_t >() );
        }// for
    }// if
    
    #else
    
    if ( lacc.rel_eps() != 0 )
        tol = lacc.rel_eps();

    compress( compress::get_config( tol ) );

    #endif // HLR_USE_APLR
    #endif // HLR_HAS_COMPRESSION
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
is_tucker ( const with_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const tucker_tensor3< value_t > * >( &t ) != nullptr;
}

// to not confuse other modules
#undef HLR_USE_APLR

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_TUCKER_TENSOR3_HH
