#ifndef __HLR_TENSOR_TUCKER_TENSOR_HH
#define __HLR_TENSOR_TUCKER_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/tucker_tensor
// Description : tensor using tucker decomposition
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/tensor/base_tensor.hh>

namespace hlr { namespace tensor {

//
// implements tensor using Tucker decomposition
//
template < typename T_value >
class tucker_tensor3 : public base_tensor3< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  super_t = base_tensor3< value_t >;

    static constexpr uint  dimension = 3;

private:
    // core tensor
    blas::tensor3< value_t >  _G;

    // per mode matrices
    blas::matrix< value_t >   _X0, _X1, _X2;
    
public:
    //
    // ctors
    //

    tucker_tensor3 ()
    {}

    tucker_tensor3 ( const tucker_tensor3 &  t )
            : super_t( t )
            , _G( t._G )
            , _X0( t._X0 )
            , _X1( t._X1 )
            , _X2( t._X2 )
    {}

    tucker_tensor3 ( tucker_tensor3 &&  t )
            : super_t( std::forward< base_tensor3< value_t > >( t ) )
            , _G(  std::move( t._G ) )
            , _X0( std::move( t._X0 ) )
            , _X1( std::move( t._X1 ) )
            , _X2( std::move( t._X2 ) )
    {}

    tucker_tensor3 ( const indexset &  is0,
                     const indexset &  is1,
                     const indexset &  is2,
                     blas::tensor3< value_t > &&  aG,
                     blas::matrix< value_t > &&   aX0,
                     blas::matrix< value_t > &&   aX1,
                     blas::matrix< value_t > &&   aX2 )
            : super_t( is0, is1, is2 )
            , _G(  std::move( aG ) )
            , _X0( std::move( aX0 ) )
            , _X1( std::move( aX1 ) )
            , _X2( std::move( aX2 ) )
    {}
    
    // dtor
    virtual ~tucker_tensor3 ()
    {}

    // assignment
    tucker_tensor3 &  operator = ( const tucker_tensor3 &  t )
    {
        super_t::operator = ( t );

        _G  = blas::copy( t._G );
        _X0 = blas::copy( t._X0 );
        _X1 = blas::copy( t._X1 );
        _X2 = blas::copy( t._X2 );
    }

    tucker_tensor3 &  operator = ( tucker_tensor3 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _G  = std::move( t._G );
        _X0 = std::move( t._X0 );
        _X1 = std::move( t._X1 );
        _X2 = std::move( t._X2 );
    }

    //
    // access internal data
    //

    blas::tensor3< value_t > &  G ()               { return _G; }
    blas::matrix< value_t > &   X ( const uint d )
    {
        switch ( d )
        {
            case 0  : return _X0;
            case 1  : return _X1;
            case 2  : return _X2;
            default : HLR_ERROR( "wrong mode" );
        }
    }
    
    const blas::tensor3< value_t > &  G () const               { return _G; }
    const blas::matrix< value_t > &   X ( const uint d ) const
    {
        switch ( d )
        {
            case 0  : return _X0;
            case 1  : return _X1;
            case 2  : return _X2;
            default : HLR_ERROR( "wrong mode" );
        }
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
        X->_X0 = blas::copy( _X0 );
        X->_X1 = blas::copy( _X1 );
        X->_X2 = blas::copy( _X2 );
        
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
        return super_t::byte_size() + _G.byte_size() + _X0.byte_size() + _X1.byte_size() + _X2.byte_size();
    }
};

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

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_TUCKER_TENSOR3_HH
