#ifndef __HLR_TENSOR_DENSE_TENSOR_HH
#define __HLR_TENSOR_DENSE_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/dense_tensor
// Description : dense (full) tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/traits.hh>

#include <hlr/tensor/base_tensor.hh>

namespace hlr { namespace tensor {

//
// implements dense (full) 3D tensor
// - storage layout is column-major
//
template < typename T_value >
class dense_tensor3 : public base_tensor3< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  super_t = base_tensor3< value_t >;

    static constexpr uint  dimension = 3;

private:
    // tensor data
    blas::tensor3< value_t >   _tensor;

public:
    //
    // ctors
    //

    dense_tensor3 ()
    {}

    dense_tensor3 ( const dense_tensor3 &  t )
            : super_t( t )
            , _tensor( t._tensor.size() )
    {
        blas::copy( t._tensor, _tensor );
    }

    dense_tensor3 ( dense_tensor3 &&  t )
            : super_t( std::forward< base_tensor3< value_t > >( t ) )
    {
        std::swap( _tensor, t._tensor );
    }

    dense_tensor3 ( const indexset &  is0,
                    const indexset &  is1,
                    const indexset &  is2 )
            : super_t( is0, is1, is2 )
            , _tensor( is0.size(), is1.size(), is2.size() )
    {}
    
    dense_tensor3 ( const std::array< indexset, 3 > &  ais )
            : super_t{ ais[0], ais[1], ais[2] }
            , _tensor( ais[0].size(), ais[1].size(), ais[2].size() )
    {}
    
    dense_tensor3 ( const indexset &             is0,
                    const indexset &             is1,
                    const indexset &             is2,
                    blas::tensor3< value_t > &&  t )
            : super_t( is0, is1, is2 )
            , _tensor( std::move( t ) )
    {}
    
    // dtor
    virtual ~dense_tensor3 ()
    {}

    // assignment
    dense_tensor3 &  operator = ( const dense_tensor3 &  t )
    {
        super_t::operator = ( t );

        _tensor = blas::copy( t._tensor );
    }

    dense_tensor3 &  operator = ( dense_tensor3 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _tensor = std::move( t._tensor );
    }

    //
    // access internal data
    //

    size_t  dim  ( const uint  d ) const { HLR_DBG_ASSERT( d < dimension ); return _tensor.size(d); }

    blas::tensor3< value_t > &        tensor ()       { return _tensor; }
    const blas::tensor3< value_t > &  tensor () const { return _tensor; }

    value_t          coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const { return this->_tensor(i,j,l); }
    value_t &        coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l )       { return this->_tensor(i,j,l); }
    
    value_t          operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const { return coeff( i, j, l ); }
    value_t &        operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l )       { return coeff( i, j, l ); }
    
    //
    // misc
    //
    
    // return copy of local object
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    copy () const
    {
        auto  T = super_t::copy();
        auto  X = ptrcast( T.get(), dense_tensor3< value_t > );

        X->_tensor = blas::copy( _tensor );
        
        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    create () const
    {
        return std::make_unique< dense_tensor3< value_t > >();
    }
    
    // return size in bytes used by this object
    virtual size_t  byte_size () const
    {
        return super_t::byte_size() + _tensor.byte_size();
    }
};

//
// type tests
//
bool
is_dense ( const with_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const dense_tensor3< value_t > * >( &t ) != nullptr;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_DENSE_TENSOR3_HH
