#ifndef __HLR_BLAS_TENSOR_HH
#define __HLR_BLAS_TENSOR_HH
//
// Project     : HLR
// Module      : blas/tensor
// Description : implements dense tensor class
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <type_traits>

#include <hpro/blas/MemBlock.hh>

namespace hlr { namespace blas {

using Hpro::copy_policy_t;
using Hpro::copy_reference;
using Hpro::copy_value;
using Hpro::real_type_t;

// trait for giving access to tensor properties
template < typename T_tensor > struct tensor_trait;

// signals, that T is of tensor type
template < typename T > struct is_tensor { static const bool  value = false; };

//
// basic tensor interface
//
template < typename T_derived >
class tensor3_base
{
public:
    // scalar value type of tensor
    using  value_t = typename tensor_trait< T_derived >::value_t;

public:
    //
    // data access
    //

    // return size per dimension
    size_t       size         ( const uint  d ) const noexcept { return derived().size( d ); }

    // return range per dimension
    blas::range  range        ( const uint  d ) const noexcept { return blas::range( 0, idx_t(size(d))-1 ); }

    // return stride of data per dimension
    size_t       stride       ( const uint  d ) const noexcept { return derived().stride(d); }

    // return coefficient (i,j,l)
    value_t      operator ()  ( const idx_t i, const idx_t j, const idx_t l ) const noexcept
    {
        return derived()(i,j,l);
    }

    // return reference to coefficient (i,j,l)
    value_t &    operator ()  ( const idx_t i, const idx_t j, const idx_t l ) noexcept
    {
        return derived()(i,j,l);
    }

    // return pointer to internal data
    value_t *    data         () const noexcept { return derived().data(); }

private:
    // convert to derived type
    T_derived &        derived  ()       noexcept { return * static_cast<       T_derived * >( this ); }
    const T_derived &  derived  () const noexcept { return * static_cast< const T_derived * >( this ); }
};

//
// signals, that T is of tensor type
//
template < typename T >
struct is_tensor< tensor3_base< T > >
{
    static const bool  value = is_tensor< T >::value;
};


//
// 3d tensor
// - storage: column major
//
template < typename T_value >
struct tensor3 : public tensor3_base< tensor3< T_value > >, public blas::MemBlock< T_value >
{
    // ensure only floating point types (or complex version)
    static_assert( std::is_floating_point< T_value >::value || Hpro::is_complex_type< T_value >::value,
                   "only floating point types supported" );
    
public:
    // internal value type
    using  value_t = T_value;

    // super class type
    using  super_t = blas::MemBlock< value_t >;
    
private:
    // dimensions of tensor
    size_t   _length[3];
    
    // strides of data in memory block (rows, columns and page)
    size_t   _stride[3];
    
public:
    //
    // constructor and destructor
    //
    
    // creates zero sized tensor
    tensor3 () noexcept
            : super_t()
            , _length{ 0, 0, 0 }
            , _stride{ 0, 0, 0 }
    {}

    // creates tensor of size \a anrows × \a ancols
    tensor3 ( const size_t  n0,
              const size_t  n1,
              const size_t  n2 )
            : super_t( n0 * n1 * n2 )
            , _length{ n0, n1, n2 }
            , _stride{ 1, n0, n0*n1 }
    {}

    // copy constructor
    tensor3 ( const tensor3 &      t,
              const copy_policy_t  p = copy_reference )
            : super_t()
            , _length{ 0, 0, 0 }
            , _stride{ 0, 0, 0 }
    {
        switch ( p )
        {
            case copy_reference :
                (*this) = t;
                break;

            case copy_value :
                _length[0] = t._length[0];
                _length[1] = t._length[1];
                _length[2] = t._length[2];
                _stride[0] = 1;
                _stride[1] = _length[0];
                _stride[1] = _length[0]*_length[1];
                super_t::alloc_wo_value( _length[0] * _length[1] * _length[2] );

                for ( idx_t p = 0; p < idx_t( _length[2] ); p++ )
                    for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                        for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                            (*this)(i,j,p) = t(i,j,p);
                
                break;
        }// switch
    }

    // move constructor
    tensor3 ( tensor3 &&  t ) noexcept
            : super_t( std::move( t ) )
            , _length{ 0, 0, 0 }
            , _stride{ 0, 0, 0 }
    {
        t._data = nullptr;
        
        std::swap( _length, t._length );
        std::swap( _stride, t._stride );
    }

    // creates tensor using part of t defined by r0 × r1 × r2
    // p defines whether data is copied or referenced
    tensor3 ( const tensor3 &      t,
              const blas::range &  ar0,
              const blas::range &  ar1,
              const blas::range &  ar2,
              const copy_policy_t  p = copy_reference )
            : super_t()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        const auto  r0( ar0 == blas::range::all ? t.range(0) : ar0 );
        const auto  r1( ar1 == blas::range::all ? t.range(1) : ar1 );
        const auto  r2( ar2 == blas::range::all ? t.range(2) : ar2 );
        
        _length[0] = r0.size() / r0.stride();
        _length[1] = r1.size() / r1.stride();
        _length[2] = r2.size() / r2.stride();

        switch ( p )
        {
            case copy_reference :
                _stride[0] = r0.stride() * t.stride(0);
                _stride[1] = r1.stride() * t.stride(1);
                _stride[2] = r2.stride() * t.stride(2);
            
                super_t::init( t.data() + r0.first() * t.stride(0) + r1.first() * t.stride(1) + r2.first() * t.stride(2) );
                break;

            case copy_value :
                super_t::alloc_wo_value( _length[0] * _length[1] * _length[2] );
                _stride[0] = 1;
                _stride[1] = _length[0];
                _stride[2] = _length[0]*_length[1];

                for ( idx_t p = 0; p < idx_t( _length[2] ); p++ )
                    for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                        for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                            (*this)(i,j,p) = t( r0.first() + i * idx_t( r0.stride() ),
                                                r1.first() + j * idx_t( r1.stride() ),
                                                r2.first() + p * idx_t( r2.stride() ) );
                break;
        }// switch
    }

    // copy operator for matrices (always copy reference! for real copy, use BLAS::copy)
    tensor3 &  operator = ( const tensor3 &  t )
    {
        super_t::init( t.data(), false );
        
        _length[0] = t._length[0];
        _length[1] = t._length[1];
        _length[2] = t._length[2];
        _stride[0] = t._stride[0];
        _stride[1] = t._stride[1];
        _stride[2] = t._stride[2];

        return *this;
    }

    // move operator
    tensor3 & operator = ( tensor3 &&  t ) noexcept
    {
        if ( this != & t ) // prohibit self-moving
        {
            super_t::init( t, t._is_owner );
            
            _length[0] = t._length[0];
            _length[1] = t._length[1];
            _length[2] = t._length[2];
            _stride[0] = t._stride[0];
            _stride[1] = t._stride[1];
            _stride[2] = t._stride[2];

            t._data      = nullptr;
            t._length[0] = 0;
            t._length[1] = 0;
            t._length[2] = 0;
            t._stride[0] = 0;
            t._stride[1] = 0;
            t._stride[2] = 0;
        }// if

        return *this;
    }
    
    //
    // data access
    //

    // return size per dimension
    size_t       size         ( const uint  d ) const noexcept { return _length[d]; }

    // return coefficient (i,j,p)
    value_t      operator ()  ( const idx_t i,
                                const idx_t j,
                                const idx_t l ) const noexcept
    {
        HLR_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) && l < idx_t(_length[2]) );
        return super_t::_data[ l * _stride[2] + j * _stride[1] + i * _stride[0] ];
    }

    // return reference to coefficient (i,j)
    value_t &    operator ()  ( const idx_t  i,
                                const idx_t  j,
                                const idx_t  l ) noexcept
    {
        HLR_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) && l < idx_t(_length[2]) );
        return super_t::_data[ l * _stride[2] + j * _stride[1] + i * _stride[0] ];
    }

    // return pointer to internal data
    value_t *    data         () const noexcept { return super_t::_data; }

    // return data stride per dimension
    size_t       stride       ( const uint  d ) const noexcept { return _stride[d]; }

    // optimised resize: only change if dimension really changes
    void         resize       ( const size_t  n0,
                                const size_t  n1,
                                const size_t  n2 )
    {
        if (( _length[0] != n0 ) ||
            ( _length[1] != n1 ) ||
            ( _length[2] != n2 ))
        {
            *this = std::move( tensor3( n0, n1, n2 ) );
        }// if
    }
    
    //
    // construction operators
    //

    // create real copy of tense
    tensor3< value_t >  copy () const
    {
        tensor3< value_t >  t( *this, copy_value );

        return t;
    }
    
    // create reference to this tensor
    tensor3< value_t >  reference () const
    {
        tensor3< value_t >  t( *this, copy_reference );

        return t;
    }
    
    // return tensor referencing sub tensor defined by \a r1 × \a r2
    tensor3< value_t >  operator () ( const blas::range & r0,
                                      const blas::range & r1,
                                      const blas::range & r2 ) const
    {
        return tensor3< value_t >( *this, r0, r1, r2 );
    }

    ////////////////////////////////////////////////////

    // d-mode unfolding
    matrix< value_t > unfold ( const uint  d ) const;
    
    // return i'th mod-d fiber
    vector< value_t >  fiber ( const uint    d,
                               const size_t  i ) const
    {
        return vector< value_t >( *this, r, j );
    }
                          
    // // return vector referencing part of row \a i defined by \a r
    // Vector< value_t >  operator () ( const idx_t  i, const range & r ) const
    // {
    //     return Vector< value_t >( *this, i, r );
    // }
                          
    // // return vector referencing column \a j
    // Vector< value_t >  column   ( const idx_t  j ) const
    // {
    //     return (*this)( range( 0, idx_t(nrows())-1 ), j );
    // }
    
    // // return vector referencing row \a i
    // Vector< value_t >  row      ( const idx_t  i ) const
    // {
    //     return (*this)( i, range( 0, idx_t(ncols())-1 ) );
    // }
    
    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof( value_t ) * _length[0] * _length[1] * _length[2] + sizeof(_length) + sizeof(_stride) + sizeof(super_t);
    }

    //
    // tests
    //

    // test data for invalid values, e.g. INF and NAN
    void  check_data  () const;
};

// trait for giving access to tensor properties
template < typename T > struct tensor_trait< tensor3< T > > { using  value_t = T; };

// signals, that T is of tensor type
template < typename T > struct is_tensor< tensor3< T > > { static const bool  value = true; };

// tensor type concept
template < typename T > concept tensor_type = is_tensor< T >::value;

//
// return real copy of given tensor
//
template < typename value_t >
tensor3< value_t >
copy ( const tensor3< value_t > &  t )
{
    return t.copy();
}

////////////////////////////////////////////////////////////////
//
// BLAS functions
//

//
// d-mode unfolding
//
template < typename value_t >
matrix< value_t >
tensor3_base< value_t >::unfold ( const uint  d ) const
{
    HLR_ASSERT( d < dimension );
    
    if ( d == 0 )
    {
        auto  M = matrix< value_t >( size(0), size(1) * size(2) );

        
    }// if
}

////////////////////////////////////////////////////////////////
//
// BLAS functions
//

template < typename value_t >
void
copy ( const tensor3< value_t > &  src,
       tensor3< value_t > &        dest )
{
    HLR_DBG_ASSERT( ( src.size(0) == dest.size(0) ) &&
                    ( src.size(1) == dest.size(1) ) &&
                    ( src.size(2) == dest.size(2) ) );
    
    std::copy( src.data(), src.data() + src.size(), dest.data() );
}

template < typename value_t >
real_type_t< value_t >
dot ( const tensor3< value_t > &  t1,
      const tensor3< value_t > &  t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) );

    using  real_t = real_type_t< value_t >;

    auto  d = real_t(0);
    auto  n = t1.size(0) * t1.size(1) * t1.size(2);

    for ( size_t  i = 0; i < n; ++i )
        d += t1.data()[i] * t2.data()[i];

    return d;
}

template < typename value_t >
real_type_t< value_t >
norm_F ( const tensor3< value_t > &  t )
{
    return std::sqrt( std::abs( dot( t, t ) ) );
}

}}// namespace hlr::blas

#endif  // __HPRO_BLAS_TENSOR_HH
