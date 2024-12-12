#ifndef __HLR_BLAS_TENSOR3_HH
#define __HLR_BLAS_TENSOR3_HH
//
// Project     : HLR
// Module      : blas/tensor
// Description : implements dense tensor class
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <type_traits>

#include <hpro/blas/MemBlock.hh>

#include <hlr/arith/blas.hh>

namespace hlr { namespace blas {

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
    size_t       size         ( const uint  d ) const { return derived().size( d ); }

    // return range per dimension
    blas::range  range        ( const uint  d ) const { return blas::range( 0, idx_t(size(d))-1 ); }

    // return stride of data per dimension
    size_t       stride       ( const uint  d ) const { return derived().stride(d); }

    // return coefficient (i,j,l)
    value_t      operator ()  ( const idx_t i, const idx_t j, const idx_t l ) const
    {
        return derived()(i,j,l);
    }

    // return reference to coefficient (i,j,l)
    value_t &    operator ()  ( const idx_t i, const idx_t j, const idx_t l )
    {
        return derived()(i,j,l);
    }

    // return pointer to internal data
    value_t *    data         () const { return derived().data(); }

private:
    // convert to derived type
    T_derived &        derived  ()       { return * static_cast<       T_derived * >( this ); }
    const T_derived &  derived  () const { return * static_cast< const T_derived * >( this ); }
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
    size_t  _length[3];
    
    // strides of data in memory block (rows, columns and page)
    size_t  _stride[3];
    
public:
    //
    // constructor and destructor
    //
    
    // creates zero sized tensor
    tensor3 ()
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
                _stride[2] = _length[0]*_length[1];
                super_t::alloc_wo_value( _length[0] * _length[1] * _length[2] );

                for ( idx_t l = 0; l < idx_t( _length[2] ); l++ )
                    for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                        for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                            (*this)(i,j,l) = t(i,j,l);
                
                break;
        }// switch
    }

    // move constructor
    tensor3 ( tensor3 &&  t )
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

                for ( idx_t l = 0; l < idx_t( _length[2] ); l++ )
                    for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                        for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                            (*this)(i,j,l) = t( r0.first() + i * idx_t( r0.stride() ),
                                                r1.first() + j * idx_t( r1.stride() ),
                                                r2.first() + l * idx_t( r2.stride() ) );
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
    tensor3 & operator = ( tensor3 &&  t )
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
    size_t       size         ( const uint  d ) const { return _length[d]; }

    // return coefficient (i,j,p)
    value_t      operator ()  ( const idx_t i,
                                const idx_t j,
                                const idx_t l ) const
    {
        HLR_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) && l < idx_t(_length[2]) );
        return super_t::_data[ l * _stride[2] + j * _stride[1] + i * _stride[0] ];
    }

    // return reference to coefficient (i,j)
    value_t &    operator ()  ( const idx_t  i,
                                const idx_t  j,
                                const idx_t  l )
    {
        HLR_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) && l < idx_t(_length[2]) );
        return super_t::_data[ l * _stride[2] + j * _stride[1] + i * _stride[0] ];
    }

    // return pointer to internal data
    value_t *    data         () const { return super_t::_data; }

    // return data stride per dimension
    size_t       stride       ( const uint  d ) const { return _stride[d]; }

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

    //
    // sub-tensors
    //

    // return slice by fixing i'th mode <mode>
    matrix< value_t >
    slice ( const uint    mode,
            const size_t  i ) const
    {
        // switch ( mode )
        // {
        //     // fixed: 0, I₁ × I₂
        //     case  0 : return matrix< value_t >( data() + i * stride(0), // pointer to first element
        //                                         size( 1 ),              // nrows
        //                                         stride(0),              // row stride
        //                                         size( 2 ),              // ncols
        //                                         stride(0) * stride(1)   // col stride );

        //     // fixed: 1, I₀ × I₂
        //     case  1 : return matrix< value_t >( data() + i * stride(0), // pointer to first element
        //                                         size( 0 ),              // nrows
        //                                         stride(0),              // row stride
        //                                         size( 2 ),              // ncols
        //                                         stride(0) * stride(1)   // col stride );
                                                
        //     // fixed: 2, I₀ × I₁
        //     case  2 : return matrix< value_t >( data() + i * stride(0), // pointer to first element
        //                                         size( 0 ),              // nrows
        //                                         stride(0),              // row stride
        //                                         size( 1 ),              // ncols
        //                                         stride(0) * stride(1)   // col stride );
                                                
        //     default : HLR_ERROR( "wrong mode" );
        // }// switch

        if      ( mode == 0 ) return matrix< value_t >( data() + i,                     size(1), size(0), size(2), size(0)*size(1) );
        else if ( mode == 1 ) return matrix< value_t >( data() + i * size(0),           size(0),       1, size(2), size(0)*size(1) );
        else if ( mode == 2 ) return matrix< value_t >( data() + i * size(0) * size(1), size(0),       1, size(1), size(0)         );
        else
            HLR_ERROR( "wrong mode" );
    }
                          
    // return (i,j)'th mode-d fiber
    vector< value_t >
    fiber ( const uint    mode,
            const size_t  i,
            const size_t  j ) const
    {
        idx_t  i0 = 0;
        idx_t  i1 = 0;
        idx_t  i2 = 0;
        
        switch ( mode )
        {
            case  0 : {         i1 = i; i2 = j; } break;
            case  1 : { i0 = i;         i2 = j; } break;
            case  2 : { i0 = i; i1 = j;         } break;
            default : HLR_ERROR( "wrong mode" );
        }// switch

        return vector< value_t >( data()
                                  + ( i2 * _stride[2] )
                                  + ( i1 * _stride[1] )
                                  + ( i0 * _stride[0] ),
                                  _length[mode], _stride[mode] );

        // if      ( mode == 0 ) return vector< value_t >( data() + j * size(0) * size(1) + i * size(0), size(0), 1 );               // i = column, j = page
        // else if ( mode == 1 ) return vector< value_t >( data() + j * size(0) * size(1) + i,           size(1), size(0) );         // i = row,    j = page
        // else if ( mode == 2 ) return vector< value_t >( data() + j * size(0) + i,                     size(2), size(0)*size(1) ); // i = row,    j = column
        // else
        //     HLR_ERROR( "wrong mode" );
    }
                          
    // unfolding
    matrix< value_t > unfold ( const uint  mode ) const;
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof( value_t ) * _length[0] * _length[1] * _length[2] + sizeof(_length) + sizeof(_stride) + sizeof(super_t);
    }

    // return size of (floating point) data in bytes handled by this object
    size_t  data_byte_size () const
    {
        return sizeof( value_t ) * _length[0] * _length[1] * _length[2];
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

//
// return real copy of given tensor
//
template < typename value_t >
tensor3< value_t >
copy ( const tensor3< value_t > &  t )
{
    return t.copy();
}

//
// print tensor to (standard) output
//
template < typename value_t >
void
print ( const tensor3< value_t > &  t,
        std::ostream &              out = std::cout )
{
    // back to front
    for ( int  i2 = t.size(2)-1; i2 >= 0; --i2 )
    {
        // top to bottom
        for ( uint  i1 = 0; i1 < t.size(1); ++i1 )
        {
            // offset for 3D effect
            for ( uint  o = 0; o < i2; ++o )
                out << "   ";
                    
            // print single row
            for ( uint  i0 = 0; i0 < t.size(0); ++i0 )
                out << t( i0, i1, i2 ) << ", ";

            out << std::endl;
        }// for

        out << std::endl;
    }// for
}

////////////////////////////////////////////////////////////////
//
// BLAS functions
//

template < typename value_t >
void
copy ( const tensor3< value_t > &  src,
       tensor3< value_t > &        dest );
using Hpro::BLAS::copy;

template < typename value_t >
value_t
dot ( const tensor3< value_t > &  t1,
      const tensor3< value_t > &  t2 );
using Hpro::BLAS::dot;

template < typename value_t >
real_type_t< value_t >
norm_F ( const tensor3< value_t > &  t );
using Hpro::BLAS::norm_F;

template < typename value_t >
real_type_t< value_t >
max_abs_val ( const tensor3< value_t > &  t );

//
// compute B := α A + β B (element wise)
//
template < typename alpha_t,
           typename value_t >
requires ( std::convertible_to< alpha_t, value_t > )
void
add ( const alpha_t               alpha,
      const tensor3< value_t > &  A,
      tensor3< value_t > &        B );

template < typename alpha_t,
           typename beta_t,
           typename value_t >
requires ( std::convertible_to< alpha_t, value_t > &&
           std::convertible_to< beta_t, value_t > )
void
add ( const alpha_t               alpha,
      const tensor3< value_t > &  A,
      const beta_t                beta,
      tensor3< value_t > &        B );
using Hpro::BLAS::add;

//
// compute d-mode tensor product X×M
//
template < typename     value_t,
           matrix_type  matrix_t >
tensor3< value_t >
tensor_product ( const tensor3< value_t > &  X,
                 const matrix_t &            M,
                 const uint                  mode );

//
// element-wise multiplication X2 := X1 * X2
//
template < typename value_t >
tensor3< value_t >
hadamard_product ( const tensor3< value_t > &  X1,
                   tensor3< value_t > &        X2 );

}}// namespace hlr::blas

#include <hlr/arith/detail/tensor3.hh>

#endif  // __HPRO_BLAS_TENSOR3_HH
