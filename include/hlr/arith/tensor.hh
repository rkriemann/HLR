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

#include <hlr/approx/traits.hh>
#include <hlr/approx/svd.hh>
#include <hlr/arith/blas.hh>

namespace hlr
{

using accuracy = Hpro::TTruncAcc;
using Hpro::idx_t;

namespace blas
{

using Hpro::copy_policy_t;
using Hpro::copy_reference;
using Hpro::copy_value;
using Hpro::real_type_t;

// trait for giving access to tensor properties
template < typename T_tensor > struct tensor_trait;

// signals, that T is of tensor type
template < typename T > struct is_tensor { static const bool  value = false; };

template < typename T > inline constexpr bool is_tensor_v = is_tensor< T >::value;

// tensor type concept
template < typename T > concept tensor_type = is_tensor_v< T >;

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

                for ( idx_t p = 0; p < idx_t( _length[2] ); p++ )
                    for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                        for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                            (*this)(i,j,p) = t(i,j,p);
                
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

    // return i'th mod-d fiber
    vector< value_t >  fiber ( const uint    mode,
                               const size_t  i,
                               const size_t  j ) const
    {
        if      ( mode == 0 ) return vector< value_t >( data() + j * size(0) * size(1) + i * size(0), size(0), 1 );               // i = column, j = page
        else if ( mode == 1 ) return vector< value_t >( data() + j * size(0) * size(1) + i,           size(1), size(0) );         // i = row,    j = page
        else if ( mode == 2 ) return vector< value_t >( data() + j * size(0) + i,                     size(2), size(0)*size(1) ); // i = row,    j = column
        else
            HLR_ERROR( "wrong mode" );
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
void
print ( const tensor_type auto &  t,
        std::ostream &            out = std::cout )
{
    for ( uint  l = 0; l < t.size(2); ++l )
    {
        for ( uint  i = 0; i < t.size(0); ++i )
        {
            for ( uint  j = 0; j < t.size(1); ++j )
                out << t( i, j, l ) << ", ";

            out << std::endl;
        }// for

        out << std::endl;
    }// for
}

std::ostream &
operator << ( std::ostream &            os,
              const tensor_type auto &  M )
{
    print( M, os );
    return os;
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
tensor3< value_t >::unfold ( const uint  d ) const
{
    HLR_ASSERT( d < 3 );
    
    if ( d == 0 )
    {
        auto    M   = matrix< value_t >( size(0), size(1) * size(2) );
        size_t  col = 0;

        for ( size_t  l = 0; l < size(2); ++l )
        {
            for ( size_t  j = 0; j < size(1); ++j )
            {
                auto  t_jl = fiber( d, j, l );
                auto  m_c  = M.column( col++ );
                    
                blas::copy( t_jl, m_c );
            }// for
        }// for
            
        return M;
    }// if
    else if ( d == 1 )
    {
        auto    M   = matrix< value_t >( size(1), size(0) * size(2) );
        size_t  col = 0;

        for ( size_t  l = 0; l < size(2); ++l )
        {
            for ( size_t  i = 0; i < size(0); ++i )
            {
                auto  t_il = fiber( d, i, l );
                auto  m_c  = M.column( col++ );
                
                blas::copy( t_il, m_c );
            }// for
        }// for
        
        return M;
    }// if
    else
    {
        auto    M   = matrix< value_t >( size(2), size(0) * size(1) );
        size_t  col = 0;
        
        for ( size_t  j = 0; j < size(1); ++j )
        {
            for ( size_t  i = 0; i < size(0); ++i )
            {
                auto  t_ij = fiber( d, i, j );
                auto  m_c  = M.column( col++ );
                
                blas::copy( t_ij, m_c );
            }// for
        }// for
        
        return M;
    }// else
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

    for ( size_t  l = 0; l < src.size(2); l++ )
        for ( size_t  j = 0; j < src.size(1); j++ )
            for ( size_t  i = 0; i < src.size(0); i++ )
                dest(i,j,l) = src(i,j,l);
}
using Hpro::BLAS::copy;

template < typename value_t >
value_t
dot ( const tensor3< value_t > &  t1,
      const tensor3< value_t > &  t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) );

    auto  d = value_t(0);

    for ( size_t  l = 0; l < t1.size(2); l++ )
        for ( size_t  j = 0; j < t1.size(1); j++ )
            for ( size_t  i = 0; i < t1.size(0); i++ )
                d += t1(i,j,l) * t2(i,j,l);

    return d;
}
using Hpro::BLAS::dot;

template < typename value_t >
real_type_t< value_t >
norm_F ( const tensor3< value_t > &  t )
{
    return std::sqrt( std::abs( dot( t, t ) ) );
}
using Hpro::BLAS::norm_F;

template < typename value_t >
real_type_t< value_t >
max_abs_val ( const tensor3< value_t > &  t )
{
    auto  v = real_type_t< value_t >(0);

    for ( size_t  l = 0; l < t.size(2); l++ )
        for ( size_t  j = 0; j < t.size(1); j++ )
            for ( size_t  i = 0; i < t.size(0); i++ )
                v += std::max( std::abs( t(i,j,l) ), v );

    return v;
}

//
// compute B := α A + β B (element wise)
//
template < typename alpha_t,
           typename value_t >
requires ( std::convertible_to< alpha_t, value_t > )
void
add ( const alpha_t               alpha,
      const tensor3< value_t > &  A,
      tensor3< value_t > &        B )
{
    HLR_DBG_ASSERT( ( A.size(0) == B.size(0) ) &&
                    ( A.size(1) == B.size(1) ) &&
                    ( A.size(2) == B.size(2) ) );
    
    for ( size_t  l = 0; l < A.size(2); l++ )
        for ( size_t  j = 0; j < A.size(1); j++ )
            for ( size_t  i = 0; i < A.size(0); i++ )
                B(i,j,l) += value_t(alpha) * A(i,j,l);
}

template < typename alpha_t,
           typename beta_t,
           typename value_t >
requires ( std::convertible_to< alpha_t, value_t > &&
           std::convertible_to< beta_t, value_t > )
void
add ( const alpha_t               alpha,
      const tensor3< value_t > &  A,
      const beta_t                beta,
      tensor3< value_t > &        B )
{
    HLR_DBG_ASSERT( ( A.size(0) == B.size(0) ) &&
                    ( A.size(1) == B.size(1) ) &&
                    ( A.size(2) == B.size(2) ) );
    
    for ( size_t  l = 0; l < A.size(2); l++ )
        for ( size_t  j = 0; j < A.size(1); j++ )
            for ( size_t  i = 0; i < A.size(0); i++ )
                B(i,j,l) = value_t(alpha) * A(i,j,l) + value_t(beta) * B(i,j,l);
}
using Hpro::BLAS::add;

//
// compute d-mode tensor product X×M
//
template < typename     value_t,
           matrix_type  matrix_t >
tensor3< value_t >
tensor_product ( const tensor3< value_t > &  X,
                 const matrix_t &            M,
                 const uint                  mode )
{
    HLR_ASSERT( X.size(mode) == M.ncols() );

    auto  Y = tensor3< value_t >( ( mode == 0 ? M.nrows() : X.size(0) ),
                                  ( mode == 1 ? M.nrows() : X.size(1) ),
                                  ( mode == 2 ? M.nrows() : X.size(2) ) );

    if ( mode == 0 )
    {
        for ( size_t  l = 0; l < X.size(2); ++l )
        {
            for ( size_t  j = 0; j < X.size(1); ++j )
            {
                auto  x_ij = X.fiber( mode, j, l );
                auto  y_ij = Y.fiber( mode, j, l );

                mulvec( M, x_ij, y_ij );
            }// for
        }// for
    }// if
    else if ( mode == 1 )
    {
        for ( size_t  l = 0; l < X.size(2); ++l )
        {
            for ( size_t  i = 0; i < X.size(0); ++i )
            {
                auto  x_ij = X.fiber( mode, i, l );
                auto  y_ij = Y.fiber( mode, i, l );

                mulvec( M, x_ij, y_ij );
            }// for
        }// for
    }// if
    else if ( mode == 2 )
    {
        for ( size_t  j = 0; j < X.size(1); ++j )
        {
            for ( size_t  i = 0; i < X.size(0); ++i )
            {
                auto  x_ij = X.fiber( mode, i, j );
                auto  y_ij = Y.fiber( mode, i, j );

                mulvec( M, x_ij, y_ij );
            }// for
        }// for
    }// if

    return Y;
}

//
// element-wise multiplication X2 := X1 * X2
//
template < typename value_t >
tensor3< value_t >
hadamard_product ( const tensor3< value_t > &  X1,
                   tensor3< value_t > &        X2 )
{
    HLR_ASSERT( ( X1.size(0) == X2.size(0) ) &&
                ( X1.size(1) == X2.size(1) ) &&
                ( X1.size(2) == X2.size(2) ) );

    for ( size_t  l = 0; l < X1.size(2); l++ )
        for ( size_t  j = 0; j < X1.size(1); j++ )
            for ( size_t  i = 0; i < X2.size(0); i++ )
                X2(i,j,l) *= X1(i,j,l);
}

////////////////////////////////////////////////////////////////
//
// truncation
//

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc,
        const approx_t &            apx )
{
    auto  X0 = X.unfold( 0 );
    auto  U0 = apx.column_basis( X0, acc );

    auto  X1 = X.unfold( 1 );
    auto  U1 = apx.column_basis( X1, acc );

    auto  X2 = X.unfold( 2 );
    auto  U2 = apx.column_basis( X2, acc );

    auto  Y0 = tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( U1 ), 1 );
    auto  G  = tensor_product( Y1, adjoint( U2 ), 2 );

    return { std::move(G), std::move(U0), std::move(U1), std::move(U2) };
}

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc )
{
    const auto  apx = approx::SVD< value_t >();

    return hosvd( X, acc, apx );
}

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
          const accuracy &            acc,
          const approx_t &            apx )
{
    auto  Y  = copy( X );
    auto  U0 = blas::matrix< value_t >();
    auto  U1 = blas::matrix< value_t >();
    auto  U2 = blas::matrix< value_t >();
    
    for ( uint  d = 0; d < 3; ++d )
    {
        auto  Yd = Y.unfold( d );
        auto  Ud = apx.column_basis( Yd, acc );
        auto  T  = tensor_product( Y, adjoint( Ud ), d );

        Y = std::move( T );

        switch ( d )
        {
            case 0 : U0 = std::move( Ud ); break;
            case 1 : U1 = std::move( Ud ); break;
            case 2 : U2 = std::move( Ud ); break;
        }// switch
    }// for

    return { std::move(Y), std::move(U0), std::move(U1), std::move(U2) };
}

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc )
{
    const auto  apx = approx::SVD< value_t >();

    return sthosvd( X, acc, apx );
}

}}// namespace hlr::blas

#endif  // __HPRO_BLAS_TENSOR_HH
