#ifndef __HLR_BLAS_DETAIL_TENSOR4_HH
#define __HLR_BLAS_DETAIL_TENSOR4_HH

namespace hlr { namespace blas {

//
// d-mode unfolding
//
template < typename value_t >
matrix< value_t >
tensor4< value_t >::unfold ( const uint  d ) const
{
    HLR_ASSERT( d < 3 );
    
    if ( d == 0 )
    {
        auto    M   = matrix< value_t >( size(0), size(1) * size(2) * size(3) );
        size_t  col = 0;

        for ( size_t  i3 = 0; i3 < size(3); ++i3 )
        {
            for ( size_t  i2 = 0; i2 < size(2); ++i2 )
            {
                for ( size_t  i1 = 0; i1 < size(1); ++i1 )
                {
                    auto  t_123 = fiber( d, i1, i2, i3 );
                    auto  m_c   = M.column( col++ );
                    
                    blas::copy( t_123, m_c );
                }// for
            }// for
        }// for
            
        return M;
    }// if
    else if ( d == 1 )
    {
        auto    M   = matrix< value_t >( size(1), size(0) * size(2) * size(3) );
        size_t  col = 0;

        for ( size_t  i3 = 0; i3 < size(3); ++i3 )
        {
            for ( size_t  i2 = 0; i2 < size(2); ++i2 )
            {
                for ( size_t  i0 = 0; i0 < size(0); ++i0 )
                {
                    auto  t_023 = fiber( d, i0, i2, i3 );
                    auto  m_c   = M.column( col++ );
                    
                    blas::copy( t_023, m_c );
                }// for
            }// for
        }// for
        
        return M;
    }// if
    else if ( d == 2 )
    {
        auto    M   = matrix< value_t >( size(2), size(0) * size(1) * size(3) );
        size_t  col = 0;
        
        for ( size_t  i3 = 0; i3 < size(3); ++i3 )
        {
            for ( size_t  i1 = 0; i1 < size(1); ++i1 )
            {
                for ( size_t  i0 = 0; i0 < size(0); ++i0 )
                {
                    auto  t_013 = fiber( d, i0, i1, i3 );
                    auto  m_c   = M.column( col++ );
                
                    blas::copy( t_013, m_c );
                }// for
            }// for
        }// for
        
        return M;
    }// else
    else // if ( d == 3 )
    {
        auto    M   = matrix< value_t >( size(3), size(0) * size(1) * size(2) );
        size_t  col = 0;
        
        for ( size_t  i2 = 0; i2 < size(2); ++i2 )
        {
            for ( size_t  i1 = 0; i1 < size(1); ++i1 )
            {
                for ( size_t  i0 = 0; i0 < size(0); ++i0 )
                {
                    auto  t_012 = fiber( d, i0, i1, i2 );
                    auto  m_c   = M.column( col++ );
                
                    blas::copy( t_012, m_c );
                }// for
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
copy ( const tensor4< value_t > &  src,
       tensor4< value_t > &        dest )
{
    HLR_DBG_ASSERT( ( src.size(0) == dest.size(0) ) &&
                    ( src.size(1) == dest.size(1) ) &&
                    ( src.size(2) == dest.size(2) ) &&
                    ( src.size(3) == dest.size(3) ) );

    for ( size_t  i3 = 0; i3 < src.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < src.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < src.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < src.size(0); i0++ )
                    dest(i0,i1,i2,i3) = src(i0,i1,i2,i3);
}
using Hpro::BLAS::copy;

template < typename value_t >
value_t
dot ( const tensor4< value_t > &  t1,
      const tensor4< value_t > &  t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) &&
                    ( t1.size(3) == t2.size(3) ) );

    auto  d = value_t(0);

    for ( size_t  i3 = 0; i3 < t1.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < t1.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < t1.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < t1.size(0); i0++ )
                    d += t1(i0,i1,i2,i3) * t2(i0,i1,i2,i3);

    return d;
}
using Hpro::BLAS::dot;

template < typename value_t >
real_type_t< value_t >
norm_F ( const tensor4< value_t > &  t )
{
    return std::sqrt( std::abs( dot( t, t ) ) );
}
using Hpro::BLAS::norm_F;

template < typename value_t >
real_type_t< value_t >
max_abs_val ( const tensor4< value_t > &  t )
{
    auto  v = real_type_t< value_t >(0);

    for ( size_t  i3 = 0; i3 < t.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < t.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < t.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < t.size(0); i0++ )
                    v += std::max( std::abs( t(i0,i1,i2,i3) ), v );

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
      const tensor4< value_t > &  t1,
      tensor4< value_t > &        t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) &&
                    ( t1.size(3) == t2.size(3) ) );
    
    for ( size_t  i3 = 0; i3 < t1.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < t1.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < t1.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < t1.size(0); i0++ )
                    t2(i0,i1,i2,i3) += value_t(alpha) * t1(i0,i1,i2,i3);
}

template < typename alpha_t,
           typename beta_t,
           typename value_t >
requires ( std::convertible_to< alpha_t, value_t > &&
           std::convertible_to< beta_t, value_t > )
void
add ( const alpha_t               alpha,
      const tensor4< value_t > &  t1,
      const beta_t                beta,
      tensor4< value_t > &        t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) &&
                    ( t1.size(3) == t2.size(3) ) );
    
    for ( size_t  i3 = 0; i3 < t1.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < t1.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < t1.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < t1.size(0); i0++ )
                    t2(i0,i1,i2,i3) = value_t(alpha) * t1(i0,i1,i2,i3) + value_t(beta) * t2(i0,i1,i2,i3);
}
using Hpro::BLAS::add;

//
// compute d-mode tensor product X×M
//
template < typename     value_t,
           matrix_type  matrix_t >
tensor4< value_t >
tensor_product ( const tensor4< value_t > &  X,
                 const matrix_t &            M,
                 const uint                  mode )
{
    HLR_ASSERT( X.size(mode) == M.ncols() );

    auto  Y = tensor4< value_t >( ( mode == 0 ? M.nrows() : X.size(0) ),
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
    else // if ( mode == 3 )
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
// element-wise multiplication t2 := t1 * t2
//
template < typename value_t >
tensor4< value_t >
hadamard_product ( const tensor4< value_t > &  t1,
                   tensor4< value_t > &        t2 )
{
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) &&
                    ( t1.size(3) == t2.size(3) ) );
    
    for ( size_t  i3 = 0; i3 < t1.size(3); i3++ )
        for ( size_t  i2 = 0; i2 < t1.size(2); i2++ )
            for ( size_t  i1 = 0; i1 < t1.size(1); i1++ )
                for ( size_t  i0 = 0; i0 < t1.size(0); i0++ )
                    t2(i0,i1,i2,i3) *= t1(i0,i1,i2,i3);
}

}}// namespace hlr::blas

#endif // __HLR_BLAS_DETAIL_TENSOR4_HH
