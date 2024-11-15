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
copy ( const tensor4< value_t > &  src,
       tensor4< value_t > &        dest )
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
dot ( const tensor4< value_t > &  t1,
      const tensor4< value_t > &  t2 )
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
      const tensor4< value_t > &  A,
      tensor4< value_t > &        B )
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
      const tensor4< value_t > &  A,
      const beta_t                beta,
      tensor4< value_t > &        B )
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
tensor4< value_t >
tensor_product ( const tensor4< value_t > &  X,
                 const matrix_t &            M,
                 const uint                  mode )
{
    HLR_ASSERT( X.size(mode) == M.ncols() );

    auto  Y = tensor4< value_t >( ( mode == 0 ? M.nrows() : X.size(0) ),
                                  ( mode == 1 ? M.nrows() : X.size(1) ),
                                  ( mode == 2 ? M.nrows() : X.size(2) ) );

    #if 0

    if ( mode == 0 )
    {
        for ( size_t  l = 0; l < X.size(2); ++l )
        {
            auto  Xl = X.slice( 2, l );
            auto  Yl = Y.slice( 2, l );

            prod( value_t(1), M, Xl, value_t(0), Yl );
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
    
    #else
    
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

    #endif

    return Y;
}

//
// element-wise multiplication X2 := X1 * X2
//
template < typename value_t >
tensor4< value_t >
hadamard_product ( const tensor4< value_t > &  X1,
                   tensor4< value_t > &        X2 )
{
    HLR_ASSERT( ( X1.size(0) == X2.size(0) ) &&
                ( X1.size(1) == X2.size(1) ) &&
                ( X1.size(2) == X2.size(2) ) );

    for ( size_t  l = 0; l < X1.size(2); l++ )
        for ( size_t  j = 0; j < X1.size(1); j++ )
            for ( size_t  i = 0; i < X2.size(0); i++ )
                X2(i,j,l) *= X1(i,j,l);
}

}}// namespace hlr::blas

#endif // __HLR_BLAS_DETAIL_TENSOR4_HH
