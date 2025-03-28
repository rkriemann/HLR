#ifndef __HLR_BLAS_DETAIL_TENSOR3_HH
#define __HLR_BLAS_DETAIL_TENSOR3_HH

namespace hlr { namespace blas {

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

    switch ( mode )
    {
        case 0 :
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
        }
        break;
        
        case 1 :
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
        }
        break;

        case 2 :
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
        }
        break;

        default :
            HLR_ERROR( "invalid mode" );
    }// switch

    #endif

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

//
// discrete wavelet transform using Haar wavelet
//
template < typename value_t >
std::array< blas::tensor3< value_t >, 8 >
dwt ( const blas::tensor3< value_t > &  t )
{
    //
    // axis 0
    //

    const size_t  n  = t.size(0);
    const size_t  n0 = t.size(0) / 2 + t.size(0) % 2;
    auto          L  = blas::tensor3< value_t >( n0, t.size(1), t.size(2) );
    auto          H  = blas::tensor3< value_t >( n0, t.size(1), t.size(2) );

    for ( size_t  i2 = 0; i2 < t.size(2); ++i2 )
    {
        for ( size_t  i1 = 0; i1 < t.size(1); ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  t0 = t(2*i0,i1,i2);
                const auto  t1 = t(2*i0+1,i1,i2);
                
                L(i0,i1,i2) = t0 + t1;
                H(i0,i1,i2) = t0 - t1;
            }// for
        }// for
    }// for

    //
    // axis 1
    //

    const size_t  n1 = t.size(1) / 2 + t.size(1) % 2;
    auto          LL = blas::tensor3< value_t >( n0, n1, t.size(2) );
    auto          LH = blas::tensor3< value_t >( n0, n1, t.size(2) );
    auto          HL = blas::tensor3< value_t >( n0, n1, t.size(2) );
    auto          HH = blas::tensor3< value_t >( n0, n1, t.size(2) );

    for ( size_t  i2 = 0; i2 < t.size(2); ++i2 )
    {
        for ( size_t  i1 = 0; i1 < n1; ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  L0 = L(i0,2*i1,i2);
                const auto  L1 = L(i0,2*i1+1,i2);
                
                LL(i0,i1,i2) = L0 + L1;
                LH(i0,i1,i2) = L0 - L1;

                const auto  H0 = H(i0,2*i1,i2);
                const auto  H1 = H(i0,2*i1+1,i2);

                HL(i0,i1,i2) = H0 + H1;
                HH(i0,i1,i2) = H0 - H1;
            }// for
        }// for
    }// for

    //
    // axis 2
    //

    const size_t  n2 = t.size(2) / 2 + t.size(2) % 2;
    auto          LLL = blas::tensor3< value_t >( n0, n1, n2 );
    auto          LLH = blas::tensor3< value_t >( n0, n1, n2 );
    auto          LHL = blas::tensor3< value_t >( n0, n1, n2 );
    auto          LHH = blas::tensor3< value_t >( n0, n1, n2 );
    auto          HLL = blas::tensor3< value_t >( n0, n1, n2 );
    auto          HLH = blas::tensor3< value_t >( n0, n1, n2 );
    auto          HHL = blas::tensor3< value_t >( n0, n1, n2 );
    auto          HHH = blas::tensor3< value_t >( n0, n1, n2 );

    for ( size_t  i2 = 0; i2 < n2; ++i2 )
    {
        for ( size_t  i1 = 0; i1 < n1; ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  LL0 = LL(i0,i1,2*i2);
                const auto  LL1 = LL(i0,i1,2*i2+1);
                
                LLL(i0,i1,i2) = LL0 + LL1;
                LLH(i0,i1,i2) = LL0 - LL1;

                const auto  LH0 = LH(i0,i1,2*i2);
                const auto  LH1 = LH(i0,i1,2*i2+1);
                
                LHL(i0,i1,i2) = LH0 + LH1;
                LHH(i0,i1,i2) = LH0 - LH1;

                const auto  HL0 = HL(i0,i1,2*i2);
                const auto  HL1 = HL(i0,i1,2*i2+1);
                
                HLL(i0,i1,i2) = HL0 + HL1;
                HLH(i0,i1,i2) = HL0 - HL1;

                const auto  HH0 = HH(i0,i1,2*i2);
                const auto  HH1 = HH(i0,i1,2*i2+1);
                
                HHL(i0,i1,i2) = HH0 + HH1;
                HHH(i0,i1,i2) = HH0 - HH1;
            }// for
        }// for
    }// for

    return { std::move( LLL ),
             std::move( LLH ),
             std::move( LHL ),
             std::move( LHH ),
             std::move( HLL ),
             std::move( HLH ),
             std::move( HHL ),
             std::move( HHH ) };
}

//
// inverse discrete wavelet transform using Haar wavelet
//
template < typename value_t >
blas::tensor3< value_t >
idwt ( const std::array< blas::tensor3< value_t >, 8 > &  coeffs )
{
    auto          LLL = coeffs[0];
    auto          LLH = coeffs[1];
    auto          LHL = coeffs[2];
    auto          LHH = coeffs[3];
    auto          HLL = coeffs[4];
    auto          HLH = coeffs[5];
    auto          HHL = coeffs[6];
    auto          HHH = coeffs[7];
    const size_t  n0  = LLL.size(0);
    const size_t  n1  = LLL.size(1);
    const size_t  n2  = LLL.size(2);

    //
    // axis 2
    //

    auto  LL = blas::tensor3< value_t >( n0, n1, 2*n2 );
    auto  LH = blas::tensor3< value_t >( n0, n1, 2*n2 );
    auto  HL = blas::tensor3< value_t >( n0, n1, 2*n2 );
    auto  HH = blas::tensor3< value_t >( n0, n1, 2*n2 );
    
    for ( size_t  i2 = 0; i2 < n2; ++i2 )
    {
        for ( size_t  i1 = 0; i1 < n1; ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  lll = LLL(i0,i1,i2);
                const auto  llh = LLH(i0,i1,i2);
                
                LL(i0,i1,2*i2)   = (lll + llh) / 2.0;
                LL(i0,i1,2*i2+1) = (lll - llh) / 2.0;

                const auto  lhl = LHL(i0,i1,i2);
                const auto  lhh = LHH(i0,i1,i2);

                LH(i0,i1,2*i2)   = (lhl + lhh) / 2.0;
                LH(i0,i1,2*i2+1) = (lhl - lhh) / 2.0;

                const auto  hll = HLL(i0,i1,i2);
                const auto  hlh = HLH(i0,i1,i2);

                HL(i0,i1,2*i2)   = (hll + hlh) / 2.0;
                HL(i0,i1,2*i2+1) = (hll - hlh) / 2.0;

                const auto  hhl = HHL(i0,i1,i2);
                const auto  hhh = HHH(i0,i1,i2);

                HH(i0,i1,2*i2)   = (hhl + hhh) / 2.0;
                HH(i0,i1,2*i2+1) = (hhl - hhh) / 2.0;
            }// for
        }// for
    }// for

    //
    // axis 1
    //

    auto  L = blas::tensor3< value_t >( n0, 2*n1, 2*n2 );
    auto  H = blas::tensor3< value_t >( n0, 2*n1, 2*n2 );

    for ( size_t  i2 = 0; i2 < 2*n2; ++i2 )
    {
        for ( size_t  i1 = 0; i1 < n1; ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  ll = LL(i0,i1,i2);
                const auto  lh = LH(i0,i1,i2);
                
                L(i0,2*i1,i2)   = (ll + lh) / 2.0;
                L(i0,2*i1+1,i2) = (ll - lh) / 2.0;

                const auto  hl = HL(i0,i1,i2);
                const auto  hh = HH(i0,i1,i2);
                
                H(i0,2*i1,i2)   = (hl + hh) / 2.0;
                H(i0,2*i1+1,i2) = (hl - hh) / 2.0;
            }// for
        }// for
    }// for

    //
    // axis 0
    //

    auto  M = blas::tensor3< value_t >( 2*n0, 2*n1, 2*n2 );

    for ( size_t  i2 = 0; i2 < 2*n2; ++i2 )
    {
        for ( size_t  i1 = 0; i1 < 2*n1; ++i1 )
        {
            for ( size_t  i0 = 0; i0 < n0; ++i0 )
            {
                const auto  l = L(i0,i1,i2);
                const auto  h = H(i0,i1,i2);
                
                M(2*i0,i1,i2)   = (l + h) / 2.0;
                M(2*i0+1,i1,i2) = (l - h) / 2.0;
            }// for
        }// for
    }// for

    return M;
}

}}// namespace hlr::blas

#endif // __HLR_BLAS_DETAIL_TENSOR3_HH
