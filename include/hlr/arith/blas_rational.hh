#ifndef __HLR_ARITH_BLAS_RATIONAL_HH
#define __HLR_ARITH_BLAS_RATIONAL_HH
//
// Project     : HLR
// Module      : arith/blas_rational
// Description : basic linear algebra functions for rational numbers
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <boost/rational.hpp>

#include <gmpxx.h>

#include <hlr/utils/math.hh>
#include <hlr/arith/blas.hh>

namespace hlr {

//////////////////////////////////////////////////////////////////////
//
// additional math functions
//
//////////////////////////////////////////////////////////////////////

namespace math
{

template < typename integer_t > boost::rational< integer_t >  abs    ( const boost::rational< integer_t >  i ) { return boost::abs( i ); }
template < typename integer_t > boost::rational< integer_t >  conj   ( const boost::rational< integer_t >  i ) { return i; }
template < typename integer_t > boost::rational< integer_t >  square ( const boost::rational< integer_t >  i ) { return i*i; }

inline mpq_class  abs    ( const mpq_class  i ) { return ::abs( i ); }
inline mpq_class  conj   ( const mpq_class  i ) { return i; }
inline mpq_class  square ( const mpq_class  i ) { return i*i; }

}// namespace math

namespace blas {

//////////////////////////////////////////////////////////////////////
//
// for boost::rational
//
//////////////////////////////////////////////////////////////////////

template < typename integer_t >
idx_t
max_idx ( const vector< boost::rational< integer_t > > &  v )
{
    const idx_t  n = v.length();
    idx_t        p = 0;
    auto         m = math::abs( v( 0 ) );

    for ( idx_t  i = 0; i < n; ++i )
    {
        if ( math::abs( v(i) ) > m )
        {
            p = i;
            m = math::abs( v(i) );
        }// if
    }// for

    return p;
}

template < typename integer_t >
boost::rational< integer_t >
sqnorm_2 ( const vector< boost::rational< integer_t > > &  v )
{
    const size_t  n = v.length();
    auto          f = boost::rational< integer_t >( 0 );

    for ( size_t  i = 0; i < n; ++i )
        f += math::square( v(i) );

    return f;
}

template < typename integer_t >
boost::rational< integer_t >
sqnorm_F ( const matrix< boost::rational< integer_t > > &  M )
{
    const size_t  nrows = M.nrows();
    const size_t  ncols = M.ncols();
    auto          v     = boost::rational< integer_t >( 0 );

    for ( size_t  j = 0; j < ncols; ++j )
        for ( size_t  i = 0; i < nrows; ++i )
            v += math::square( M(i,j) );

    return v;
}

template < typename integer_t >
void
scale ( const boost::rational< integer_t >        f,
        vector< boost::rational< integer_t > > &  v )
{
    if ( f != boost::rational< integer_t >( 1 ) )
    {
        const auto  n = v.length();

        for ( size_t  i = 0; i < n; ++i )
            v(i) *= f;
    }// if
}

template < typename integer_t >
void
copy ( const vector< boost::rational< integer_t > > &  vs,
       vector< boost::rational< integer_t > > &        vd )
{
    const auto  n = vs.length();

    assert( n == vd.length() );

    for ( size_t  i = 0; i < n; ++i )
        vd(i) = vs(i);
}

template < typename integer_t >
void
add ( const boost::rational< integer_t >              alpha,
      const vector< boost::rational< integer_t > > &  x,
      vector< boost::rational< integer_t > > &        y )
{
    const auto  n = x.length();

    assert( n == y.length() );

    for ( size_t  i = 0; i < n; ++i )
        y(i) += alpha * x(i);
}

template < typename integer_t >
boost::rational< integer_t >
dot ( const vector< boost::rational< integer_t > > &  v1,
      const vector< boost::rational< integer_t > > &  v2 )
{
    const auto  n = v1.length();

    assert( n == v2.length() );

    auto  d = boost::rational< integer_t >( 0 );
    
    for ( size_t  i = 0; i < n; ++i )
        d += v1(i) * v2(i);

    return  d;
}

template < typename integer_t,
           typename matrix_A_t,
           typename matrix_B_t >
void
prod ( const boost::rational< integer_t >        alpha,
       const matrix_A_t &                        A,
       const matrix_B_t &                        B,
       const boost::rational< integer_t >        beta,
       matrix< boost::rational< integer_t > > &  C )
{
    static_assert( is_matrix< matrix_A_t >::value && is_matrix< matrix_B_t >::value, "only matrix types supported for A/B" );
    static_assert( std::is_same< typename matrix_A_t::value_t, boost::rational< integer_t > >::value, "need to have same value type" );
    static_assert( std::is_same< typename matrix_B_t::value_t, boost::rational< integer_t > >::value, "need to have same value type" );
    
    assert( A.nrows() == C.nrows() );
    assert( B.ncols() == C.ncols() );
    assert( A.ncols() == B.nrows() );

    const auto  nrows_C = C.nrows();
    const auto  ncols_C = C.ncols();
    const auto  ncols_A = A.ncols();

    for ( size_t  i = 0; i < nrows_C; ++i )
    {
        for ( size_t  j = 0; j < ncols_C; ++j )
        {
            auto  r = boost::rational< integer_t >( 0 );
            
            for ( size_t  k = 0; k < ncols_A; ++k )
                r += A(i,k) * B(k,j);

            C(i,j) = beta * C(i,j) + alpha * r;
        }// for
    }// for
}

//////////////////////////////////////////////////////////////////////
//
// for GMP
//
//////////////////////////////////////////////////////////////////////

inline
idx_t
max_idx ( const vector< mpq_class > &  v )
{
    const idx_t  n = v.length();
    idx_t        p = 0;
    auto         m = math::abs( v( 0 ) );

    for ( idx_t  i = 0; i < n; ++i )
    {
        if ( math::abs( v(i) ) > m )
        {
            p = i;
            m = math::abs( v(i) );
        }// if
    }// for

    return p;
}

inline
mpq_class
sqnorm_2 ( const vector< mpq_class > &  v )
{
    const size_t  n = v.length();
    auto          f = mpq_class( 0 );

    for ( size_t  i = 0; i < n; ++i )
        f += math::square( v(i) );

    return f;
}

inline
mpq_class
sqnorm_F ( const matrix< mpq_class > &  M )
{
    const size_t  nrows = M.nrows();
    const size_t  ncols = M.ncols();
    auto          v     = mpq_class( 0 );

    for ( size_t  j = 0; j < ncols; ++j )
        for ( size_t  i = 0; i < nrows; ++i )
            v += math::square( M(i,j) );

    return v;
}

inline
void
scale ( const mpq_class        f,
        vector< mpq_class > &  v )
{
    if ( f != mpq_class( 1 ) )
    {
        const auto  n = v.length();

        for ( size_t  i = 0; i < n; ++i )
            v(i) *= f;
    }// if
}

inline
void
copy ( const vector< mpq_class > &  vs,
       vector< mpq_class > &        vd )
{
    const auto  n = vs.length();

    assert( n == vd.length() );

    for ( size_t  i = 0; i < n; ++i )
        vd(i) = vs(i);
}

inline
void
add ( const mpq_class              alpha,
      const vector< mpq_class > &  x,
      vector< mpq_class > &        y )
{
    const auto  n = x.length();

    assert( n == y.length() );

    for ( size_t  i = 0; i < n; ++i )
        y(i) += alpha * x(i);
}

inline
mpq_class
dot ( const vector< mpq_class > &  v1,
      const vector< mpq_class > &  v2 )
{
    const auto  n = v1.length();

    assert( n == v2.length() );

    auto  d = mpq_class( 0 );
    
    for ( size_t  i = 0; i < n; ++i )
        d += v1(i) * v2(i);

    return  d;
}

template < typename matrix_A_t,
           typename matrix_B_t >
void
prod ( const mpq_class        alpha,
       const matrix_A_t &     A,
       const matrix_B_t &     B,
       const mpq_class        beta,
       matrix< mpq_class > &  C )
{
    static_assert( is_matrix< matrix_A_t >::value && is_matrix< matrix_B_t >::value, "only matrix types supported for A/B" );
    static_assert( std::is_same< typename matrix_A_t::value_t, mpq_class >::value, "need to have same value type" );
    static_assert( std::is_same< typename matrix_B_t::value_t, mpq_class >::value, "need to have same value type" );
    
    assert( A.nrows() == C.nrows() );
    assert( B.ncols() == C.ncols() );
    assert( A.ncols() == B.nrows() );

    const auto  nrows_C = C.nrows();
    const auto  ncols_C = C.ncols();
    const auto  ncols_A = A.ncols();

    for ( size_t  i = 0; i < nrows_C; ++i )
    {
        for ( size_t  j = 0; j < ncols_C; ++j )
        {
            auto  r = mpq_class( 0 );
            
            for ( size_t  k = 0; k < ncols_A; ++k )
                r += A(i,k) * B(k,j);

            C(i,j) = beta * C(i,j) + alpha * r;
        }// for
    }// for
}

}}// namespace hlr::blas

#endif // __HLR_ARITH_BLAS_RATIONAL_HH
