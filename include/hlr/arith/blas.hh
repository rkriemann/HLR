#ifndef __HLR_ARITH_BLAS_HH
#define __HLR_ARITH_BLAS_HH
//
// Project     : HLR
// Module      : arith/blas
// Description : basic linear algebra functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>

#include <hpro/blas/Matrix.hh>
#include <hpro/blas/Vector.hh>
#include <hpro/blas/Algebra.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/math.hh>
#include <hlr/arith/blas_def.hh>

namespace hlr { namespace blas {

namespace hpro = HLIB;

using hpro::blas_int_t;
using hpro::matop_t;

//
// import functions from HLIBpro and adjust naming
//

using namespace HLIB::BLAS;

using range = HLIB::BLAS::Range;

template < typename value_t > using vector = HLIB::BLAS::Vector< value_t >;
template < typename value_t > using matrix = HLIB::BLAS::Matrix< value_t >;

//////////////////////////////////////////////////////////////////////
//
// template wrappers for low-rank factors as U and V
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
blas::matrix< value_t > &
mat_U ( hpro::TRkMatrix * A )
{
    assert( ! is_null( A ) );
    return hpro::blas_mat_A< value_t >( A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_U ( hpro::TRkMatrix *    A,
        const hpro::matop_t  op )
{
    assert( ! is_null( A ) );

    if ( op == hpro::apply_normal )
        return hpro::blas_mat_A< value_t >( A );
    else
        return hpro::blas_mat_B< value_t >( A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_V ( hpro::TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return hpro::blas_mat_B< value_t >( A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_V ( hpro::TRkMatrix *    A,
        const hpro::matop_t  op )
{
    assert( ! is_null( A ) );

    if ( op == hpro::apply_normal )
        return hpro::blas_mat_B< value_t >( A );
    else
        return hpro::blas_mat_A< value_t >( A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_U ( const hpro::TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return hpro::blas_mat_A< value_t >( A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_U ( const hpro::TRkMatrix *  A,
        const hpro::matop_t      op )
{
    assert( ! is_null( A ) );

    if ( op == hpro::apply_normal )
        return hpro::blas_mat_A< value_t >( A );
    else
        return hpro::blas_mat_B< value_t >( A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_V ( const hpro::TRkMatrix *  A )
{
    assert( ! is_null( A ) );
    return hpro::blas_mat_B< value_t >( A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_V ( const hpro::TRkMatrix *  A,
        const hpro::matop_t      op )
{
    assert( ! is_null( A ) );

    if ( op == hpro::apply_normal )
        return hpro::blas_mat_B< value_t >( A );
    else
        return hpro::blas_mat_A< value_t >( A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_U ( hpro::TRkMatrix &    A )
{
    return mat_U< value_t >( & A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_U ( hpro::TRkMatrix &    A,
        const hpro::matop_t  op )
{
    return mat_U< value_t >( & A, op );
}

template < typename value_t >
blas::matrix< value_t > &
mat_V ( hpro::TRkMatrix &    A )
{
    return mat_V< value_t >( & A );
}

template < typename value_t >
blas::matrix< value_t > &
mat_V ( hpro::TRkMatrix &    A,
        const hpro::matop_t  op )
{
    return mat_V< value_t >( & A, op );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_U ( const hpro::TRkMatrix &  A )
{
    return mat_U< value_t >( & A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_U ( const hpro::TRkMatrix &  A,
        const hpro::matop_t      op )
{
    return mat_U< value_t >( & A, op );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_V ( const hpro::TRkMatrix &  A )
{
    return mat_V< value_t >( & A );
}

template < typename value_t >
const blas::matrix< value_t > &
mat_V ( const hpro::TRkMatrix &  A,
        const hpro::matop_t      op )
{
    return mat_V< value_t >( & A, op );
}


//////////////////////////////////////////////////////////////////////
//
// general copy method
//
//////////////////////////////////////////////////////////////////////

template < typename T_vector >
typename hpro::enable_if_res< is_vector< T_vector >::value,
                              vector< typename T_vector::value_t > >::result
copy ( const T_vector &  v )
{
    using  value_t = typename T_vector::value_t;

    vector< value_t >  w( v.length() );

    hpro::BLAS::copy( v, w );

    return w;
}

template < typename T_matrix >
typename hpro::enable_if_res< is_matrix< T_matrix >::value,
                              matrix< typename T_matrix::value_t > >::result
copy ( const T_matrix &  A )
{
    using  value_t = typename T_matrix::value_t;

    matrix< value_t >  M( A.nrows(), A.ncols() );

    hpro::BLAS::copy( A, M );

    return M;
}

using hpro::BLAS::copy;

//////////////////////////////////////////////////////////////////////
//
// various fill methods
//
//////////////////////////////////////////////////////////////////////

template < typename T_vector,
           typename T_fill_fn >
void
fill ( blas::VectorBase< T_vector > &   v,
       T_fill_fn &  fill_fn )
{
    for ( size_t  i = 0; i < v.length(); ++i )
        v(i) = fill_fn();
}
       
template < typename T_vector >
void
fill ( blas::VectorBase< T_vector > &    v,
       const typename T_vector::value_t  f )
{
    for ( size_t  i = 0; i < v.length(); ++i )
        v(i) = f;
}
       
template < typename T_matrix,
           typename T_fill_fn >
void
fill ( blas::MatrixBase< T_matrix > &   M,
       T_fill_fn &  fill_fn )
{
    for ( size_t  i = 0; i < M.nrows(); ++i )
        for ( size_t  j = 0; j < M.ncols(); ++j )
            M(i,j) = fill_fn();
}
       
template < typename T_matrix >
void
fill ( blas::MatrixBase< T_matrix > &    M,
       const typename T_matrix::value_t  f )
{
    for ( size_t  i = 0; i < M.nrows(); ++i )
        for ( size_t  j = 0; j < M.ncols(); ++j )
            M(i,j) = f;
}

//////////////////////////////////////////////////////////////////////
//
// norm computations
//
//////////////////////////////////////////////////////////////////////

extern "C"
{
float
slange_ ( const char *        norm,
          const blas_int_t *  nrows,
          const blas_int_t *  ncols,
          const float *       M,
          const blas_int_t *  ldM,
          float *             work );
double
dlange_ ( const char *        norm,
          const blas_int_t *  nrows,
          const blas_int_t *  ncols,
          const double *      M,
          const blas_int_t *  ldM,
          double *            work );
double
clange_ ( const char *                   norm,
          const blas_int_t *             nrows,
          const blas_int_t *             ncols,
          const std::complex< float > *  M,
          const blas_int_t *             ldM,
          float *                        work );
double
zlange_ ( const char *                    norm,
          const blas_int_t *              nrows,
          const blas_int_t *              ncols,
          const std::complex< double > *  M,
          const blas_int_t *              ldM,
          double *                        work );
}

#define  HLR_BLAS_NORM1( type, func )                                   \
    inline                                                              \
    typename hpro::real_type< type >::type_t                            \
    norm_1 ( const matrix< type > &  M )                                \
    {                                                                   \
        typename hpro::real_type< type >::type_t  work = 0;             \
        const blas_int_t                          nrows = M.nrows();    \
        const blas_int_t                          ncols = M.ncols();    \
        const blas_int_t                          ldM   = M.col_stride(); \
                                                                        \
        return func( "1", & nrows, & ncols, M.data(), & ldM, & work );  \
    }

HLR_BLAS_NORM1( float,                  slange_ )
HLR_BLAS_NORM1( double,                 dlange_ )
HLR_BLAS_NORM1( std::complex< float >,  clange_ )
HLR_BLAS_NORM1( std::complex< double >, zlange_ )
#undef HLR_BLAS_NORM1

#define  HLR_BLAS_NORMI( type, func )                                   \
    inline                                                              \
    typename hpro::real_type< type >::type_t                            \
    norm_inf ( const matrix< type > &  M )                              \
    {                                                                   \
        typename hpro::real_type< type >::type_t  work = 0;             \
        const blas_int_t                          nrows = M.nrows();    \
        const blas_int_t                          ncols = M.ncols();    \
        const blas_int_t                          ldM   = M.col_stride(); \
                                                                        \
        return func( "I", & nrows, & ncols, M.data(), & ldM, & work );  \
    }

HLR_BLAS_NORMI( float,                  slange_ )
HLR_BLAS_NORMI( double,                 dlange_ )
HLR_BLAS_NORMI( std::complex< float >,  clange_ )
HLR_BLAS_NORMI( std::complex< double >, zlange_ )
#undef HLR_BLAS_NORMI

#define  HLR_BLAS_NORMM( type, func )                                   \
    inline                                                              \
    typename hpro::real_type< type >::type_t                            \
    norm_max ( const matrix< type > &  M )                              \
    {                                                                   \
        typename hpro::real_type< type >::type_t  work = 0;             \
        const blas_int_t                          nrows = M.nrows();    \
        const blas_int_t                          ncols = M.ncols();    \
        const blas_int_t                          ldM   = M.col_stride(); \
                                                                        \
        return func( "M", & nrows, & ncols, M.data(), & ldM, & work );  \
    }

HLR_BLAS_NORMM( float,                  slange_ )
HLR_BLAS_NORMM( double,                 dlange_ )
HLR_BLAS_NORMM( std::complex< float >,  clange_ )
HLR_BLAS_NORMM( std::complex< double >, zlange_ )
#undef HLR_BLAS_NORMM

//////////////////////////////////////////////////////////////////////
//
// functions related to QR factorization
//
//////////////////////////////////////////////////////////////////////

//
// compute QR factorisation M = Q·R with orthonormal Q
// and upper triangular R. Upon exit, M will hold Q.
//
// ASSUMPTION: nrows(M) ≥ ncols(M)
//
template < typename value_t >
void
qr2  ( matrix< value_t > &  M,
       matrix< value_t > &  R )
{
    const blas_int_t        nrows = M.nrows();
    const blas_int_t        ncols = M.ncols();
    std::vector< value_t >  tau( ncols );
    std::vector< value_t >  work( ncols );

    HLR_ASSERT( ncols <= nrows );

    #if 1
    
    blas_int_t  info = 0;

    geqr2( nrows, ncols, M.data(), nrows, tau.data(), work.data(), info );

    for ( blas_int_t  i = 0; i < ncols; ++i )
        for ( blas_int_t  j = 0; j <= i; ++j )
            R(j,i) = M(j,i);

    ung2r( nrows, ncols, ncols, M.data(), nrows, tau.data(), work.data(), info );
    
    #else
    
    for ( blas_int_t  i = 0; i < ncols; ++i )
    {
        auto  m_i = M.column( i );

        //
        // generate elementary reflector H(i) to annihilate M(i+1:m,i)
        //
        
        larfg( m_i.length()-i, M(i,i), m_i.data()+i+1, 1, tau[i] );

        //
        // apply H(i) to M(i:nrows, i+1:ncols) from the left
        //
        
        if ( i < ncols-1 )
        {
            const auto  m_ii = M(i,i);
            matrix      M_sub( M, range( i, nrows-1 ), range( i+1, ncols-1 ) );
            
            M(i,i) = value_t(1);
            larf( 'L', nrows-i, ncols-i-1, m_i.data() + i, 1, tau[i], M_sub.data(), M.col_stride(), work.data() );
            M(i,i) = m_ii;
        }// if

        //
        // copy upper part to R
        //
        
        for ( blas_int_t  j = 0; j <= i; ++j )
            R(j,i) = M(j,i);
    }// for

    //
    // compute Q
    //

    for ( blas_int_t  i = ncols-1; i >= 0; --i )
    {
        auto  m_i = M.column( i );
        
        // 
        // apply H(i) to M( i:nrows, i:ncols ) from the left
        //
        
        if ( i < ncols-1 )
        {
            matrix  M_sub( M, range( i, nrows-1 ), range( i+1, ncols-1 ) );
            
            M(i,i) = value_t(1);
            larf( 'L', nrows-i, ncols-i-1, m_i.data() + i, 1, tau[i], M_sub.data(), M.col_stride(), work.data() );
        }// if
        
        vector  m_i1_i( M.column(i), range( i+1, nrows-1 ) );
            
        scale( -tau[i], m_i1_i );

        M(i,i) = value_t(1) - tau[i];

        //
        // zero part above diagonal
        //

        for ( blas_int_t  j = 0; j < i; ++j )
            M(j,i) = value_t(0);
    }// for

    #endif
}

//
// to switch between different QR implementations
//
template < typename value_t >
void
qr_wrapper ( matrix< value_t > &  M,
             matrix< value_t > &  R )
{
    blas::qr2( M, R );
}

//
// compute QR factorisation A = Q·R with orthonormal Q
// and upper triangular R. Upon exit, A will hold Q
// implicitly together with tau.
//
template < typename value_t >
void
qr_impl  ( matrix< value_t > &       A,
           matrix< value_t > &       R,
           std::vector< value_t > &  T )
{
    const auto  nrows = blas_int_t( A.nrows() );
    const auto  ncols = blas_int_t( A.ncols() );
    const auto  minrc = std::min( nrows, ncols );
    blas_int_t  info  = 0;

    #if 1

    if ( blas_int_t( T.size() ) != minrc )
        T.resize( minrc );
    
    //
    // workspace query
    //

    auto  work_query = value_t(0);

    geqrf< value_t >( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to geqrf failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );
              
    //
    // compute QR
    //

    geqrf< value_t >( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), work.data(), work.size(), info );
    
    if ( info < 0 )
        HLR_ERROR( "geqrf failed" );

    #else
    
    //
    // workspace query
    //

    value_t  t_query[5] = { value_t(0), value_t(0), value_t(0), value_t(0), value_t(0) };
    auto     work_query = value_t(0);

    geqr( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), t_query, -1, & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to geqr failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );

    T.resize( blas_int_t( std::real( t_query[0] ) ) );
              
    //
    // compute QR
    //

    geqr( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), T.size(), work.data(), work.size(), info );
    
    if ( info < 0 )
        HLR_ERROR( "geqr failed" );

    #endif
    
    //
    // copy upper triangular matrix to R
    //

    if (( blas_int_t( R.nrows() ) != minrc ) || ( blas_int_t( R.ncols() ) != ncols ))
        R = std::move( Matrix< value_t >( minrc, ncols ) );
    else
        fill( value_t(0), R );
    
    for ( blas_int_t  i = 0; i < ncols; i++ )
    {
        vector< value_t >  colA( A, range( 0, i ), i );
        vector< value_t >  colR( R, range( 0, i ), i );

        copy( colA, colR );
    }// for
}

//
// compute op(Q)·M or M·op(Q) with Q from implicit QR factorization
// where op is apply_normal or apply_transposed for real valued matrices
// and apply_normal and apply_adjoint for complex valued matrices.
//
template < typename value_t >
void
prod_Q ( const eval_side_t               side,
         const hpro::matop_t             op_Q,
         const matrix< value_t > &       Q,
         const std::vector< value_t > &  T,
         matrix< value_t > &             M )
{
    const auto  nrows = blas_int_t( M.nrows() );
    const auto  ncols = blas_int_t( M.ncols() );
    const auto  k     = blas_int_t( Q.ncols() );
    blas_int_t  info  = 0;

    if ( hpro::is_complex_type< value_t >::value && ( op_Q == hpro::apply_trans ) )
        HLR_ERROR( "only normal and adjoint mode supported for complex valued matrices" );
    
    //
    // workspace query
    //

    char  op         = ( op_Q == hpro::apply_normal ? 'N' :
                         ( hpro::is_complex_type< value_t >::value ? 'C' : 'T' ) );
    auto  work_query = value_t(0);

    unmqr( char(side), op, nrows, ncols, k,
                   Q.data(), blas_int_t( Q.col_stride() ), T.data(),
                   M.data(), blas_int_t( M.col_stride() ),
                   & work_query, -1, info );

    // gemqr( char(side), op, nrows, ncols, k,
    //                Q.data(), blas_int_t( Q.col_stride() ), T.data(), T.size(),
    //                M.data(), blas_int_t( M.col_stride() ),
    //                & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to gemqr failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );

    //
    // multiply with Q
    //

    unmqr( char(side), op, nrows, ncols, k,
                   Q.data(), blas_int_t( Q.col_stride() ), T.data(),
                   M.data(), blas_int_t( M.col_stride() ),
                   work.data(), work.size(), info );

    // gemqr( char(side), op, nrows, ncols, k,
    //                Q.data(), blas_int_t( Q.col_stride() ), T.data(), T.size(),
    //                M.data(), blas_int_t( M.col_stride() ),
    //                work.data(), work.size(), info );
    
    if ( info < 0 )
        HLR_ERROR( "gemqr failed" );
}

//
// form explicit Q from given Householder vectors in Q and tau
// - result is overwritten in Q
//
template < typename value_t >
matrix< value_t >
compute_Q ( const matrix< value_t > &       Q,
            const std::vector< value_t > &  T )
{
    #if 0

    const auto         ncols = blas_int_t( Q.ncols() );
    matrix< value_t >  M( Q.nrows(), ncols );

    for ( blas_int_t  i = 0; i < ncols; ++i )
        M( i, i ) = value_t(1);

    prod_Q( from_left, hpro::apply_normal, Q, T, M );

    return M;
    
    #else
    
    //
    // workspace query
    //

    const auto  nrows = blas_int_t( Q.nrows() );
    const auto  ncols = blas_int_t( Q.ncols() );
    const auto  minrc = std::min( nrows, ncols );
    blas_int_t  info  = 0;
    auto        work_query = value_t(0);

    orgqr< value_t >( nrows, ncols, minrc, Q.data(), blas_int_t( Q.col_stride() ), T.data(), & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to orgqr failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );

    //
    // multiply with Q
    //

    auto  M = copy( Q );
    
    orgqr< value_t >( nrows, ncols, minrc, M.data(), blas_int_t( M.col_stride() ), T.data(), work.data(), work.size(), info );
    
    if ( info < 0 )
        HLR_ERROR( "orgqr failed" );

    return M;
    
    #endif
}

//
// compute QR factorisation of the tall-and-skinny nrows × ncols matrix M,
// ncols ≪ nrows, with n×m matrix Q and mxm matrix R (n >= m)
// Upon exit, M will be overwritten with Q
//
template < typename value_t >
void
tsqr  ( matrix< value_t > &  M,
        matrix< value_t > &  R )
{
    const size_t  nrows = M.nrows();
    const size_t  ncols = M.ncols();
    const size_t  ntile = 256;

    HLR_ASSERT( nrows >= ncols );
    
    if (( nrows > ntile ) && ( nrows >= 4 * ncols ))
    {
        auto  mid   = nrows / 2;
        auto  rows0 = range( 0, mid-1 );
        auto  rows1 = range( mid, nrows-1 );
        auto  Q0    = matrix< value_t >( M, rows0, range::all, hpro::copy_value );
        auto  Q1    = matrix< value_t >( M, rows1, range::all, hpro::copy_value );
        auto  R0    = matrix< value_t >( ncols, ncols );
        auto  R1    = matrix< value_t >( ncols, ncols );

        //
        // M = | Q0 R0 | = | Q0   | | R0 | = | Q0   | Q2 R
        //     | Q1 R1 |   |   Q1 | | R1 |   |   Q1 | 
        //
        
        tsqr( Q0, R0 );
        tsqr( Q1, R1 );

        auto  Q2  = matrix< value_t >( 2*ncols, ncols );
        auto  Q20 = matrix< value_t >( Q2, Range( 0,     ncols-1   ), Range::all );
        auto  Q21 = matrix< value_t >( Q2, Range( ncols, 2*ncols-1 ), Range::all );

        copy( R0, Q20 );
        copy( R1, Q21 );

        qr2( Q2, R );

        //
        // Q = | Q0    | Q    (overwrite M)
        //     |    Q1 |
        //
        
        auto  Q_0  = matrix< value_t >( M, rows0, Range::all );
        auto  Q_1  = matrix< value_t >( M, rows1, Range::all );

        prod( value_t(1), Q0, Q20, value_t(0), Q_0 );
        prod( value_t(1), Q1, Q21, value_t(0), Q_1 );
    }// if
    else
    {
        qr2( M, R );
    }// else
}

//
// construct approximate factorisation M = Q·R with orthonormal Q
//
template < typename value_t >
std::pair< matrix< value_t >,
           matrix< value_t > >
factorise_ortho ( const matrix< value_t > &  M )
{
    auto  Q = std::move( copy( M ) );
    auto  R = matrix< value_t >();

    hpro::BLAS::factorise_ortho( Q, R );

    return { std::move( Q ), std::move( R ) };
}

//
// construct approximate factorisation M = Q·R with orthonormal Q
//
template < typename value_t >
std::pair< matrix< value_t >,
           matrix< value_t > >
factorise_ortho ( const matrix< value_t > &  M,
                  const hpro::TTruncAcc &    acc )
{
    using  real_t  = typename hpro::real_type< value_t >::type_t;
    
    const size_t  nrows = M.nrows();
    const size_t  ncols = M.ncols();

    if ( nrows >= ncols )
    {
        //
        // M = Q R
        //   = Q U S V^H
        //   ≈ Q U(:,1:k) S(1:k,:) V^H
        //   = Q' S(1:k,:) V^H  with Q' = Q U(:,1:k)
        // R ≔ Q'^H M
        //
    
        // compute QR of A
        auto  Q = std::move( copy( M ) );
        auto  R = matrix< value_t >();
        
        qr( Q, R );

        // compute SVD of R
        auto  U = std::move( R );
        auto  V = matrix< value_t >();
        auto  S = vector< real_t >();

        svd( U, S, V );

        // compute new rank
        const auto  k   = acc.trunc_rank( S );
        auto        U_k = matrix< value_t >( U, range::all, range( 0, k-1 ) );
        auto        OQ  = prod( value_t(1), Q, U_k );
        auto        OR  = prod( value_t(1), adjoint( OQ ), M );

        return { std::move( OQ ), std::move( OR ) };
    }// if
    else
    {
        //
        // M^H = Q R  =>
        //   M = R^H Q^H
        //     = (U S V^H)^H Q^H
        //     = V S^H U^H Q^H
        //     ≈ V(:,1:k) S(1:k,:) U^H Q^H
        //     = Q' S(1:k,:) Q'^H  with Q' = V(:,1:k)
        // R   = Q'^H M 
        //
    
        // compute QR of M^H
        auto  Q = std::move( copy( adjoint( M ) ) );
        auto  R = matrix< value_t >();
        
        qr( Q, R );

        // compute SVD of R^H
        auto  U = std::move( R );
        auto  V = matrix< value_t >();
        auto  S = vector< real_t >();
        
        svd( U, S, V );

        // compute new rank
        const auto  k   = acc.trunc_rank( S );
        auto        V_k = matrix< value_t >( V, range::all, range( 0, k-1 ) );
        auto        OQ  = std::move( blas::copy( V_k ) );
        auto        OR  = prod( value_t(1), adjoint( OQ ), M );

        return { std::move( OQ ), std::move( OR ) };
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// functions for eigenvalue computations
//
//////////////////////////////////////////////////////////////////////

//
// compute eigen values and eigen vectors of M using two-sided Jacobi iteration.
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_jac ( matrix< value_t > &                                M,
            const size_t                                       amax_sweeps = 0,
            const typename hpro::real_type< value_t >::type_t  atolerance  = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto         nrows      = M.nrows();
    const auto         ncols      = M.ncols();
    const auto         minrc      = std::min( nrows, ncols );
    const size_t       max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 15 );
    const real_t       tolerance  = ( atolerance > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool               converged  = false;
    uint               sweep      = 0;
    matrix< value_t >  V( minrc, ncols );

    // initialise V with identity
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    while ( ! converged && ( sweep < max_sweeps ))
    {
        real_t  max_err = 0.0;
        
        sweep++;
        converged = true;
                
        for ( size_t  i = 0; i < nrows-1; i++ )
        {
            for ( size_t j = i + 1; j < ncols; j++ )
            {
                //
                // compute Jacobi rotation diagonalizing ⎧ M_ii  M_ij ⎫
                //                                       ⎩ M_ji  M_jj ⎭
                //

                const auto  c = M(i,j);

                if ( std::abs( c ) == value_t(0) )
                    continue;

                const auto  a   = M(i,i);
                const auto  b   = M(j,j);
                const auto  err = std::abs( c ) / std::real( std::sqrt( a*b ) );
                
                if (  err > tolerance )
                    converged = false;

                if ( ! std::isnormal( err ) )
                    std::cout << std::endl;
                
                max_err = std::max( err, max_err );
                
                //
                // compute Jacobi rotation which diagonalises │a c│
                //                                            │c b│
                //

                const auto  xi = (b - a) / ( value_t(2) * c );
                const auto  t  = ( math::sign( xi ) / ( std::abs(xi) + std::sqrt( 1.0 + xi*xi ) ) );
                const auto  cs = value_t(1) / std::sqrt( 1.0 + t*t );
                const auto  sn = cs * t;

                M(i,i) = a - c * t;
                M(j,j) = b + c * t;
                M(i,j) = M(j,i) = 0;
                
                //
                // update columns i and j of A (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    if (( k == i ) || ( k == j ))
                        continue;
                    
                    const auto  m_ik = M(i,k);
                    const auto  m_jk = M(j,k);

                    M(k,i) = M(i,k) = cs * m_ik - sn * m_jk;
                    M(k,j) = M(j,k) = sn * m_ik + cs * m_jk;
                }// for

                //
                // update V (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  v_ki = V(k,i);
                    const auto  v_kj = V(k,j);

                    V(k,i) = cs * v_ki - sn * v_kj;
                    V(k,j) = sn * v_ki + cs * v_kj;
                }// for
            }// for
        }// for

        //
        // determine diagonal dominance ( Σ_j≠i a_ij ) / a_ii
        //

        real_t  diag_dom = real_t(0);
        real_t  avg_dom  = real_t(0);
        
        for ( size_t  i = 0; i < nrows-1; i++ )
        {
            real_t  row_sum = real_t(0);
            
            for ( size_t j = 0; j < ncols; j++ )
            {
                if ( i != j )
                    row_sum += std::abs( M(i,j) );
            }// for

            const auto  dom = row_sum / std::abs( M(i,i) );
            
            diag_dom = std::max( diag_dom, dom );
            avg_dom += dom;
        }// for

        avg_dom /= nrows;
        
        std::cout << "sweeps " << sweep << " : "
                  << "error = " << max_err << ", "
                  << "diag_dom = " << diag_dom << ", "
                  << "avg_dom = " << avg_dom 
                  << std::endl;

        if (( diag_dom <= 2.0 ) && ( avg_dom <= 0.05 ))
            break;
    }// while

    std::cout << "#sweeps = " << sweep << std::endl;
    
    //
    // extract eigenvalues as diagonal elements of M
    //

    vector< value_t >  E( minrc );
    
    for ( size_t  i = 0; i < minrc; i++ )
        E(i) = M(i,i);

    return { std::move( E ), std::move( V ) };
}

//
// compute eigen values and eigen vectors of M using DPT iteration.
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
std::pair< blas::vector< value_t >,
           blas::matrix< value_t > >
eigen_dpt ( matrix< value_t > &                                M,
            const size_t                                       amax_sweeps = 0,
            const typename hpro::real_type< value_t >::type_t  atolerance  = 0,
            const std::string &                                error_type  = "frobenius",
            const int                                          verbosity   = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;

    // assumption
    HLR_ASSERT( M.nrows() == M.ncols() );
    
    const auto    nrows      = M.nrows();
    const real_t  tolerance  = ( atolerance  > 0 ? atolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    const uint    max_sweeps = ( amax_sweeps > 0 ? amax_sweeps : 100 );

    vector< value_t >  diag_T( nrows );
    vector< value_t >  diag_M( nrows );
    matrix< value_t >  Delta( M );  // reuse M
    matrix< value_t >  V( nrows, nrows );
    matrix< value_t >  T( nrows, nrows );

    for ( size_t  i = 0; i < nrows; ++i )
    {
        diag_M(i)     = M( i, i );
        Delta( i, i ) = value_t(0);     // diag(Δ) = 0
        V(i,i)        = value_t(1);     // V = I before iteration
    }// for

    //
    // compute I - Θ⊗M with Θ_ij = 1 / ( m_ii - m_jj )
    //
    auto  hmul_theta =  [&diag_M] ( matrix< value_t > &  A )
                        {
                            for ( size_t  j = 0; j < A.ncols(); ++j )
                                for ( size_t  i = 0; i < A.nrows(); ++i )
                                {
                                    if ( i == j )
                                        A(i,j)  =   value_t(1);
                                    else
                                        A(i,j) *= - value_t(1) / ( diag_M(i) - diag_M(j) );
                                }// for
                        };

    //
    // iteration
    //
    
    real_t  old_error = real_t(0);
    uint    sweep     = 0;
    
    do
    {
        // T = Δ·V
        prod( value_t(1), Delta, V, value_t(0), T );
        
        // T = Δ·V - V·diag(Δ·V) = T - V·diag(T) 
        // computed as T(i,:) = T(i,:) - T(i,i) · V(i,:)
        for ( size_t  i = 0; i < nrows; ++i )
            diag_T(i) = T(i,i);
        
        for ( size_t  i = 0; i < nrows; ++i )
        {
            auto  V_i = V.column(i);
            auto  T_i = T.column(i);

            add( -diag_T(i), V_i, T_i );
        }// for

        // I - Θ ∗ (Δ·V - V·diag(Δ·V)) = I - Θ ∗ T
        hmul_theta( T );

        //
        // compute error ||V-T||_F
        //

        real_t  error = 0;

        if (( error_type == "frobenius" ) || ( error_type == "fro" ))
        {
            add( value_t(-1), T, V );
            error = norm_F( V );
        }// if
        else if (( error_type == "maximum" ) || ( error_type == "max" ))
        {
            add( value_t(-1), T, V );
            error = norm_max( V );
        }// if
        else if (( error_type == "residual" ) || ( error_type == "res" ))
        {
            // // extract eigenvalues as diag( M + Δ·V ) and eigenvectors as V
            // // (T holds new V)
            // std::vector< value_t >  E( n );

            // for ( int  i = 0; i < n; ++i )
            //     E[i] = diag_M[ i ] + dot( n, Delta + i, n, T.data() + i*n, 1 );

            // // copy diagonal back to M
            // copy( n, diag_M.data(), 1, M.data(), n+1 );
            // gemm( 'N', 'N', n, n, n, value_t(1), M.data(), n, T.data(), n, value_t(0), V.data(), n );
            // for ( int  i = 0; i < n; ++i )
            // {
            //     axpy( n, -E[i], T.data() + i*n, 1, V.data() + i*n, 1 );
            //     M[ i*n+i ] = 0.0; // reset diagonal for Delta
            // }// for
            
            // error = normF( n, n, V ) / ( M_norm * norm1( n, n, T ) );
        }// if
        else
            HLR_ERROR( "unknown error type" );

        //
        // test stop criterion
        //

        copy( T, V );

        if ( verbosity >= 1 )
        {
            std::cout << "    sweep " << sweep << " : error = " << error;

            if ( sweep > 0 )
                std::cout << ", reduction = " << error / old_error;
            
            std::cout << std::endl;
        }// if
        
        old_error = error;
        
        ++sweep;
        
        if ( error < tolerance )
            break;

        if ( ! std::isnormal( error ) )
            break;
        
    } while ( sweep < max_sweeps );

    //
    // eigenvalues  : diag( M + Δ·V )
    // eigenvectors : V
    //

    vector< value_t >  E( nrows );

    for ( size_t  i = 0; i < nrows; ++i )
    {
        auto  Delta_i = Delta.row( i );
        auto  V_i     = V.column( i );
        
        E(i) = diag_M( i ) + dot( Delta_i, V_i );
    }// for
    
    return { std::move( E ), std::move( V ) };
}

//
// compute singular value decomposition M = U S V^T of the
// nrows × ncols matrix M,
// - upon exit, M contains U
// - algorithm from "Lapack Working Notes 15"
//
template < typename value_t >
void
svd_jac ( matrix< value_t > &                                      M,
          vector< typename hpro::real_type< value_t >::type_t > &  S,
          matrix< value_t > &                                      V,
          const size_t                                             max_sweeps = 0,
          const typename hpro::real_type< value_t >::type_t        tolerance  = 0 )
{
    using  real_t = typename hpro::real_type< value_t >::type_t;
    
    const auto    nrows     = M.nrows();
    const auto    ncols     = M.ncols();
    const auto    minrc     = std::min( nrows, ncols );
    const real_t  tol       = ( tolerance > 0 ? tolerance : real_t(10) * std::numeric_limits< real_t >::epsilon() );
    bool          converged = false;
    uint          sweep     = 0;

    // initialise V with identity
    if (( V.nrows() != minrc ) || ( V.ncols() != ncols ) )
        V = std::move( matrix< value_t >( minrc, ncols ) );
    
    for ( size_t  i = 0; i < minrc; i++ )
        V(i,i) = 1.0;

    while ( ! converged and (( max_sweeps > 0 ) && ( sweep < max_sweeps )) )
    {
        sweep++;
        converged = true;
                
        for ( size_t  i = 0; i < nrows-1; i++ )
        {
            for ( size_t j = i + 1; j < ncols; j++ )
            {
                //
                // compute |a c| = (i,j) submatrix of A^T A
                //         |c b|
                //

                auto  m_i = M.column( i );
                auto  m_j = M.column( j );
                
                const auto  a = dot( m_i, m_i );
                const auto  b = dot( m_j, m_j );
                const auto  c = dot( m_i, m_j );

                if ( std::abs( c ) / std::real( std::sqrt( a*b ) ) > tol )
                    converged = false;
                        
                //
                // compute Jacobi rotation which diagonalises │a c│
                //                                            │c b│
                //

                const auto  xi = (b - a) / ( value_t(2) * c );
                const auto  t  = ( math::sign( xi ) / ( std::abs(xi) + std::sqrt( 1.0 + xi*xi ) ) );
                const auto  cs = value_t(1) / std::sqrt( 1.0 + t*t );
                const auto  sn = cs * t;

                //
                // update columns i and j of A (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  m_ki = M(k,i);
                    const auto  m_kj = M(k,j);

                    M(k,i) = cs * m_ki - sn * m_kj;
                    M(k,j) = sn * m_ki + cs * m_kj;
                }// for

                //
                // update V (apply rotation)
                //

                for ( size_t  k = 0; k < nrows; k++ )
                {
                    const auto  v_ki = V(k,i);
                    const auto  v_kj = V(k,j);

                    V(k,i) = cs * v_ki - sn * v_kj;
                    V(k,j) = sn * v_ki + cs * v_kj;
                }// for
            }// for
        }// for
    }// while

    //
    // extract singular values and update U
    //

    if ( S.length() != minrc )
        S = std::move( vector< real_t >( minrc ) );
    
    for ( size_t  i = 0; i < minrc; i++ )
    {
        auto  m_i = M.column(i);
        
        S(i) = norm2( m_i );

        if ( std::abs( S(i) ) > 1e-14 )
        {
            scale( value_t(1) / S(i), m_i );
        }// if
        else
        {
            S(i) = 0.0;
            fill( value_t(0), m_i );

            auto  v_i = V.column(i);

            fill( value_t(0), v_i );
        }// else
    }// for
}

}}// namespace hlr::blas

#endif // __HLR_ARITH_BLAS_HH
