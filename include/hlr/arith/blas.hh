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

namespace hlr
{

namespace hpro = HLIB;

//
// import into general namespace
//

using hpro::eval_side_t;
using hpro::from_left;
using hpro::from_right;

using hpro::diag_type_t;
using hpro::unit_diag;
using hpro::general_diag;

using hpro::matop_t;
using hpro::apply_normal;
using hpro::apply_conjugate;
using hpro::apply_transposed;
using hpro::apply_adjoint;

namespace blas
{

//
// import functions from HLIBpro and adjust naming
//

using namespace HLIB::BLAS;

using hpro::blas_int_t;
using range = HLIB::BLAS::Range;

template < typename value_t > using vector = HLIB::BLAS::Vector< value_t >;
template < typename value_t > using matrix = HLIB::BLAS::Matrix< value_t >;

//////////////////////////////////////////////////////////////////////
//
// template wrappers for low-rank factors as U and V
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >       blas::matrix< value_t > & mat ( hpro::TDenseMatrix *       A ) { assert( ! is_null( A ) ); return hpro::blas_mat< value_t >( A ); }
template < typename value_t > const blas::matrix< value_t > & mat ( const hpro::TDenseMatrix * A ) { assert( ! is_null( A ) ); return hpro::blas_mat< value_t >( A ); }
template < typename value_t >       blas::matrix< value_t > & mat ( hpro::TDenseMatrix &       A ) { return hpro::blas_mat< value_t >( A ); }
template < typename value_t > const blas::matrix< value_t > & mat ( const hpro::TDenseMatrix & A ) { return hpro::blas_mat< value_t >( A ); }
template < typename value_t >       blas::matrix< value_t > & mat ( std::unique_ptr< hpro::TDenseMatrix > & A ) { assert( ! is_null( A.get() ) ); return hpro::blas_mat< value_t >( *A ); }


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

template < typename value1_t,
           typename value2_t >
matrix< value2_t >
copy ( const matrix< value1_t > &  A )
{
    matrix< value2_t >  M( A.nrows(), A.ncols() );
    const size_t        n = M.nrows() * M.ncols();

    for ( size_t  i = 0; i < n; ++i )
        M.data()[i] = value2_t( A.data()[i] );

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
       T_fill_fn &                      fill_fn )
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
fill ( blas::MatrixBase< T_matrix > &  M,
       T_fill_fn &                     fill_fn )
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

}}// namespace hlr::blas

#endif // __HLR_ARITH_BLAS_HH
