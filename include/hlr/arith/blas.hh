#ifndef __HLR_ARITH_BLAS_HH
#define __HLR_ARITH_BLAS_HH
//
// Project     : HLR
// Module      : arith/blas
// Description : basic linear algebra functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/blas/Matrix.hh>
#include <hpro/blas/Vector.hh>
#include <hpro/blas/Algebra.hh>

#include <hlr/utils/log.hh>

namespace hlr { namespace blas {

namespace hpro = HLIB;

using hpro::blas_int_t;

//
// definition of ormqr/unmqr functions
//
extern "C"
{

void sgeqr_ ( const blas_int_t *  n,
              const blas_int_t *  m,
              float *             A,
              const blas_int_t *  ldA,
              float *             T,
              const blas_int_t *  tsize,
              float *             work,
              const blas_int_t *  lwork,
              blas_int_t *        info );

void dgeqr_ ( const blas_int_t *  n,
              const blas_int_t *  m,
              double *            A,
              const blas_int_t *  ldA,
              double *            T,
              const blas_int_t *  tsize,
              double *            work,
              const blas_int_t *  lwork,
              blas_int_t *        info );

void cgeqr_ ( const blas_int_t *         n,
              const blas_int_t *         m,
              hpro::Complex< float > *   A,
              const blas_int_t *         ldA,
              hpro::Complex< float > *   T,
              const blas_int_t *         tsize,
              hpro::Complex< float > *   work,
              const blas_int_t *         lwork,
              blas_int_t *               info );

void zgeqr_ ( const blas_int_t *         n,
              const blas_int_t *         m,
              hpro::Complex< double > *  A,
              const blas_int_t *         ldA,
              hpro::Complex< double > *  T,
              const blas_int_t *         tsize,
              hpro::Complex< double > *  work,
              const blas_int_t *         lwork,
              blas_int_t *               info );


void sgemqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const float *       A,
               const blas_int_t *  ldA,
               const float *       T,
               const blas_int_t *  tsize,
               float *             C,
               const blas_int_t *  ldC,
               float *             work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void dgemqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const double *      A,
               const blas_int_t *  ldA,
               const double *      T,
               const blas_int_t *  tsize,
               double *            C,
               const blas_int_t *  ldC,
               double *            work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void cgemqr_ ( const char *                    side,
               const char *                    trans,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const blas_int_t *              k,
               const hpro::Complex< float > *  A,
               const blas_int_t *              ldA,
               const hpro::Complex< float > *  T,
               const blas_int_t *              tsize,
               hpro::Complex< float > *        C,
               const blas_int_t *              ldC,
               hpro::Complex< float > *        work,
               const blas_int_t *              lwork,
               blas_int_t *                    info );

void zgemqr_ ( const char *                    side,
               const char *                    trans,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const blas_int_t *              k,
               const hpro::Complex< double > * A,
               const blas_int_t *              ldA,
               const hpro::Complex< double > * T,
               const blas_int_t *              tsize,
               hpro::Complex< double > *       C,
               const blas_int_t *              ldC,
               hpro::Complex< double > *       work,
               const blas_int_t *              lwork,
               blas_int_t *                    info );


void sormqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const float *       A,
               const blas_int_t *  ldA,
               const float *       tau,
               float *             C,
               const blas_int_t *  ldC,
               float *             work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void dormqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const double *      A,
               const blas_int_t *  ldA,
               const double *      tau,
               double *            C,
               const blas_int_t *  ldC,
               double *            work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void cunmqr_ ( const char *                     side,
               const char *                     trans,
               const blas_int_t *               n,
               const blas_int_t *               m,
               const blas_int_t *               k,
               const hpro::Complex< float > *   A,
               const blas_int_t *               ldA,
               const hpro::Complex< float > *   tau,
               hpro::Complex< float > *         C,
               const blas_int_t *               ldC,
               hpro::Complex< float > *         work,
               const blas_int_t *               lwork,
               blas_int_t *                     info );

void zunmqr_ ( const char *                     side,
               const char *                     trans,
               const blas_int_t *               n,
               const blas_int_t *               m,
               const blas_int_t *               k,
               const hpro::Complex< double > *  A,
               const blas_int_t *               ldA,
               const hpro::Complex< double > *  tau,
               hpro::Complex< double > *        C,
               const blas_int_t *               ldC,
               hpro::Complex< double > *        work,
               const blas_int_t *               lwork,
               blas_int_t *                     info );

}

//
// import functions from HLIBpro and adjust naming
//

using namespace HLIB::BLAS;

using range = HLIB::BLAS::Range;

template < typename value_t > using vector = HLIB::BLAS::Vector< value_t >;
template < typename value_t > using matrix = HLIB::BLAS::Matrix< value_t >;

//
// general copy method
//
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

//
// various fill methods
//
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

namespace detail
{

#define HLR_BLAS_GEQR( type, func )                                     \
    inline                                                              \
    void geqr ( const blas_int_t  n,                                    \
                const blas_int_t  m,                                    \
                type *            A,                                    \
                const blas_int_t  ldA,                                  \
                type *            T,                                    \
                const blas_int_t  tsize,                                \
                type *            work,                                 \
                const blas_int_t  lwork,                                \
                blas_int_t &      info )                                \
    {                                                                   \
        func( & n, & m, A, & ldA, T, & tsize, work, & lwork, & info );  \
    }

HLR_BLAS_GEQR( float,                   sgeqr_ )
HLR_BLAS_GEQR( double,                  dgeqr_ )
HLR_BLAS_GEQR( hpro::Complex< float >,  cgeqr_ )
HLR_BLAS_GEQR( hpro::Complex< double >, zgeqr_ )

#undef HLR_BLAS_GEQR

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
    
    std::vector< value_t >  work( blas_int_t( hpro::re( work_query ) ) );
              
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

    detail::geqr( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), t_query, -1, & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to geqr failed" );
    
    std::vector< value_t >  work( blas_int_t( hpro::re( work_query ) ) );

    T.resize( blas_int_t( hpro::re( t_query[0] ) ) );
              
    //
    // compute QR
    //

    detail::geqr( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), T.size(), work.data(), work.size(), info );
    
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

namespace detail
{

#define HLR_BLAS_ORMQR( type, func )                                    \
    inline                                                              \
    void ormqr ( const char        side,                                \
                 const char        trans,                               \
                 const blas_int_t  n,                                   \
                 const blas_int_t  m,                                   \
                 const blas_int_t  k,                                   \
                 const type *      A,                                   \
                 const blas_int_t  ldA,                                 \
                 const type *      tau,                                 \
                 type *            C,                                   \
                 const blas_int_t  ldC,                                 \
                 type *            work,                                \
                 const blas_int_t  lwork,                               \
                 blas_int_t &      info )                               \
    {                                                                   \
        func( & side, & trans, & n, & m, & k, A, & ldA, tau, C, & ldC, work, & lwork, & info ); \
    }

HLR_BLAS_ORMQR( float,                   sormqr_ )
HLR_BLAS_ORMQR( double,                  dormqr_ )
HLR_BLAS_ORMQR( hpro::Complex< float >,  cunmqr_ )
HLR_BLAS_ORMQR( hpro::Complex< double >, zunmqr_ )

#undef HLR_BLAS_ORMQR

#define HLR_BLAS_GEMQR( type, func )                                    \
    inline                                                              \
    void gemqr ( const char        side,                                \
                 const char        trans,                               \
                 const blas_int_t  n,                                   \
                 const blas_int_t  m,                                   \
                 const blas_int_t  k,                                   \
                 const type *      A,                                   \
                 const blas_int_t  ldA,                                 \
                 const type *      T,                                   \
                 const blas_int_t  tsize,                               \
                 type *            C,                                   \
                 const blas_int_t  ldC,                                 \
                 type *            work,                                \
                 const blas_int_t  lwork,                               \
                 blas_int_t &      info )                               \
    {                                                                   \
        func( & side, & trans, & n, & m, & k, A, & ldA, T, & tsize, C, & ldC, work, & lwork, & info ); \
    }

HLR_BLAS_GEMQR( float,                   sgemqr_ )
HLR_BLAS_GEMQR( double,                  dgemqr_ )
HLR_BLAS_GEMQR( hpro::Complex< float >,  cgemqr_ )
HLR_BLAS_GEMQR( hpro::Complex< double >, zgemqr_ )

#undef HLR_BLAS_GEMQR

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

    detail::ormqr( char(side), op, nrows, ncols, k,
                   Q.data(), blas_int_t( Q.col_stride() ), T.data(),
                   M.data(), blas_int_t( M.col_stride() ),
                   & work_query, -1, info );

    // detail::gemqr( char(side), op, nrows, ncols, k,
    //                Q.data(), blas_int_t( Q.col_stride() ), T.data(), T.size(),
    //                M.data(), blas_int_t( M.col_stride() ),
    //                & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to gemqr failed" );
    
    std::vector< value_t >  work( blas_int_t( hpro::re( work_query ) ) );

    //
    // multiply with Q
    //

    detail::ormqr( char(side), op, nrows, ncols, k,
                   Q.data(), blas_int_t( Q.col_stride() ), T.data(),
                   M.data(), blas_int_t( M.col_stride() ),
                   work.data(), work.size(), info );

    // detail::gemqr( char(side), op, nrows, ncols, k,
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
    
    std::vector< value_t >  work( blas_int_t( hpro::re( work_query ) ) );

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
