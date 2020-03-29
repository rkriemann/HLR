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

namespace hlr { namespace blas {

//
// import functions from HLIBpro and adjust naming
//

namespace hpro = HLIB;

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
