#ifndef __HLR_ARITH_BLAS_HH
#define __HLR_ARITH_BLAS_HH
//
// Project     : HLR
// Module      : arith/blas
// Description : basic linear algebra functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cassert>
#include <type_traits>
#include <variant>

#include <hpro/blas/Matrix.hh>
#include <hpro/blas/Vector.hh>
#include <hpro/blas/Algebra.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/utils/traits.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/math.hh>
#include <hlr/arith/blas_def.hh>

namespace hlr
{

//
// import into general namespace
//

using Hpro::eval_side_t;
using Hpro::from_left;
using Hpro::from_right;

using Hpro::diag_type_t;
using Hpro::unit_diag;
using Hpro::general_diag;

using Hpro::tri_type_t;
using Hpro::lower_triangular;
using Hpro::upper_triangular;

using Hpro::matop_t;
using Hpro::apply_normal;
using Hpro::apply_conjugate;
using Hpro::apply_transposed;
using Hpro::apply_adjoint;

// to print out update statistics in approximation functions (used in external script)
#define HLR_APPROX_RANK_STAT( msg ) // std::cout << msg << std::endl

namespace blas
{

//
// import functions from HLIBpro and adjust naming
//

using namespace Hpro::BLAS;

using Hpro::blas_int_t;
using range = Hpro::BLAS::Range;

template < typename value_t > using vector = Hpro::BLAS::Vector< value_t >;
template < typename value_t > using matrix = Hpro::BLAS::Matrix< value_t >;

template < typename type_t > inline constexpr bool is_vector_v = is_vector< type_t >::value;
template < typename type_t > inline constexpr bool is_matrix_v = is_matrix< type_t >::value;

template < typename T > concept  vector_type = is_vector_v< T >;
template < typename T > concept  matrix_type = is_matrix_v< T >;

//////////////////////////////////////////////////////////////////////
//
// template wrappers for vectors, matrices and
// low-rank factors as U and V
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >       vector< value_t > & vec ( Hpro::TScalarVector< value_t > *       v ) { HLR_DBG_ASSERT( ! is_null( v ) ); return v->blas_vec(); }
template < typename value_t > const vector< value_t > & vec ( const Hpro::TScalarVector< value_t > * v ) { HLR_DBG_ASSERT( ! is_null( v ) ); return v->blas_vec(); }
template < typename value_t >       vector< value_t > & vec ( Hpro::TScalarVector< value_t > &       v ) { return v.blas_vec(); }
template < typename value_t > const vector< value_t > & vec ( const Hpro::TScalarVector< value_t > & v ) { return v.blas_vec(); }
template < typename value_t >       vector< value_t > & vec ( std::unique_ptr< Hpro::TScalarVector< value_t > > & v ) { HLR_DBG_ASSERT( ! is_null( v.get() ) ); return v->blas_vec(); }

template < typename value_t >       matrix< value_t > & mat ( Hpro::TDenseMatrix< value_t > *       A ) { HLR_DBG_ASSERT( ! is_null( A ) ); return A->blas_mat(); }
template < typename value_t > const matrix< value_t > & mat ( const Hpro::TDenseMatrix< value_t > * A ) { HLR_DBG_ASSERT( ! is_null( A ) ); return A->blas_mat(); }
template < typename value_t >       matrix< value_t > & mat ( Hpro::TDenseMatrix< value_t > &       A ) { return A.blas_mat(); }
template < typename value_t > const matrix< value_t > & mat ( const Hpro::TDenseMatrix< value_t > & A ) { return A.blas_mat(); }
template < typename value_t >       matrix< value_t > & mat ( std::unique_ptr< Hpro::TDenseMatrix< value_t > > & A ) { HLR_DBG_ASSERT( ! is_null( A.get() ) ); return A->blas_mat(); }


template < typename value_t >
matrix< value_t > &
mat_U ( Hpro::TRkMatrix< value_t > *  A )
{
    HLR_DBG_ASSERT( ! is_null( A ) );
    return A->blas_mat_A();
}

template < typename value_t >
matrix< value_t > &
mat_U ( Hpro::TRkMatrix< value_t > *  A,
        const Hpro::matop_t           op )
{
    HLR_DBG_ASSERT( ! is_null( A ) );

    if ( op == Hpro::apply_normal )
        return A->blas_mat_A();
    else
        return A->blas_mat_B();
}

template < typename value_t >
matrix< value_t > &
mat_V ( Hpro::TRkMatrix< value_t > *  A )
{
    HLR_DBG_ASSERT( ! is_null( A ) );
    return A->blas_mat_B();
}

template < typename value_t >
matrix< value_t > &
mat_V ( Hpro::TRkMatrix< value_t > *    A,
        const Hpro::matop_t             op )
{
    HLR_DBG_ASSERT( ! is_null( A ) );

    if ( op == Hpro::apply_normal )
        return A->blas_mat_B();
    else
        return A->blas_mat_A();
}

template < typename value_t >
const matrix< value_t > &
mat_U ( const Hpro::TRkMatrix< value_t > *  A )
{
    HLR_DBG_ASSERT( ! is_null( A ) );
    return A->blas_mat_A();
}

template < typename value_t >
const matrix< value_t > &
mat_U ( const Hpro::TRkMatrix< value_t > *  A,
        const Hpro::matop_t                 op )
{
    HLR_DBG_ASSERT( ! is_null( A ) );

    if ( op == Hpro::apply_normal )
        return A->blas_mat_A();
    else
        return A->blas_mat_B();
}

template < typename value_t >
const matrix< value_t > &
mat_V ( const Hpro::TRkMatrix< value_t > *  A )
{
    HLR_DBG_ASSERT( ! is_null( A ) );
    return A->blas_mat_B();
}

template < typename value_t >
const matrix< value_t > &
mat_V ( const Hpro::TRkMatrix< value_t > *  A,
        const Hpro::matop_t                 op )
{
    HLR_DBG_ASSERT( ! is_null( A ) );

    if ( op == Hpro::apply_normal )
        return A->blas_mat_B();
    else
        return A->blas_mat_A();
}

template < typename value_t >
matrix< value_t > &
mat_U ( Hpro::TRkMatrix< value_t > &  A )
{
    return mat_U< value_t >( & A );
}

template < typename value_t >
matrix< value_t > &
mat_U ( Hpro::TRkMatrix< value_t > &  A,
        const Hpro::matop_t  op )
{
    return mat_U< value_t >( & A, op );
}

template < typename value_t >
matrix< value_t > &
mat_V ( Hpro::TRkMatrix< value_t > &  A )
{
    return mat_V< value_t >( & A );
}

template < typename value_t >
matrix< value_t > &
mat_V ( Hpro::TRkMatrix< value_t > &  A,
        const Hpro::matop_t           op )
{
    return mat_V< value_t >( & A, op );
}

template < typename value_t >
const matrix< value_t > &
mat_U ( const Hpro::TRkMatrix< value_t > &  A )
{
    return mat_U< value_t >( & A );
}

template < typename value_t >
const matrix< value_t > &
mat_U ( const Hpro::TRkMatrix< value_t > &  A,
        const Hpro::matop_t                 op )
{
    return mat_U< value_t >( & A, op );
}

template < typename value_t >
const matrix< value_t > &
mat_V ( const Hpro::TRkMatrix< value_t > &  A )
{
    return mat_V< value_t >( & A );
}

template < typename value_t >
const matrix< value_t > &
mat_V ( const Hpro::TRkMatrix< value_t > &  A,
        const Hpro::matop_t                 op )
{
    return mat_V< value_t >( & A, op );
}

template < typename value_t >
matrix< value_t > &
mat_U ( std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A )
{
    return mat_U< value_t >( *A );
}

template < typename value_t >
const matrix< value_t > &
mat_V ( std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A )
{
    return mat_V< value_t >( *A );
}

template < typename value_t >
matrix< value_t > &
mat_U ( std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A,
        const Hpro::matop_t                              op )
{
    return mat_U< value_t >( *A, op );
}

template < typename value_t >
const matrix< value_t > &
mat_V (  std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A,
         const Hpro::matop_t                              op )
{
    return mat_V< value_t >( *A, op );
}

template < typename value_t >
matrix< value_t > &
mat_U ( const std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A,
        const Hpro::matop_t                                    op )
{
    return mat_U< value_t >( *A, op );
}

template < typename value_t >
const matrix< value_t > &
mat_V ( const std::unique_ptr< Hpro::TRkMatrix< value_t > > &  A,
        const Hpro::matop_t                                    op )
{
    return mat_V< value_t >( *A, op );
}

//////////////////////////////////////////////////////////////////////
//
// general helpers
//
//////////////////////////////////////////////////////////////////////

//
// print matrix
//
void
print ( const matrix_type auto &  M,
        std::ostream &            out = std::cout )
{
    for ( uint  i = 0; i < M.nrows(); ++i )
    {
        for ( uint  j = 0; j < M.ncols(); ++j )
            out << M( i, j ) << ", ";

        out << std::endl;
    }// for

    out << std::endl;
}

//
// create identity matrix
//
template < typename value_t >
matrix< value_t >
eye ( const size_t  nrows,
      const size_t  ncols )
{
    auto  I = matrix< value_t >( nrows, ncols );

    for ( size_t  i = 0; i < std::min( nrows, ncols ); ++i )
        I(i,i) = value_t(1);

    return I;
}

//
// create null matrix
//
template < typename value_t >
matrix< value_t >
zeros ( const size_t  nrows,
        const size_t  ncols )
{
    return matrix< value_t >( nrows, ncols );
}

//
// extend given matrix M by nrows × ncols, e.g., resulting matrix
// has dimensions nrows(M) + nrows × ncols(M) + ncols
//
template < typename value_t >
matrix< value_t >
extend ( const matrix< value_t > &  M,
         const size_t               nrows,
         const size_t               ncols )
{
    auto  T  = matrix< value_t >( M.nrows() + nrows, M.ncols() + ncols );
    auto  TM = matrix< value_t >( T, range( 0, M.nrows()-1 ), range( 0, M.ncols()-1 ) );

    copy( M, TM );

    return T;
}

//
// join given matrices M_i row-wise, e.g., return [ M_0, M_1, ..., M_n-1 ]
//
template < typename value_t >
matrix< value_t >
join_row ( const std::list< matrix< value_t > > &  matrices )
{
    //
    // determine dimension of result
    //

    size_t  nrows = 0;
    size_t  ncols = 0;

    for ( auto  M_i : matrices )
    {
        if ( nrows == 0 )
            nrows = M_i.nrows();
        else
            HLR_DBG_ASSERT( nrows == M_i.nrows() );

        ncols += M_i.ncols();
    }// for

    //
    // put all matrices together
    //

    auto    M   = matrix< value_t >( nrows, ncols );
    size_t  pos = 0;
    
    for ( auto  M_i : matrices )
    {
        const auto  ncols_i = M_i.ncols();
        auto        dest_i  = matrix< value_t >( M, range::all, range( pos, pos + ncols_i - 1 ) );

        copy( M_i, dest_i );
        pos += ncols_i;
    }// for

    return M;
}

//
// join given matrices M_i column-wise, e.g., return [ M_0; M_1; ..., M_n-1 ]
//
template < typename value_t >
matrix< value_t >
join_col ( const std::list< matrix< value_t > > &  matrices )
{
    //
    // determine dimension of result
    //

    size_t  nrows = 0;
    size_t  ncols = 0;

    for ( auto  M_i : matrices )
    {
        if ( ncols == 0 )
            ncols = M_i.ncols();
        else if ( ncols != M_i.ncols() )
            HLR_ERROR( "matrices have different column sizes" );

        nrows += M_i.nrows();
    }// for

    //
    // put all matrices together
    //

    auto    M   = matrix< value_t >( nrows, ncols );
    size_t  pos = 0;
    
    for ( auto  M_i : matrices )
    {
        const auto  nrows_i = M_i.nrows();
        auto        dest_i  = matrix< value_t >( M, range( pos, pos + nrows_i - 1 ), range::all );

        copy( M_i, dest_i );
        pos += nrows_i;
    }// for

    return M;
}

//
// convert given vector into diagonal matrix
//
template < typename valueM_t,
           typename valueD_t >
matrix< valueM_t >
diag ( const vector< valueD_t > &  d )
{
    const auto  n = d.length();
    auto        M = matrix< valueM_t >( n, n );

    for ( size_t  i = 0; i < n; ++i )
        M(i,i) = d(i);
    
    return M;
}

//
// construct block-diagonal matrix out of given matrices M_i
//
template < typename value_t >
matrix< value_t >
diag ( const std::list< matrix< value_t > > &  matrices )
{
    //
    // determine dimension of result
    //

    size_t  nrows = 0;
    size_t  ncols = 0;

    for ( auto  M_i : matrices )
    {
        nrows += M_i.nrows();
        ncols += M_i.ncols();
    }// for

    //
    // put all matrices together
    //

    auto    M     = matrix< value_t >( nrows, ncols );
    size_t  pos_r = 0;
    size_t  pos_c = 0;
    
    for ( auto  M_i : matrices )
    {
        const auto  nrows_i = M_i.nrows();
        const auto  ncols_i = M_i.ncols();
        auto        dest_i  = matrix< value_t >( M,
                                                 range( pos_r, pos_r + nrows_i - 1 ),
                                                 range( pos_c, pos_c + ncols_i - 1 ) );

        copy( M_i, dest_i );
        pos_r += nrows_i;
        pos_c += ncols_i;
    }// for

    return M;
}

//////////////////////////////////////////////////////////////////////
//
// general copy method
//
//////////////////////////////////////////////////////////////////////

template < vector_type T_vector >
vector< typename T_vector::value_t >
copy ( const T_vector &  v )
{
    using  value_t = typename T_vector::value_t;

    vector< value_t >  w( v.length() );

    Hpro::BLAS::copy( v, w );

    return w;
}

template < typename value_dest_t,
           typename value_src_t >
requires ( std::convertible_to< value_src_t, value_dest_t > )
vector< value_dest_t >
copy ( const vector< value_src_t > &  v )
{
    const size_t            n = v.length();
    vector< value_dest_t >  x( n );

    for ( size_t  i = 0; i < n; ++i )
        x(i) = value_dest_t( v(i) );

    return x;
}

template < matrix_type T_matrix >
matrix< typename T_matrix::value_t >
copy ( const T_matrix &  A )
{
    using  value_t = typename T_matrix::value_t;

    matrix< value_t >  M( A.nrows(), A.ncols() );

    Hpro::BLAS::copy( A, M );

    return M;
}

template < typename value_dest_t,
           typename value_src_t >
requires ( std::convertible_to< value_src_t, value_dest_t > )
matrix< value_dest_t >
copy ( const matrix< value_src_t > &  A )
{
    matrix< value_dest_t >  M( A.nrows(), A.ncols() );
    const size_t            n = M.nrows() * M.ncols();

    for ( size_t  i = 0; i < n; ++i )
        M.data()[i] = value_dest_t( A.data()[i] );

    return M;
}

template < typename value_src_t,
           typename value_dest_t >
requires ( std::convertible_to< value_src_t, value_dest_t > )
void
copy ( const matrix< value_src_t > &   A,
       const matrix< value_dest_t > &  B )
{
    HLR_DBG_ASSERT(( A.nrows() == B.nrows() ) && ( A.ncols() == B.ncols() ));
    
    const size_t  n = A.nrows() * A.ncols();

    for ( size_t  i = 0; i < n; ++i )
        B.data()[i] = value_dest_t( A.data()[i] );
}

using Hpro::BLAS::copy;

//////////////////////////////////////////////////////////////////////
//
// various fill methods
//
//////////////////////////////////////////////////////////////////////

template < typename T_vector,
           typename T_value >
void
fill ( VectorBase< T_vector > &  v,
       const T_value             f )
{
    using value_v_t = typename T_vector::value_t;
    
    for ( size_t  i = 0; i < v.length(); ++i )
        v(i) = value_v_t(f);
}
       
template < typename T_matrix,
           typename T_value >
void
fill ( MatrixBase< T_matrix > &    M,
       const T_value               f )
{
    using value_M_t = typename T_matrix::value_t;
    
    for ( size_t  i = 0; i < M.nrows(); ++i )
        for ( size_t  j = 0; j < M.ncols(); ++j )
            M(i,j) = value_M_t(f);
}

template < typename T_vector,
           typename T_func >
void
fill_fn ( VectorBase< T_vector > &   v,
          T_func &&                  fill_fn )
{
    for ( size_t  i = 0; i < v.length(); ++i )
        v(i) = fill_fn();
}
       
template < typename T_matrix,
           typename T_func >
void
fill_fn ( MatrixBase< T_matrix > &  M,
          T_func &&                 func )
{
    for ( size_t  i = 0; i < M.nrows(); ++i )
        for ( size_t  j = 0; j < M.ncols(); ++j )
            M(i,j) = func();
}
       
//
// determine maximal absolute value in M
//
template < matrix_type  matrix_t >
typename matrix_t::value_t
max_abs_val ( const matrix_t &  M )
{
    HLR_DBG_ASSERT( M.nrows() * M.ncols() > 0 );

    // todo
    HLR_ASSERT(( M.col_stride() == M.nrows() ) &&
                   ( M.row_stride() == 1 ));
        
    const auto  res = max_idx( blas_int_t(M.nrows() * M.ncols()), M.data(), 1 )-1;

    return std::abs( M.data()[res] );
}

//
// determine minimal absolute value in M
//
template < matrix_type  matrix_t >
typename matrix_t::value_t
min_abs_val ( const matrix_t &  M )
{
    HLR_DBG_ASSERT( M.nrows() * M.ncols() > 0 );

    // todo
    HLR_ASSERT(( M.col_stride() == M.nrows() ) &&
               ( M.row_stride() == 1 ));

    using  value_t = typename matrix_t::value_t;
    using  real_t  = Hpro::real_type_t< value_t >;
    
    value_t *     ptr  = M.data();
    real_t        vmin = std::abs( ptr[0] );
    const size_t  n    = M.nrows() * M.ncols();
    
    for ( size_t  i = 0; i < n; ++i )
        vmin = std::min( vmin, std::abs( ptr[i] ) );

    return vmin;
}

//////////////////////////////////////////////////////////////////////
//
// norm computations
//
//////////////////////////////////////////////////////////////////////

#define  HLR_BLAS_NORM1( type, func )                                   \
    inline                                                              \
    typename Hpro::real_type_t< type >                                  \
    norm_1 ( const matrix< type > &  M )                                \
    {                                                                   \
        typename Hpro::real_type_t< type >  work = 0;                   \
        const blas_int_t                    nrows = M.nrows();          \
        const blas_int_t                    ncols = M.ncols();          \
        const blas_int_t                    ldM   = M.col_stride();     \
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
    typename Hpro::real_type_t< type >                                  \
    norm_inf ( const matrix< type > &  M )                              \
    {                                                                   \
        typename Hpro::real_type_t< type >  work = 0;                   \
        const blas_int_t                    nrows = M.nrows();          \
        const blas_int_t                    ncols = M.ncols();          \
        const blas_int_t                    ldM   = M.col_stride();     \
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
    typename Hpro::real_type_t< type >                                  \
    norm_max ( const matrix< type > &  M )                              \
    {                                                                   \
        typename Hpro::real_type_t< type >  work = 0;                   \
        const blas_int_t                    nrows = M.nrows();          \
        const blas_int_t                    ncols = M.ncols();          \
        const blas_int_t                    ldM   = M.col_stride();     \
                                                                        \
        return func( "M", & nrows, & ncols, M.data(), & ldM, & work );  \
    }

HLR_BLAS_NORMM( float,                  slange_ )
HLR_BLAS_NORMM( double,                 dlange_ )
HLR_BLAS_NORMM( std::complex< float >,  clange_ )
HLR_BLAS_NORMM( std::complex< double >, zlange_ )
#undef HLR_BLAS_NORMM

//
// Frobenius norm for lowrank matrices
//
template < typename value_t >
long double
sqnorm_F ( const matrix< value_t > &  U,
           const matrix< value_t > &  V )
{
    //
    // ∑_ij (M_ij)² = ∑_ij (∑_k u_ik v_jk')²
    //              = ∑_ij (∑_k u_ik v_jk') (∑_l u_il v_jl')'
    //              = ∑_ij ∑_k ∑_l u_ik v_jk' u_il' v_jl
    //              = ∑_k ∑_l ∑_i u_ik u_il' ∑_j v_jk' v_jl
    //              = ∑_k ∑_l (u_l)^H · u_k  v_k^H · v_l
    //
    
    long double  res = 0;
    
    for ( size_t  k = 0; k < U.ncols(); k++ )
    {
        auto  u_k = U.column( k );
        auto  v_k = V.column( k );
                
        for ( size_t  l = 0; l < V.ncols(); l++ )
        {
            auto  u_l = U.column( l );
            auto  v_l = V.column( l );

            res += std::abs( dot( u_k, u_l ) * dot( v_k, v_l ) );
        }// for
    }// for

    return math::abs( res );
}

template < typename value_t >
typename Hpro::real_type_t< value_t >
norm_F ( const matrix< value_t > &  U,
         const matrix< value_t > &  V )
{
    return math::sqrt( sqnorm_F( U, V ) );
}

// make sure, standard norm_F is found
using Hpro::BLAS::norm_F;

//////////////////////////////////////////////////////////////////////
//
// enhanced versions regarding scalar parameters
//
//////////////////////////////////////////////////////////////////////

template < typename     alpha_t,
           vector_type  vector_t >
requires ( std::convertible_to< alpha_t, typename vector_t::value_t > )
void
scale ( const alpha_t  alpha,
        vector_t &     v )
{
    const auto  n = Hpro::idx_t(v.length());
    
    for ( Hpro::idx_t  i = 0; i < n; ++i )
        v(i) *= alpha;
}

template < typename     alpha_t,
           matrix_type  matrix_t >
requires ( std::convertible_to< alpha_t, typename matrix_t::value_t > )
void
scale ( const alpha_t  alpha,
        matrix_t &     M )
{
    const auto  n = Hpro::idx_t(M.nrows());
    const auto  m = Hpro::idx_t(M.ncols());
    
    for ( Hpro::idx_t  j = 0; j < m; ++j )
        for ( Hpro::idx_t  i = 0; i < n; ++i )
            M(i,j) *= alpha;
}

//////////////////////////////////////////////////////////////////////
//
// various simplified forms of matrix addition, multiplication
//
//////////////////////////////////////////////////////////////////////

void
add ( const auto                alpha,
      const vector_type auto &  x,
      vector_type auto &        y )
{
    using value_t = typename std::decay_t< decltype(x) >::value_t;
    
    Hpro::BLAS::add( value_t( alpha ), x, y );
}

void
add ( const auto                alpha,
      const matrix_type auto &  A,
      matrix_type auto &        B )
{
    using value_t = typename std::decay_t< decltype(A) >::value_t;
    
    Hpro::BLAS::add( value_t( alpha ), A, B );
}

template < matrix_type matrixA_t,
           vector_type vectorX_t >
requires ( std::same_as< typename matrixA_t::value_t, typename vectorX_t::value_t > )
vector< typename matrixA_t::value_t >
mulvec ( const matrixA_t &  A,
         const vectorX_t &  x )
{
    HLR_DBG_ASSERT( A.ncols() == x.length() );
    
    return Hpro::BLAS::mulvec( typename matrixA_t::value_t(1), A, x );
}

template < matrix_type matrixA_t,
           vector_type vectorX_t,
           vector_type vectorY_t >
requires ( std::same_as< typename matrixA_t::value_t, typename vectorX_t::value_t > &&
           std::same_as< typename matrixA_t::value_t, typename vectorY_t::value_t > )
void
mulvec ( const matrixA_t &  A,
         const vectorX_t &  x,
         vectorY_t &        y )
{
    HLR_DBG_ASSERT( A.ncols() == x.length() );
    HLR_DBG_ASSERT( A.nrows() == y.length() );
    
    Hpro::BLAS::mulvec( typename matrixA_t::value_t(1), A, x, typename matrixA_t::value_t(1), y );
}

using Hpro::BLAS::mulvec;

//
// compute op(U·V')·x = y
//
template < typename T_alpha,
           typename T_value >
requires ( std::convertible_to< T_alpha, T_value > )
void
mulvec_lr ( const T_alpha                    alpha,
            const blas::matrix< T_value > &  U,
            const blas::matrix< T_value > &  V,
            const matop_t                    op,
            const blas::vector< T_value > &  x,
            blas::vector< T_value > &        y )
{
    using  value_t = T_value;
    
    HLR_DBG_ASSERT( U.ncols() == V.ncols() );

    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·V^H x
        //
        
        // t := V^H x
        auto  t = blas::mulvec( blas::adjoint( V ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // y := y + U t
        blas::mulvec( U, t, y );
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·V^H)^T x
        //   = y + conj(V)·U^T x
        //
        
        // t := U^T x
        auto  t = blas::mulvec( blas::transposed( U ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // r := conj(V) t = conj( V · conj(t) )
        blas::conj( t );
            
        auto  r = blas::mulvec( V, t );

        blas::conj( r );

        // y = y + r
        blas::add( value_t(1), r, y );
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·V^H)^H x
        //   = y + V·U^H x
        //
        
        // t := U^H x
        auto  t = blas::mulvec( blas::adjoint( U ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // y := t + V t
        blas::mulvec( V, t, y );
    }// if
}

//
// compute op(U·S·V')·x = y
//
template < typename T_alpha,
           typename T_value >
requires ( std::convertible_to< T_alpha, T_value > )
void
mulvec_lr ( const T_alpha                    alpha,
            const blas::matrix< T_value > &  U,
            const blas::matrix< T_value > &  S,
            const blas::matrix< T_value > &  V,
            const matop_t                    op,
            const blas::vector< T_value > &  x,
            blas::vector< T_value > &        y )
{
    using  value_t = T_value;
    
    HLR_DBG_ASSERT( U.ncols() == S.nrows() );
    HLR_DBG_ASSERT( V.ncols() == S.ncols() );
    
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·S·V^H x
        //
        
        // t := V^H x
        auto  t = blas::mulvec( blas::adjoint( V ), x );

        // s := α S t
        auto  s = blas::mulvec( alpha, S, t );

        // y := y + U s
        blas::mulvec( U, s, y );
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·S·V^H)^T x
        //   = y + conj(V)·S^T·U^T x
        //
        
        // t := U^T x
        auto  t = blas::mulvec( blas::transposed( U ), x );

        // s := α S^T t
        auto  s = blas::mulvec( alpha, blas::transposed( S ), t );
        
        // r := conj(V) s = conj( V · conj(s) )
        blas::conj( s );
            
        auto  r = blas::mulvec( V, s );

        blas::conj( r );

        // y = y + r
        blas::add( value_t(1), r, y );
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·S·V^H)^H x
        //   = y + V·S^H·U^H x
        //
        
        // t := U^H x
        auto  t = blas::mulvec( blas::adjoint( U ), x );

        // s := α S^T t
        auto  s = blas::mulvec( alpha, blas::adjoint( S ), t );
        
        // y := t + V t
        blas::mulvec( V, s, y );
    }// if
}

//
// compute op(U·S·V')·x = y with diagonal S
//
template < typename T_alpha,
           typename T_value,
           typename T_value_S >
requires ( std::convertible_to< T_value_S, T_value > && std::convertible_to< T_alpha, T_value > )
void
mulvec_lr ( const T_alpha                      alpha,
            const blas::matrix< T_value > &    U,
            const blas::vector< T_value_S > &  S,
            const blas::matrix< T_value > &    V,
            const matop_t                      op,
            const blas::vector< T_value > &    x,
            blas::vector< T_value > &          y )
{
    using  value_t = T_value;

    HLR_DBG_ASSERT( U.ncols() == S.length() );
    HLR_DBG_ASSERT( V.ncols() == S.length() );
    
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + U·S·V^H x
        //
        
        // t := V^H x
        auto  t = blas::mulvec( blas::adjoint( V ), x );

        // t := α S t
        for ( uint  i = 0; i < t.length(); ++i )
            t(i) *= alpha * S(i);
        
        // y := y + U s
        blas::mulvec( U, t, y );
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·S·V^H)^T x
        //   = y + conj(V)·S^T·U^T x
        //
        
        // t := U^T x
        auto  t = blas::mulvec( blas::transposed( U ), x );

        // t := conj( α S^T t )
        for ( uint  i = 0; i < t.length(); ++i )
            t(i) = math::conj( alpha * S(i) * t(i) );
        
        // r := conj(V) t = conj( V · conj(t) )  (conj(t) already before)
        auto  r = blas::mulvec( V, t );

        blas::conj( r );

        // y = y + r
        blas::add( value_t(1), r, y );
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + (U·S·V^H)^H x
        //   = y + V·S^H·U^H x
        //
        
        // t := U^H x
        auto  t = blas::mulvec( blas::adjoint( U ), x );

        // s := α S^H t
        for ( uint  i = 0; i < t.length(); ++i )
            t(i) *= alpha * math::conj( S(i) );
        
        // y := t + V t
        blas::mulvec( V, t, y );
    }// if
}

//
// simplifications for default BLAS::prod
//
template < typename T_beta,
           matrix_type matrixA_t,
           matrix_type matrixB_t,
           matrix_type matrixC_t >
requires ( std::same_as< typename matrixA_t::value_t, typename matrixB_t::value_t > &&
           std::same_as< typename matrixA_t::value_t, typename matrixC_t::value_t > &&
           std::convertible_to< T_beta, typename matrixC_t::value_t > )
void
prod ( const matrixA_t &  A,
       const matrixB_t &  B,
       const T_beta       beta,
       matrixC_t &        C )
{
    HLR_DBG_ASSERT(( A.ncols() == B.nrows() ) &&
                   ( A.nrows() == C.nrows() ) &&
                   ( B.ncols() == C.ncols() ));
    
    return Hpro::BLAS::prod( typename matrixC_t::value_t(1), A, B, typename matrixC_t::value_t(beta), C );
}

template < typename T_alpha,
           typename T_beta,
           matrix_type matrixA_t,
           matrix_type matrixB_t,
           matrix_type matrixC_t >
requires ( std::same_as< typename matrixA_t::value_t, typename matrixB_t::value_t > &&
           std::same_as< typename matrixA_t::value_t, typename matrixC_t::value_t > &&
           std::convertible_to< T_alpha, typename matrixA_t::value_t > &&
           std::convertible_to< T_beta,  typename matrixC_t::value_t > )
void
prod ( const T_alpha      alpha,
       const matrixA_t &  A,
       const matrixB_t &  B,
       const T_beta       beta,
       matrixC_t &        C )
{
    HLR_DBG_ASSERT(( A.ncols() == B.nrows() ) &&
                   ( A.nrows() == C.nrows() ) &&
                   ( B.ncols() == C.ncols() ));
    
    return prod( typename matrixC_t::value_t(alpha), A, B, typename matrixC_t::value_t(beta), C );
}

template < matrix_type matrixA_t,
           matrix_type matrixB_t >
requires ( std::same_as< typename matrixA_t::value_t, typename matrixB_t::value_t > )
matrix< typename matrixA_t::value_t >
prod ( const matrixA_t &  A,
       const matrixB_t &  B )
{
    HLR_DBG_ASSERT( A.ncols() == B.nrows() );
    
    return prod( typename matrixA_t::value_t(1), A, B );
}

using Hpro::BLAS::mulvec;
using Hpro::BLAS::prod;

//
// compute M ≔ M·D with diagonal D
//
template < matrix_type matrix_t,
           vector_type vector_t >
requires ( std::convertible_to< value_type_t< vector_t >, value_type_t< matrix_t > > )
void
prod_diag_ip ( matrix_t &        M,
               const vector_t &  D )
{
    HLR_DBG_ASSERT( M.ncols() == D.length() );
    
    using  value_t = typename matrix_t::value_t;
    
    for ( Hpro::idx_t  i = 0; i < M.ncols(); ++i )
    {
        auto  Mi = M.column( i );

        scale( value_t(D(i)), Mi );
    }// for
}

template < matrix_type matrix_t,
           vector_type vector_t >
requires ( std::convertible_to< value_type_t< vector_t >, value_type_t< matrix_t > > )
matrix< typename matrix_t::value_t >    
prod_diag ( const matrix_t &  M,
            const vector_t &  D )
{
    HLR_DBG_ASSERT( M.ncols() == D.length() );
    
    using  value_t = typename matrix_t::value_t;

    auto  T = copy( M );
    
    for ( Hpro::idx_t  i = 0; i < T.ncols(); ++i )
    {
        auto  Ti = T.column( i );

        scale( value_t(D(i)), Ti );
    }// for

    return T;
}

//
// compute M ≔ D·M with diagonal D
//
template < vector_type vector_t,
           matrix_type matrix_t >
requires ( std::convertible_to< value_type_t< vector_t >, value_type_t< matrix_t > > )
void
prod_diag_ip ( const vector_t &  D,
               matrix_t &        M )
{
    HLR_DBG_ASSERT( M.nrows() == D.length() );
    
    using  value_t = value_type_t< matrix_t >;
    
    for ( Hpro::idx_t  i = 0; i < M.nrows(); ++i )
    {
        auto  Mi = M.row( i );

        scale( value_t(D(i)), Mi );
    }// for
}

template < vector_type vector_t,
           matrix_type matrix_t >
requires ( std::convertible_to< value_type_t< vector_t >, value_type_t< matrix_t > > )
matrix< typename matrix_t::value_t >    
prod_diag ( const vector_t &  D,
            const matrix_t &  M )
{
    HLR_DBG_ASSERT( M.nrows() == D.length() );
    
    using  value_t = value_type_t< matrix_t >;
    
    auto  T = copy( M );
    
    for ( Hpro::idx_t  i = 0; i < T.nrows(); ++i )
    {
        auto  Ti = T.row( i );

        scale( value_t(D(i)), Ti );
    }// for

    return T;
}

using Hpro::BLAS::prod_diag;

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
       matrix< value_t > &  R,
       const bool           comp_Q = true )
{
    const auto              nrows = M.nrows();
    const auto              ncols = M.ncols();
    const auto              minrc = std::min( nrows, ncols );
    std::vector< value_t >  tau( ncols );
    std::vector< value_t >  work( ncols );

    // // DEBUG {
    // auto  DM = copy( M );
    // // DEBUG }
    
    #if 1
    
    blas_int_t  info = 0;

    geqr2( nrows, ncols, M.data(), nrows, tau.data(), work.data(), info );

    if (( R.nrows() != minrc ) || ( R.ncols() != ncols ))
        R = std::move( matrix< value_t >( minrc, ncols ) );
    
    if ( comp_Q )
    {
        if ( ncols > nrows )
        {
            //
            // copy M to R, resize M, copy M back and nullify R in
            //

            copy( M, R );
            M = std::move( matrix< value_t >( nrows, nrows ) );

            auto  RM = matrix< value_t >( R, range::all, range( 0, nrows-1 ) );

            copy( RM, M );

            for ( size_t  j = 0; j < nrows; ++j )
                for ( size_t  i = j+1; i < nrows; ++i )
                    R(i,j) = value_t(0);

            ung2r( nrows, nrows, nrows, M.data(), nrows, tau.data(), work.data(), info );
        }// if
        else
        {
            // just copy R from M
            for ( size_t  j = 0; j < ncols; ++j )
                for ( size_t  i = 0; i <= j; ++i )
                    R(i,j) = M(i,j);

            ung2r( nrows, ncols, ncols, M.data(), nrows, tau.data(), work.data(), info );
        }// else
    }// if
    else
    {
        for ( size_t  j = 0; j < ncols; ++j )
            for ( size_t  i = 0; i <= std::min( j, minrc-1 ); ++i )
                R(i,j) = M(i,j);
    }// else

    // // DEBUG {
    // auto  M1 = prod( M, R );

    // hlr::add( value_t(-1), DM, M1 );

    // const auto  err = norm_2( M1 ) / norm_2( DM );

    // if ( err > 1e-15 )
    //     std::cout << "qr : " << err << std::endl;
    // // DEBUG }
    
    #else
    
    if (( R.nrows() != ncols ) || ( R.ncols() != ncols ))
        R = std::move( matrix< value_t >( ncols, ncols ) );

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

    if ( comp_Q )
    {
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
    }// if

    #endif
}

//
// compute QR factorisation M = Q·R with orthonormal Q
// and upper triangular R. Upon exit, M will hold Q.
//
// ASSUMPTION: nrows(M) ≥ ncols(M)
//
template < typename value_t >
void
qrt  ( matrix< value_t > &  M,
       matrix< value_t > &  R,
       const bool           comp_Q = true )
{
    const blas_int_t        nrows = M.nrows();
    const blas_int_t        ncols = M.ncols();
    const blas_int_t        minrc = std::min( nrows, ncols );
    const blas_int_t        nb    = minrc;
    std::vector< value_t >  T( nb * minrc );
    std::vector< value_t >  work( nb * ncols );

    HLR_DBG_ASSERT( ncols <= nrows );

    blas_int_t  info = 0;

    // compute QR with H = I - V·T·V'
    geqrt( nrows, ncols, nb, M.data(), nrows, T.data(), nb, work.data(), info );

    if (( R.nrows() != ncols ) || ( R.ncols() != ncols ))
        R = std::move( matrix< value_t >( ncols, ncols ) );
    
    // copy R
    for ( blas_int_t  i = 0; i < ncols; ++i )
        for ( blas_int_t  j = 0; j <= i; ++j )
            R(j,i) = M(j,i);

    if ( comp_Q )
    {
        // compute Q
        matrix< value_t >  Q( nrows, minrc );

        for ( blas_int_t  i = 0; i < minrc; ++i )
            Q(i,i) = value_t(1);
        
        larfb( 'L', 'N', 'F', 'C', nrows, ncols, minrc, M.data(), nrows, T.data(), nb, Q.data(), nrows, work.data(), ncols );

        copy( Q, M );
    }// if
}

//
// compute QR factorisation M = Q·R with orthonormal Q
// and upper triangular R. Upon exit, M will hold Q.
//
// ASSUMPTION: nrows(M) > 2·ncols(M)
//
template < typename value_t >
void
qrts  ( matrix< value_t > &  M,
        matrix< value_t > &  R,
        const bool           comp_Q = true )
{
    const blas_int_t        nrows = M.nrows();
    const blas_int_t        ncols = M.ncols();
    const blas_int_t        nbrow = 2*ncols;
    const blas_int_t        nbcol = ncols;
    std::vector< value_t >  T( nbcol * ncols * ( 1 + ( nrows - ncols ) / ( nbrow - ncols ) ) );

    HLR_DBG_ASSERT( 2*ncols < nrows );

    //
    // work size query
    //

    auto  wquery = value_t(0);
    auto  wsize  = blas_int_t(0);
    auto  info   = blas_int_t(0);
    
    latsqr( nrows, ncols, nbrow, nbcol, M.data(), nrows, T.data(), nbcol, & wquery, -1, info );

    wsize = blas_int_t( std::real( wquery ) );
    
    if ( comp_Q )
    {
        ungtsqr( nrows, ncols, nbrow, nbcol, M.data(), nrows, T.data(), nbcol, & wquery, -1, info );
        wsize = std::max( wsize, blas_int_t( std::real( wquery ) ) );
    }// if
        
    auto  work = std::vector< value_t >( wsize );

    // compute QR with H = I - V·T·V'
    latsqr( nrows, ncols, nbrow, nbcol, M.data(), nrows, T.data(), nbcol, work.data(), work.size(), info );

    // copy R
    if (( blas_int_t( R.nrows() ) != ncols ) || ( blas_int_t( R.ncols() ) != ncols ))
        R = std::move( matrix< value_t >( ncols, ncols ) );
    
    for ( blas_int_t  i = 0; i < ncols; ++i )
        for ( blas_int_t  j = 0; j <= i; ++j )
            R(j,i) = M(j,i);

    if ( comp_Q )
    {
        // compute Q
        ungtsqr( nrows, ncols, nbrow, nbcol, M.data(), nrows, T.data(), nbcol, work.data(), work.size(), info );
    }// if
}

//
// to switch between different QR implementations
//
template < typename value_t >
void
qr ( matrix< value_t > &  M,
     matrix< value_t > &  R,
     const bool           comp_Q )
{
    // if ( M.nrows() > 2*M.ncols() ) // not efficient in general
    //    qrts( M, R, comp_Q );
    // else
        qr2( M, R, comp_Q );
}

template < typename value_t >
std::pair< matrix< value_t >,
           matrix< value_t > >
qr ( const matrix< value_t > &  M )
{
    auto  Q = copy( M );
    auto  R = matrix< value_t >();

    qr2( Q, R, true );

    return { std::move( Q ), std::move( R ) };
}

using Hpro::BLAS::qr;

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

    geqrf( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to geqrf failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );
              
    //
    // compute QR
    //

    geqrf( nrows, ncols, A.data(), blas_int_t( A.col_stride() ), T.data(), work.data(), work.size(), info );
    
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
         const Hpro::matop_t             op_Q,
         const matrix< value_t > &       Q,
         const std::vector< value_t > &  T,
         matrix< value_t > &             M )
{
    const auto  nrows = blas_int_t( M.nrows() );
    const auto  ncols = blas_int_t( M.ncols() );
    const auto  k     = blas_int_t( Q.ncols() );
    blas_int_t  info  = 0;

    if ( Hpro::is_complex_type< value_t >::value && ( op_Q == Hpro::apply_trans ) )
        HLR_ERROR( "only normal and adjoint mode supported for complex valued matrices" );
    
    //
    // workspace query
    //

    char  op         = ( op_Q == Hpro::apply_normal ? 'N' :
                         ( Hpro::is_complex_type< value_t >::value ? 'C' : 'T' ) );
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

    prod_Q( from_left, Hpro::apply_normal, Q, T, M );

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

    orgqr( nrows, ncols, minrc, Q.data(), blas_int_t( Q.col_stride() ), T.data(), & work_query, -1, info );

    if ( info < 0 )
        HLR_ERROR( "workspace query to orgqr failed" );
    
    std::vector< value_t >  work( blas_int_t( std::real( work_query ) ) );

    //
    // multiply with Q
    //

    auto  M = copy( Q );
    
    orgqr( nrows, ncols, minrc, M.data(), blas_int_t( M.col_stride() ), T.data(), work.data(), work.size(), info );
    
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

    HLR_DBG_ASSERT( nrows >= ncols );
    
    if (( nrows > ntile ) && ( nrows >= 4 * ncols ))
    {
        auto  mid   = nrows / 2;
        auto  rows0 = range( 0, mid-1 );
        auto  rows1 = range( mid, nrows-1 );
        auto  Q0    = matrix< value_t >( M, rows0, range::all, Hpro::copy_value );
        auto  Q1    = matrix< value_t >( M, rows1, range::all, Hpro::copy_value );
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

        qr_wrapper( Q2, R );

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
        qr_wrapper( M, R );
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

    Hpro::BLAS::factorise_ortho( Q, R );

    return { std::move( Q ), std::move( R ) };
}

//
// construct approximate factorisation M = Q·R with orthonormal Q
//
template < typename value_t >
std::pair< matrix< value_t >,
           matrix< value_t > >
factorise_ortho ( const matrix< value_t > &  M,
                  const Hpro::TTruncAcc &    acc )
{
    using  real_t  = typename Hpro::real_type< value_t >::type_t;
    
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
        auto        OQ  = std::move( copy( V_k ) );
        auto        OR  = prod( value_t(1), adjoint( OQ ), M );

        return { std::move( OQ ), std::move( OR ) };
    }// else
}

//
// compute QR with column pivoting, i.e., M·P = Q·R
// - upon return M holds Q
//
template < typename value_t >
void
qrp ( matrix< value_t > &   M,
      matrix< value_t > &   R,
      std::vector< int > &  P )
{
    // // DEBUG {
    // auto  CM = copy( M );
    // // DEBUG }

    std::vector< Hpro::blas_int_t >  P2;
        
    Hpro::BLAS::qrp( M, R, P2 );

    P.resize( P2.size() );
    
    for ( size_t  i = 0; i < P2.size(); ++i )
        P[i] = int(P2[i]);
    
    // // DEBUG {
    // auto  PR = copy( R );
    
    // for ( size_t  i = 0; i < P.size(); ++i )
    // {
    //     auto  j    = P[i];
    //     auto  R_i  = R.column( i );
    //     auto  PR_j = PR.column( j );

    //     copy( R_i, PR_j );
    // }// for

    // Hpro::DBG::write( PR, "PR.mat", "PR" );
    
    // auto  TM = prod( M, PR );

    // add( value_t(-1), CM, TM );

    // const auto  err = norm_2( TM ) / norm_2( CM );

    // if ( err > 1e-15 )
    //     std::cout << "qrp : " << err << std::endl;
    // // DEBUG }
}
    
//
// construct SVD of bidiagonal matrix with diagonal D and off-diagonal E
//
template < typename value_t >
std::tuple< matrix< value_t >,
            vector< value_t >,
            matrix< value_t > >
bdsvd ( const vector< value_t > &  D,
        const vector< value_t > &  E )
{
    const blas_int_t           n   = D.length();
    blas_int_t                 nsv = 0; // number of singular values found
    matrix< value_t >          Z( 2*n, n+1 );
    std::vector< value_t >     work( 14 * n );
    std::vector< blas_int_t >  iwork( 12 * n );
    blas_int_t                 info = 0;
    auto                       S2 = vector< value_t >( 2*n ); // bdsvd actually needs 2*n space here

    bdsvd( 'L', 'V', 'A', D.length(), D.data(), E.data(),
           value_t(0), value_t(0), blas_int_t(0), blas_int_t(0),
           nsv, S2.data(), Z.data(), 2*n, work.data(), iwork.data(), info );

    auto  U  = matrix< value_t >( n, nsv );
    auto  S  = vector< value_t >( n );
    auto  V  = matrix< value_t >( n, nsv );
    auto  SS = vector< value_t >( S2, range( 0, n-1 ) );
    auto  ZU = matrix< value_t >( Z, range( 0,   n-1 ), range( 0, nsv-1 ) );
    auto  ZV = matrix< value_t >( Z, range( n, 2*n-1 ), range( 0, nsv-1 ) );

    copy( ZU, U );
    copy( SS, S );
    copy( ZV, V );
    
    return { std::move( U ), std::move( S ), std::move( V ) };
}

//
// compute SVD of M
//
template < matrix_type  matrix_t >
std::tuple< matrix< typename matrix_t::value_t >,
            vector< Hpro::real_type_t< typename matrix_t::value_t > >,
            matrix< typename matrix_t::value_t > >
svd ( const matrix_t &  M )
{
    using  value_t = typename matrix_t::value_t;
    using  real_t  = Hpro::real_type_t< value_t >;
    
    auto  U = copy( M );
    auto  S = vector< real_t >( std::min( U.nrows(), U.ncols() ) );
    auto  V = matrix< value_t >( S.length(), M.ncols() );

    svd( U, S, V );

    return { std::move( U ), std::move( S ), std::move( V ) };
}

//
// compute singular values and left/right singular vectors of M
//
template < matrix_type  matrix_t >
std::tuple< matrix< typename matrix_t::value_t >,
            vector< Hpro::real_type_t< typename matrix_t::value_t > > >
svd ( const matrix_t &  M,
      const bool        left )
{
    using  value_t = typename matrix_t::value_t;
    using  real_t  = Hpro::real_type_t< value_t >;
    
    auto  U = copy( M );
    auto  S = vector< real_t >( std::min( U.nrows(), U.ncols() ) );

    Hpro::BLAS::svd( U, S, left );

    return { std::move( U ), std::move( S ) };
}

// but use also all other SVD functions
using Hpro::BLAS::svd;

//
// compute singular values of M
//
template < typename value_t >
vector< Hpro::real_type_t< value_t > >
sv ( const matrix< value_t > &  M )
{
    auto  T = blas::copy( M );
    auto  S = blas::vector< Hpro::real_type_t< value_t > >();
    
    Hpro::BLAS::sv( T, S );
    
    return S;
}

//
// compute singular values of U·V'
//
template < typename value_t >
vector< real_type_t< value_t > >
sv ( const matrix< value_t > &  U,
     const matrix< value_t > &  V )
{
    using  real_t = real_type_t< value_t >;
    
    const auto   nrows_U = U.nrows();
    const auto   nrows_V = V.nrows();
    const auto   rank    = U.ncols();
    const auto   minrc   = std::min( nrows_U, nrows_V );
    auto         S       = vector< real_t >( minrc );

    if ( rank >= minrc )
    {
        auto  M = prod( value_t(1), U, adjoint(V) );

        Hpro::BLAS::sv( M, S );
    }// if
    else
    {
        auto  QU = copy( U );
        auto  QV = copy( V );
        auto  RU = matrix< value_t >( rank, rank );
        auto  RV = matrix< value_t >( rank, rank );

        qr( QU, RU );
        qr( QV, RV );
        
        auto  R = prod( value_t(1), RU, adjoint(RV) );
            
        Hpro::BLAS::sv( R, S );
    }// else

    return S;
}

using Hpro::BLAS::sv;

//
// compute eigenvalues of M
//
template < matrix_type matrix_t >
std::pair< matrix< typename matrix_t::value_t >,
           vector< typename matrix_t::value_t > >
eigen ( matrix_t &  M )
{
    using value_t = typename matrix_t::value_t;
    
    auto  E = vector< value_t >();
    auto  V = matrix< value_t >();

    Hpro::BLAS::eigen( M, E, V );

    return { std::move( V ), std::move( E ) };
}
using Hpro::BLAS::eigen;

//
// compute eigenvalues of hermitian M
//
template < matrix_type matrix_t >
std::pair< matrix< typename matrix_t::value_t >,
           vector< real_type_t< typename matrix_t::value_t > > >
eigen_herm ( matrix_t &  M )
{
    using value_t = typename matrix_t::value_t;
    using real_t  = real_type_t< typename matrix_t::value_t >;
    
    auto  E = vector< real_t >();
    auto  V = matrix< value_t >();

    Hpro::BLAS::eigen_herm( M, E, V );

    return { std::move( V ), std::move( E ) };
}
using Hpro::BLAS::eigen_herm;

}}// namespace hlr::blas

//
// stream output for matrices in global namespace
//
template < typename value_t >
std::ostream &
operator << ( std::ostream &                        os,
              const hlr::blas::matrix< value_t > &  M )
{
    hlr::blas::print( M, os );
    return os;
}

#endif // __HLR_ARITH_BLAS_HH
