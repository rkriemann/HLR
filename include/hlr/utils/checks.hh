#ifndef __HLR_UTILS_CHECKS_HH
#define __HLR_UTILS_CHECKS_HH
//
// Project     : HLib
// File        : checks.hh
// Description : testing/checking functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/config.hh>

namespace hlr
{

//
// extend standard test to "all" and "any" of the arguments
//
#define HLR_TEST_ALL( name, type )                                      \
    template < typename value_t >                                       \
    bool name ## _all  ( const type *  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename value_t, typename... T >                        \
    bool name ## _all  ( const type *  p, T&&...  args ) noexcept { return name( p ) && name ## _all( std::forward< T >( args )... ); } \
                                                                        \
    template < typename value_t >                                       \
    bool name ## _all  ( const type &  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename value_t, typename... T >                        \
    bool name ## _all  ( const type &  p, T&&...  args ) noexcept { return name( p ) && name ## _all( std::forward< T >( args )... ); }
    
#define HLR_TEST_ANY( name, type )                                      \
    template < typename value_t >                                       \
    bool name ## _any  ( const type *  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename value_t, typename... T >                        \
    bool name ## _any  ( const type *  p, T&&...  args ) noexcept { return name( p ) || name ## _any( std::forward< T >( args )... ); } \
                                                                        \
    template < typename value_t >                                       \
    bool name ## _any  ( const type &  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename value_t, typename... T >                        \
    bool name ## _any  ( const type &  p, T&&...  args ) noexcept { return name( p ) || name ## _any( std::forward< T >( args )... ); }
    
//
// return true, if pointer is null
//
template < typename T >
bool is_null      ( T *  p )               noexcept { return (p == nullptr); }

template < typename T >
bool is_null_all  ( T *  p )               noexcept { return is_null( p ); }

template < typename T1, typename... T2 >
bool is_null_all  ( T1 *  p, T2...  ptrs ) noexcept { return is_null( p ) && is_null_all( ptrs... ); }

template < typename T >
bool is_null_any  ( T *  p )               noexcept { return is_null( p ); }

template < typename T1, typename... T2 >
bool is_null_any  ( T1 *  p, T2...  ptrs ) noexcept { return is_null( p ) || is_null_any( ptrs... ); }

//
// same for smart pointers
//
template < typename T >
bool is_null      ( const std::shared_ptr< T > &   p ) noexcept { return (p.get() == nullptr); }

template < typename T >
bool is_null      ( const std::unique_ptr< T > &   p ) noexcept { return (p.get() == nullptr); }

template < typename T >
bool is_null_all  ( const std::shared_ptr< T > &   p ) noexcept { return is_null( p ); }

template < typename T >
bool is_null_all  ( const std::unique_ptr< T > &   p ) noexcept { return is_null( p ); }

template < typename T >
bool is_null_any  ( const std::shared_ptr< T > &   p ) noexcept { return is_null( p ); }

template < typename T >
bool is_null_any  ( const std::unique_ptr< T > &   p ) noexcept { return is_null( p ); }

template < typename T1, typename... T2 >
bool is_null_all  ( const std::shared_ptr< T1 > &  p,
                    T2...                          ptrs ) noexcept { return is_null( p ) && is_null_all( ptrs... ); }

template < typename T1, typename... T2 >
bool is_null_all  ( const std::unique_ptr< T1 > &  p,
                    T2...                          ptrs ) noexcept { return is_null( p ) && is_null_all( ptrs... ); }

template < typename T1, typename... T2 >
bool is_null_any  ( const std::shared_ptr< T1 > &  p,
                    T2...                          ptrs ) noexcept { return is_null( p ) || is_null_any( ptrs... ); }

template < typename T1, typename... T2 >
bool is_null_any  ( const std::unique_ptr< T1 > &  p,
                    T2...                          ptrs ) noexcept { return is_null( p ) || is_null_any( ptrs... ); }

//
// return true if A is corresponding to leaf matrix block
//
template < typename T >
bool is_leaf      ( T *  A )               noexcept { return ! is_null( A ) && ! A->is_blocked(); }

template < typename T >
bool is_leaf_any  ( T *  A )               noexcept { return is_leaf( A ); }

template < typename T1, typename... T2 >
bool is_leaf_any  ( T1 *  A, T2...  mtrs ) noexcept { return is_leaf( A ) || is_leaf_any( mtrs... ); }

template < typename T >
bool is_leaf_all  ( T *  A )               noexcept { return is_leaf( A ); }

template < typename T1, typename... T2 >
bool is_leaf_all  ( T1 *  A, T2...  mtrs ) noexcept { return is_leaf( A ) && is_leaf_all( mtrs... ); }


template < typename T >
bool is_leaf      ( T &  A )               noexcept { return ! A.is_blocked(); }

template < typename T >
bool is_leaf_any  ( T &  A )               noexcept { return is_leaf( A ); }

template < typename T1, typename... T2 >
bool is_leaf_any  ( T1 &  A, T2...  mtrs ) noexcept { return is_leaf( A ) || is_leaf_any( mtrs... ); }

template < typename T >
bool is_leaf_all  ( T &  A )               noexcept { return is_leaf( A ); }

template < typename T1, typename... T2 >
bool is_leaf_all  ( T1 &  A, T2...  mtrs ) noexcept { return is_leaf( A ) && is_leaf_all( mtrs... ); }

//
// return true if A is corresponding to a small matrix block
//
template < typename T >
bool is_small      ( const size_t  n,
                     T *           A )     noexcept { return ! is_null( A ) && ( std::min( A->rows(), A->cols() ) <= n ); }

template < typename T >
bool is_small_any  ( const size_t  n,
                     T *           A )     noexcept { return is_small( n, A ); }

template < typename T1, typename... T2 >
bool is_small_any  ( const size_t  n,
                     T1 *          A,
                     T2...         mtrs )  noexcept { return is_small( n, A ) || is_small_any( n, mtrs... ); }

template < typename T >
bool is_small_all  ( const size_t  n,
                     T *           A )     noexcept { return is_small( n, A ); }

template < typename T1, typename... T2 >
bool is_small_all  ( const size_t  n,
                     T1 *          A,
                     T2...         mtrs )  noexcept { return is_small( n, A ) && is_small_all( n, mtrs... ); }


template < typename T >
bool is_small      ( const size_t  n,
                     T &           A )     noexcept { return is_small( n, & A ); }

template < typename T >
bool is_small_any  ( const size_t  n,
                     T &           A )     noexcept { return is_small( n, A ); }

template < typename T1, typename... T2 >
bool is_small_any  ( const size_t  n,
                     T1 &          A,
                     T2...         mtrs )  noexcept { return is_small( n, A ) || is_small_any( n, mtrs... ); }

template < typename T >
bool is_small_all  ( const size_t  n,
                     T &           A )     noexcept { return is_small( n, A ); }

template < typename T1, typename... T2 >
bool is_small_all  ( const size_t  n,
                     T1 &          A,
                     T2...         mtrs )  noexcept { return is_small( n, A ) && is_small_all( n, mtrs... ); }



template < typename T >
bool is_small      ( T *           A )     noexcept { return ! is_null( A ) && Hpro::is_small( A ); }

template < typename T >
bool is_small_any  ( T *           A )     noexcept { return is_small( A ); }

template < typename T1, typename... T2 >
bool is_small_any  ( T1 *          A,
                     T2...         mtrs )  noexcept { return is_small( A ) || is_small_any( mtrs... ); }

template < typename T >
bool is_small_all  ( T *           A )     noexcept { return is_small( A ); }

template < typename T1, typename... T2 >
bool is_small_all  ( T1 *          A,
                     T2...         mtrs )  noexcept { return is_small( A ) && is_small_all( mtrs... ); }

template < typename T >
bool is_small      ( T &           A )     noexcept { return Hpro::is_small( & A ); }

template < typename T >
bool is_small_any  ( T &           A )     noexcept { return is_small( A ); }

template < typename T1, typename... T2 >
bool is_small_any  ( T1 &          A,
                     T2...         mtrs )  noexcept { return is_small( A ) || is_small_any( mtrs... ); }

template < typename T >
bool is_small_all  ( T &           A )     noexcept { return is_small( A ); }

template < typename T1, typename... T2 >
bool is_small_all  ( T1 &          A,
                     T2...         mtrs )  noexcept { return is_small( A ) && is_small_all( mtrs... ); }


//
// return true if given matrix is a structured (blocked) matrix
//

// template < typename value_t > bool is_blocked      ( const Hpro::TMatrix< value_t > *  A ) noexcept { return (A != nullptr) && A->is_blocked(); }
// template < typename value_t > bool is_blocked      ( const Hpro::TMatrix< value_t > &  A ) noexcept { return A.is_blocked(); }

template < typename value_t > bool is_blocked_all  ( const Hpro::TMatrix< value_t > *  A ) noexcept { return is_blocked( A ); }
template < typename value_t > bool is_blocked_all  ( const Hpro::TMatrix< value_t > &  A ) noexcept { return is_blocked( A ); }

template < typename value_t > bool is_blocked_any  ( const Hpro::TMatrix< value_t > *  A ) noexcept { return is_blocked( A ); }
template < typename value_t > bool is_blocked_any  ( const Hpro::TMatrix< value_t > &  A ) noexcept { return is_blocked( A ); }

template < typename value_t, typename... T >
bool is_blocked_all ( const Hpro::TMatrix< value_t > &  A,
                      T&&...                 mtrs )  noexcept { return is_blocked( A ) && hlr::is_blocked_all( std::forward< T >( mtrs )... ); }

template < typename value_t, typename... T >
bool is_blocked_any ( const Hpro::TMatrix< value_t > &  A,
                      T&&...                 mtrs )  noexcept { return is_blocked( A ) || hlr::is_blocked_any( std::forward< T >( mtrs )... ); }

//
// return true if given matrix is a dense matrix
//

// template < typename value_t >
// bool is_dense  ( const Hpro::TMatrix< value_t > *  A ) noexcept { return Hpro::is_dense( A ); }

// template < typename value_t >
// bool is_dense  ( const Hpro::TMatrix< value_t > &  A ) noexcept { return hlr::is_dense( & A ); }

HLR_TEST_ALL( is_dense, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_dense, Hpro::TMatrix< value_t > )

//
// return true if given matrix is a low-rank matrix
//

// template < typename value_t >
// bool is_lowrank  ( const Hpro::TMatrix< value_t > *  A ) noexcept { return Hpro::is_lowrank( A ); }

// template < typename value_t >
// bool is_lowrank  ( const Hpro::TMatrix< value_t > &  A ) noexcept { return hlr::is_lowrank( &A ); }

HLR_TEST_ALL( is_lowrank, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_lowrank, Hpro::TMatrix< value_t > )

//
// return true if given matrix is a sparse matrix
//

// template < typename value_t >
// bool is_sparse  ( const Hpro::TMatrix< value_t > *  A ) noexcept { return Hpro::is_sparse( A ); }

// template < typename value_t >
// bool is_sparse  ( const Hpro::TMatrix< value_t > &  A ) noexcept { return hlr::is_sparse( & A ); }

HLR_TEST_ALL( is_sparse, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_sparse, Hpro::TMatrix< value_t > )

//
// return true if given matrix has nested dissection structure
//

template < typename value_t >
bool is_nd     ( const Hpro::TMatrix< value_t > &  A )     noexcept { return Hpro::is_dd( & A ); }

template < typename value_t >
bool is_nd     ( const Hpro::TMatrix< value_t > *  A )     noexcept { return Hpro::is_dd( A ); }

HLR_TEST_ALL( is_nd, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_nd, Hpro::TMatrix< value_t > )

//
// return true if given vector is a scalar vector
//

template < typename value_t >
bool is_scalar      ( const Hpro::TVector< value_t > &  v )     noexcept { return Hpro::is_scalar( v ); }

template < typename value_t >
bool is_scalar      ( const Hpro::TVector< value_t > *  v )     noexcept { return Hpro::is_scalar( v ); }

template < typename value_t >
bool is_scalar_all  ( const Hpro::TVector< value_t > *  p ) noexcept { return hlr::is_scalar( p ); } 

template < typename value_t, typename... T >
bool is_scalar_all  ( const Hpro::TVector< value_t > *  p, T&&...  args ) noexcept { return hlr::is_scalar( p ) && hlr::is_scalar_all( std::forward< T >( args )... ); }

template < typename value_t >
bool is_scalar_all  ( const Hpro::TVector< value_t > &  p ) noexcept { return hlr::is_scalar( p ); }

template < typename value_t, typename... T >
bool is_scalar_all  ( const Hpro::TVector< value_t > &  p, T&&...  args ) noexcept { return hlr::is_scalar( p ) && hlr::is_scalar_all( std::forward< T >( args )... ); }

// HLR_TEST_ALL( is_scalar, Hpro::TVector< value_t > )
// HLR_TEST_ANY( is_scalar, Hpro::TVector< value_t > )

}// namespace hlr

#endif // __HLR_UTILS_CHECKS_HH
