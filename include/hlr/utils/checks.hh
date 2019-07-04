#ifndef __HLR_UTILS_CHECKS_HH
#define __HLR_UTILS_CHECKS_HH
//
// Project     : HLib
// File        : log.hh
// Description : testing/checking functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/structure.hh>
#include <base/config.hh>

namespace hlr
{

//!
//! return true, if pointer is null
//!
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
bool is_small      ( T *           A )     noexcept { return ! is_null( A ) && HLIB::is_small( A ); }

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
bool is_small      ( T &           A )     noexcept { return HLIB::is_small( & A ); }

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

}// namespace hlr

#endif // __HLR_UTILS_CHECKS_HH
