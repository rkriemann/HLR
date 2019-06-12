#ifndef __HLR_UTILS_CHECKS_HH
#define __HLR_UTILS_CHECKS_HH
//
// Project     : HLib
// File        : log.hh
// Description : testing/checking functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

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

}// namespace hlr

#endif // __HLR_UTILS_CHECKS_HH
