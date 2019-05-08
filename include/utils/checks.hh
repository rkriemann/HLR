#ifndef __HLR_UTILS_CHECKS_HH
#define __HLR_UTILS_CHECKS_HH
//
// Project     : HLib
// File        : log.hh
// Description : testing/checking functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

namespace HLR
{

//!
//! return true, if pointer is null
//!
template < typename T >
bool
is_null ( T *  p ) noexcept
{
    return (p == nullptr);
}

//!
//! return true, if all given pointers are null
//!
template < typename T >
bool
is_null_all  ( T *  p ) noexcept
{
    return is_null( p );
}

template < typename T1, typename... T2 >
bool
is_null_all  ( T1 *  p, T2...  ptrs )
{
    return is_null( p ) && is_null_all( ptrs... );
}

//!
//! return true, if any given matrix pointers is null
//!
template < typename T >
bool
is_null_any  ( T *  p ) noexcept
{
    return is_null( p );
}

template < typename T1, typename... T2 >
bool
is_null_any  ( T1 *  p, T2...  ptrs )
{
    return is_null( p ) || is_null_any( ptrs... );
}

}// namespace HLR

#endif // __HLR_UTILS_CHECKS_HH
