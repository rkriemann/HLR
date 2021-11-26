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

namespace hpro = HLIB;

//
// extend standard test to "all" and "any" of the arguments
//
#define HLR_TEST_ALL( name, type )                                      \
    inline                                                              \
    bool name ## _all  ( const type *  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename... T >                                          \
    bool name ## _all  ( const type *  p, T&&...  args ) noexcept { return name( p ) && name ## _all( std::forward< T >( args )... ); } \
                                                                        \
    inline                                                              \
    bool name ## _all  ( const type &  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename... T >                                          \
    bool name ## _all  ( const type &  p, T&&...  args ) noexcept { return name( p ) && name ## _all( std::forward< T >( args )... ); }
    
#define HLR_TEST_ANY( name, type )                                      \
    inline                                                              \
    bool name ## _any  ( const type *  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename... T >                                          \
    bool name ## _any  ( const type *  p, T&&...  args ) noexcept { return name( p ) || name ## _any( std::forward< T >( args )... ); } \
                                                                        \
    inline                                                              \
    bool name ## _any  ( const type &  p ) noexcept { return name( p ); } \
                                                                        \
    template < typename... T >                                          \
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
bool is_small      ( T *           A )     noexcept { return ! is_null( A ) && hpro::is_small( A ); }

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
bool is_small      ( T &           A )     noexcept { return hpro::is_small( & A ); }

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

using hpro::is_blocked;
using hpro::is_blocked_all;

inline
bool is_blocked_any ( const hpro::TMatrix &  A )     noexcept { return is_blocked( A ); }

template < typename... T >
bool is_blocked_any ( const hpro::TMatrix &  A,
                      T...                   mtrs )  noexcept { return is_blocked( A ) || hlr::is_blocked_any( mtrs... ); }

template < typename... T >
bool is_blocked_all ( const hpro::TMatrix &  A,
                      T&&...                 mtrs )  noexcept { return is_blocked( A ) && hlr::is_blocked_all( std::forward< T >( mtrs )... ); }

//
// return true if given matrix is a dense matrix
//

using hpro::is_dense;

inline
bool is_dense       ( const hpro::TMatrix &  A )     noexcept { return is_dense( & A ); }

HLR_TEST_ALL( is_dense, hpro::TMatrix )
HLR_TEST_ANY( is_dense, hpro::TMatrix )

//
// return true if given matrix is a low-rank matrix
//

using hpro::is_lowrank;

inline
bool is_lowrank     ( const hpro::TMatrix &  A )     noexcept { return is_lowrank( & A ); }

HLR_TEST_ALL( is_lowrank, hpro::TMatrix )
HLR_TEST_ANY( is_lowrank, hpro::TMatrix )

//
// return true if given matrix is a sparse matrix
//

using hpro::is_sparse;

inline
bool is_sparse     ( const hpro::TMatrix &  A )     noexcept { return is_sparse( & A ); }

HLR_TEST_ALL( is_sparse, hpro::TMatrix )
HLR_TEST_ANY( is_sparse, hpro::TMatrix )

//
// return true if given matrix has nested dissection structure
//

inline
bool is_nd     ( const hpro::TMatrix &  A )     noexcept { return HLIB::is_dd( & A ); }

inline
bool is_nd     ( const hpro::TMatrix *  A )     noexcept { return HLIB::is_dd( A ); }

HLR_TEST_ALL( is_nd, hpro::TMatrix )
HLR_TEST_ANY( is_nd, hpro::TMatrix )

//
// return true if given vector is a scalar vector
//

inline
bool is_scalar      ( const hpro::TVector &  v )     noexcept { return hpro::is_scalar( v ); }

inline
bool is_scalar      ( const hpro::TVector *  v )     noexcept { return hpro::is_scalar( v ); }

inline bool is_scalar_all  ( const hpro::TVector *  p ) noexcept { return hlr::is_scalar( p ); } 

template < typename... T >
bool is_scalar_all  ( const hpro::TVector *  p, T&&...  args ) noexcept { return hlr::is_scalar( p ) && hlr::is_scalar_all( std::forward< T >( args )... ); }

inline
bool is_scalar_all  ( const hpro::TVector &  p ) noexcept { return hlr::is_scalar( p ); }

template < typename... T >
bool is_scalar_all  ( const hpro::TVector &  p, T&&...  args ) noexcept { return hlr::is_scalar( p ) && hlr::is_scalar_all( std::forward< T >( args )... ); }

// HLR_TEST_ALL( is_scalar, hpro::TVector )
// HLR_TEST_ANY( is_scalar, hpro::TVector )

}// namespace hlr

#endif // __HLR_UTILS_CHECKS_HH
