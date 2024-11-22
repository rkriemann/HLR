#ifndef __HLR_UTILS_LOG_HH
#define __HLR_UTILS_LOG_HH
//
// Project     : HLR
// Module      : log.hh
// Description : logging functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <iostream>
#include <mutex>
//#include <source_location>

#include <hpro/base/config.hh>

#include <hlr/utils/term.hh>

namespace hlr
{

//
// breakpoint function as entry point for debugging
//
void
breakpoint ();

//
// logging function
//
template < typename msg_t >
void
error ( const msg_t &  msg )
{
    std::cout << msg << std::endl;
    std::exit( 1 );
}

// throw exception with file info
#define HLR_ERROR( msg )                                                \
    {                                                                   \
        /* auto  location = std::source_location::current(); */         \
        hlr::breakpoint();                                              \
        throw std::runtime_error( hlr::term::italic( __FILE__ + HLIB::to_string( ":%d", __LINE__ ) ) + \
                                  std::string( " : " ) + hlr::term::red( msg ) ); \
    }
//                            std::string( " in " ) + hlr::term::italic( __PRETTY_FUNCTION__ ) + 

// debug assert
#if defined(NDEBUG)
#  define HLR_DBG_ASSERT( expr )
#else
#  define HLR_DBG_ASSERT( expr )                                        \
    if ( ! ( expr ) )                                                   \
    {                                                                   \
        breakpoint();                                                   \
        HLR_ERROR( ( hlr::term::bold( #expr ) + " failed" ) );          \
    }
#endif

// always-on-assert
#define HLR_ASSERT( expr )                                              \
    if ( ! ( expr ) )                                                   \
    {                                                                   \
        breakpoint();                                                   \
        HLR_ERROR( ( hlr::term::bold( #expr ) + " failed" ) );          \
    }

// mutex for log function
extern std::mutex  __LOG_MUTEX;

//
// logging function
//
template < typename msg_t >
void
log ( const int      lvl,
      const msg_t &  msg )
{
    if ( HLIB::verbose( lvl ) )
    {
        std::scoped_lock  lock( __LOG_MUTEX );
        
        std::cout << msg << std::endl;
    }// if
}

void
log ( const int            lvl,
      const std::string &  msg );

//
// logging enabled/disabled via NDEBUG
//
#ifdef NDEBUG
#  define HLR_LOG( lvl, msg )                                           \
    if ( HLIB::verbose( lvl ) ) {                                       \
        hlr::log( lvl,                                                  \
                  std::string( "                                          " ) + msg ); }
#else
#  define HLR_LOG( lvl, msg )                                           \
    if ( HLIB::verbose( lvl ) ) {                                       \
        hlr::log( lvl,                                                  \
                  std::string( "                                          " ) + msg ); }
#endif
              // hlr::term::green( __FILE__  ) + HLIB::to_string( ":%d", __LINE__ ) +
              // std::string( " in " ) + hlr::term::yellow( __func__ ) + 
              // std::string( " : " ) + msg )

}// namespace hlr

#endif // __HLR_UTILS_LOG_HH
