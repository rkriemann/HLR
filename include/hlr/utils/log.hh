#ifndef __HLR_UTILS_LOG_HH
#define __HLR_UTILS_LOG_HH
//
// Project     : HLib
// File        : log.hh
// Description : logging functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <mutex>

#include <base/config.hh>

namespace hlr
{

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

//
// logging enabled/disabled via NDEBUG
//
#ifdef NDEBUG
#  define HLR_LOG( lvl, msg ) 
#else
#  define HLR_LOG( lvl, msg ) hlr::log( lvl, __FILE__ + std::string( " / " ) + __func__ + std::string( " : " ) + msg )
#endif

}// namespace hlr

#endif // __HLR_UTILS_LOG_HH
