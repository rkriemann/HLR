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

#include <base/config.hh>

namespace HLR
{

//
// logging function
//
template < typename msg_t >
void
log ( const int      lvl,
      const msg_t &  msg )
{
    if ( HLIB::verbose( lvl ) )
        std::cout << msg << std::endl;
}

}// namespace HLR

#endif // __HLR_UTILS_LOG_HH
