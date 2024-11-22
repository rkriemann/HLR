//
// Project     : HLR
// Module      : log.cc
// Description : logging functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/log.hh>

namespace hlr
{

//
// breakpoint function as entry point for debugging
//
void
breakpoint ()
{
    ;
}

// mutex for log function
std::mutex  __LOG_MUTEX;

void
log ( const int            lvl,
      const std::string &  msg )
{
    if ( HLIB::verbose( lvl ) )
    {
        std::scoped_lock  lock( __LOG_MUTEX );

        // print string without end-of-line characters
        for ( auto  c : msg )
            if ( c != '\n' )
                std::cout << c;

        std::cout << std::endl;
    }// if
}

}// namespace hlr
