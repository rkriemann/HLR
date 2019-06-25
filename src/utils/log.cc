//
// Project     : HLib
// File        : log.cc
// Description : logging functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlr/utils/log.hh>

namespace hlr
{

// mutex for log function
std::mutex  __LOG_MUTEX;

}// namespace hlr
