#ifndef __HLR_UTILS_MACH_HH
#define __HLR_UTILS_MACH_HH
//
// Project     : HLR
// Module      : mach.hh
// Description : machine related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <string>

namespace hlr { namespace mach
{

//
// return info about associated CPU cores 
//
std::string  cpuset   ();

//
// return CPU name
//
std::string  cpu ();

//
// return hostname
//
std::string  hostname ();

}}// namespace hlr::mach

#endif  // __HLR_UTILS_MACH_HH
