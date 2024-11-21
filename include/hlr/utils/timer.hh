#pragma once
#ifndef __HLR_UTILS_TIMER_HH
#define __HLR_UTILS_TIMER_HH
//
// Project     : HLR
// Module      : timer
// Description : timing functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/base/System.hh>

namespace hlr
{

namespace timer
{

using Hpro::Time::Wall::now;
using Hpro::Time::Wall::since;

}// namespace timer

//
// measure time from instantiation to destruction and store
// value in given variable
//
struct auto_timer
{
    hpro::Time::Wall::TTimePoint  start;
    double &                      time_val;

    auto_timer ( double &  atime_val )
            : time_val( atime_val )
    {
        start = hpro::Time::Wall::now();
    }

    ~auto_timer ()
    {
        time_val += hpro::Time::Wall::since( start ).seconds();
    }
};

}// namespace hlr

#endif // __HLR_TOOLS_TIMER_HH
