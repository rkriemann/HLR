//
// Project     : HLR
// Module      : mach.cc
// Description : machine related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif

#if defined(__linux)
#  include <sched.h>
#endif

#include <unistd.h>

#include <fstream>
#include <sstream>

//
// in case of MKL: make it think an Intel CPU is running
// (for AMD processors)
//
extern "C" { int mkl_serv_intel_cpu_true () { return 1; } }

namespace hlr { namespace mach
{

//
// return info about associated CPU cores 
//
std::string
cpuset ()
{
    #if defined(__linux)
    
    cpu_set_t  cset;

    CPU_ZERO( & cset );
    sched_getaffinity( 0, sizeof(cset), & cset );

    uint                ncores  = 0;
    std::ostringstream  out;
    int                 first   = -1;
    int                 last    = -1;
    bool                comma   = false;
    auto                prn_set = [&] ()
    {
        if ( comma ) out << ',';
                
        if      ( first == last    ) out << first;
        else if ( last  == first+1 ) out << first << ',' << last;
        else                         out << first << '-' << last;
    };

    for ( int  i = 0; i < 1024; ++i )
    {
        if ( CPU_ISSET( i, & cset ) )
        {
            ++ncores;
            
            // first initialization
            if ( first == -1 )
            {
                first = i;
                last  = i;
            }// if

            // new interval 
            if ( last < i-1 )
            {
                prn_set();
                
                first = i;
                last  = i;

                comma = true;
            }// if
            
            last = i;
        }// if
    }// for

    // finish expr
    prn_set();

    // add number of cores
    out << " (#" << ncores << ')';
                
    return out.str();

    #else

    return "unknown";
    
    #endif
}

//
// return CPU name
//
std::string
cpu ()
{
    #if defined(__linux)
    
    //
    // look in /proc/cpuinfo for "model name"
    //

    std::ifstream  cpuinfo( "/proc/cpuinfo" );

    if ( ! cpuinfo )
        return "";

    std::string  line, model_name;
                    
    line.reserve( 256 );
                    
    while ( cpuinfo )
    {
        getline( cpuinfo, line );

        if ( line.substr( 0, 10 ) == "model name" )
        {
            auto  pos = line.find( ':' );

            if ( pos == std::string::npos )
                return "";

            model_name = line.substr( pos + 2 );
            break;
        }// if
    }// while

    // remove (R), (TM), etc.
    std::string   cpu;
    const size_t  nlen = model_name.size();

    for ( size_t  pos = 0; pos < nlen; ++pos )
    {
        if      (( pos + 3  <= nlen) && (model_name.substr( pos,   3 ) == "(R)"        )) pos += 2;
        else if (( pos + 4  <= nlen) && (model_name.substr( pos,   4 ) == "(TM)"       )) pos += 3;
        else if (( pos + 7  <= nlen) && (model_name.substr( pos+2, 5 ) == "-Core"      )) pos += 6;
        else if (( pos + 6  <= nlen) && (model_name.substr( pos+1, 5 ) == "-Core"      )) pos += 5;
        else if (( pos + 5  <= nlen) && (model_name.substr( pos,   5 ) == " Core"      )) pos += 4;
        else if (( pos + 4  <= nlen) && (model_name.substr( pos,   4 ) == " CPU"       )) pos += 3;
        else if (( pos + 10 <= nlen) && (model_name.substr( pos,  10 ) == " Processor" )) pos += 9;
        else if (( pos + 4  <= nlen) && (model_name.substr( pos,   4 ) == " PRO"       )) pos += 3;
        else if (( pos + 21 <= nlen) && (model_name.substr( pos,  21 ) == " with Radeon Graphics" )) pos += 20;
        else if (( pos + 2  <= nlen) && (model_name.substr( pos,   2 ) == " @"         )) pos = nlen;
        else
            cpu += model_name[ pos ];
    }// for
    
    return cpu;

    #else

    return "unknown";

    #endif
}

//
// return hostname
//
std::string
hostname ()
{
    char  buf[256];

    gethostname( buf, sizeof(buf) );

    return buf;
}

}}// namespace hlr::mach
