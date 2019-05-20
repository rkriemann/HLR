#ifndef __HLR_TOOLS_HH
#define __HLR_TOOLS_HH
//
// Project     : HLib
// File        : tools.hh
// Description : misc. functions to simplify life
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <sstream>
#include <algorithm>

#include <base/config.hh>

namespace hlr
{

//
// simplifies test if <val> is in <cont>
//
template < typename container_t >
bool
contains ( container_t const &                    cont,
           typename container_t::const_reference  val )
{
    return std::find( cont.begin(), cont.end(), val ) != cont.end();
}

//
// apply for_each to container instead of iterators
//
template < typename container_t,
           typename function_t >
void
for_each ( container_t const &  cont,
           function_t           f )
{
    std::for_each( cont.begin(), cont.end(), f );
}

//
// return first item and remove it from container
//
template <typename T_container >
typename T_container::value_type
behead ( T_container &  acont )
{
    typename T_container::value_type  t = acont.front();

    acont.pop_front();
    
    return t;
}

//
// return string representation of items in container
//
template < template < typename value_t > typename container_t, typename value_t >
std::string
to_string ( container_t< value_t > const &  cont )
{
    std::ostringstream  out;
    size_t              pos  = 0;
    const size_t        size = cont.size();

    for ( auto &&  e : cont )
    {
        out << e;

        if ( ++pos < size )
            out << ",";
    }// for

    return out.str();
}

}// namespace hlr

#endif // __HLR_TOOLS_HH
