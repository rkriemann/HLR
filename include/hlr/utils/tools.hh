#ifndef __HLR_UTILS_TOOLS_HH
#define __HLR_UTILS_TOOLS_HH
//
// Project     : HLib
// File        : tools.hh
// Description : misc. functions to simplify life
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>

namespace hlr
{

//
// simplifies test if <val> is in <cont>
//
template < typename cont_t,
           typename arg_t >
bool
contains ( const std::vector< cont_t > &  cont,
           const arg_t &                  val )
{
    return std::find( cont.begin(), cont.end(), cont_t( val ) ) != cont.end();
}

template < typename cont_t,
           typename arg_t >
bool
contains ( const std::list< cont_t > &  cont,
           const arg_t &                val )
{
    return std::find( cont.begin(), cont.end(), cont_t( val ) ) != cont.end();
}

template < typename cont_t,
           typename arg_t >
bool
contains ( const std::set< cont_t > &  cont,
           const arg_t &               val )
{
    return cont.find( cont_t( val ) ) != cont.end();
}

template < typename cont_t,
           typename arg_t >
bool
contains ( const std::unordered_set< cont_t > &  cont,
           const arg_t &                         val )
{
    return cont.find( cont_t( val ) ) != cont.end();
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

}// namespace hlr

#endif // __HLR_TOOLS_HH
