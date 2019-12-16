#ifndef __HLR_UTILS_TEXT_HH
#define __HLR_UTILS_TEXT_HH
//
// Project     : HLib
// File        : text.hh
// Description : text functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <string>
#include <sstream>

namespace hlr
{

//
// return <n> in unicode subscript representation
//
std::string  subscript   ( size_t  n );

//
// return <n> in superscript representation
//
std::string  superscript ( size_t  n );

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

#endif  // __HLR_UTILS_TEXT_HH
