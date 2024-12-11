//
// Project     : HLR
// Module      : text.cc
// Description : text functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <string>
#include <sstream>

namespace hlr
{

namespace
{

//
// return smallest power of 10 less than <n>
//
size_t
largest_power10_smaller_than ( const size_t  n )
{
    size_t  p = 1;

    while (( 10*p ) <= n )
        p *= 10;

    return p;
}

}// namespace anonymous

//
// return <n> in subscript representation
//
std::string
subscript ( size_t  n )
{
    static const std::string   subs[] = { "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉" };

    if ( n == 0 )
        return subs[0];

    std::ostringstream  out;
    size_t              pot = largest_power10_smaller_than( n );

    while ( pot > 0 )
    {
        const auto  n_i = n / pot;

        out << subs[ n_i ];

        n    = n - ( n_i * pot );
        pot /= 10;
    }// while

    return out.str();
}

//
// return <n> in superscript representation
//
std::string
superscript ( size_t  n )
{
    static const std::string   sups[] = { "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹" };

    if ( n == 0 )
        return sups[0];

    std::ostringstream  out;
    size_t              pot = largest_power10_smaller_than( n );

    while ( pot > 0 )
    {
        const auto  n_i = n / pot;

        out << sups[ n_i ];

        n    = n - ( n_i * pot );
        pot /= 10;
    }// while

    return out.str();
}

}// namespace hlr
