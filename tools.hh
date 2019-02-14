#ifndef __HLR_TOOLS_HH
#define __HLR_TOOLS_HH

//
// simplifies test if <val> is in <cont>
//
template < typename container_t >
bool
contains ( container_t const &                    cont,
           typename container_t::const_reference  val )
{
    for ( const auto &  c : cont )
    {
        if ( c == val )
            return true;
    }// for

    return false;
    
    return std::find( cont.begin(), cont.end(), val ) != cont.end();
}

template < template < typename value_t > typename container_t, typename value_t >
std::string
to_string ( container_t< value_t > const &  cont )
{
    ostringstream  out;

    for ( auto &&  e : cont )
        out << e << ",";

    return out.str();
}

#endif // __HLR_TOOLS_HH
