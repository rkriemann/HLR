//
// Project     : HLR
// File        : term.cc
// Description : basic functions to handle/modify terminal output
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>

#if ! ( defined(WINDOWS) || defined(_WIN32) || defined(_WIN64) )
#include <unistd.h>
#endif

#include "hlr/utils/term.hh"

namespace hlr
{

namespace term
{

namespace
{

constexpr bool use_colors = false;

//
// return true, if given stream supported color codes
//
bool
supports_color  ( const std::ostream &  os )
{
    #if defined(WINDOWS) || defined(_WIN32) || defined(_WIN64)
    
    return false;
    
    #else

    if ( ! use_colors )
        return false;
    
    //
    // trivial test for cout/cerr
    //
    
    if      ( & os == & std::cout ) return ( ::isatty( fileno( stdout ) ) == 1 );
    else if ( & os == & std::cerr ) return ( ::isatty( fileno( stderr ) ) == 1 );
    else                            return false;
    
    #endif
}

}// namespace anonymous

//
// modify given string
//
std::string  black        ( const std::string &  text ) { return black()        + text + reset(); }
std::string  red          ( const std::string &  text ) { return red()          + text + reset(); }
std::string  green        ( const std::string &  text ) { return green()        + text + reset(); }
std::string  yellow       ( const std::string &  text ) { return yellow()       + text + reset(); }
std::string  blue         ( const std::string &  text ) { return blue()         + text + reset(); }
std::string  magenta      ( const std::string &  text ) { return magenta()      + text + reset(); }
std::string  cyan         ( const std::string &  text ) { return cyan()         + text + reset(); }
std::string  grey         ( const std::string &  text ) { return grey()         + text + reset(); }

std::string  ltgrey       ( const std::string &  text ) { return ltgrey()       + text + reset(); }
std::string  ltred        ( const std::string &  text ) { return ltred()        + text + reset(); }
std::string  ltgreen      ( const std::string &  text ) { return ltgreen()      + text + reset(); }
std::string  ltyellow     ( const std::string &  text ) { return ltyellow()     + text + reset(); }
std::string  ltblue       ( const std::string &  text ) { return ltblue()       + text + reset(); }
std::string  ltmagenta    ( const std::string &  text ) { return ltmagenta()    + text + reset(); }
std::string  ltcyan       ( const std::string &  text ) { return ltcyan()       + text + reset(); }
std::string  white        ( const std::string &  text ) { return white()        + text + reset(); }

std::string  on_black     ( const std::string &  text ) { return on_black()     + text + reset(); }
std::string  on_red       ( const std::string &  text ) { return on_red()       + text + reset(); }
std::string  on_green     ( const std::string &  text ) { return on_green()     + text + reset(); }
std::string  on_yellow    ( const std::string &  text ) { return on_yellow()    + text + reset(); }
std::string  on_blue      ( const std::string &  text ) { return on_blue()      + text + reset(); }
std::string  on_magenta   ( const std::string &  text ) { return on_magenta()   + text + reset(); }
std::string  on_cyan      ( const std::string &  text ) { return on_cyan()      + text + reset(); }
std::string  on_grey      ( const std::string &  text ) { return on_grey()      + text + reset(); }
                                                                            
std::string  on_ltgrey    ( const std::string &  text ) { return on_ltgrey()    + text + reset(); }
std::string  on_ltred     ( const std::string &  text ) { return on_ltred()     + text + reset(); }
std::string  on_ltgreen   ( const std::string &  text ) { return on_ltgreen()   + text + reset(); }
std::string  on_ltyellow  ( const std::string &  text ) { return on_ltyellow()  + text + reset(); }
std::string  on_ltblue    ( const std::string &  text ) { return on_ltblue()    + text + reset(); }
std::string  on_ltmagenta ( const std::string &  text ) { return on_ltmagenta() + text + reset(); }
std::string  on_ltcyan    ( const std::string &  text ) { return on_ltcyan()    + text + reset(); }
std::string  on_white     ( const std::string &  text ) { return on_white()     + text + reset(); }
                                                                            
std::string  bold         ( const std::string &  text ) { return bold()         + text + reset(); }
std::string  dark         ( const std::string &  text ) { return dark()         + text + reset(); }
std::string  underline    ( const std::string &  text ) { return underline()    + text + reset(); }
std::string  reverse      ( const std::string &  text ) { return reverse()      + text + reset(); }

//
// return modifier string for terminal color/attribute
//

#if defined(WINDOWS) || defined(_WIN32) || defined(_WIN64)

const char *  reset        () { return "";  }

const char *  black        () { return ""; }
const char *  red          () { return ""; }
const char *  green        () { return ""; }
const char *  yellow       () { return ""; }
const char *  blue         () { return ""; }
const char *  magenta      () { return ""; }
const char *  cyan         () { return ""; }
const char *  grey         () { return ""; }

const char *  ltgrey       () { return ""; }
const char *  ltred        () { return ""; }
const char *  ltgreen      () { return ""; }
const char *  ltyellow     () { return ""; }
const char *  ltblue       () { return ""; }
const char *  ltmagenta    () { return ""; }
const char *  ltcyan       () { return ""; }
const char *  white        () { return ""; }

const char *  on_black     () { return ""; }
const char *  on_red       () { return ""; }
const char *  on_green     () { return ""; }
const char *  on_yellow    () { return ""; }
const char *  on_blue      () { return ""; }
const char *  on_magenta   () { return ""; }
const char *  on_cyan      () { return ""; }
const char *  on_grey      () { return ""; }

const char *  on_ltgrey    () { return ""; }
const char *  on_ltred     () { return ""; }
const char *  on_ltgreen   () { return ""; }
const char *  on_ltyellow  () { return ""; }
const char *  on_ltblue    () { return ""; }
const char *  on_ltmagenta () { return ""; }
const char *  on_ltcyan    () { return ""; }
const char *  on_white     () { return ""; }

const char *  bold         () { return "";  }
const char *  dark         () { return "";  }
const char *  underline    () { return "";  }
const char *  reverse      () { return "";  }

#else

const char *  reset        () { return ( use_colors ? "\033[0m" : "" ); }

const char *  black        () { return ( use_colors ? "\033[30m" : "" ); }
const char *  red          () { return ( use_colors ? "\033[31m" : "" ); }
const char *  green        () { return ( use_colors ? "\033[32m" : "" ); }
const char *  yellow       () { return ( use_colors ? "\033[33m" : "" ); }
const char *  blue         () { return ( use_colors ? "\033[34m" : "" ); }
const char *  magenta      () { return ( use_colors ? "\033[35m" : "" ); }
const char *  cyan         () { return ( use_colors ? "\033[36m" : "" ); }
const char *  grey         () { return ( use_colors ? "\033[90m" : "" ); }

const char *  ltgrey       () { return ( use_colors ? "\033[37m" : "" ); }
const char *  ltred        () { return ( use_colors ? "\033[91m" : "" ); }
const char *  ltgreen      () { return ( use_colors ? "\033[92m" : "" ); }
const char *  ltyellow     () { return ( use_colors ? "\033[93m" : "" ); }
const char *  ltblue       () { return ( use_colors ? "\033[94m" : "" ); }
const char *  ltmagenta    () { return ( use_colors ? "\033[95m" : "" ); }
const char *  ltcyan       () { return ( use_colors ? "\033[96m" : "" ); }
const char *  white        () { return ( use_colors ? "\033[97m" : "" ); }

const char *  on_black     () { return ( use_colors ? "\033[40m" : "" ); }
const char *  on_red       () { return ( use_colors ? "\033[41m" : "" ); }
const char *  on_green     () { return ( use_colors ? "\033[42m" : "" ); }
const char *  on_yellow    () { return ( use_colors ? "\033[43m" : "" ); }
const char *  on_blue      () { return ( use_colors ? "\033[44m" : "" ); }
const char *  on_magenta   () { return ( use_colors ? "\033[45m" : "" ); }
const char *  on_cyan      () { return ( use_colors ? "\033[46m" : "" ); }
const char *  on_grey      () { return ( use_colors ? "\033[47m" : "" ); }

const char *  on_ltgrey    () { return ( use_colors ? "\033[100m" : "" ); }
const char *  on_ltred     () { return ( use_colors ? "\033[101m" : "" ); }
const char *  on_ltgreen   () { return ( use_colors ? "\033[102m" : "" ); }
const char *  on_ltyellow  () { return ( use_colors ? "\033[103m" : "" ); }
const char *  on_ltblue    () { return ( use_colors ? "\033[104m" : "" ); }
const char *  on_ltmagenta () { return ( use_colors ? "\033[105m" : "" ); }
const char *  on_ltcyan    () { return ( use_colors ? "\033[106m" : "" ); }
const char *  on_white     () { return ( use_colors ? "\033[107m" : "" ); }

const char *  bold         () { return ( use_colors ? "\033[1m" : "" ); }
const char *  dark         () { return ( use_colors ? "\033[2m" : "" ); }
const char *  underline    () { return ( use_colors ? "\033[4m" : "" ); }
const char *  reverse      () { return ( use_colors ? "\033[7m" : "" ); }

#endif

//
// change color or attribute of output stream
//

#define IS_TTY( stream, color )  if ( supports_color( stream ) ) { return stream << color(); } else { return stream; }

std::ostream &  reset        ( std::ostream &  os ) { IS_TTY( os, reset ); }

std::ostream &  black        ( std::ostream &  os ) { IS_TTY( os, black ); }
std::ostream &  red          ( std::ostream &  os ) { IS_TTY( os, red ); }
std::ostream &  green        ( std::ostream &  os ) { IS_TTY( os, green ); }
std::ostream &  yellow       ( std::ostream &  os ) { IS_TTY( os, yellow ); }
std::ostream &  blue         ( std::ostream &  os ) { IS_TTY( os, blue ); }
std::ostream &  magenta      ( std::ostream &  os ) { IS_TTY( os, magenta ); }
std::ostream &  cyan         ( std::ostream &  os ) { IS_TTY( os, cyan ); }
std::ostream &  grey         ( std::ostream &  os ) { IS_TTY( os, grey ); }

std::ostream &  ltgrey       ( std::ostream &  os ) { IS_TTY( os, ltgrey ); }
std::ostream &  ltred        ( std::ostream &  os ) { IS_TTY( os, ltred ); }
std::ostream &  ltgreen      ( std::ostream &  os ) { IS_TTY( os, ltgreen ); }
std::ostream &  ltyellow     ( std::ostream &  os ) { IS_TTY( os, ltyellow ); }
std::ostream &  ltblue       ( std::ostream &  os ) { IS_TTY( os, ltblue ); }
std::ostream &  ltmagenta    ( std::ostream &  os ) { IS_TTY( os, ltmagenta ); }
std::ostream &  ltcyan       ( std::ostream &  os ) { IS_TTY( os, ltcyan ); }
std::ostream &  white        ( std::ostream &  os ) { IS_TTY( os, white ); }

std::ostream &  on_black     ( std::ostream &  os ) { IS_TTY( os, on_black ); }
std::ostream &  on_red       ( std::ostream &  os ) { IS_TTY( os, on_red ); }
std::ostream &  on_green     ( std::ostream &  os ) { IS_TTY( os, on_green ); }
std::ostream &  on_yellow    ( std::ostream &  os ) { IS_TTY( os, on_yellow ); }
std::ostream &  on_blue      ( std::ostream &  os ) { IS_TTY( os, on_blue ); }
std::ostream &  on_magenta   ( std::ostream &  os ) { IS_TTY( os, on_magenta ); }
std::ostream &  on_cyan      ( std::ostream &  os ) { IS_TTY( os, on_cyan ); }
std::ostream &  on_grey      ( std::ostream &  os ) { IS_TTY( os, on_grey ); }

std::ostream &  on_ltgrey    ( std::ostream &  os ) { IS_TTY( os, on_ltgrey ); }
std::ostream &  on_ltred     ( std::ostream &  os ) { IS_TTY( os, on_ltred ); }
std::ostream &  on_ltgreen   ( std::ostream &  os ) { IS_TTY( os, on_ltgreen ); }
std::ostream &  on_ltyellow  ( std::ostream &  os ) { IS_TTY( os, on_ltyellow ); }
std::ostream &  on_ltblue    ( std::ostream &  os ) { IS_TTY( os, on_ltblue ); }
std::ostream &  on_ltmagenta ( std::ostream &  os ) { IS_TTY( os, on_ltmagenta ); }
std::ostream &  on_ltcyan    ( std::ostream &  os ) { IS_TTY( os, on_ltcyan ); }
std::ostream &  on_white     ( std::ostream &  os ) { IS_TTY( os, on_white ); }

std::ostream &  bold         ( std::ostream &  os ) { IS_TTY( os, bold ); }
std::ostream &  dark         ( std::ostream &  os ) { IS_TTY( os, dark ); }
std::ostream &  underline    ( std::ostream &  os ) { IS_TTY( os, underline ); }
std::ostream &  reverse      ( std::ostream &  os ) { IS_TTY( os, reverse ); }

std::ostream &  alert        ( std::ostream &  os )
{
    return os << bold << red << on_white;
}

std::ostream &  bullet       ( std::ostream &  os )
{
    return os << yellow << bold << "∙ " << reset;
}

}// namespace term

}// namespace hlr
