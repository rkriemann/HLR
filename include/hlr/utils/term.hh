#ifndef __HLR_UTILS_TERM_HH
#define __HLR_UTILS_TERM_HH
//
// Project     : HLR
// File        : term.hh
// Description : basic functions to handle/modify terminal output
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <ostream>
#include <string>

namespace hlr { namespace term {

//
// change color or attribute of output stream
//
std::ostream &  reset        ( std::ostream &  os );

std::ostream &  black        ( std::ostream &  os );
std::ostream &  red          ( std::ostream &  os );
std::ostream &  green        ( std::ostream &  os );
std::ostream &  yellow       ( std::ostream &  os );
std::ostream &  blue         ( std::ostream &  os );
std::ostream &  magenta      ( std::ostream &  os );
std::ostream &  cyan         ( std::ostream &  os );
std::ostream &  white        ( std::ostream &  os );

std::ostream &  ltgrey       ( std::ostream &  os );
std::ostream &  ltred        ( std::ostream &  os );
std::ostream &  ltgreen      ( std::ostream &  os );
std::ostream &  ltyellow     ( std::ostream &  os );
std::ostream &  ltblue       ( std::ostream &  os );
std::ostream &  ltmagenta    ( std::ostream &  os );
std::ostream &  ltcyan       ( std::ostream &  os );
std::ostream &  ltwhite      ( std::ostream &  os );

std::ostream &  on_black     ( std::ostream &  os );
std::ostream &  on_red       ( std::ostream &  os );
std::ostream &  on_green     ( std::ostream &  os );
std::ostream &  on_yellow    ( std::ostream &  os );
std::ostream &  on_blue      ( std::ostream &  os );
std::ostream &  on_magenta   ( std::ostream &  os );
std::ostream &  on_cyan      ( std::ostream &  os );
std::ostream &  on_white     ( std::ostream &  os );

std::ostream &  on_ltgrey    ( std::ostream &  os );
std::ostream &  on_ltred     ( std::ostream &  os );
std::ostream &  on_ltgreen   ( std::ostream &  os );
std::ostream &  on_ltyellow  ( std::ostream &  os );
std::ostream &  on_ltblue    ( std::ostream &  os );
std::ostream &  on_ltmagenta ( std::ostream &  os );
std::ostream &  on_ltcyan    ( std::ostream &  os );
std::ostream &  on_ltwhite   ( std::ostream &  os );

std::ostream &  bold         ( std::ostream &  os );
std::ostream &  dark         ( std::ostream &  os );
std::ostream &  italic       ( std::ostream &  os );
std::ostream &  underline    ( std::ostream &  os );
std::ostream &  reverse      ( std::ostream &  os );

std::ostream &  alert        ( std::ostream &  os );
std::ostream &  bullet       ( std::ostream &  os );
std::ostream &  dash         ( std::ostream &  os );

//
// modify given string
//
std::string  reset        ( const std::string &  text );

std::string  black        ( const std::string &  text );
std::string  red          ( const std::string &  text );
std::string  green        ( const std::string &  text );
std::string  yellow       ( const std::string &  text );
std::string  blue         ( const std::string &  text );
std::string  magenta      ( const std::string &  text );
std::string  cyan         ( const std::string &  text );
std::string  grey         ( const std::string &  text );

std::string  ltgrey       ( const std::string &  text );
std::string  ltred        ( const std::string &  text );
std::string  ltgreen      ( const std::string &  text );
std::string  ltyellow     ( const std::string &  text );
std::string  ltblue       ( const std::string &  text );
std::string  ltmagenta    ( const std::string &  text );
std::string  ltcyan       ( const std::string &  text );
std::string  white        ( const std::string &  text );

std::string  on_black     ( const std::string &  text );
std::string  on_red       ( const std::string &  text );
std::string  on_green     ( const std::string &  text );
std::string  on_yellow    ( const std::string &  text );
std::string  on_blue      ( const std::string &  text );
std::string  on_magenta   ( const std::string &  text );
std::string  on_cyan      ( const std::string &  text );
std::string  on_grey      ( const std::string &  text );

std::string  on_ltgrey    ( const std::string &  text );
std::string  on_ltred     ( const std::string &  text );
std::string  on_ltgreen   ( const std::string &  text );
std::string  on_ltyellow  ( const std::string &  text );
std::string  on_ltblue    ( const std::string &  text );
std::string  on_ltmagenta ( const std::string &  text );
std::string  on_ltcyan    ( const std::string &  text );
std::string  on_white     ( const std::string &  text );

std::string  bold         ( const std::string &  text );
std::string  dark         ( const std::string &  text );
std::string  italic       ( const std::string &  text );
std::string  underline    ( const std::string &  text );
std::string  reverse      ( const std::string &  text );

//
// return modifier string for terminal color/attribute
//
const char *  reset        ();

const char *  black        ();
const char *  red          ();
const char *  green        ();
const char *  yellow       ();
const char *  blue         ();
const char *  magenta      ();
const char *  cyan         ();
const char *  grey         ();

const char *  ltgrey       ();
const char *  ltred        ();
const char *  ltgreen      ();
const char *  ltyellow     ();
const char *  ltblue       ();
const char *  ltmagenta    ();
const char *  ltcyan       ();
const char *  white        ();

const char *  on_black     ();
const char *  on_red       ();
const char *  on_green     ();
const char *  on_yellow    ();
const char *  on_blue      ();
const char *  on_magenta   ();
const char *  on_cyan      ();
const char *  on_grey      ();

const char *  on_ltgrey    ();
const char *  on_ltred     ();
const char *  on_ltgreen   ();
const char *  on_ltyellow  ();
const char *  on_ltblue    ();
const char *  on_ltmagenta ();
const char *  on_ltcyan    ();
const char *  on_white     ();

const char *  bold         ();
const char *  dark         ();
const char *  italic       ();
const char *  underline    ();
const char *  reverse      ();

}}// namespace hlr::term

#endif  // __HLR_UTILS_TERM_HH
