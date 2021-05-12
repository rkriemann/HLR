#ifndef __HLR_UTILS_EPS_PRINTER_HH
#define __HLR_UTILS_EPS_PRINTER_HH
//
// Project     : HLR
// Module      : utils/eps_printer
// Description : provides printing class for EPS output
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <ostream>

namespace hlr
{

//
// provides (limited) drawing functions in PostScript format
//
struct eps_printer
{
    //
    // construct printer for given output stream
    //

    eps_printer ( std::ostream &  outstream );

    //
    // drawing methods
    //

    // begin and end drawing
    void begin ( const double  width,
                 const double  height );
    void end   ();

    //
    // save and restore current drawing state
    //
    
    void save    ();
    void restore ();
    
    //
    // transformation methods
    //
    
    // scale output
    void scale        ( const double  x,
                        const double  y );
    
    // translate output
    void translate    ( const double  x,
                        const double  y );
    
    //
    // 2D - drawing
    //

    void draw_line    ( const double  x1,
                        const double  y1,
                        const double  x2,
                        const double  y2 );
    
    void draw_rect    ( const double  x1,
                        const double  y1,
                        const double  x2,
                        const double  y2 );
    
    void fill_rect    ( const double  x1,
                        const double  y1,
                        const double  x2,
                        const double  y2 );

    void draw_text    ( const double           x,
                        const double           y,
                        const std::string &    text,
                        const char             justification = 'l' ); // l/r/c

    //
    // set styles
    //

    void set_gray        ( const int  g );

    void set_rgb         ( const int  r,
                           const int  g,
                           const int  b );

    void set_line_width  ( const double         width );
    
    void set_font        ( const std::string &  font,
                           const double         size );

private:
    // internal output stream
    std::ostream &  _output;
};

}// namespace hlr

#endif  // __HLR_UTILS_EPS_PRINTER_HH
