//
// Project     : HLR
// Module      : utils/eps_printer
// Description : provides printing class for EPS output
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cmath>

#include <hlr/utils/eps_printer.hh>

namespace hlr
{

//
// constructor and destructor
//

eps_printer::eps_printer ( std::ostream &  outstream )
        : _output( outstream )
{}

//
// methods for drawing
//

// begin and end drawing
void
eps_printer::begin ( const double  width,
                    const double  height )
{
    const int  border = 5;
    const int  ofs    = 20;

    // print preamble
    _output << "%!PS-Adobe-3.0 EPSF-3.0" << std::endl
            << "%Title: " << std::endl
            << "%Creator: HLR" << std::endl
            << "%Orientation: Portrait" << std::endl
            << "%%BoundingBox: " << (-border+ofs) << " " << (-border+ofs) << " " << (width+border+ofs) << " " << (height+border+ofs) << std::endl
            << "%Pages: 0" << std::endl
            << "%BeginSetup" << std::endl
            << "%EndSetup" << std::endl
            << "%Magnification: 1.0000" << std::endl
            << "%EndComments" << std::endl
        
            << "/PSPrinterDict 200 dict def" << std::endl
            << "PSPrinterDict begin" << std::endl
        
            << "/BD   { bind def } bind def" << std::endl 
            << "/X    { exch } bind def" << std::endl
            << std::endl
        
            << "/NP   { newpath } bind def" << std::endl 
            << "/M    { moveto } bind def" << std::endl 
            << "/L    { lineto } bind def" << std::endl 
            << "/RL   { rlineto } bind def" << std::endl 
            << "/DA   { NP arc closepath } bind def" << std::endl 
            << "/CP   { closepath } bind def" << std::endl 
            << "/S    { stroke } bind def" << std::endl 
            << "/F    { fill } bind def" << std::endl
            << "/R    { roll } bind def" << std::endl
            << std::endl
        
            << "/DL   { 4 2 roll" << std::endl
            << "        newpath" << std::endl
            << "        moveto lineto" << std::endl
            << "        closepath" << std::endl
            << "        stroke } bind def" << std::endl
            << "/DR   { newpath" << std::endl
            << "        4 2 roll" << std::endl
            << "        moveto exch dup 0 rlineto" << std::endl
            << "        exch 0 exch rlineto" << std::endl
            << "        -1 mul 0 rlineto" << std::endl
            << "        closepath stroke } bind def" << std::endl
            << "/FR   { newpath" << std::endl
            << "        4 2 roll" << std::endl
            << "        moveto exch dup 0 rlineto" << std::endl
            << "        exch 0 exch rlineto" << std::endl
            << "        -1 mul 0 rlineto" << std::endl
            << "        closepath fill } bind def" << std::endl
            << std::endl
        
            << "/SW   { setlinewidth } bind def" << std::endl 
            << "/SRGB { setrgbcolor } bind def" << std::endl 
            << "/SG   { setgray } bind def" << std::endl 
            << "/FF   { findfont } bind def" << std::endl 
            << "/SF   { setfont } bind def" << std::endl 
            << "/SCF  { scalefont } bind def" << std::endl 
            << "/TB   { 3 1 roll" << std::endl
            << "        translate 1 -1 scale" << std::endl
            << "      } bind def" << std::endl
            << "/TL   { gsave" << std::endl
            << "          TB newpath 0 0 moveto show" << std::endl
            << "        grestore } bind def" << std::endl
            << "/TR   { gsave" << std::endl
            << "          TB" << std::endl
            << "          dup stringwidth pop" << std::endl
            << "          newpath neg 0 moveto show" << std::endl
            << "        grestore } bind def" << std::endl
            << "/TC   { gsave" << std::endl
            << "          TB" << std::endl
            << "          dup stringwidth pop 2 div" << std::endl
            << "          newpath neg 0 moveto show" << std::endl
            << "        grestore } bind def" << std::endl

            << "/GS   { gsave } bind def" << std::endl 
            << "/GR   { grestore } bind def" << std::endl 
            << std::endl
        
            << "end" << std::endl
            << "%EndProlog" << std::endl
            << "PSPrinterDict begin" << std::endl
            << "/psprnsavedpage save def" << std::endl;

    //
    // setup additional details
    //
    
    save();
    set_line_width( 1 );
    set_gray( 0 );
    scale( 1, -1 );
    translate( 0, -int(height) );
    translate( ofs, -int(ofs) );
}

void
eps_printer::end ()
{
    restore();
        
    _output << "psprnsavedpage restore" << std::endl
            << "end" << std::endl
            << "showpage" << std::endl
            << "%EOF" << std::endl
            << std::flush;
}

//
// save and restore current state
//
void
eps_printer::save ()
{
    _output << "GS" << std::endl;
}

void
eps_printer::restore ()
{
    _output << "GR" << std::endl;
}
    
//
// transformation methods
//

//
// scale output
//
void
eps_printer::scale ( const double  x,
                     const double  y )
{
    _output << x << " " << y << " scale" << std::endl;
}

//
// translate output
//
void
eps_printer::translate ( const double  x,
                         const double  y )
{
    _output << x << " " << y << " translate" << std::endl;
}

//
// rotate output
//
void
eps_printer::rotate ( const double  angle )
{
    _output << angle << " rotate" << std::endl;
}

//
// 2D - drawing
//

void
eps_printer::draw_line ( const double  x1,
                         const double  y1,
                         const double  x2,
                         const double  y2 )
{
    _output << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << " DL" << std::endl;
}

void
eps_printer::draw_rect ( const double  x1,
                        const double  y1,
                        const double  x2,
                        const double  y2 )
{
    _output << x1 << ' ' << y1 << ' ' << x2-x1 << ' ' << y2-y1 << " DR" << std::endl;
}

void
eps_printer::fill_rect ( const double x1, const double y1, const double x2, const double y2 )
{
    _output << x1 << ' ' << y1 << ' ' << x2-x1 << ' ' << y2-y1 << " FR" << std::endl;
}

void
eps_printer::draw_text ( const double         x,
                         const double         y,
                         const std::string &  text,
                         const char           justification )
{
    _output << x << ' ' << y << " (" << text << ") ";

    switch ( justification )
    {
        default  :
        case 'l' : _output << "TL" << std::endl; break;
        case 'r' : _output << "TR" << std::endl; break;
        case 'c' : _output << "TC" << std::endl; break;
    }// switch
}

//
// set color
//
void
eps_printer::set_gray ( const int g )
{
    _output << std::max( 0, std::min( 255, g ) ) / 256.0 << " SG" << std::endl;
}

void
eps_printer::set_rgb ( const int r, const int g, const int b )
{
    _output << std::max( 0, std::min( 255, r ) ) / 256.0 << ' ' 
            << std::max( 0, std::min( 255, g ) ) / 256.0 << ' '
            << std::max( 0, std::min( 255, b ) ) / 256.0 << ' '
            << "SRGB" << std::endl;
}

void
eps_printer::set_rgb ( const uint rgb )
{
    const auto  r = ( rgb & 0xFF0000 ) >> 16;
    const auto  g = ( rgb & 0x00FF00 ) >> 8;
    const auto  b =   rgb & 0x0000FF;

    set_rgb( r, g, b );
}

void
eps_printer::set_line_width ( const double width )
{
    _output << width << " SW" << std::endl;
}

void
eps_printer::set_font( const std::string & font, const double size )
{
    _output << "/" + font + " FF " << std::max( 0.01, size ) << " SCF SF" << std::endl;
}

}// namespace hlr
