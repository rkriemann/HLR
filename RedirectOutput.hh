#ifndef  __REDIRECTOUTPUT_HH
#define  __REDIRECTOUTPUT_HH
//
// Project     : HLib
// File        : RedirectOutput.hh
// Description : redirects stdout/stderr to file
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <fstream>

//
// redirect stdout/stderr
//
struct RedirectOutput
{
    std::unique_ptr< std::ofstream >  file_out;
    std::streambuf *                  orig_cout;
    std::streambuf *                  orig_cerr ;

    RedirectOutput ( const std::string &  filename )
    {
        orig_cout = std::cout.rdbuf();
        orig_cerr = std::cerr.rdbuf();
        
        file_out  = std::make_unique< std::ofstream >( filename );
        
        std::cout.rdbuf( file_out->rdbuf() );
        std::cerr.rdbuf( file_out->rdbuf() );
    }// if

    ~RedirectOutput ()
    {
        if ( std::cout.rdbuf() != orig_cout ) std::cout.rdbuf( orig_cout );
        if ( std::cerr.rdbuf() != orig_cerr ) std::cerr.rdbuf( orig_cerr );
    }
};

#endif //  __REDIRECTOUTPUT_HH
