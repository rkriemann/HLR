
#include <iostream>
#include <string>

using std::string;

#include <boost/format.hpp>

using boost::format;

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

#include "cmdline.hh"
#include "gen_problem.hh"

using namespace HLR;

// main function specific to arithmetic
template < typename problem_t >
void
mymain ( int, char ** );

int
main ( int argc, char ** argv )
{
    parse_cmdline( argc, argv );
    
    try
    {
        INIT();

        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel"  ) mymain< HLR::Apps::LogKernel >( argc, argv );
        else if ( appl == "matern"     ) mymain< HLR::Apps::MaternCov >( argc, argv );
        else if ( appl == "laplaceslp" ) mymain< HLR::Apps::LaplaceSLP >( argc, argv );
        else
            throw "unknown application";

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}

// Local Variables:
// mode: c++
// End:
