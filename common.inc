
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

using namespace hlr;

// main function specific to arithmetic
template < typename problem_t >
void
mymain ( int, char ** );

int
hlrmain ( int argc, char ** argv )
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

        if      ( appl == "logkernel"  ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( appl == "matern"     ) mymain< hlr::apps::matern_cov >( argc, argv );
        else if ( appl == "laplaceslp" ) mymain< hlr::apps::laplace_slp >( argc, argv );
        else
            throw "unknown application";

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}

//
// generate accuracy
//
TTruncAcc
gen_accuracy ()
{
    if ( eps < 0 )
        return fixed_rank( k );
    else
        return fixed_prec( eps );
}

// Local Variables:
// mode: c++
// End:
