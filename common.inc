
#include <iostream>
#include <string>
#include <algorithm>

using std::string;

#include <boost/format.hpp>

using boost::format;

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

#include "cmdline.hh"
#include "gen_problem.hh"

using namespace hlr;

// return main memory usage as a string
std::string
mem_usage ()
{
    return term::ltgrey( " [" + Mem::to_string( Mem::usage() ) + "]" );
}

// return min/max/median of elements in container
template < typename T > T  min ( const std::vector< T > &  vec ) { return *std::min_element( vec.begin(), vec.end() ); }
template < typename T > T  max ( const std::vector< T > &  vec ) { return *std::max_element( vec.begin(), vec.end() ); }

template < typename T >
T
median ( const std::vector< T > &  vec )
{
    std::vector< T >  tvec( vec );

    std::sort( tvec.begin(), tvec.end() );

    if ( tvec.size() % 2 == 1 )
        return tvec[ tvec.size() / 2 ];
    else
        return T( ( tvec[ tvec.size() / 2 - 1 ] + tvec[ tvec.size() / 2 ] ) / T(2) );
}

// main function specific to arithmetic
template < typename problem_t >
void
mymain ( int, char ** );

int
hlrmain ( int argc, char ** argv )
{
    cmdline::parse( argc, argv );
    
    try
    {
        INIT();

        std::cout << term::bullet << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel"  ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( appl == "materncov"  ) mymain< hlr::apps::matern_cov >( argc, argv );
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
        return fixed_rank( cmdline::k );
    else
        return fixed_prec( cmdline::eps );
}

// Local Variables:
// mode: c++
// End:
