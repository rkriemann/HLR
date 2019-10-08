
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

//
// default formating
//

// return main memory usage as a string
inline
std::string
mem_usage ()
{
    return term::yellow( " [" + Mem::to_string( Mem::usage() ) + "]" );
}

// return default formated memory string
inline
std::string
format_mem ( const size_t  m )
{
    return term::black( HLIB::Mem::to_string( m ) ) + mem_usage();
}

// return default formated timing string
std::string
format_time ( const double  t )
{
    return term::cyan( str( format( "%.3e s" ) % t ) );
}

template < typename duration_t >
std::string
format_time ( const duration_t  t )
{
    return format_time( t.seconds() );
}

template < typename... T >
std::string
format_time ( const double  t, const T... ts )
{
    return term::cyan( str( format( "%.3e s" ) % t ) ) + " / " + format_time( ts... );
}

// return default formated error string
inline
std::string
format_error ( const double  e )
{
    return term::red( str( format( "%.4e s" ) % e ) );
}

//
// return min/max/median of elements in container
//

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

//
// main function specific to arithmetic
//

template < typename problem_t >
void
mymain ( int, char ** );

inline
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
inline
TTruncAcc
gen_accuracy ()
{
    if ( eps < 0 ) return fixed_rank( cmdline::k );
    else           return fixed_prec( cmdline::eps );
}

// Local Variables:
// mode: c++
// End:
