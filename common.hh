
#include <iostream>
#include <string>
#include <algorithm>

using std::string;

#include <boost/format.hpp>

using boost::format;

#include <hpro/base/init.hh>
#include <hpro/base/System.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/algebra/mat_norm.hh>
#include <hpro/algebra/mat_conv.hh>
#include <hpro/io/TMatrixIO.hh>
#include <hpro/io/TMatrixVis.hh>
#include <hpro/io/TClusterVis.hh>

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

#include "cmdline.hh"
#include "gen_problem.hh"

// using namespace hlr;

//
// default formating
//

// return main memory usage as a string
inline
std::string
mem_usage ()
{
    return hlr::term::yellow( " [" + hpro::Mem::to_string( hpro::Mem::usage() ) + "]" );
}

// return default formated memory string
inline
std::string
format_mem ( const size_t  m )
{
    return hlr::term::black( hpro::Mem::to_string( m ) ) + mem_usage();
}

// return default formated timing string
std::string
format_time ( const double  t )
{
    return hlr::term::cyan( str( format( "%.3e s" ) % t ) );
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
    return hlr::term::cyan( str( format( "%.3e s" ) % t ) ) + " / " + format_time( ts... );
}

// return default formated error string
inline
std::string
format_error ( const double  e )
{
    return hlr::term::red( str( format( "%.4e s" ) % e ) );
}

//
// timing
//

namespace timer = hpro::Time::Wall;

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
    hlr::cmdline::parse( argc, argv );
    
    try
    {
        hpro::INIT();

        std::cout << hlr::term::bullet << hlr::term::bold << hpro::Mach::hostname() << hlr::term::reset << std::endl
                  << "    CPU cores : " << hpro::Mach::cpuset() << std::endl;
        
        hpro::CFG::set_verbosity( hlr::verbosity );

        if ( hlr::nthreads != 0 )
            hpro::CFG::set_nthreads( hlr::nthreads );

        if      ( hlr::appl == "logkernel"  ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( hlr::appl == "materncov"  ) mymain< hlr::apps::matern_cov >( argc, argv );
        else if ( hlr::appl == "laplaceslp" ) mymain< hlr::apps::laplace_slp >( argc, argv );
        else
            throw "unknown application";

        hpro::DONE();
    }// try
    catch ( char const *   e ) { std::cout << e << std::endl; }
    catch ( hpro::Error &  e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}

//
// generate accuracy
//
inline
hpro::TTruncAcc
gen_accuracy ()
{
    if ( hlr::cmdline::eps < 0 ) return hpro::fixed_rank( hlr::cmdline::k );
    else                         return hpro::fixed_prec( hlr::cmdline::eps );
}

// Local Variables:
// mode: c++
// End:
