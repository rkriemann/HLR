#ifndef __HLR_CMDLINE_INC
#define __HLR_CMDLINE_INC

#include <iostream>
#include <string>

using std::string;

#include <boost/format.hpp>
#include <boost/program_options.hpp>

using boost::format;
using namespace boost::program_options;

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

size_t  n         = 512;
size_t  ntile     = 64;
size_t  k         = 5;
string  appl      = "logkernel";
uint    nthreads  = 0;
uint    verbosity = 1;

void
parse_cmdline ( int argc, char ** argv )
{
    //
    // define command line options
    //

    options_description  opts( string( "usage: " ) + argv[0] + " [options]\n  where options include" );
    variables_map        vm;

    // standard options
    opts.add_options()
        ( "help,h",                       ": print this help text" )
        ( "ntile",       value<int>(),    ": set tile size" )
        ( "nprob,n",     value<int>(),    ": set problem size" )
        ( "nodag",                        ": do not use DAG in arithmetic" )
        ( "app",         value<string>(), ": application type (logkernel,matern)" )
        ( "rank,k",      value<uint>(),   ": set H-algebra rank k" )
        ( "threads,t",   value<int>(),    ": number of parallel threads" )
        ( "verbosity,v", value<int>(),    ": verbosity level" )
        ;

    //
    // parse command line options
    //

    try
    {
        store( command_line_parser( argc, argv ).options( opts ).run(), vm );
        notify( vm );
    }// try
    catch ( required_option &  e )
    {
        std::cout << e.get_option_name() << " requires an argument, try \"-h\"" << std::endl;
        exit( 1 );
    }// catch
    catch ( unknown_option &  e )
    {
        std::cout << e.what() << ", try \"-h\"" << std::endl;
        exit( 1 );
    }// catch

    //
    // eval command line options
    //

    if ( vm.count( "help") )
    {
        std::cout << opts << std::endl;
        exit( 1 );
    }// if

    if ( vm.count( "threads"   ) ) nthreads  = vm["threads"].as<int>();
    if ( vm.count( "verbosity" ) ) verbosity = vm["verbosity"].as<int>();
    if ( vm.count( "nodag"     ) ) HLIB::CFG::Arith::use_dag = false;
    if ( vm.count( "nprob"     ) ) n         = vm["nprob"].as<int>();
    if ( vm.count( "ntile"     ) ) ntile     = vm["ntile"].as<int>();
    if ( vm.count( "rank"      ) ) k         = vm["rank"].as<uint>();
    if ( vm.count( "app"       ) ) appl      = vm["app"].as<string>();
}

#endif // __HLR_CMDLINE_INC

// Local Variables:
// mode: c++
// End:
