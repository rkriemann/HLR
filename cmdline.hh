#ifndef __HLR_CMDLINE_HH
#define __HLR_CMDLINE_HH

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

size_t  n          = 1024;
size_t  ntile      = 128;
size_t  k          = 16;
double  eps        = -1;
string  appl       = "logkernel";
string  distr      = "cyclic2d";
uint    nthreads   = 0;
uint    verbosity  = 1;
bool    noredir    = false;
string  gridfile   = "";
string  matrixfile = "";
    
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
        ( "app",         value<string>(), ": application type (logkernel,matern,laplaceslp)" )
        ( "grid",        value<string>(), ": grid file to use (intern: sphere,sphere2,cube,square)" )
        ( "matrix",      value<string>(), ": matrix file use" )
        ( "distr",       value<string>(), ": block cluster distribution (cyclic2d,shiftcycrow)" )
        ( "rank,k",      value<uint>(),   ": set H-algebra rank k" )
        ( "eps,e",       value<double>(), ": set H-algebra precision ε" )
        ( "accu",                         ": use accumulator arithmetic" )
        ( "threads,t",   value<int>(),    ": number of parallel threads" )
        ( "verbosity,v", value<int>(),    ": verbosity level" )
        ( "noredir",                      ": do not redirect output (MPI only)" )
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

    if ( vm.count( "nodag"     ) ) HLIB::CFG::Arith::use_dag = false;
    if ( vm.count( "accu"      ) ) HLIB::CFG::Arith::use_accu = true;
    if ( vm.count( "threads"   ) ) nthreads   = vm["threads"].as<int>();
    if ( vm.count( "verbosity" ) ) verbosity  = vm["verbosity"].as<int>();
    if ( vm.count( "nprob"     ) ) n          = vm["nprob"].as<int>();
    if ( vm.count( "ntile"     ) ) ntile      = vm["ntile"].as<int>();
    if ( vm.count( "rank"      ) ) k          = vm["rank"].as<uint>();
    if ( vm.count( "eps"       ) ) eps        = vm["eps"].as<double>();
    if ( vm.count( "app"       ) ) appl       = vm["app"].as<string>();
    if ( vm.count( "grid"      ) ) gridfile   = vm["grid"].as<string>();
    if ( vm.count( "matrix"    ) ) matrixfile = vm["matrix"].as<string>();
    if ( vm.count( "distr"     ) ) distr      = vm["distr"].as<string>();
    if ( vm.count( "noredir"   ) ) noredir    = true;

    assert( ( appl == "logkernel" ) || ( appl == "materncov" ) || ( appl == "laplaceslp" ) );
    assert( ( distr == "cyclic2d" ) || ( distr == "shiftcycrow" ) );
}

#endif // __HLR_CMDLINE_HH

// Local Variables:
// mode: c++
// End:
