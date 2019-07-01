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
bool    onlydag    = false;
bool    docopy     = true;
bool    levelwise  = false;
int     coarse     = 0;
int     nbench     = 1;

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
        ( "onlydag",                      ": only compute DAG but do not execute it" )
        ( "nocopy",                       ": do not copy matrix before arithmetic" )
        ( "lvlwise",                      ": do level-wise LU" )
        ( "coarse",      value<int>(),    ": use coarse DAG for LU" )
        ( "bench",       value<int>(),    ": number of benchmark iterations" )
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
    if ( vm.count( "onlydag"   ) ) onlydag    = true;
    if ( vm.count( "nocopy"    ) ) docopy     = false;
    if ( vm.count( "lvlwise"   ) ) levelwise  = true;
    if ( vm.count( "coarse"    ) ) coarse     = vm["coarse"].as<int>();
    if ( vm.count( "bench"     ) ) nbench     = vm["bench"].as<int>();

    if ( appl == "help" )
    {
        std::cout << "Applications:" << std::endl
                  << "  - logkernel  : 1D integral equation over [0,1] with log |x-y| kernel;" << std::endl
                  << "                 n defines number of DoFs" << std::endl
                  << "  - materncov  : Matérn covariance over given number of spatial points;" << std::endl
                  << "                 if grid is defined, use grid points otherwise n random points in 3D" << std::endl
                  << "  - laplaceslp : 3D integral equation with Laplace SLP and piecewise constant elements" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( gridfile == "help" )
    {
        std::cout << "Internal Grids:" << std::endl
                  << "  - sphere, sphere2 : spherical grid with radius 1 (different initial grid)" << std::endl
                  << "  - cube            : unit cube" << std::endl
                  << "  - square          : unit square in 3D" << std::endl
                  << std::endl
                  << "Refinement level l is defined by appending \"-l\", e.g. sphere-3" << std::endl
                  << std::endl
                  << "If grid is not an internal grid, it is assumed to be a file name" << std::endl;

        std::exit( 0 );
    }// if
    
    assert( ( appl == "logkernel" ) || ( appl == "materncov" ) || ( appl == "laplaceslp" ) );
    assert( ( distr == "cyclic2d" ) || ( distr == "shiftcycrow" ) );
}

#endif // __HLR_CMDLINE_HH

// Local Variables:
// mode: c++
// End:
