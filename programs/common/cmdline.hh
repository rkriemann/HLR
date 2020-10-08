#ifndef __HLR_CMDLINE_HH
#define __HLR_CMDLINE_HH

#include <iostream>
#include <string>

using std::string;

#include <boost/format.hpp>
#include <boost/program_options.hpp>

using boost::format;
using namespace boost::program_options;

#include <hpro/base/config.hh>

namespace hpro = HLIB;

namespace hlr { namespace cmdline {

size_t  n          = 1024;         // problem size
size_t  ntile      = 128;          // tile size (nmin)
size_t  nseq       = 0;            // use sequential arithmetic below
size_t  nlvl       = 0;            // number of level, e.g., for Tile-H or MBLR
size_t  k          = 16;           // constant rank
double  eps        = -1;           // constant precision
double  tol        = 0;            // tolerance
string  appl       = "logkernel";  // application
string  distr      = "cyclic2d";   // block distribution
uint    nthreads   = 0;            // number of threads to use (prefer "taskset" or "numactl")
uint    verbosity  = 1;            // verbosity level
bool    noredir    = false;        // prevent stdout redirection in distributed mode
string  gridfile   = "sphere-4";   // gridfile for corresponding applications
string  matrixfile = "";           // use matrix from file instead of application
string  sparsefile = "";           // sparse matrix instead of application
bool    onlydag    = false;        // only compute task graph, no DAG execution
bool    docopy     = true;         // copy matrix before further comp. to distr. memory
bool    levelwise  = false;        // use levelwise task graph construction
bool    oop_lu     = false;        // use out-of-place task graph
bool    fused      = false;        // compute fused DAG for LU and accumulators
bool    nosparsify = false;        // do not sparsify task graph
int     coarse     = 0;            // use coarse sparse graph
int     nbench     = 1;            // perform computations <nbench> times
string  ref        = "";           // reference matrix, algorithm, etc.
auto    kappa      = hpro::complex( 2, 0 ); // wave number for helmholtz problems
string  cluster    = "h";          // clustering technique (h,tlr,mblr,hodlr)
string  adm        = "weak";       // admissibility (std,weak,hodlr)
string  approx     = "default";    // low-rank approximation method (svd,rrqr,randsvd,randlr,aca,lanczos)

void
parse ( int argc, char ** argv )
{
    //
    // define command line options
    //

    options_description  opts( hlr::term::bold( "usage: " ) + argv[0] + " [options]\n\n" +
                               "where " + hlr::term::italic( "options" ) + " include" );
    options_description  gen_opts( hlr::term::bold( "General Options" ) );
    options_description  app_opts( hlr::term::bold( "Application Options" ) );
    options_description  ari_opts( hlr::term::bold( "Arithmetic Options" ) );
    variables_map        vm;

    // standard options
    gen_opts.add_options()
        ( "help,h",                       ": print this help text" )
        ( "noredir",                      ": do not redirect output (MPI only)" )
        ( "threads,t",   value<int>(),    ": number of parallel threads" )
        ( "verbosity,v", value<int>(),    ": verbosity level" )
        ;

    app_opts.add_options()
        ( "adm",         value<string>(), ": admissibility (std,weak,offdiag,hodlr)" )
        ( "app",         value<string>(), ": application type (logkernel,matern,laplaceslp)" )
        ( "cluster",     value<string>(), ": clustering technique (tlr,blr,mblr(-n),tileh,bsp,h)" )
        ( "grid",        value<string>(), ": grid file to use (intern: sphere,sphere2,cube,square)" )
        ( "kappa",       value<double>(), ": wavenumber for Helmholtz problems" )
        ( "matrix",      value<string>(), ": matrix file use" )
        ( "nprob,n",     value<int>(),    ": set problem size" )
        ( "sparse",      value<string>(), ": sparse matrix file use" )
        ;

    ari_opts.add_options()
        ( "accu",                         ": use accumulator arithmetic" )
        ( "approx",      value<string>(), ": LR approximation to use (svd,rrqr,randsvd,randlr,aca,lanczos)" )
        ( "bench",       value<int>(),    ": number of benchmark iterations" )
        ( "coarse",      value<int>(),    ": use coarse DAG for LU" )
        ( "distr",       value<string>(), ": block cluster distribution (cyclic2d,shiftcycrow)" )
        ( "eps,e",       value<double>(), ": set H-algebra precision ε" )
        ( "fused",                        ": compute fused DAG for LU and accumulators" )
        ( "lvl",                          ": do level-wise LU" )
        ( "nlvl",        value<int>(),    ": number of levels, e.g. for Tile-H or MBLR" )
        ( "nseq",        value<int>(),    ": set size of sequential arithmetic" )
        ( "ntile",       value<int>(),    ": set tile size" )
        ( "nocopy",                       ": do not copy matrix before arithmetic" )
        ( "nodag",                        ": do not use DAG in arithmetic" )
        ( "nosparsify",                   ": do not sparsify DAG" )
        ( "onlydag",                      ": only compute DAG but do not execute it" )
        ( "oop",                          ": do out-of-place LU" )
        ( "rank,k",      value<uint>(),   ": set H-algebra rank k" )
        ( "ref",         value<string>(), ": reference matrix or algorithm" )
        ( "tol",         value<double>(), ": tolerance for some algorithms" )
        ;

    opts.add( gen_opts ).add( app_opts ).add( ari_opts );
    
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

    if ( vm.count( "nodag"      ) ) hpro::CFG::Arith::use_dag = false;
    if ( vm.count( "accu"       ) ) hpro::CFG::Arith::use_accu = true;
    if ( vm.count( "approx"     ) ) approx     = vm["approx"].as<string>();
    if ( vm.count( "threads"    ) ) nthreads   = vm["threads"].as<int>();
    if ( vm.count( "verbosity"  ) ) verbosity  = vm["verbosity"].as<int>();
    if ( vm.count( "nprob"      ) ) n          = vm["nprob"].as<int>();
    if ( vm.count( "ntile"      ) ) ntile      = vm["ntile"].as<int>();
    if ( vm.count( "nseq"       ) ) nseq       = vm["nseq"].as<int>();
    if ( vm.count( "nlvl"       ) ) nlvl       = vm["nlvl"].as<int>();
    if ( vm.count( "rank"       ) ) k          = vm["rank"].as<uint>();
    if ( vm.count( "eps"        ) ) eps        = vm["eps"].as<double>();
    if ( vm.count( "tol"        ) ) tol        = vm["tol"].as<double>();
    if ( vm.count( "app"        ) ) appl       = vm["app"].as<string>();
    if ( vm.count( "grid"       ) ) gridfile   = vm["grid"].as<string>();
    if ( vm.count( "matrix"     ) ) matrixfile = vm["matrix"].as<string>();
    if ( vm.count( "sparse"     ) ) sparsefile = vm["sparse"].as<string>();
    if ( vm.count( "distr"      ) ) distr      = vm["distr"].as<string>();
    if ( vm.count( "noredir"    ) ) noredir    = true;
    if ( vm.count( "onlydag"    ) ) onlydag    = true;
    if ( vm.count( "nocopy"     ) ) docopy     = false;
    if ( vm.count( "lvl"        ) ) levelwise  = true;
    if ( vm.count( "oop"        ) ) oop_lu     = true;
    if ( vm.count( "fused"      ) ) fused      = true;
    if ( vm.count( "nosparsify" ) ) nosparsify = true;
    if ( vm.count( "coarse"     ) ) coarse     = vm["coarse"].as<int>();
    if ( vm.count( "bench"      ) ) nbench     = vm["bench"].as<int>();
    if ( vm.count( "ref"        ) ) ref        = vm["ref"].as<string>();
    if ( vm.count( "kappa"      ) ) kappa      = vm["kappa"].as<double>();
    if ( vm.count( "cluster"    ) ) cluster    = vm["cluster"].as<string>();
    if ( vm.count( "adm"        ) ) adm        = vm["adm"].as<string>();

    if ( appl == "help" )
    {
        std::cout << "Applications:" << std::endl
                  << "  - logkernel    : 1D integral equation ∫_[0,1] log |x-y| dx;" << std::endl
                  << "                   n defines number of DoFs" << std::endl
                  << "  - materncov    : Matérn covariance over given number of spatial points;" << std::endl
                  << "                   if grid is defined use grid points, otherwise n random points in 3D" << std::endl
                  << "  - laplaceslp   : 3D integral equation with Laplace SLP and piecewise constant elements" << std::endl
                  << "  - helmholtzslp : 3D integral equation with Helmholz SLP and piecewise constant elements" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( approx == "help" )
    {
        std::cout << "Low-rank approximation methods:" << std::endl
                  << "  - svd     : low-rank singular value decomposition (best approximation)" << std::endl
                  << "  - rrqr    : rank-revealing QR" << std::endl
                  << "  - randsvd : randomized SVD" << std::endl
                  << "  - randlr  : randomized low-rank" << std::endl
                  << "  - aca     : adaptive cross approximation" << std::endl
                  << "  - lanczos : Lanczos bidiagonalization" << std::endl
                  << "  - all     : use all for comparison" << std::endl
                  << "  - default : use default approximation (SVD)" << std::endl;

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
    
    if ( cluster == "help" )
    {
        std::cout << "Clustering Techniques:" << std::endl
                  << "  - tlr/blr : flat clustering, i.e., without hierarchy" << std::endl
                  << "  - mblr    : MBLR clustering with <nlvl> level" << std::endl
                  << "  - tileh   : Tile-H / LatticeH for first level and BSP for rest" << std::endl
                  << "  - bsp/h   : binary space partitioning" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( adm == "help" )
    {
        std::cout << "Clustering Techniques:" << std::endl
                  << "  - std           : standard geometric admissibility min(diam(t),diam(s)) ≤ η dist(t,s)" << std::endl
                  << "  - weak          : weak geometric admissibility" << std::endl
                  << "  - offdiag/hodlr : off-diagonal addmissibility" << std::endl
                  << "  - hilo          : Hi/Low frequency addmissibility" << std::endl
                  << "  - none          : nothing admissible" << std::endl;

        std::exit( 0 );
    }// if
    
    assert( ( appl == "logkernel" ) || ( appl == "materncov" ) || ( appl == "laplaceslp" ) ||
            ( appl == "random"    ) || ( appl == "randcond"  ) || ( appl == "randprob"   ));
    assert( ( distr == "cyclic2d" ) || ( distr == "cyclic1d" ) || ( distr == "shiftcycrow" ) );
    assert( ( approx == "svd"    ) || ( approx == "rrqr"    ) || ( approx == "randsvd" ) ||
            ( approx == "randlr" ) || ( approx == "aca"     ) || ( approx == "lanczos" ) ||
            ( approx == "all"    ) || ( approx == "default" ) );
}

}}// namespace hlr::cmdline

#endif // __HLR_CMDLINE_HH
