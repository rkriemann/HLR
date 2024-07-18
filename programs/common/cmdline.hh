#ifndef __HLR_CMDLINE_HH
#define __HLR_CMDLINE_HH

#include <iostream>
#include <string>
#include <filesystem>

using std::string;

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using boost::format;
using namespace boost::program_options;

#include <hpro/base/config.hh>

namespace hpro = HLIB;

#include <hlr/utils/log.hh>

namespace hlr { namespace cmdline {

size_t  n          = 1024;         // problem size
size_t  ntile      = 128;          // tile size (nmin)
size_t  nseq       = 0;            // use sequential arithmetic below
size_t  nlvl       = 0;            // number of level, e.g., for Tile-H or MBLR
size_t  k          = 16;           // constant rank
double  eps        = 1e-4;         // constant precision
double  tol        = 0;            // tolerance
string  appl       = "logkernel";  // application
string  kernel     = "newton";     // (radial) kernel to use
string  distr      = "cyclic2d";   // block distribution
uint    nthreads   = 0;            // number of threads to use (prefer "taskset" or "numactl")
uint    verbosity  = 1;            // verbosity level
bool    noredir    = false;        // prevent stdout redirection in distributed mode
string  gridfile   = "sphere-4";   // gridfile for corresponding applications
string  matrixfile = "";           // use matrix from file instead of application
string  sparsefile = "";           // sparse matrix instead of application
string  datafile   = "";           // general data file for different programs
bool    onlydag    = false;        // only compute task graph, no DAG execution
bool    docopy     = true;         // copy matrix before further comp. to distr. memory
bool    levelwise  = false;        // use levelwise task graph construction
bool    ip_lu      = false;        // use in-place task graph
bool    oop_lu     = false;        // use out-of-place task graph
bool    fused      = false;        // compute fused DAG for LU and accumulators
bool    nosparsify = false;        // do not sparsify task graph
bool    coarsen    = false;        // coarsen matrix structure
bool    tohodlr    = false;        // convert matrix to HODLR format
int     coarse     = 0;            // use coarse sparse graph
uint    nbench     = 1;            // perform computations <nbench> times (at most)
double  tbench     = 1;            // minimal time for benchmark runs
string  ref        = "";           // reference matrix, algorithm, etc.
string  cluster    = "h";          // clustering technique (h,tlr,mblr,hodlr)
string  adm        = "strong";     // admissibility (strong,vertex,weak,offdiagonal)
double  eta        = 2.0;          // admissibility parameter
string  capprox    = "default";    // construction low-rank approximation method (aca,hca,dense)
string  aapprox    = "default";    // arithmetic low-rank approximation method (svd,rrqr,randsvd,randlr,aca,lanczos)
string  tapprox    = "default";    // tensor low-rank approximation method (hosvd,sthosvd,ghosvd,hhosvd,tcafull)
string  arith      = "std";        // which kind of arithmetic to use
double  compress   = 0;            // apply SZ/ZFP compression with rate (1…, ZFP only) or accuracy (0,1] (0 = off)
auto    kappa      = std::complex< double >( 2, 0 ); // wave number for helmholtz problems
double  sigma      = 1;            // parameter for matern covariance and gaussian kernel

void
read_config ( const std::string &  filename )
{
    if ( ! std::filesystem::exists( filename ) )
    {
        HLR_ERROR( "file not found : " + filename );
        return;
    }// if
    
    auto  cfg = boost::property_tree::ptree();
    
    boost::property_tree::ini_parser::read_ini( filename, cfg );

    appl       = cfg.get( "app.appl",    appl );
    kernel     = cfg.get( "app.kernel",  kernel );
    n          = cfg.get( "app.n",       n );
    adm        = cfg.get( "app.adm",     adm );
    eta        = cfg.get( "app.eta",     eta );
    cluster    = cfg.get( "app.cluster", cluster );
    gridfile   = cfg.get( "app.grid",    gridfile );
    kappa      = cfg.get( "app.kappa",   kappa );
    sigma      = cfg.get( "app.sigma",   sigma );
    matrixfile = cfg.get( "app.matrix",  matrixfile );
    sparsefile = cfg.get( "app.sparse",  sparsefile );
    datafile   = cfg.get( "app.data",    datafile );
        
    ntile      = cfg.get( "arith.ntile",  ntile );
    nbench     = cfg.get( "arith.nbench", nbench );
    tbench     = cfg.get( "arith.tbench", tbench );
    eps        = cfg.get( "arith.eps",    eps );
    k          = cfg.get( "arith.rank",   k );
        
    nthreads   = cfg.get( "misc.nthreads",  nthreads );
    verbosity  = cfg.get( "misc.verbosity", verbosity );
}

void
parse ( int argc, char ** argv )
{
    //
    // read from config file
    //

    if ( std::filesystem::exists( ".config" ) )
        read_config( ".config" );
    
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
        ( "config",      value<string>(), ": config file to read from" )
        ( "help,h",                       ": print this help text" )
        ( "noredir",                      ": do not redirect output (MPI only)" )
        ( "threads,t",   value<int>(),    ": number of parallel threads" )
        ( "verbosity,v", value<int>(),    ": verbosity level" )
        ;

    app_opts.add_options()
        ( "adm",         value<string>(), ": admissibility (strong,vertex,weak,offdiag)" )
        ( "app",         value<string>(), ": application type (logkernel,matern,laplaceslp,helmholtzslp,exp)" )
        ( "kernel",      value<string>(), ": kernel to use (newton,log,exp,...)" )
        ( "cluster",     value<string>(), ": clustering technique (tlr,blr,mblr(-n),tileh,bsp,h,sfc)" )
        ( "data",        value<string>(), ": data file to use" )
        ( "eta",         value<double>(), ": admissibility parameter for \"std\" and \"weak\"" )
        ( "grid",        value<string>(), ": grid file to use (intern: sphere,sphere2,cube,square)" )
        ( "kappa",       value<double>(), ": wavenumber for Helmholtz problems" )
        ( "sigma",       value<double>(), ": parameter σ for Matérn and Gaussian" )
        ( "matrix",      value<string>(), ": matrix file to use" )
        ( "nprob,n",     value<int>(),    ": set problem size" )
        ( "sparse",      value<string>(), ": sparse matrix file to use" )
        ( "compress",    value<double>(), ": apply SZ/ZFP compression with rate [1,…; ZFP only) or accuracy (0,1) (default: 0 = off)" )
        ( "coarsen",                      ": coarsen matrix structure" )
        ( "tohodlr",                      ": convert matrix to HODLR format" )
        ( "capprox",     value<string>(), ": LR approximation to use (aca,hca,dense)" )
        ;

    ari_opts.add_options()
        ( "accu",                         ": use accumulator arithmetic" )
        ( "aapprox",     value<string>(), ": LR approximation to use (svd,rrqr,randsvd,randlr,aca,lanczos)" )
        ( "tapprox",     value<string>(), ": tensor LR approximation to use (hosvd,sthosvd,ghosvd,hhosvd,tcafull)" )
        ( "arith",       value<string>(), ": which arithmetic to use (hpro, std, accu, lazy, all)" )
        ( "nbench",      value<uint>(),   ": (maximal) number of benchmark iterations" )
        ( "tbench",      value<double>(), ": minimal time for benchmark loop" )
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
        ( "ip",                           ": do in-place LU" )
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

    // first read from config file
    if ( vm.count( "config" ) )
        read_config( vm["config"].as<string>() );
    
    if ( vm.count( "nodag"      ) ) hpro::CFG::Arith::use_dag = false;
    if ( vm.count( "accu"       ) ) hpro::CFG::Arith::use_accu = true;
    if ( vm.count( "capprox"    ) ) capprox    = vm["capprox"].as<string>();
    if ( vm.count( "aapprox"    ) ) aapprox    = vm["aapprox"].as<string>();
    if ( vm.count( "tapprox"    ) ) tapprox    = vm["tapprox"].as<string>();
    if ( vm.count( "arith"      ) ) arith      = vm["arith"].as<string>();
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
    if ( vm.count( "kernel"     ) ) kernel     = vm["kernel"].as<string>();
    if ( vm.count( "grid"       ) ) gridfile   = vm["grid"].as<string>();
    if ( vm.count( "matrix"     ) ) matrixfile = vm["matrix"].as<string>();
    if ( vm.count( "sparse"     ) ) sparsefile = vm["sparse"].as<string>();
    if ( vm.count( "data"       ) ) datafile   = vm["data"].as<string>();
    if ( vm.count( "distr"      ) ) distr      = vm["distr"].as<string>();
    if ( vm.count( "noredir"    ) ) noredir    = true;
    if ( vm.count( "onlydag"    ) ) onlydag    = true;
    if ( vm.count( "nocopy"     ) ) docopy     = false;
    if ( vm.count( "lvl"        ) ) levelwise  = true;
    if ( vm.count( "ip"         ) ) ip_lu      = true;
    if ( vm.count( "oop"        ) ) oop_lu     = true;
    if ( vm.count( "fused"      ) ) fused      = true;
    if ( vm.count( "nosparsify" ) ) nosparsify = true;
    if ( vm.count( "coarsen"    ) ) coarsen    = true;
    if ( vm.count( "tohodlr"    ) ) tohodlr    = true;
    if ( vm.count( "coarse"     ) ) coarse     = vm["coarse"].as<int>();
    if ( vm.count( "nbench"     ) ) nbench     = vm["nbench"].as<uint>();
    if ( vm.count( "tbench"     ) ) tbench     = vm["tbench"].as<double>();
    if ( vm.count( "ref"        ) ) ref        = vm["ref"].as<string>();
    if ( vm.count( "kappa"      ) ) kappa      = vm["kappa"].as<double>();
    if ( vm.count( "sigma"      ) ) sigma      = vm["sigma"].as<double>();
    if ( vm.count( "cluster"    ) ) cluster    = vm["cluster"].as<string>();
    if ( vm.count( "adm"        ) ) adm        = vm["adm"].as<string>();
    if ( vm.count( "eta"        ) ) eta        = vm["eta"].as<double>();
    if ( vm.count( "compress"   ) ) compress   = vm["compress"].as<double>();

    if ( appl == "help" )
    {
        std::cout << "Applications:" << std::endl
                  << "  - logkernel    : 1D integral equation ∫_[0,1] log |x-y| dx;" << std::endl
                  << "                   n defines number of DoFs" << std::endl
                  << "  - materncov    : Matérn covariance over given number of spatial points;" << std::endl
                  << "                   if grid is defined use grid points, otherwise n random points in 3D" << std::endl
                  << "  - laplaceslp   : 3D integral equation with Laplace SLP and piecewise constant elements" << std::endl
                  << "  - helmholtzslp : 3D integral equation with Helmholz SLP and piecewise constant elements" << std::endl
                  << "  - exp          : 3D integral equation with exponential kernel and piecewise constant elements" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( capprox == "help" )
    {
        std::cout << "Construction low-rank approximation methods:" << std::endl
                  << "  - aca     : adaptive cross approximation" << std::endl
                  << "  - hca     : hybrid cross approximation" << std::endl
                  << "  - dense   : no approximation" << std::endl
                  << "  - default : use default approximation (ACA)" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( aapprox == "help" )
    {
        std::cout << "Arithmetic low-rank approximation methods:" << std::endl
                  << "  - svd     : low-rank singular value decomposition (best approximation)" << std::endl
                  << "  - rrqr    : rank-revealing QR" << std::endl
                  << "  - randsvd : randomized SVD" << std::endl
                  << "  - randlr  : randomized low-rank" << std::endl
                  << "  - aca     : adaptive cross approximation" << std::endl
                  << "  - lanczos : Lanczos bidiagonalization" << std::endl
                  << "  - hpro    : purely use Hpro functions for approximation" << std::endl
                  << "  - all     : use all for comparison" << std::endl
                  << "  - default : use default approximation (SVD)" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( tapprox == "help" )
    {
        std::cout << "Tensor Low-rank approximation methods:" << std::endl
                  << "  - hosvd   : standard HOSVD" << std::endl
                  << "  - sthosvd : sequentially truncated HOSVD" << std::endl
                  << "  - ghosvd  : greedy HOSVD" << std::endl
                  << "  - hhosvd  : hierarchical HOSVD" << std::endl
                  << "  - tcafull : full tensor cross approximation" << std::endl
                  << "  - default : use default approximation (HHOSVD)" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( arith == "help" )
    {
        std::cout << "Arithmetic to use:" << std::endl
                  << "  - (dag)std  : standard H-arithmetic (immediate updates)" << std::endl
                  << "  - (dag)accu : accumulator based H-arithmetic" << std::endl
                  << "  - (dag)lazy : lazy evaluation in H-arithmetic" << std::endl
                  << "  - all       : use all types" << std::endl
                  << "  - default   : use default arithmetic (std)" << std::endl;

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
                  << "  - bsp/h   : binary space partitioning" << std::endl
                  << "  - sfc     : Hilber curve base partitioning (optional: -binary/-blr)" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( adm == "help" )
    {
        std::cout << "Admissibility Conditions:" << std::endl
                  << "  - strong/std    : standard geometric admissibility (min(diam(t),diam(s)) ≤ η dist(t,s))" << std::endl
                  << "  - vertex        : vertex geometric admissibility" << std::endl
                  << "  - weak          : weak geometric admissibility (dist(t,s) > 0)" << std::endl
                  << "  - offdiag/hodlr : off-diagonal addmissibility (t ≠ s)"<< std::endl
                  << "  - hilo          : Hi/Low frequency addmissibility" << std::endl
                  << "  - none          : nothing admissible" << std::endl;

        std::exit( 0 );
    }// if
    
    if ( ! ( ( appl == "logkernel" )    ||
             ( appl == "log" )          ||
             ( appl == "laplaceslp" )   ||
             ( appl == "helmholtzslp" ) ||
             ( appl == "exp" )          ||
             ( appl == "materncov" )    ||  
             ( appl == "gaussian" ) ) )
        HLR_ERROR( "unknown application : " + appl );
    
    if ( ! ( ( distr == "cyclic2d" ) ||
             ( distr == "cyclic1d" ) ||
             ( distr == "shiftcycrow" ) ) )
        HLR_ERROR( "unknown distribution : " + distr );
    
    if ( ! ( ( capprox == "aca" ) || ( capprox == "dense"   ) ||
             ( capprox == "hca" ) || ( capprox == "default" ) ) )
        HLR_ERROR( "unknown construction LR approximation : " + capprox );

    if ( ! ( ( aapprox == "svd"    ) || ( aapprox == "rrqr" ) || ( aapprox == "randsvd" ) ||
             ( aapprox == "randlr" ) || ( aapprox == "aca"  ) || ( aapprox == "lanczos" ) ||
             ( aapprox == "hpro"   ) || ( aapprox == "all"  ) || ( aapprox == "default" ) ) )
        HLR_ERROR( "unknown arithmetic LR approximation : " + aapprox );

    for ( auto &  entry : split( cmdline::tapprox, "," ) )
        if ( ! ( ( entry == "hosvd"  ) || ( entry == "sthosvd" ) || ( entry == "ghosvd" ) ||
                 ( entry == "hhosvd" ) || ( entry == "tcafull" ) || ( entry == "default" ) ) )
            HLR_ERROR( "unknown tensor approximation : " + entry );

    if ( ! ( ( arith == "std" )  || ( arith == "dagstd" )  ||
             ( arith == "accu" ) || ( arith == "dagaccu" ) ||
             ( arith == "lazy" ) || ( arith == "daglazy" ) ||
             ( arith == "all" )  || ( arith == "default" ) ) )
        HLR_ERROR( "unknown arithmetic : " + arith );
}

}}// namespace hlr::cmdline

#endif // __HLR_CMDLINE_HH
