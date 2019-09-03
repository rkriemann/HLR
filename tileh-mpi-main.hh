
#include "hlr/mpi/mpi.hh"
#include "hlr/cluster/distr.hh"
#include "hlr/cluster/tileh.hh"
#include "cmdline.hh"
#include "gen_problem.hh"
#include "hlr/utils/RedirectOutput.hh"
#include "hlr/utils/compare.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/tbb/dag.hh"
#include "hlr/tbb/matrix.hh"

using namespace hlr;

// return main memory usage as a string
std::string
mem_usage ()
{
    return term::ltgrey( " [" + Mem::to_string( Mem::usage() ) + "]" );
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

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();

    auto  tic     = Time::Wall::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tileh::cluster( coord.get(), ntile, 4 ); // std::max< uint >( 3, std::log2( nprocs )+2 ) );
    auto  bct     = cluster::tileh::blockcluster( ct.get(), ct.get() );

    // assign blocks to nodes
    if      ( distr == "cyclic2d"    ) cluster::distribution::cyclic_2d( nprocs, bct->root() );
    else if ( distr == "cyclic1d"    ) cluster::distribution::cyclic_1d( nprocs, bct->root() );
    else if ( distr == "shiftcycrow" ) cluster::distribution::shifted_cyclic_1d( nprocs, bct->root() );
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "bct_distr" );
    }// if
    
    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = mpi::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    mem   = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "A_%03d", pid ) );
    }// if

    // TLR::MPI::RANK = k;

    // no sparsification
    hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
    
    {
        std::cout << term::bullet << term::bold << "LU ( Tile-H MPI, "
                  << impl_name << ", "
                  << acc.to_string() << " )" << term::reset << std::endl;
        
        auto  C = std::shared_ptr( hlr::tbb::matrix::copy( *A ) );
        
        tic = Time::Wall::now();

        if ( nprocs == 1 )
        {
            auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, hlr::tbb::dag::refine ) );

            hlr::tbb::dag::run( dag, acc );
        }// if
        else
        {
            impl::lu< HLIB::real >( C.get(), acc );
        }// else
        
        toc = Time::Wall::since( tic );
        
        std::cout << "    done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset() << std::endl;
        std::cout << "    mem    = " << Mem::to_string( C->byte_size() ) << mem_usage() << std::endl;

        if ( nprocs == 1 )
        {
            write_matrix( C.get(), "LU.hm" );
            
            matrix::luinv_eval  A_inv( C, hlr::tbb::dag::refine, hlr::tbb::dag::run );
            // TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
            std::cout << "    error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv )
                      << term::reset << std::endl;
        }// if
        else
        {
            // compare with otherwise computed result
            std::cout << "    error  = ";
            compare_ref_file( C.get(), "LU.hm", ( eps != -1 ? eps : 1e-10 ) );
        }// else
        
    }
}

int
main ( int argc, char ** argv )
{
    // init MPI before anything else
    mpi::environment   env{ argc, argv };
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    // redirect output for all except proc 0
    std::unique_ptr< RedirectOutput >  redir_out = ( pid != 0
                                                     ? std::make_unique< RedirectOutput >( to_string( "tileh-mpi_%03d.out", pid ) )
                                                     : nullptr );

    parse_cmdline( argc, argv );
    
    try
    {
        INIT();

        // adjust HLIB network data
        NET::set_nprocs( nprocs );
        NET::set_pid( pid );
    
        std::cout << term::bullet << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel"  ) mymain< hlr::apps::log_kernel  >( argc, argv );
        else if ( appl == "matern"     ) mymain< hlr::apps::matern_cov  >( argc, argv );
        else if ( appl == "laplaceslp" ) mymain< hlr::apps::laplace_slp >( argc, argv );
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
